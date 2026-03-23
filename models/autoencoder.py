# models/autoencoder.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Joint Autoencoder — Full Secure E2E Communication System

Wires together:
    Encoder → Legitimate Channel → Decoder  (Alice → Bob)
                    ↓
              Eavesdropper Channel           (Alice → Eve)

During training, both paths are active simultaneously.
The joint loss drives:
    - Bob   → correct decoding (minimize cross-entropy)
    - Eve   → random guessing  (maximize entropy)

During evaluation, only the legitimate path is used.
"""

import torch
import torch.nn as nn
from config import M, LEGIT_SNR_DB
from device import DEVICE, dtype
from models.encoder import Encoder
from models.decoder import Decoder
from channel import (
    LegitimateChannel,
    EavesdropperChannel,
    AWGNOnlyChannel
)


class SecureAutoencoder(nn.Module):
    """
    Full Secure End-to-End Communication System.

    Encapsulates the complete Alice-Bob-Eve setup from the paper.
    A single forward pass returns both Bob's and Eve's received
    signals, enabling joint loss computation.
    """

    def __init__(self, snr_db: float = LEGIT_SNR_DB):
        super().__init__()

        # Core trainable components
        self.encoder = Encoder()
        self.decoder = Decoder()

        # Channel models (not trainable — simulation only)
        self.legit_channel = LegitimateChannel(snr_db=snr_db)
        self.eve_channel   = EavesdropperChannel(snr_db=snr_db)

        # Current SNR setting
        self.snr_db = snr_db

    def forward(self, s: torch.Tensor):
        """
        Full forward pass through Alice → Bob and Alice → Eve.

        Args:
            s : Message indices [batch]  (integers in 0..M-1)

        Returns:
            x           : Transmitted signal        [batch, 2n]
            y           : Bob's received signal     [batch, 2n]
            z           : Eve's received signal     [batch, 2n]
            bob_probs   : Bob's symbol probs        [batch, M]
            eve_probs   : Eve's symbol probs        [batch, M]
        """
        # Encoder: message → transmitted signal
        x = self.encoder(s)                     # [batch, 2n]

        # Channels: apply impairments
        y = self.legit_channel(x)               # Bob's received signal
        z = self.eve_channel(x)                 # Eve's received signal

        # Decoder: received signal → symbol probabilities
        bob_probs = self.decoder(y)             # [batch, M]
        eve_probs = self.decoder(z)             # [batch, M]

        return x, y, z, bob_probs, eve_probs

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        """
        Inference only — legitimate path (Bob) only.
        Used during evaluation.

        Args:
            s : Message indices [batch]

        Returns:
            Predicted message indices [batch]
        """
        with torch.no_grad():
            x = self.encoder(s)
            y = self.legit_channel(x)
            return self.decoder.predict(y)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Return raw transmitted signal for a given message."""
        with torch.no_grad():
            return self.encoder(s)

    def set_snr(self, snr_db: float):
        """
        Dynamically update SNR for both channels.
        Used during BER evaluation sweeps.
        """
        self.snr_db = snr_db
        self.legit_channel.set_snr(snr_db)
        self.eve_channel.set_snr(snr_db)

    def get_trainable_params(self):
        """
        Return only encoder + decoder parameters.
        Channel models have no trainable parameters.
        """
        return list(self.encoder.parameters()) + \
               list(self.decoder.parameters())

    def count_parameters(self) -> dict:
        """Parameter count breakdown for reporting."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder"   : enc_params,
            "decoder"   : dec_params,
            "total"     : enc_params + dec_params
        }


# ── SANITY TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("   autoencoder.py — Sanity Tests")
    print("=" * 55)

    # Instantiate full system
    model = SecureAutoencoder(snr_db=LEGIT_SNR_DB).to(DEVICE)

    # Parameter counts
    param_counts = model.count_parameters()
    print(f"\n  Parameter Breakdown:")
    print(f"  Encoder params       : {param_counts['encoder']:>10,}")
    print(f"  Decoder params       : {param_counts['decoder']:>10,}")
    print(f"  Total params         : {param_counts['total']:>10,}")

    # Full forward pass
    batch_size = 256
    messages   = torch.randint(0, M, (batch_size,)).to(DEVICE)

    x, y, z, bob_probs, eve_probs = model(messages)

    print(f"\n  [Forward Pass Shapes]")
    print(f"  Input messages       : {messages.shape}")
    print(f"  Transmitted x        : {x.shape}      ← encoder output")
    print(f"  Bob received y       : {y.shape}      ← legitimate channel")
    print(f"  Eve received z       : {z.shape}      ← eavesdropper channel")
    print(f"  Bob probabilities    : {bob_probs.shape}   ← decoder output")
    print(f"  Eve probabilities    : {eve_probs.shape}   ← decoder output")

    # Verify untrained system is near-random (expected before training)
    bob_preds      = torch.argmax(bob_probs, dim=1)
    eve_preds      = torch.argmax(eve_probs, dim=1)
    bob_acc_before = (bob_preds == messages).float().mean().item()
    eve_acc_before = (eve_preds == messages).float().mean().item()

    print(f"\n  [Untrained System Accuracy — Both Should Be ~{1/M:.3f}]")
    print(f"  Bob accuracy         : {bob_acc_before:.4f}  (random chance = {1/M:.3f})")
    print(f"  Eve accuracy         : {eve_acc_before:.4f}  (random chance = {1/M:.3f})")

    # Verify SNR update works
    model.set_snr(10.0)
    print(f"\n  SNR update test      : set to 10.0 dB")
    print(f"  Legit channel SNR    : {model.legit_channel.snr_db} dB ✓")
    print(f"  Eve channel SNR      : {model.eve_channel.snr_db} dB ✓")

    # Verify predict() method
    preds = model.predict(messages)
    print(f"\n  predict() output     : {preds.shape}  ✓")

    # Gradient flow check — critical for training
    print(f"\n  [Gradient Flow Check]")
    model.set_snr(LEGIT_SNR_DB)
    messages2     = torch.randint(0, M, (batch_size,)).to(DEVICE)
    _, _, _, b_p, e_p = model(messages2)

    dummy_loss = b_p.sum() + e_p.sum()
    dummy_loss.backward()

    enc_grad = next(iter(model.encoder.parameters())).grad
    dec_grad = next(iter(model.decoder.parameters())).grad

    enc_grad_ok = enc_grad is not None and not torch.isnan(enc_grad).any()
    dec_grad_ok = dec_grad is not None and not torch.isnan(dec_grad).any()

    print(f"  Encoder gradients    : {'✅ flowing' if enc_grad_ok else '❌ FAILED'}")
    print(f"  Decoder gradients    : {'✅ flowing' if dec_grad_ok else '❌ FAILED'}")

    all_ok = all([
        x.shape == (batch_size, 4),
        bob_probs.shape == (batch_size, M),
        eve_probs.shape == (batch_size, M),
        enc_grad_ok,
        dec_grad_ok
    ])

    print(f"\n  {'✅  autoencoder.py passed all sanity tests.' if all_ok else '❌  Some checks failed.'}\n")
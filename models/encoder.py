# models/encoder.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Encoder (Transmitter / Alice)

Architecture:
    Message s (integer) 
    → Embedding(M, 256)
    → Dense(256, 256, ReLU)
    → Dense(256, 2n)
    → Normalization (unit power constraint)
    → Transmitted signal x ∈ R^(2n)

The normalization layer is physically critical — it enforces
the power constraint so the transmitted signal has unit average power.
"""

import torch
import torch.nn as nn
import numpy as np
from config import M, n, ENC_EMBEDDING_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM
from device import DEVICE, dtype


class NormalizationLayer(nn.Module):
    """
    Enforces unit average power constraint on transmitted signal.

    Physical meaning: the transmitter cannot exceed a power budget.
    Without this, the encoder could 'cheat' by amplifying signals
    to arbitrary power levels to overcome channel noise.

    Normalization: x_norm = x / sqrt(mean(||x||^2)) * sqrt(n)
    This ensures E[||x||^2] = n (n = number of channel uses)
    """

    def __init__(self, n_channel_uses: int = n):
        super().__init__()
        self.n = n_channel_uses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS power across batch
        # x shape: [batch, 2n]
        power = torch.sqrt(torch.mean(x ** 2))
        # Normalize to unit power, scale by sqrt(n)
        return x / (power + 1e-8) * np.sqrt(self.n)


class Encoder(nn.Module):
    """
    End-to-End Autoencoder Encoder (Transmitter).

    Takes an integer message index, embeds it, and outputs
    a normalized complex signal representation for transmission.
    """

    def __init__(self):
        super().__init__()

        # Embedding: maps integer message index to dense vector
        # Equivalent to one-hot + linear, but more efficient
        self.embedding = nn.Embedding(M, ENC_EMBEDDING_DIM)

        # Dense layers with ReLU activation
        self.dense1 = nn.Linear(ENC_EMBEDDING_DIM, ENC_HIDDEN_DIM)
        self.dense2 = nn.Linear(ENC_HIDDEN_DIM, ENC_OUTPUT_DIM)

        # Activations
        self.relu = nn.ReLU()

        # Power normalization — must be last
        self.normalize = NormalizationLayer(n)

        # Initialize weights cleanly
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier initialization for dense layers.
        Critical for the paper's 'symbol-level fingerprint' concept —
        different random initializations produce different encoders,
        which is the basis of the authentication scheme.
        """
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.zeros_(self.dense2.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: message index → normalized transmitted signal.

        Args:
            s : Integer message indices [batch] (values in 0..M-1)

        Returns:
            x : Normalized transmitted signal [batch, 2n]
        """
        # Embed message index to dense vector
        x = self.embedding(s)               # [batch, 256]

        # Dense layers
        x = self.relu(self.dense1(x))       # [batch, 256]
        x = self.dense2(x)                  # [batch, 2n=4]

        # Enforce power constraint
        x = self.normalize(x)               # [batch, 2n=4]

        return x

    def get_constellation(self) -> torch.Tensor:
        """
        Generate full constellation — all M symbol representations.
        Used for visualization and authentication fingerprinting.

        Returns:
            constellation : [M, 2n] — one point per message symbol
        """
        all_messages = torch.arange(M).to(DEVICE)
        with torch.no_grad():
            constellation = self.forward(all_messages)
        return constellation


# ── SANITY TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 50)
    print("   encoder.py — Sanity Tests")
    print("=" * 50)

    # Instantiate encoder on correct device
    encoder = Encoder().to(DEVICE)

    # Print architecture
    print(f"\n  Architecture:")
    print(f"  {encoder}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n  Total parameters     : {total_params:,}")

    # Test forward pass
    batch_size = 256
    messages   = torch.randint(0, M, (batch_size,)).to(DEVICE)
    x          = encoder(messages)

    print(f"\n  Input  shape         : {messages.shape}  (message indices)")
    print(f"  Output shape         : {x.shape}  (transmitted signal)")
    print(f"  Output device        : {x.device}")

    # Verify power constraint
    mean_power = (x ** 2).mean().item()
    print(f"\n  Mean signal power    : {mean_power:.4f}  (expect ~{n/2:.1f})")

    # Verify all M symbols produce distinct outputs
    constellation = encoder.get_constellation()
    print(f"\n  Constellation shape  : {constellation.shape}")
    print(f"  (All {M} symbols mapped to {constellation.shape[1]}D space)")

    # Check no two symbols are identical
    diffs = []
    for i in range(M):
        for j in range(i+1, M):
            diff = (constellation[i] - constellation[j]).norm().item()
            diffs.append(diff)
    min_dist = min(diffs)
    print(f"  Min inter-symbol dist: {min_dist:.4f}  (must be > 0)")

    # Quick constellation plot
    const_np = constellation.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(const_np[:, 0], const_np[:, 1], s=100, c='blue', zorder=5)
    for i in range(M):
        plt.annotate(str(i), (const_np[i, 0], const_np[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.title("Initial Constellation (Before Training)")
    plt.xlabel("I (In-phase)")
    plt.ylabel("Q (Quadrature)")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/constellation_initial.png", dpi=150)
    plt.close()
    print(f"\n  Constellation plot saved to results/plots/")

    if min_dist > 0:
        print("\n  ✅  encoder.py passed all sanity tests.\n")
    else:
        print("\n  ❌  FAIL: Two symbols mapped to same point.\n")
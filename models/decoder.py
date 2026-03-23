# models/decoder.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Decoder (Receiver / Bob)

Architecture:
    Received signal y ∈ R^(2n)
    → Dense(2n, 256, ReLU)
    → Dense(256, 256, ReLU)
    → Dense(256, M, Softmax)
    → Estimated message ŝ

The decoder includes a synchronization sub-network (Phase Estimator +
Feature Extractor) as described in the paper's Fig.1, which compensates
for dynamic channel impairments before final decoding.
"""

import torch
import torch.nn as nn
import numpy as np
from config import (
    M, n, ENC_OUTPUT_DIM,
    DEC_HIDDEN_DIM, DEC_OUTPUT_DIM,
    SYNC_HIDDEN_DIM
)
from device import DEVICE, dtype


class PhaseEstimator(nn.Module):
    """
    Synchronization sub-network — Phase Estimator.

    Estimates the phase offset introduced by the channel
    and outputs a phase correction signal.
    Mirrors the paper's Fig.1 synchronization block.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENC_OUTPUT_DIM, SYNC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),        # Add dropout — prevents over-generalising to Eve's channel
            nn.Linear(SYNC_HIDDEN_DIM, 2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate and correct phase offset.

        Args:
            y : Received signal [batch, 2n]

        Returns:
            y_corrected : Phase-corrected signal [batch, 2n]
        """
        # Estimate phase components
        phase_est = self.net(y)             # [batch, 2]
        cos_est   = phase_est[:, 0:1]       # [batch, 1]
        sin_est   = phase_est[:, 1:2]       # [batch, 1]

        # Normalize to unit circle for valid rotation
        norm      = torch.sqrt(cos_est**2 + sin_est**2 + 1e-8)
        cos_est   = cos_est / norm
        sin_est   = sin_est / norm

        # Apply inverse phase rotation to compensate
        I = y[:, 0::2]   # In-phase components
        Q = y[:, 1::2]   # Quadrature components

        I_corr =  I * cos_est + Q * sin_est
        Q_corr = -I * sin_est + Q * cos_est

        # Reconstruct corrected signal
        y_corr = torch.zeros_like(y)
        y_corr[:, 0::2] = I_corr
        y_corr[:, 1::2] = Q_corr

        return y_corr


class FeatureExtractor(nn.Module):
    """
    Feature Extractor sub-network.

    Extracts robust features from the received signal before
    passing to the main decoder. Mirrors the paper's CNN-based
    feature extractor (Fig.1). We use a lightweight dense version
    for software simulation.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENC_OUTPUT_DIM, SYNC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SYNC_HIDDEN_DIM, SYNC_HIDDEN_DIM),
            nn.ReLU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)          # [batch, SYNC_HIDDEN_DIM]


class Decoder(nn.Module):
    """
    End-to-End Autoencoder Decoder (Receiver / Bob).

    Full pipeline:
        Received y
        → Phase Estimator (sync correction)
        → Feature Extractor
        → Concatenate [phase-corrected y, features]
        → Dense(ReLU) × 2
        → Dense(Softmax) → class probabilities over M symbols
    """

    def __init__(self):
        super().__init__()

        # Synchronization sub-networks
        self.phase_estimator   = PhaseEstimator()
        self.feature_extractor = FeatureExtractor()

        # After concatenation: phase-corrected y (2n) + features (256)
        concat_dim = ENC_OUTPUT_DIM + SYNC_HIDDEN_DIM  # 4 + 256 = 260

        # Main decoder dense layers
        self.dense1  = nn.Linear(concat_dim,    DEC_HIDDEN_DIM)
        self.dense2  = nn.Linear(DEC_HIDDEN_DIM, DEC_HIDDEN_DIM)
        self.out     = nn.Linear(DEC_HIDDEN_DIM, DEC_OUTPUT_DIM)

        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.dense1, self.dense2, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: received signal → symbol probabilities.

        Args:
            y      : Received signal [batch, 2n]

        Returns:
            probs  : Symbol probabilities [batch, M]
                     probs[i][j] = P(message j | received y_i)
        """
        # Synchronization: estimate and correct phase
        y_corr   = self.phase_estimator(y)      # [batch, 2n]

        # Feature extraction
        features = self.feature_extractor(y)    # [batch, 256]

        # Concatenate corrected signal with extracted features
        combined = torch.cat([y_corr, features], dim=1)  # [batch, 260]

        # Dense classification layers
        out = self.relu(self.dense1(combined))  # [batch, 256]
        out = self.relu(self.dense2(out))       # [batch, 256]
        out = self.softmax(self.out(out))       # [batch, M=16]

        return out

    def predict(self, y: torch.Tensor) -> torch.Tensor:
        """
        Hard decision: return the most likely message index.

        Args:
            y : Received signal [batch, 2n]

        Returns:
            predicted message indices [batch]
        """
        probs = self.forward(y)
        return torch.argmax(probs, dim=1)


# ── SANITY TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 50)
    print("   decoder.py — Sanity Tests")
    print("=" * 50)

    # Instantiate decoder
    decoder = Decoder().to(DEVICE)

    # Print architecture summary
    print(f"\n  Architecture:")
    print(f"  {decoder}\n")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters     : {total_params:,}")

    # Test forward pass with dummy received signal
    batch_size = 256
    y_test     = torch.randn(batch_size, ENC_OUTPUT_DIM,
                              dtype=dtype()).to(DEVICE)

    probs      = decoder(y_test)
    preds      = decoder.predict(y_test)

    print(f"\n  Input shape          : {y_test.shape}  (received signal)")
    print(f"  Output shape         : {probs.shape}  (symbol probabilities)")
    print(f"  Predictions shape    : {preds.shape}  (message indices)")
    print(f"  Output device        : {probs.device}")

    # Verify probabilities sum to 1
    prob_sums  = probs.sum(dim=1)
    max_dev    = (prob_sums - 1.0).abs().max().item()
    print(f"\n  Prob sum deviation   : {max_dev:.6f}  (expect < 1e-5)")

    # Verify output is valid probability distribution
    min_prob   = probs.min().item()
    max_prob   = probs.max().item()
    print(f"  Prob range           : [{min_prob:.4f}, {max_prob:.4f}]"
          f"  (expect [0, 1])")

    # Before training, should predict roughly uniformly (1/M per class)
    pred_counts = torch.bincount(preds, minlength=M).float()
    uniformity  = (pred_counts / batch_size)
    print(f"\n  Prediction spread    : min={uniformity.min():.3f}"
          f"  max={uniformity.max():.3f}")
    print(f"  (Expect roughly uniform ~{1/M:.3f} before training)")

    # Test full encoder → channel → decoder pipeline
    print(f"\n  [Pipeline Test: Encoder → Channel → Decoder]")
    from models.encoder import Encoder
    from channel import LegitimateChannel, EavesdropperChannel

    encoder  = Encoder().to(DEVICE)
    legit_ch = LegitimateChannel()
    eve_ch   = EavesdropperChannel()

    messages = torch.randint(0, M, (batch_size,)).to(DEVICE)
    x        = encoder(messages)
    y        = legit_ch(x)
    z        = eve_ch(x)

    legit_probs = decoder(y)
    eve_probs   = decoder(z)

    print(f"  Messages shape       : {messages.shape}")
    print(f"  Encoded x shape      : {x.shape}")
    print(f"  Legit received y     : {y.shape}")
    print(f"  Eve received z       : {z.shape}")
    print(f"  Legit probs shape    : {legit_probs.shape}")
    print(f"  Eve probs shape      : {eve_probs.shape}")

    checks = [
        max_dev < 1e-4,
        min_prob >= 0.0,
        max_prob <= 1.0,
        probs.shape == (batch_size, M),
        legit_probs.shape == (batch_size, M)
    ]

    if all(checks):
        print(f"\n  ✅  decoder.py passed all sanity tests.\n")
    else:
        print(f"\n  ❌  Some checks failed. Review output above.\n")
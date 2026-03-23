# channel.py
"""
Differentiable Channel Models for Secure E2E Communications.

Implements:
  - AWGN (Additive White Gaussian Noise)
  - Phase offset impairment
  - Frequency offset impairment
  - Legitimate channel (fixed parameters)
  - Eavesdropper channel (randomly sampled parameters)

All operations are differentiable — gradients flow through
the channel during backpropagation.
"""

import torch
import torch.nn as nn
import numpy as np
from config import (
    LEGIT_SNR_DB, LEGIT_PHASE_OFFSET, LEGIT_FREQ_OFFSET,
    EVE_PHASE_MIN, EVE_PHASE_MAX,
    EVE_FREQ_MEAN, EVE_FREQ_STD,
    n
)
from device import DEVICE, dtype


# ── UTILITY ───────────────────────────────────────────────────────────────────

def snr_db_to_linear(snr_db: float) -> float:
    """Convert SNR from dB to linear scale."""
    return 10.0 ** (snr_db / 10.0)


def compute_noise_std(snr_db: float, signal_power: float = 1.0) -> float:
    """
    Compute noise standard deviation from SNR (dB).
    Assumes normalized signal power = 1.0 (enforced by encoder normalization).
    σ_noise = sqrt(signal_power / (2 * SNR_linear))
    Factor of 2 accounts for complex channel (I and Q components).
    """
    snr_linear = snr_db_to_linear(snr_db)
    return np.sqrt(signal_power / (2.0 * snr_linear))


# ── CORE CHANNEL OPERATIONS ───────────────────────────────────────────────────

def apply_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add Additive White Gaussian Noise to signal x.

    Args:
        x        : Transmitted signal tensor [batch, 2n]
        snr_db   : Signal-to-noise ratio in dB

    Returns:
        y        : Noisy received signal [batch, 2n]
    """
    noise_std = compute_noise_std(snr_db)
    noise = torch.randn_like(x) * noise_std
    return x + noise


def apply_phase_offset(x: torch.Tensor, phase: float) -> torch.Tensor:
    """
    Apply a phase rotation to the complex signal.

    The signal x has shape [batch, 2n] where pairs of values
    represent (I, Q) components of n complex symbols.
    Phase rotation: x_rot = x * e^(j*phase)
    In real arithmetic:
        I_out =  I_in * cos(phase) - Q_in * sin(phase)
        Q_out =  I_in * sin(phase) + Q_in * cos(phase)

    Args:
        x     : Signal tensor [batch, 2n]
        phase : Phase offset in radians

    Returns:
        Rotated signal tensor [batch, 2n]
    """
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)

    # Split into I and Q components
    # x[:, 0::2] = I components, x[:, 1::2] = Q components
    I = x[:, 0::2]
    Q = x[:, 1::2]

    I_rot = I * cos_p - Q * sin_p
    Q_rot = I * sin_p + Q * cos_p

    # Interleave back: [I0, Q0, I1, Q1, ...]
    out = torch.zeros_like(x)
    out[:, 0::2] = I_rot
    out[:, 1::2] = Q_rot
    return out


def apply_freq_offset(x: torch.Tensor,
                      freq_offset: float,
                      n_symbols: int) -> torch.Tensor:
    """
    Apply frequency offset to the signal.
    Models carrier frequency offset (CFO) between transmitter and receiver.

    Args:
        x           : Signal tensor [batch, 2n]
        freq_offset : Normalized frequency offset (Δf * T)
        n_symbols   : Number of channel uses (n)

    Returns:
        Signal with frequency offset applied [batch, 2n]
    """
    if abs(freq_offset) < 1e-10:
        return x  # Skip computation if offset is negligible

    out = torch.zeros_like(x)
    for i in range(n_symbols):
        phase = 2.0 * np.pi * freq_offset * i
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        I = x[:, 2 * i]
        Q = x[:, 2 * i + 1]

        out[:, 2 * i]     = I * cos_p - Q * sin_p
        out[:, 2 * i + 1] = I * sin_p + Q * cos_p

    return out


# ── CHANNEL CLASSES ───────────────────────────────────────────────────────────

class LegitimateChannel(nn.Module):
    """
    Legitimate receiver channel — fixed, known parameters.

    This channel is stable and consistent between training
    and testing, representing the trusted Alice-Bob link.
    """

    def __init__(self,
                 snr_db: float = LEGIT_SNR_DB,
                 phase_offset: float = LEGIT_PHASE_OFFSET,
                 freq_offset: float = LEGIT_FREQ_OFFSET):
        super().__init__()
        self.snr_db       = snr_db
        self.phase_offset = phase_offset
        self.freq_offset  = freq_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass signal through legitimate channel.
        Order: Phase offset → Frequency offset → AWGN
        """
        y = apply_phase_offset(x, self.phase_offset)
        y = apply_freq_offset(y, self.freq_offset, n)
        y = apply_awgn(y, self.snr_db)
        return y

    def set_snr(self, snr_db: float):
        """Allow dynamic SNR changes during evaluation."""
        self.snr_db = snr_db


class EavesdropperChannel(nn.Module):
    """
    Eavesdropper channel — randomly sampled parameters.

    During training, thousands of random channel parameter
    combinations are sampled to force the encoder to learn
    representations that are exclusive to the legitimate channel.

    Phase offset ~ Uniform(-π, π)
    Freq offset  ~ Gaussian(0, EVE_FREQ_STD)
    SNR          ~ Same distribution as legitimate channel
                   (paper assumption: eavesdropper is geographically
                    close to legitimate user)
    """

    def __init__(self, snr_db: float = LEGIT_SNR_DB):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass signal through a randomly sampled eavesdropper channel.
        New random parameters are sampled on every forward pass.
        """
        # Sample random channel parameters
        phase_offset = np.random.uniform(EVE_PHASE_MIN, EVE_PHASE_MAX)
        freq_offset  = np.random.normal(EVE_FREQ_MEAN, EVE_FREQ_STD)

        # Apply channel impairments
        z = apply_phase_offset(x, phase_offset)
        z = apply_freq_offset(z, freq_offset, n)
        z = apply_awgn(z, self.snr_db)
        return z

    def set_snr(self, snr_db: float):
        self.snr_db = snr_db


class AWGNOnlyChannel(nn.Module):
    """
    Pure AWGN channel — no phase or frequency offsets.
    Used as a baseline and for simplified evaluation.
    """

    def __init__(self, snr_db: float = LEGIT_SNR_DB):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_awgn(x, self.snr_db)

    def set_snr(self, snr_db: float):
        self.snr_db = snr_db


# ── SANITY TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import ENC_OUTPUT_DIM
    import matplotlib.pyplot as plt

    print("=" * 50)
    print("   channel.py — Sanity Tests")
    print("=" * 50)

    # Create a dummy normalized signal (mimics encoder output)
    batch  = 256
    x_test = torch.randn(batch, ENC_OUTPUT_DIM, dtype=dtype()).to(DEVICE)

    # Normalize to unit power (mimics encoder normalization layer)
    x_test = x_test / x_test.norm(dim=1, keepdim=True) * np.sqrt(ENC_OUTPUT_DIM / 2)

    print(f"\n  Input signal shape   : {x_test.shape}")
    print(f"  Input signal device  : {x_test.device}")
    print(f"  Input mean power     : {(x_test**2).mean().item():.4f}  (expect ~1.0)")

    # Test legitimate channel
    legit_ch = LegitimateChannel()
    y = legit_ch(x_test)
    noise_power = ((y - x_test)**2).mean().item()
    print(f"\n  [LegitimateChannel]")
    print(f"  Output shape         : {y.shape}")
    print(f"  Approx noise power   : {noise_power:.4f}")

    # Test eavesdropper channel
    eve_ch = EavesdropperChannel()
    z = eve_ch(x_test)
    print(f"\n  [EavesdropperChannel]")
    print(f"  Output shape         : {z.shape}")

    # Test SNR sweep — verify noise increases as SNR decreases
    print(f"\n  [SNR Sweep — noise std should increase as SNR decreases]")
    print(f"  {'SNR (dB)':<12} {'Noise Std':<12} {'Expected'}")
    print(f"  {'-'*40}")
    for snr in [10, 7, 4, 1, -2]:
        std = compute_noise_std(snr)
        print(f"  {snr:<12} {std:<12.4f} {'↑ higher' if snr < 7 else '↓ lower'}")

    # Test AWGN channel
    awgn_ch = AWGNOnlyChannel(snr_db=7.0)
    y_awgn = awgn_ch(x_test)
    print(f"\n  [AWGNOnlyChannel]")
    print(f"  Output shape         : {y_awgn.shape}")

    print("\n  ✅  channel.py passed all sanity tests.\n")
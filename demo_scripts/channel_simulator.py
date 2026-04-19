#!/usr/bin/env python3
"""
Channel Simulator for Demo Video
==================================

This script simulates realistic wireless channel effects WITHOUT actual SDR hardware.
It reads tx_symbols.txt and generates rx_symbols.txt with realistic:
- Phase offsets
- Frequency offsets  
- AWGN noise
- Power scaling

Two modes:
1. LEGITIMATE (Bob): Fixed known channel → Bob decodes correctly
2. EAVESDROPPER (Eve): Random unknown channel → Eve fails to decode
"""

import numpy as np
import argparse
import sys
from pathlib import Path


# Make project root importable so demo channel stays consistent with training config.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LEGIT_PHASE_OFFSET as TRAIN_LEGIT_PHASE_OFFSET,
    LEGIT_FREQ_OFFSET as TRAIN_LEGIT_FREQ_OFFSET,
    LEGIT_SNR_DB as TRAIN_LEGIT_SNR_DB,
    n as TRAIN_CHANNEL_USES,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CHANNEL PARAMETERS (matching your config.py and channel.py)
# ══════════════════════════════════════════════════════════════════════════════

# Legitimate Channel (Bob): exactly match training-time constants
LEGIT_PHASE_OFFSET = TRAIN_LEGIT_PHASE_OFFSET
LEGIT_FREQ_OFFSET = TRAIN_LEGIT_FREQ_OFFSET
LEGIT_SNR_DB = TRAIN_LEGIT_SNR_DB

# Eavesdropper Channel (Eve): slight mismatch around Bob channel
EVE_PHASE_JITTER = 0.55            # +/- radians around Bob's fixed phase
EVE_FREQ_JITTER_STD = 0.08         # Gaussian jitter around Bob's fixed CFO
EVE_SNR_DB = LEGIT_SNR_DB - 3.5    # Slightly lower SNR than Bob

# File paths
TX_FILE = "tx_symbols.txt"
RX_FILE = "rx_symbols.txt"


# ══════════════════════════════════════════════════════════════════════════════
#  CHANNEL SIMULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def read_symbols(filename: str) -> np.ndarray:
    """
    Read complex symbols from file.
    
    Format: Each line is "real,imag"
    Returns: Complex array
    """
    symbols = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            real_part = float(parts[0])
            imag_part = float(parts[1])
            symbols.append(real_part + 1j * imag_part)
    
    return np.array(symbols)


def write_symbols(filename: str, symbols: np.ndarray):
    """
    Write complex symbols to file.
    
    Format: Each line is "real,imag"
    """
    with open(filename, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol.real:.6f},{symbol.imag:.6f}\n")


def apply_phase_offset(symbols: np.ndarray, phase_rad: float) -> np.ndarray:
    """Apply phase rotation to symbols."""
    return symbols * np.exp(1j * phase_rad)


def apply_frequency_offset(symbols: np.ndarray, freq_offset: float) -> np.ndarray:
    """
    Apply frequency offset (phase drift over time).
    
    Phase increases linearly: φ(t) = 2π * freq_offset * t
    """
    n_symbols = len(symbols)
    phase_drift = 2 * np.pi * freq_offset * np.arange(n_symbols)
    return symbols * np.exp(1j * phase_drift)


def apply_channel_like_training(symbols: np.ndarray,
                                phase_offset: float,
                                freq_offset: float,
                                snr_db: float,
                                block_size: int = TRAIN_CHANNEL_USES) -> np.ndarray:
    """
    Apply channel impairments in the same order/shape pattern as training.

    Training applies phase/frequency/noise over one codeword of n complex uses.
    For repeated demo symbols, we therefore apply the same channel per n-symbol
    block instead of drifting across the entire repeated stream.
    """
    out_blocks = []
    total = len(symbols)

    for start in range(0, total, block_size):
        block = symbols[start:start + block_size]

        # Same order as channel.py: phase -> frequency -> AWGN
        block = apply_phase_offset(block, phase_offset)
        block = apply_frequency_offset(block, freq_offset)
        block = add_awgn(block, snr_db)
        out_blocks.append(block)

    return np.concatenate(out_blocks) if out_blocks else symbols


def add_awgn(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise.
    
    SNR = 10 * log10(P_signal / P_noise)
    """
    # Match training assumption from channel.py: fixed signal_power=1.0
    # when converting SNR to additive noise variance.
    signal_power = 1.0

    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex (I and Q)
    noise_real = np.random.normal(0, noise_std, len(symbols))
    noise_imag = np.random.normal(0, noise_std, len(symbols))
    noise = noise_real + 1j * noise_imag
    
    return symbols + noise


def simulate_legitimate_channel(tx_symbols: np.ndarray, 
                                snr_db: float = LEGIT_SNR_DB) -> np.ndarray:
    """
    Simulate the LEGITIMATE channel (Bob).
    
    Fixed known parameters:
    - Phase offset: 0.5 rad
    - Frequency offset: 0.02
    - High SNR
    
    Returns: Received symbols that Bob can decode correctly
    """
    return apply_channel_like_training(
        tx_symbols,
        phase_offset=LEGIT_PHASE_OFFSET,
        freq_offset=LEGIT_FREQ_OFFSET,
        snr_db=snr_db,
    )


def simulate_eavesdropper_channel(tx_symbols: np.ndarray,
                                  snr_db: float = EVE_SNR_DB,
                                  seed: int = None) -> np.ndarray:
    """
    Simulate the EAVESDROPPER channel (Eve).
    
    Random unknown parameters:
    - Phase offset: Uniform(-π, π)
    - Frequency offset: Gaussian(0, 0.15)
    - Same SNR but different channel
    
    Returns: Received symbols that Eve cannot decode (different channel)
    """
    if seed is not None:
        np.random.seed(seed)
    
    phase_offset = LEGIT_PHASE_OFFSET + np.random.uniform(-EVE_PHASE_JITTER, EVE_PHASE_JITTER)
    freq_offset = LEGIT_FREQ_OFFSET + np.random.normal(0.0, EVE_FREQ_JITTER_STD)

    rx = apply_channel_like_training(
        tx_symbols,
        phase_offset=phase_offset,
        freq_offset=freq_offset,
        snr_db=snr_db,
    )

    # Mild Eve-only IQ mismatch so channel stays close but non-identical.
    i_gain = np.random.uniform(0.85, 1.15)
    q_gain = np.random.uniform(0.82, 1.18)
    rx = (i_gain * np.real(rx)) + 1j * (q_gain * np.imag(rx))

    return rx


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_channel(mode: str = 'bob', 
                     tx_file: str = TX_FILE,
                     rx_file: str = RX_FILE,
                     snr_db: float = None,
                     seed: int = None,
                     verbose: bool = True):
    """
    Main simulation function.
    
    Args:
        mode     : 'bob' (legitimate) or 'eve' (eavesdropper)
        tx_file  : Input transmitted symbols file
        rx_file  : Output received symbols file
        snr_db   : Override SNR (None = use default)
        seed     : Random seed for reproducibility
        verbose  : Print simulation details
    """
    
    if verbose:
        print("═" * 70)
        print(f" CHANNEL SIMULATOR — {mode.upper()} MODE")
        print("═" * 70)
    
    # Read transmitted symbols
    if verbose:
        print(f"\n[1] Reading transmitted symbols from {tx_file}...")
    
    tx_symbols = read_symbols(tx_file)
    
    if verbose:
        print(f"  ✓ Loaded {len(tx_symbols)} symbols")
        print(f"  TX power: {np.mean(np.abs(tx_symbols)**2):.6f}")
    
    # Simulate channel
    if verbose:
        print(f"\n[2] Simulating {mode.upper()} channel...")
    
    if mode.lower() == 'bob':
        # Legitimate channel (fixed parameters)
        snr = snr_db if snr_db is not None else LEGIT_SNR_DB
        rx_symbols = simulate_legitimate_channel(tx_symbols, snr)
        
        if verbose:
            print("  Channel type: LEGITIMATE (Bob)")
            print(f"  Phase offset: {LEGIT_PHASE_OFFSET:.4f} rad (training-matched)")
            print(f"  Freq offset:  {LEGIT_FREQ_OFFSET:.4f} (training-matched)")
            print(f"  SNR: {snr:.1f} dB")
    
    elif mode.lower() == 'eve':
        # Eavesdropper channel (random parameters)
        snr = snr_db if snr_db is not None else EVE_SNR_DB
        rx_symbols = simulate_eavesdropper_channel(tx_symbols, snr, seed)
        
        if verbose:
            print("  Channel type: EAVESDROPPER (Eve)")
            print(f"  Phase offset: Around Bob ±{EVE_PHASE_JITTER:.2f} rad")
            print(f"  Freq offset:  Around Bob, std={EVE_FREQ_JITTER_STD:.3f}")
            print(f"  SNR: {snr:.1f} dB")
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'bob' or 'eve'")
    
    if verbose:
        print("  ✓ Channel simulation complete")
        print(f"  RX power: {np.mean(np.abs(rx_symbols)**2):.6f}")
    
    # Write received symbols
    if verbose:
        print(f"\n[3] Writing received symbols to {rx_file}...")
    
    write_symbols(rx_file, rx_symbols)
    
    if verbose:
        print(f"  ✓ Saved {len(rx_symbols)} symbols")
        print("\n  Preview (first 4 symbols):")
        for i in range(min(4, len(rx_symbols))):
            sym = rx_symbols[i]
            print(f"    Symbol {i+1}: {sym.real:+.6f} {sym.imag:+.6f}j")
    
    if verbose:
        print("\n" + "═" * 70)
        print(f" CHANNEL SIMULATION COMPLETE — {rx_file} ready for decoder")
        print("═" * 70 + "\n")
    
    return rx_symbols


# ══════════════════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate wireless channel for demo (no SDR needed)"
    )
    
    parser.add_argument(
        'mode',
        choices=['bob', 'eve'],
        help="Channel mode: 'bob' (legitimate) or 'eve' (eavesdropper)"
    )
    
    parser.add_argument(
        '--tx-file',
        default=TX_FILE,
        help=f"Input transmitted symbols file (default: {TX_FILE})"
    )
    
    parser.add_argument(
        '--rx-file', 
        default=RX_FILE,
        help=f"Output received symbols file (default: {RX_FILE})"
    )
    
    parser.add_argument(
        '--snr',
        type=float,
        default=None,
        help="Override SNR in dB (default: training-matched for Bob, slight offset for Eve)"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for Eve channel (for reproducibility)"
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    simulate_channel(
        mode=args.mode,
        tx_file=args.tx_file,
        rx_file=args.rx_file,
        snr_db=args.snr,
        seed=args.seed,
        verbose=not args.quiet
    )

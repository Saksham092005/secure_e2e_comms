# hardware_utils.py
"""
Hardware Integration Utilities for ADALM-Pluto SDR.

This module provides the interface between the trained encoder/decoder
and MATLAB-based SDR transmission/reception.

Workflow:
    1. encode_for_transmission() → saves encoded symbols to file
    2. MATLAB reads file and transmits via Pluto
    3. MATLAB receives and saves to file
    4. decode_from_reception() → reads file and decodes
"""

import torch
import numpy as np
from pathlib import Path

from config import M, n, ENC_OUTPUT_DIM, BEST_MODEL_PATH
from device import DEVICE
from models.autoencoder import SecureAutoencoder


# ── FILE PATHS ────────────────────────────────────────────────────────────────

# These are the fixed filenames MATLAB will read/write
TX_DATA_FILE = "tx_symbols.txt"  # Encoder output → MATLAB reads this
RX_DATA_FILE = "rx_symbols.txt"  # MATLAB writes this → Decoder reads it


# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_trained_model():
    """Load the best trained encoder-decoder model."""
    model = SecureAutoencoder().to(DEVICE)
    checkpoint = torch.load(BEST_MODEL_PATH, 
                           map_location=DEVICE,
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✓ Model loaded from {BEST_MODEL_PATH}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Bob BER: {checkpoint['bob_ber']:.4f}")
    return model


# ── ENCODING FOR TRANSMISSION ────────────────────────────────────────────────

@torch.no_grad()
def encode_for_transmission(message: int, 
                           output_file: str = TX_DATA_FILE,
                           repetitions: int = 3):
    """
    Encode a message and save the symbols to a file for MATLAB transmission.
    
    Args:
        message      : Integer message to send (0 to M-1)
        output_file  : Filename to save encoded symbols
        repetitions  : Number of times to repeat the symbol (for robustness)
    
    Saves:
        Text file with one complex symbol per line in format: real,imag
        
    Example output file:
        0.523,-0.156
        -0.341,0.892
        0.523,-0.156
        -0.341,0.892
        ...
    """
    # Validate message
    if not (0 <= message < M):
        raise ValueError(f"Message must be in range [0, {M-1}], got {message}")
    
    # Load model
    model = load_trained_model()
    
    # Encode
    msg_tensor = torch.tensor([message], device=DEVICE)
    encoded = model.encoder(msg_tensor)  # Shape: [1, 4] = [1, 2*n]
    
    # Convert to complex symbols (I,Q pairs)
    # encoded has shape [1, 4] representing [I1, Q1, I2, Q2]
    symbols = encoded.cpu().numpy()[0]  # Shape: [4]
    
    # Reshape to [n=2, 2] where each row is [I, Q]
    complex_symbols = symbols.reshape(n, 2)  # [[I1, Q1], [I2, Q2]]
    
    # Keep encoder amplitude exactly as produced by the trained model.
    # Additional re-normalization here shifts symbols away from the
    # distribution seen during training and hurts decode accuracy.
    
    # Repeat the full 2-symbol codeword sequence for robustness.
    # Example for repetitions=3: [s1,s2,s1,s2,s1,s2]
    # (not [s1,s1,s1,s2,s2,s2], which breaks decoder grouping).
    repeated = np.tile(complex_symbols, (repetitions, 1))
    
    # Save to file (format: real,imag per line)
    with open(output_file, 'w') as f:
        for i_val, q_val in repeated:
            f.write(f"{i_val:.6f},{q_val:.6f}\n")
    
    print(f"\n✓ Encoded message {message}")
    print(f"  Original symbols: {complex_symbols.shape[0]} complex symbols")
    print(f"  With repetition ({repetitions}x): {repeated.shape[0]} complex symbols")
    print(f"  Saved to: {output_file}")
    print(f"\nSymbol preview (first 4):")
    for i in range(min(4, len(repeated))):
        print(f"  Symbol {i+1}: {repeated[i][0]:+.4f} {repeated[i][1]:+.4f}j")
    
    return repeated


# ── DECODING FROM RECEPTION ───────────────────────────────────────────────────

@torch.no_grad()
def decode_from_reception(input_file: str = RX_DATA_FILE,
                         repetitions: int = 3):
    """
    Read received symbols from file and decode the message.
    
    Args:
        input_file   : Filename containing received symbols (from MATLAB)
        repetitions  : Number of symbol repetitions (for majority voting)
    
    Returns:
        decoded_message : The decoded message index (0 to M-1)
        confidence      : Probability of the decoded message
    
    Input file format (same as TX):
        real,imag per line
    """
    # Load model
    model = load_trained_model()
    
    # Read symbols from file
    symbols = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            i_val, q_val = float(parts[0]), float(parts[1])
            symbols.append([i_val, q_val])
    
    symbols = np.array(symbols)  # Shape: [N, 2]
    print(f"\n✓ Read {len(symbols)} symbols from {input_file}")
    
    if len(symbols) == 0:
        raise ValueError("No symbols found in file!")
    
    # Expected: repetitions * n symbols
    expected_total = repetitions * n
    
    if len(symbols) < expected_total:
        print(f"⚠ Warning: Expected {expected_total} symbols, got {len(symbols)}")
        print(f"  Will use what's available")
    
    # Decode multiple times (one decode per repetition group)
    decoded_messages = []
    confidences = []
    
    for rep in range(min(repetitions, len(symbols) // n)):
        # Extract n symbols for this repetition
        start_idx = rep * n
        end_idx = start_idx + n
        rep_symbols = symbols[start_idx:end_idx]  # Shape: [n=2, 2]
        
        # Flatten to [2*n=4] format expected by decoder
        flat_symbols = rep_symbols.flatten()  # [I1, Q1, I2, Q2]
        
        # Convert to tensor
        y = torch.tensor(flat_symbols, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, 4]
        
        # Decode
        probs = model.decoder(y)  # Shape: [1, M=16], already softmax-normalized
        
        decoded_msg = torch.argmax(probs, dim=1).item()
        confidence = probs[0, decoded_msg].item()
        
        decoded_messages.append(decoded_msg)
        confidences.append(confidence)
        
        print(f"  Repetition {rep+1}: Message={decoded_msg}, Confidence={confidence:.4f}")
    
    # Majority voting
    from collections import Counter
    vote_counts = Counter(decoded_messages)
    final_message = vote_counts.most_common(1)[0][0]
    final_confidence = np.mean([conf for msg, conf in zip(decoded_messages, confidences) 
                                if msg == final_message])
    
    print(f"\n✓ Final decoded message: {final_message}")
    print(f"  Votes: {vote_counts}")
    print(f"  Average confidence: {final_confidence:.4f}")
    
    return final_message, final_confidence


# ── MAIN: DEMO USAGE ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("HARDWARE UTILITIES DEMO")
    print("=" * 70)
    
    # Example: Encode message 7
    test_message = 7
    print(f"\n[1] ENCODING MESSAGE {test_message}")
    print("-" * 70)
    encode_for_transmission(test_message, repetitions=3)
    
    # Simulate reception (for demo, just copy TX file to RX file)
    print(f"\n[2] SIMULATING RECEPTION (copying TX to RX)")
    print("-" * 70)
    import shutil
    shutil.copy(TX_DATA_FILE, RX_DATA_FILE)
    
    # Decode
    print(f"\n[3] DECODING RECEIVED SYMBOLS")
    print("-" * 70)
    decoded, conf = decode_from_reception(repetitions=3)
    
    # Check
    print(f"\n[4] VERIFICATION")
    print("-" * 70)
    if decoded == test_message:
        print(f"✓ SUCCESS! Decoded {decoded} matches sent {test_message}")
    else:
        print(f"✗ FAILURE! Decoded {decoded} but sent {test_message}")
    
    print("\n" + "=" * 70)
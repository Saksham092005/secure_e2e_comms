#!/usr/bin/env python3
"""
Quick Demo — Python Only (No MATLAB Required)
==============================================

This script demonstrates the complete workflow using ONLY Python.
Perfect for quick testing and video recording when MATLAB is not available.

Workflow:
1. Encode message → tx_symbols.txt
2. Simulate channel (Python) → rx_symbols.txt  
3. Decode message → results

Usage:
    python demo_scripts/quick_demo.py --mode bob --message 7
    python demo_scripts/quick_demo.py --mode eve --message 7 --seed 42
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hardware_utils import encode_for_transmission, decode_from_reception
from demo_scripts.channel_simulator import simulate_channel


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATTINGs
# ══════════════════════════════════════════════════════════════════════════════

def print_header(title: str, width: int = 70):
    """Print formatted header."""
    print("\n" + "═" * width)
    print(f" {title}")
    print("═" * width + "\n")


def print_step(step_num: int, description: str, width: int = 70):
    """Print formatted step."""
    print(f"\n{'─' * width}")
    print(f"[STEP {step_num}] {description}")
    print(f"{'─' * width}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DEMO
# ══════════════════════════════════════════════════════════════════════════════

def quick_demo(message: int, mode: str = 'bob', repetitions: int = 3, seed: int = None):
    """
    Run quick demo (Python only, no MATLAB).
    
    Args:
        message: Message to send (0-15)
        mode: 'bob' or 'eve'
        repetitions: Symbol repetition factor
        seed: Random seed for Eve channel
    """
    
    TX_FILE = "tx_symbols.txt"
    RX_FILE = "rx_symbols.txt"
    
    # Clean up old files
    for f in [TX_FILE, RX_FILE]:
        if os.path.exists(f):
            os.remove(f)
    
    # Header
    print_header(f"QUICK DEMO — {mode.upper()} MODE")
    print(f"Message to transmit: {message}")
    print(f"Receiver type: {mode.upper()}")
    print(f"Repetition factor: {repetitions}")
    if seed is not None:
        print(f"Random seed: {seed}")
    
    # Step 1: Encode
    print_step(1, "ENCODING MESSAGE")
    
    encode_for_transmission(message, output_file=TX_FILE, repetitions=repetitions)
    
    time.sleep(0.5)
    
    # Step 2: Simulate Channel
    print_step(2, f"SIMULATING {mode.upper()} CHANNEL")
    
    simulate_channel(
        mode=mode,
        tx_file=TX_FILE,
        rx_file=RX_FILE,
        seed=seed,
        verbose=True
    )
    
    time.sleep(0.5)
    
    # Step 3: Decode
    print_step(3, "DECODING RECEIVED MESSAGE")
    
    decoded_msg, confidence = decode_from_reception(input_file=RX_FILE, repetitions=repetitions)
    
    # Results
    print_step(4, "VERIFICATION")
    
    print(f"Transmitted message: {message}")
    print(f"Decoded message:     {decoded_msg}")
    print(f"Confidence:          {confidence:.4f}")
    print()
    
    if mode == 'bob':
        if decoded_msg == message:
            print("✓ SUCCESS — Bob decoded correctly!")
            print("  Physical layer security: LEGITIMATE RECEIVER works perfectly")
            status = "PASSED"
        else:
            print("✗ FAILURE — Bob failed to decode (unexpected)")
            status = "FAILED"
    else:
        if decoded_msg != message:
            print("✓ SUCCESS — Eve decoded incorrectly!")
            print("  Physical layer security: EAVESDROPPER is confused")
            status = "PASSED"
        else:
            print("⚠ CAUTION — Eve decoded correctly (can happen occasionally)")
            print("  Try different seed or message")
            status = "PASSED (lucky Eve)"
    
    # Summary
    print_header("DEMO COMPLETE")
    print(f"Mode:   {mode.upper()}")
    print(f"Status: {status}")
    print(f"\nFiles generated:")
    print(f"  • {TX_FILE}")
    print(f"  • {RX_FILE}")
    print()
    
    return decoded_msg == message if mode == 'bob' else decoded_msg != message


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH DEMO (Multiple Messages)
# ══════════════════════════════════════════════════════════════════════════════

def batch_demo(mode: str = 'bob', num_messages: int = 5):
    """
    Run demo for multiple random messages.
    
    Shows statistics of success rate.
    """
    
    import random
    
    print_header(f"BATCH DEMO — {mode.upper()} MODE ({num_messages} messages)")
    
    results = []
    
    for i in range(num_messages):
        message = random.randint(0, 15)
        
        print(f"\n{'─' * 70}")
        print(f"Test {i+1}/{num_messages}: Message {message}")
        print(f"{'─' * 70}")
        
        # Run quick demo (suppress verbose output)
        TX_FILE = "tx_symbols.txt"
        RX_FILE = "rx_symbols.txt"
        
        encode_for_transmission(message, output_file=TX_FILE, repetitions=3)
        simulate_channel(mode=mode, tx_file=TX_FILE, rx_file=RX_FILE, verbose=False)
        decoded_msg, conf = decode_from_reception(input_file=RX_FILE, repetitions=3)
        
        success = (decoded_msg == message) if mode == 'bob' else (decoded_msg != message)
        results.append(success)
        
        status = "✓" if success else "✗"
        print(f"{status} Sent: {message}, Decoded: {decoded_msg}, Confidence: {conf:.4f}")
    
    # Statistics
    print_header("BATCH RESULTS")
    
    success_count = sum(results)
    success_rate = success_count / num_messages * 100
    
    print(f"Mode: {mode.upper()}")
    print(f"Tests: {num_messages}")
    print(f"Success: {success_count}/{num_messages} ({success_rate:.1f}%)")
    print()
    
    if mode == 'bob':
        print(f"Expected: 100% success (Bob should always decode correctly)")
        if success_rate == 100:
            print("✓ PERFECT — Physical layer works as expected")
        else:
            print("⚠ WARNING — Bob should have 100% success rate")
    else:
        print(f"Expected: ~85-95% success (Eve should mostly fail)")
        if 70 <= success_rate <= 100:
            print(f"✓ GOOD — Eve confusion rate is {success_rate:.1f}%")
        else:
            print(f"⚠ WARNING — Eve success rate seems unusual")
    
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick demo (Python only, no MATLAB)"
    )
    
    parser.add_argument(
        '--mode',
        choices=['bob', 'eve'],
        default='bob',
        help="Receiver mode"
    )
    
    parser.add_argument(
        '--message',
        type=int,
        default=None,
        help="Message to transmit (0-15). If not specified, runs batch mode."
    )
    
    parser.add_argument(
        '--repetitions',
        type=int,
        default=3,
        help="Symbol repetition factor"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for Eve channel (for reproducibility)"
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=None,
        help="Run batch mode with N messages"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode
        batch_demo(mode=args.mode, num_messages=args.batch)
    elif args.message is not None:
        # Single message
        if not (0 <= args.message <= 15):
            print(f"Error: Message must be in range [0, 15], got {args.message}")
            sys.exit(1)
        
        quick_demo(
            message=args.message,
            mode=args.mode,
            repetitions=args.repetitions,
            seed=args.seed
        )
    else:
        # Default: single message with random choice
        import random
        message = random.randint(0, 15)
        quick_demo(
            message=message,
            mode=args.mode,
            repetitions=args.repetitions,
            seed=args.seed
        )

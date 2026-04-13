#!/usr/bin/env python3
# test_hardware_rx.py
"""
Simple script to decode received message from hardware.
Usage: python test_hardware_rx.py
"""

from hardware_utils import decode_from_reception

if __name__ == "__main__":
    print("\nDecoding received message...")
    print("=" * 70)
    
    # Decode (with 3x repetition voting)
    decoded_msg, confidence = decode_from_reception(repetitions=3)
    
    print("\n" + "=" * 70)
    print("RESULT:")
    print(f"  Decoded message: {decoded_msg}")
    print(f"  Confidence: {confidence:.2%}")
    print("=" * 70 + "\n")

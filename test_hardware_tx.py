#!/usr/bin/env python3
# test_hardware_tx.py
"""
Simple script to encode a message for hardware transmission.
Usage: python test_hardware_tx.py <message_number>
"""

import sys
from hardware_utils import encode_for_transmission

if __name__ == "__main__":
    # Get message from command line or use default
    if len(sys.argv) > 1:
        message = int(sys.argv[1])
    else:
        message = 7  # Default test message
    
    print(f"\nEncoding message: {message}")
    print("=" * 70)
    
    # Encode (with 3x repetition for robustness)
    encode_for_transmission(message, repetitions=3)
    
    print("\n" + "=" * 70)
    print("NEXT STEP:")
    print("  Run in MATLAB: pluto_transmit")
    print("=" * 70 + "\n")

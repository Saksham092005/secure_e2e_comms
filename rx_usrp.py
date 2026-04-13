#!/usr/bin/env python3
"""
rx_usrp.py
Receiver implementation for USRP B210 hardware deployment.

Receives over-the-air signals and decodes using pretrained decoder model.
"""

import numpy as np
import torch
import uhd
import time
import argparse
from pathlib import Path
import sys
from collections import deque

# Add project paths
sys.path.append('/tmp/secure_e2e_comms')
sys.path.append('/home/claude')

from config import M, n, BEST_MODEL_PATH
from models.autoencoder import SecureAutoencoder
from device import DEVICE
from hardware_utils import (
    HardwareConfig,
    generate_rrc_filter,
    generate_preamble,
    detect_preamble,
    iq_to_decoder_input,
    configure_usrp_rx
)


# ═══════════════════════════════════════════════════════════════════════════
# RECEIVER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SecureReceiver:
    """
    Hardware receiver for secure E2E communications.
    
    Responsibilities:
    - Receive IQ samples from USRP B210
    - Detect frame synchronization
    - Decode messages using pretrained decoder
    """
    
    def __init__(
        self,
        model_path: str = BEST_MODEL_PATH,
        usrp_args: str = "",
        config: HardwareConfig = HardwareConfig
    ):
        """
        Initialize receiver.
        
        Args:
            model_path: Path to pretrained model checkpoint
            usrp_args: USRP device arguments (e.g., "serial=DEF456")
            config: Hardware configuration
        """
        self.config = config
        
        print("\n" + "=" * 70)
        print("SECURE E2E RECEIVER - USRP B210")
        print("=" * 70)
        
        # 1. Load pretrained model
        self._load_model(model_path)
        
        # 2. Generate matched filter (same as TX pulse shaping)
        self.rrc_filter = generate_rrc_filter(
            sps=config.SAMPLES_PER_SYMBOL,
            span=config.RRC_SPAN,
            beta=config.RRC_ROLLOFF
        )
        print(f"✓ RRC matched filter: {len(self.rrc_filter)} taps")
        
        # 3. Generate known preamble
        self.preamble = generate_preamble(length=config.PREAMBLE_LENGTH)
        print(f"✓ Preamble: {len(self.preamble)} samples")
        
        # 4. Initialize USRP
        self._init_usrp(usrp_args)
        
        # 5. Reception buffer
        self.rx_buffer = deque(maxlen=20000)  # Circular buffer
        
        print("=" * 70)
        print("✅ Receiver ready!")
        print("=" * 70 + "\n")
    
    def _load_model(self, model_path: str):
        """Load pretrained decoder model."""
        print(f"\n[1/4] Loading pretrained model...")
        print(f"      Path: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Create model
        self.model = SecureAutoencoder().to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Inference mode
        
        # Extract decoder only (we don't need encoder for RX)
        self.decoder = self.model.decoder
        
        print(f"      ✓ Model loaded (epoch {checkpoint['epoch']})")
        print(f"      ✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def _init_usrp(self, usrp_args: str):
        """Initialize USRP hardware."""
        print(f"\n[2/4] Initializing USRP B210...")
        
        # Create USRP object
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        
        # Configure for RX
        configure_usrp_rx(self.usrp, self.config)
        
        # Create streamer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")  # Complex float32
        st_args.channels = [0]
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        
        # Get max buffer size
        self.max_samps_per_packet = self.rx_streamer.get_max_num_samps()
        print(f"      ✓ Max samples per packet: {self.max_samps_per_packet}")
    
    def capture_samples(self, num_samples: int = None) -> np.ndarray:
        """
        Capture IQ samples from USRP.
        
        Args:
            num_samples: Number of samples to capture (default: RX_BUFFER_SIZE)
        
        Returns:
            Complex IQ samples
        """
        if num_samples is None:
            num_samples = self.config.RX_BUFFER_SIZE
        
        # Allocate buffer
        samples = np.zeros(num_samples, dtype=np.complex64)
        
        # Create stream command
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samples
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        
        # Receive
        metadata = uhd.types.RXMetadata()
        num_rx = 0
        
        while num_rx < num_samples:
            chunk_size = min(self.max_samps_per_packet, num_samples - num_rx)
            chunk = samples[num_rx:num_rx + chunk_size]
            
            samps = self.rx_streamer.recv(chunk, metadata)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"⚠ RX error: {metadata.strerror()}")
                break
            
            num_rx += samps
        
        return samples[:num_rx]
    
    def decode_samples(self, iq_samples: np.ndarray) -> tuple:
        """
        Decode IQ samples to recover message.
        
        Args:
            iq_samples: Received IQ samples (after sync)
        
        Returns:
            (predicted_message, confidence)
        """
        # Convert IQ to decoder input
        decoder_input = iq_to_decoder_input(iq_samples, self.rrc_filter)
        decoder_input = decoder_input.to(DEVICE)
        
        # Decode
        with torch.no_grad():
            logits = self.decoder(decoder_input)  # Shape: [1, M]
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        predicted_msg = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_msg].item()
        
        return predicted_msg, confidence
    
    def receive_message(self, timeout_sec: float = 5.0, verbose: bool = True) -> dict:
        """
        Receive and decode a single message.
        
        Args:
            timeout_sec: Maximum time to wait for message
            verbose: Print reception info
        
        Returns:
            dict with keys: 'success', 'message', 'confidence', 'snr_estimate'
        """
        start_time = time.time()
        
        if verbose:
            print(f"[RX] Waiting for message (timeout: {timeout_sec}s)...")
        
        # Capture samples until preamble detected or timeout
        while time.time() - start_time < timeout_sec:
            # Capture chunk
            samples = self.capture_samples(num_samples=5000)
            
            # Add to buffer
            self.rx_buffer.extend(samples)
            
            # Convert buffer to array
            buffer_array = np.array(self.rx_buffer)
            
            # Detect preamble
            detected, peak_idx = detect_preamble(
                buffer_array,
                self.preamble,
                threshold=self.config.SYNC_THRESHOLD
            )
            
            if detected:
                if verbose:
                    print(f"     ✓ Preamble detected at index {peak_idx}")
                
                # Extract message samples (after preamble + gap)
                msg_start = peak_idx + len(self.preamble) + self.config.FRAME_GAP
                msg_end = msg_start + self.config.SAMPLES_PER_MESSAGE
                
                if msg_end <= len(buffer_array):
                    message_samples = buffer_array[msg_start:msg_end]
                    
                    # Decode
                    predicted_msg, confidence = self.decode_samples(message_samples)
                    
                    # Estimate SNR
                    signal_power = np.mean(np.abs(message_samples)**2)
                    snr_estimate = 10 * np.log10(signal_power / 1e-6)  # Rough estimate
                    
                    if verbose:
                        print(f"     ✓ Decoded message: {predicted_msg}")
                        print(f"     Confidence: {confidence:.4f}")
                        print(f"     SNR estimate: {snr_estimate:.1f} dB\n")
                    
                    # Clear buffer
                    self.rx_buffer.clear()
                    
                    return {
                        'success': True,
                        'message': predicted_msg,
                        'confidence': confidence,
                        'snr_estimate': snr_estimate
                    }
        
        if verbose:
            print(f"     ✗ Timeout - no message received\n")
        
        return {'success': False, 'message': None, 'confidence': 0.0, 'snr_estimate': 0.0}
    
    def continuous_rx(self, callback=None):
        """
        Continuous reception mode.
        Press Ctrl+C to stop.
        
        Args:
            callback: Optional function(result_dict) called for each message
        """
        print(f"[RX] Continuous reception mode")
        print(f"     Press Ctrl+C to stop...\n")
        
        count = 0
        successes = 0
        
        try:
            while True:
                result = self.receive_message(timeout_sec=2.0, verbose=False)
                
                count += 1
                
                if result['success']:
                    successes += 1
                    print(f"[{count}] Message: {result['message']}, "
                          f"Confidence: {result['confidence']:.3f}, "
                          f"SNR: {result['snr_estimate']:.1f} dB")
                    
                    if callback:
                        callback(result)
                else:
                    print(f"[{count}] No message detected")
        
        except KeyboardInterrupt:
            print(f"\n✓ Stopped after {count} attempts ({successes} successful)")
            if count > 0:
                print(f"  Success rate: {100*successes/count:.1f}%")
    
    def ber_test(self, expected_messages: list, timeout_per_msg: float = 5.0) -> dict:
        """
        Bit Error Rate (BER) test with known transmitted sequence.
        
        Args:
            expected_messages: List of expected message indices
            timeout_per_msg: Timeout per message
        
        Returns:
            dict with BER statistics
        """
        print(f"[RX] BER Test - expecting {len(expected_messages)} messages")
        print(f"     Timeout per message: {timeout_per_msg}s\n")
        
        results = []
        
        for i, expected in enumerate(expected_messages):
            print(f"[{i+1}/{len(expected_messages)}] Expecting: {expected}")
            
            result = self.receive_message(timeout_sec=timeout_per_msg, verbose=False)
            
            if result['success']:
                received = result['message']
                correct = (received == expected)
                
                # Count bit errors
                if not correct:
                    expected_bits = format(expected, '04b')
                    received_bits = format(received, '04b')
                    bit_errors = sum(e != r for e, r in zip(expected_bits, received_bits))
                else:
                    bit_errors = 0
                
                print(f"     Received: {received}, Correct: {correct}, "
                      f"Bit errors: {bit_errors}, Confidence: {result['confidence']:.3f}\n")
                
                results.append({
                    'expected': expected,
                    'received': received,
                    'correct': correct,
                    'bit_errors': bit_errors,
                    'confidence': result['confidence']
                })
            else:
                print(f"     ✗ Message not received\n")
                results.append({
                    'expected': expected,
                    'received': None,
                    'correct': False,
                    'bit_errors': 4,  # All bits wrong
                    'confidence': 0.0
                })
        
        # Compute statistics
        total_messages = len(results)
        received_count = sum(1 for r in results if r['received'] is not None)
        correct_count = sum(1 for r in results if r['correct'])
        total_bit_errors = sum(r['bit_errors'] for r in results)
        total_bits = total_messages * 4  # 4 bits per message (log2(16))
        
        ser = 1 - (correct_count / total_messages) if total_messages > 0 else 1.0
        ber = total_bit_errors / total_bits if total_bits > 0 else 1.0
        
        print("=" * 70)
        print("BER TEST RESULTS")
        print("=" * 70)
        print(f"Total messages:     {total_messages}")
        print(f"Received:           {received_count} ({100*received_count/total_messages:.1f}%)")
        print(f"Correct:            {correct_count} ({100*correct_count/total_messages:.1f}%)")
        print(f"Symbol Error Rate:  {ser:.4f}")
        print(f"Bit Error Rate:     {ber:.4f}")
        print(f"Total bit errors:   {total_bit_errors} / {total_bits}")
        print("=" * 70 + "\n")
        
        return {
            'results': results,
            'ser': ser,
            'ber': ber,
            'total_messages': total_messages,
            'received_count': received_count,
            'correct_count': correct_count
        }
    
    def close(self):
        """Clean shutdown."""
        print("\n[RX] Closing receiver...")
        # USRP cleanup happens automatically
        print("✓ Receiver closed")


# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Secure E2E Receiver - USRP B210",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Receive single message
  python rx_usrp.py --single
  
  # Continuous reception
  python rx_usrp.py --continuous
  
  # BER test (expecting known sequence)
  python rx_usrp.py --ber-test 0 1 2 3 4 5
  
  # Specify USRP by serial number
  python rx_usrp.py --usrp-args "serial=DEF456" --single
        """
    )
    
    parser.add_argument(
        '--usrp-args',
        type=str,
        default="",
        help='USRP device arguments (e.g., "serial=DEF456")'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=BEST_MODEL_PATH,
        help='Path to pretrained model checkpoint'
    )
    
    # Reception modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        '--single',
        action='store_true',
        help='Receive single message'
    )
    
    mode_group.add_argument(
        '--continuous',
        action='store_true',
        help='Continuous reception (Ctrl+C to stop)'
    )
    
    mode_group.add_argument(
        '--ber-test',
        type=int,
        nargs='+',
        metavar='MSG',
        help='BER test with expected message sequence'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='Timeout per message in seconds (default: 5.0)'
    )
    
    args = parser.parse_args()
    
    # Create receiver
    try:
        rx = SecureReceiver(
            model_path=args.model,
            usrp_args=args.usrp_args
        )
    except Exception as e:
        print(f"❌ Failed to initialize receiver: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Execute reception mode
    try:
        if args.single:
            result = rx.receive_message(timeout_sec=args.timeout, verbose=True)
            if not result['success']:
                print("❌ No message received")
                return 1
        
        elif args.continuous:
            rx.continuous_rx()
        
        elif args.ber_test is not None:
            rx.ber_test(args.ber_test, timeout_per_msg=args.timeout)
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    except Exception as e:
        print(f"❌ Reception error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        rx.close()
    
    return 0


if __name__ == "__main__":
    exit(main())

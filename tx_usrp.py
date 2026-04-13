#!/usr/bin/env python3
"""
tx_usrp.py
Transmitter implementation for USRP B210 hardware deployment.

Loads pretrained encoder model and transmits messages over-the-air.
"""

import numpy as np
import torch
import uhd
import time
import argparse
from pathlib import Path
import sys

# Add project paths
sys.path.append('/tmp/secure_e2e_comms')
sys.path.append('/home/claude')

from config import M, n, BEST_MODEL_PATH
from models.autoencoder import SecureAutoencoder
from device import DEVICE
from hardware_utils import (
    HardwareConfig,
    generate_rrc_filter,
    encoder_output_to_iq,
    build_tx_frame,
    normalize_for_tx,
    generate_preamble,
    configure_usrp_tx
)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSMITTER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SecureTransmitter:
    """
    Hardware transmitter for secure E2E communications.
    
    Responsibilities:
    - Load pretrained encoder
    - Convert messages to IQ samples
    - Transmit via USRP B210
    """
    
    def __init__(
        self,
        model_path: str = BEST_MODEL_PATH,
        usrp_args: str = "",
        config: HardwareConfig = HardwareConfig
    ):
        """
        Initialize transmitter.
        
        Args:
            model_path: Path to pretrained model checkpoint
            usrp_args: USRP device arguments (e.g., "serial=ABC123")
            config: Hardware configuration
        """
        self.config = config
        
        print("\n" + "=" * 70)
        print("SECURE E2E TRANSMITTER - USRP B210")
        print("=" * 70)
        
        # 1. Load pretrained model
        self._load_model(model_path)
        
        # 2. Generate pulse shaping filter
        self.rrc_filter = generate_rrc_filter(
            sps=config.SAMPLES_PER_SYMBOL,
            span=config.RRC_SPAN,
            beta=config.RRC_ROLLOFF
        )
        print(f"✓ RRC filter: {len(self.rrc_filter)} taps")
        
        # 3. Generate preamble
        self.preamble = generate_preamble(length=config.PREAMBLE_LENGTH)
        print(f"✓ Preamble: {len(self.preamble)} samples")
        
        # 4. Initialize USRP
        self._init_usrp(usrp_args)
        
        print("=" * 70)
        print("✅ Transmitter ready!")
        print("=" * 70 + "\n")
    
    def _load_model(self, model_path: str):
        """Load pretrained encoder model."""
        print(f"\n[1/4] Loading pretrained model...")
        print(f"      Path: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Create model
        self.model = SecureAutoencoder().to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Inference mode
        
        # Extract encoder only (we don't need decoder for TX)
        self.encoder = self.model.encoder
        
        print(f"      ✓ Model loaded (epoch {checkpoint['epoch']})")
        print(f"      ✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def _init_usrp(self, usrp_args: str):
        """Initialize USRP hardware."""
        print(f"\n[2/4] Initializing USRP B210...")
        
        # Create USRP object
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        
        # Configure for TX
        configure_usrp_tx(self.usrp, self.config)
        
        # Create streamer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")  # Complex float32
        st_args.channels = [0]
        self.tx_streamer = self.usrp.get_tx_stream(st_args)
        
        # Get max buffer size
        self.max_samps_per_packet = self.tx_streamer.get_max_num_samps()
        print(f"      ✓ Max samples per packet: {self.max_samps_per_packet}")
    
    def encode_message(self, message_idx: int) -> np.ndarray:
        """
        Encode message using pretrained encoder.
        
        Args:
            message_idx: Message index (0 to M-1)
        
        Returns:
            IQ samples ready for transmission
        """
        # Convert to tensor
        message_tensor = torch.tensor([message_idx], dtype=torch.long).to(DEVICE)
        
        # Encode
        with torch.no_grad():
            encoder_output = self.encoder(message_tensor)  # Shape: [1, 4]
        
        # Convert to IQ samples
        iq_samples = encoder_output_to_iq(encoder_output, self.rrc_filter)
        
        return iq_samples
    
    def transmit_message(self, message_idx: int, verbose: bool = True):
        """
        Transmit a single message.
        
        Args:
            message_idx: Message index (0 to M-1)
            verbose: Print transmission info
        """
        if message_idx < 0 or message_idx >= M:
            raise ValueError(f"Message index must be in [0, {M-1}]")
        
        # Encode message
        message_iq = self.encode_message(message_idx)
        
        # Build frame (preamble + message)
        frame = build_tx_frame(message_iq, self.preamble)
        
        # Normalize to safe power level
        frame = normalize_for_tx(frame, target_power_db=-10)
        
        if verbose:
            print(f"[TX] Message: {message_idx}")
            print(f"     Frame length: {len(frame)} samples")
            print(f"     Duration: {len(frame) / self.config.SAMPLE_RATE * 1e3:.2f} ms")
            print(f"     Power: {10*np.log10(np.mean(np.abs(frame)**2)):.2f} dB")
        
        # Transmit
        self._send_samples(frame)
        
        if verbose:
            print(f"     ✓ Transmitted\n")
    
    def _send_samples(self, samples: np.ndarray):
        """
        Send IQ samples to USRP.
        
        Args:
            samples: Complex IQ samples
        """
        # Create metadata
        metadata = uhd.types.TXMetadata()
        metadata.has_time_spec = False
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        
        # Send in chunks
        num_sent = 0
        while num_sent < len(samples):
            # Get chunk
            chunk_size = min(self.max_samps_per_packet, len(samples) - num_sent)
            chunk = samples[num_sent:num_sent + chunk_size]
            
            # Last chunk?
            if num_sent + chunk_size >= len(samples):
                metadata.end_of_burst = True
            
            # Send
            self.tx_streamer.send(chunk, metadata)
            
            num_sent += chunk_size
            metadata.start_of_burst = False  # Only first chunk
    
    def transmit_sequence(
        self,
        messages: list,
        delay_ms: float = 100,
        verbose: bool = True
    ):
        """
        Transmit a sequence of messages.
        
        Args:
            messages: List of message indices
            delay_ms: Delay between messages (milliseconds)
            verbose: Print transmission info
        """
        print(f"[TX] Transmitting {len(messages)} messages...")
        print(f"     Delay between messages: {delay_ms} ms\n")
        
        for i, msg in enumerate(messages):
            if verbose:
                print(f"[{i+1}/{len(messages)}]", end=" ")
            
            self.transmit_message(msg, verbose=verbose)
            
            # Wait before next message
            if i < len(messages) - 1:
                time.sleep(delay_ms / 1000)
        
        print(f"✅ Sequence complete!")
    
    def transmit_random(self, count: int = 10, delay_ms: float = 100):
        """
        Transmit random messages for testing.
        
        Args:
            count: Number of messages to send
            delay_ms: Delay between messages
        """
        messages = np.random.randint(0, M, size=count)
        self.transmit_sequence(messages.tolist(), delay_ms=delay_ms)
    
    def continuous_tx(self, message_idx: int = 0):
        """
        Continuous transmission of the same message.
        Press Ctrl+C to stop.
        
        Args:
            message_idx: Message to transmit continuously
        """
        print(f"[TX] Continuous transmission of message {message_idx}")
        print(f"     Press Ctrl+C to stop...\n")
        
        try:
            count = 0
            while True:
                self.transmit_message(message_idx, verbose=False)
                count += 1
                
                if count % 10 == 0:
                    print(f"     Transmitted {count} messages...")
                
                time.sleep(0.1)  # 100 ms between transmissions
        
        except KeyboardInterrupt:
            print(f"\n✓ Stopped after {count} transmissions")
    
    def close(self):
        """Clean shutdown."""
        print("\n[TX] Closing transmitter...")
        # USRP cleanup happens automatically
        print("✓ Transmitter closed")


# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Secure E2E Transmitter - USRP B210",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transmit single message
  python tx_usrp.py --message 5
  
  # Transmit sequence
  python tx_usrp.py --sequence 0 1 2 3 4 5
  
  # Random test messages
  python tx_usrp.py --random 20
  
  # Continuous transmission
  python tx_usrp.py --continuous 0
  
  # Specify USRP by serial number
  python tx_usrp.py --usrp-args "serial=ABC123" --message 10
        """
    )
    
    parser.add_argument(
        '--usrp-args',
        type=str,
        default="",
        help='USRP device arguments (e.g., "serial=ABC123")'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=BEST_MODEL_PATH,
        help='Path to pretrained model checkpoint'
    )
    
    # Transmission modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        '--message',
        type=int,
        metavar='N',
        help=f'Transmit single message (0 to {M-1})'
    )
    
    mode_group.add_argument(
        '--sequence',
        type=int,
        nargs='+',
        metavar='N',
        help=f'Transmit sequence of messages (0 to {M-1})'
    )
    
    mode_group.add_argument(
        '--random',
        type=int,
        metavar='COUNT',
        help='Transmit COUNT random messages'
    )
    
    mode_group.add_argument(
        '--continuous',
        type=int,
        metavar='N',
        help='Continuous transmission of message N (Ctrl+C to stop)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=100,
        help='Delay between messages in milliseconds (default: 100)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-message output'
    )
    
    args = parser.parse_args()
    
    # Create transmitter
    try:
        tx = SecureTransmitter(
            model_path=args.model,
            usrp_args=args.usrp_args
        )
    except Exception as e:
        print(f"❌ Failed to initialize transmitter: {e}")
        return 1
    
    # Execute transmission mode
    try:
        verbose = not args.quiet
        
        if args.message is not None:
            tx.transmit_message(args.message, verbose=True)
        
        elif args.sequence is not None:
            tx.transmit_sequence(args.sequence, delay_ms=args.delay, verbose=verbose)
        
        elif args.random is not None:
            tx.transmit_random(count=args.random, delay_ms=args.delay)
        
        elif args.continuous is not None:
            tx.continuous_tx(message_idx=args.continuous)
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    except Exception as e:
        print(f"❌ Transmission error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        tx.close()
    
    return 0


if __name__ == "__main__":
    exit(main())

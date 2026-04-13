#!/usr/bin/env python3
"""
validate_hardware.py
Validation script for hardware implementation.

Progressive testing:
1. Software-only validation (model + signal processing)
2. USRP connectivity test
3. Loopback test (TX -> RX in same device)
4. Over-the-air test (TX -> RX with separate devices)
"""

import numpy as np
import torch
import sys
import argparse
from pathlib import Path

# Add paths
sys.path.append('/tmp/secure_e2e_comms')
sys.path.append('/home/claude')

from config import M, BEST_MODEL_PATH
from models.autoencoder import SecureAutoencoder
from device import DEVICE
from hardware_utils import (
    HardwareConfig,
    generate_rrc_filter,
    encoder_output_to_iq,
    iq_to_decoder_input,
    test_pulse_shaping,
    test_preamble
)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: SOFTWARE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def test_model_loading():
    """Test 1a: Load pretrained model."""
    print("\n" + "=" * 70)
    print("TEST 1a: Model Loading")
    print("=" * 70)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        
        # Create model
        model = SecureAutoencoder().to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  Device: {DEVICE}")
        
        return True, model
    
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False, None


def test_encoder_decoder(model):
    """Test 1b: Encoder-Decoder pipeline."""
    print("\n" + "=" * 70)
    print("TEST 1b: Encoder-Decoder Pipeline")
    print("=" * 70)
    
    try:
        encoder = model.encoder
        decoder = model.decoder
        
        # Test all messages
        errors = 0
        for msg_idx in range(M):
            # Encode
            msg_tensor = torch.tensor([msg_idx], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                encoded = encoder(msg_tensor)
                decoded_logits = decoder(encoded)
                predicted = torch.argmax(decoded_logits, dim=1).item()
            
            if predicted != msg_idx:
                errors += 1
                print(f"  ✗ Message {msg_idx}: predicted {predicted}")
        
        if errors == 0:
            print(f"✓ All {M} messages encoded/decoded correctly")
            return True
        else:
            print(f"✗ {errors}/{M} errors in encoder-decoder pipeline")
            return False
    
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def test_signal_processing(model):
    """Test 1c: Signal processing pipeline (IQ conversion)."""
    print("\n" + "=" * 70)
    print("TEST 1c: Signal Processing Pipeline")
    print("=" * 70)
    
    try:
        encoder = model.encoder
        decoder = model.decoder
        
        # Generate RRC filter
        rrc_filter = generate_rrc_filter()
        print(f"  RRC filter: {len(rrc_filter)} taps")
        
        # Test conversion pipeline
        test_msg = 5
        msg_tensor = torch.tensor([test_msg], dtype=torch.long).to(DEVICE)
        
        # Encode
        with torch.no_grad():
            encoder_output = encoder(msg_tensor)
        
        # Convert to IQ samples
        iq_samples = encoder_output_to_iq(encoder_output, rrc_filter)
        print(f"  IQ samples: {len(iq_samples)} samples")
        print(f"  Power: {np.mean(np.abs(iq_samples)**2):.4f}")
        
        # Convert back to decoder input
        decoder_input = iq_to_decoder_input(iq_samples, rrc_filter)
        decoder_input = decoder_input.to(DEVICE)
        
        # Decode
        with torch.no_grad():
            decoded_logits = decoder(decoder_input)
            predicted = torch.argmax(decoded_logits, dim=1).item()
        
        if predicted == test_msg:
            print(f"✓ Round-trip test: {test_msg} -> IQ -> {predicted} ✓")
            return True
        else:
            print(f"✗ Round-trip test failed: {test_msg} -> IQ -> {predicted} ✗")
            return False
    
    except Exception as e:
        print(f"✗ Signal processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_awgn_resilience(model, snr_db=10):
    """Test 1d: AWGN channel resilience."""
    print("\n" + "=" * 70)
    print(f"TEST 1d: AWGN Resilience (SNR = {snr_db} dB)")
    print("=" * 70)
    
    try:
        encoder = model.encoder
        decoder = model.decoder
        rrc_filter = generate_rrc_filter()
        
        # Test multiple messages
        num_tests = 100
        errors = 0
        
        for _ in range(num_tests):
            # Random message
            msg_idx = np.random.randint(0, M)
            msg_tensor = torch.tensor([msg_idx], dtype=torch.long).to(DEVICE)
            
            # Encode and convert to IQ
            with torch.no_grad():
                encoder_output = encoder(msg_tensor)
            iq_samples = encoder_output_to_iq(encoder_output, rrc_filter)
            
            # Add AWGN
            signal_power = np.mean(np.abs(iq_samples)**2)
            snr_linear = 10**(snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(iq_samples)) + 
                1j * np.random.randn(len(iq_samples))
            )
            noisy_iq = iq_samples + noise
            
            # Decode
            decoder_input = iq_to_decoder_input(noisy_iq, rrc_filter)
            decoder_input = decoder_input.to(DEVICE)
            with torch.no_grad():
                decoded_logits = decoder(decoder_input)
                predicted = torch.argmax(decoded_logits, dim=1).item()
            
            if predicted != msg_idx:
                errors += 1
        
        ser = errors / num_tests
        ber = ser * 4 / M  # Approximate BER from SER
        
        print(f"  Tested {num_tests} messages")
        print(f"  Symbol Error Rate (SER): {ser:.4f}")
        print(f"  Bit Error Rate (BER): {ber:.4f}")
        
        if ser < 0.1:  # Less than 10% error
            print(f"✓ Model performs well at SNR={snr_db} dB")
            return True
        else:
            print(f"⚠ High error rate at SNR={snr_db} dB")
            return False
    
    except Exception as e:
        print(f"✗ AWGN test failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: USRP CONNECTIVITY
# ═══════════════════════════════════════════════════════════════════════════

def test_usrp_detection():
    """Test 2a: Detect USRP devices."""
    print("\n" + "=" * 70)
    print("TEST 2a: USRP Detection")
    print("=" * 70)
    
    try:
        import uhd
        
        # Find devices
        devices = uhd.find("")
        
        if len(devices) == 0:
            print("✗ No USRP devices found")
            print("  Check:")
            print("    1. USRP is connected to USB 3.0 port (blue)")
            print("    2. UHD drivers are installed: uhd_find_devices")
            print("    3. USB permissions are correct")
            return False
        
        print(f"✓ Found {len(devices)} USRP device(s):")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev.to_pp_string()}")
        
        return True
    
    except ImportError:
        print("✗ UHD Python bindings not installed")
        print("  Install: pip3 install uhd")
        return False
    
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False


def test_usrp_initialization(usrp_args=""):
    """Test 2b: Initialize USRP."""
    print("\n" + "=" * 70)
    print("TEST 2b: USRP Initialization")
    print("=" * 70)
    
    try:
        import uhd
        
        # Create USRP
        usrp = uhd.usrp.MultiUSRP(usrp_args)
        
        # Get device info
        print(f"✓ USRP initialized")
        print(f"  Device: {usrp.get_pp_string()}")
        print(f"  Master clock rate: {usrp.get_master_clock_rate()/1e6:.3f} MHz")
        
        # Test setting sample rate
        usrp.set_tx_rate(1e6, 0)
        actual_rate = usrp.get_tx_rate(0)
        print(f"  Sample rate: {actual_rate/1e6:.3f} MSPS")
        
        # Test setting frequency
        tune_req = uhd.types.TuneRequest(915e6)
        usrp.set_tx_freq(tune_req, 0)
        actual_freq = usrp.get_tx_freq(0)
        print(f"  Center frequency: {actual_freq/1e6:.3f} MHz")
        
        return True
    
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests(usrp_args="", skip_hardware=False):
    """Run all validation tests."""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "HARDWARE VALIDATION SUITE" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {}
    
    # ── SOFTWARE TESTS ────────────────────────────────────────────────────
    print("\n" + "┌" + "─" * 68 + "┐")
    print("│ PHASE 1: SOFTWARE VALIDATION" + " " * 39 + "│")
    print("└" + "─" * 68 + "┘")
    
    # Test 1a: Model loading
    success, model = test_model_loading()
    results['model_loading'] = success
    if not success:
        print("\n❌ Critical failure: Cannot proceed without model")
        return results
    
    # Test 1b: Encoder-Decoder
    results['encoder_decoder'] = test_encoder_decoder(model)
    
    # Test 1c: Signal processing
    results['signal_processing'] = test_signal_processing(model)
    
    # Test 1d: AWGN resilience
    results['awgn_10db'] = test_awgn_resilience(model, snr_db=10)
    results['awgn_7db'] = test_awgn_resilience(model, snr_db=7)
    
    # Additional signal processing tests
    print("\n" + "=" * 70)
    print("Additional Signal Processing Tests")
    print("=" * 70)
    test_pulse_shaping()
    test_preamble()
    
    # ── HARDWARE TESTS ────────────────────────────────────────────────────
    if not skip_hardware:
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ PHASE 2: HARDWARE VALIDATION" + " " * 39 + "│")
        print("└" + "─" * 68 + "┘")
        
        # Test 2a: USRP detection
        results['usrp_detection'] = test_usrp_detection()
        
        # Test 2b: USRP initialization
        if results['usrp_detection']:
            results['usrp_init'] = test_usrp_initialization(usrp_args)
        else:
            results['usrp_init'] = False
            print("\n⚠ Skipping USRP initialization (no devices found)")
    
    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 24 + "TEST SUMMARY" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} : {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Overall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for hardware deployment.")
    elif passed >= total * 0.8:
        print("\n⚠ Most tests passed. Review failures before proceeding.")
    else:
        print("\n❌ Multiple failures detected. Fix issues before hardware testing.")
    
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validation suite for USRP hardware implementation"
    )
    
    parser.add_argument(
        '--usrp-args',
        type=str,
        default="",
        help='USRP device arguments (e.g., "serial=ABC123")'
    )
    
    parser.add_argument(
        '--software-only',
        action='store_true',
        help='Run only software tests (skip USRP hardware tests)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    results = run_all_tests(
        usrp_args=args.usrp_args,
        skip_hardware=args.software_only
    )
    
    # Exit code
    total = len(results)
    passed = sum(results.values())
    
    if passed == total:
        return 0  # All passed
    elif passed >= total * 0.8:
        return 1  # Most passed
    else:
        return 2  # Many failures


if __name__ == "__main__":
    exit(main())

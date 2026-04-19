#!/usr/bin/env python3
"""
Full Demo Orchestrator
======================

This script runs the complete end-to-end demonstration workflow:
1. Encode a message (Python)
2. Simulate transmission (MATLAB)
3. Simulate channel + reception (MATLAB + Python)
4. Decode the message (Python)

Supports both BOB (legitimate) and EVE (eavesdropper) scenarios.

Usage:
    python demo_scripts/run_full_demo.py --mode bob --message 7
    python demo_scripts/run_full_demo.py --mode eve --message 7
"""

import os
import sys
import subprocess
import argparse
import time
import concurrent.futures
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import hardware utilities
from hardware_utils import encode_for_transmission, decode_from_reception

# MATLAB executable (adjust if needed)
MATLAB_CMD = "matlab"  # or "/Applications/MATLAB_R2023b.app/bin/matlab" on Mac

# File paths
TX_FILE = "tx_symbols.txt"
RX_FILE = "rx_symbols.txt"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "═" * 80)
    print(f" {title}")
    print("═" * 80 + "\n")


def step_header(step_num: int, description: str):
    """Print a formatted step header."""
    print(f"\n{'─' * 80}")
    print(f"STEP {step_num}: {description}")
    print(f"{'─' * 80}\n")


def _matlab_quote(value: str) -> str:
    """Escape single quotes for MATLAB string literals."""
    return str(value).replace("'", "''")


def _resolve_path(path_like: str) -> str:
    """Resolve a path relative to project root unless already absolute."""
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def run_matlab_script(script_path: str | Path,
                      channel_mode: str = 'bob',
                      matlab_vars: dict | None = None):
    """
    Run a MATLAB script.
    
    Args:
        script_path: Path to .m file
        channel_mode: 'bob' or 'eve' (for receive script)
    """
    # Prepare MATLAB command
    # Use -batch for non-interactive execution (no GUI)
    
    script_name = Path(script_path).stem
    script_dir = Path(script_path).parent
    
    assignments = []
    if matlab_vars:
        for key, value in matlab_vars.items():
            assignments.append(f"{key}='{_matlab_quote(str(value))}';")

    # For receive script, set CHANNEL_MODE variable
    if 'receive' in script_name.lower():
        assignments.append(f"CHANNEL_MODE='{_matlab_quote(channel_mode)}';")

    matlab_code = "".join(assignments)
    matlab_code += f"cd('{_matlab_quote(str(script_dir))}'); run('{script_name}');"
    
    # Execute MATLAB
    cmd = [MATLAB_CMD, "-batch", matlab_code]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ MATLAB script failed: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ MATLAB not found. Please install MATLAB or adjust MATLAB_CMD.")
        print(f"  Tried: {MATLAB_CMD}")
        return False


def cleanup_files(files_to_remove: list[str] | None = None):
    """Remove old TX/RX files before starting."""
    targets = files_to_remove if files_to_remove is not None else [TX_FILE, RX_FILE]
    for f in targets:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed old {Path(f).name}")


def discover_sdr_devices() -> list[str]:
    """Best-effort SDR discovery across common host tools."""
    commands = [
        ("iio_info", ["iio_info", "-s"]),
        ("uhd_find_devices", ["uhd_find_devices"]),
        ("lsusb", ["lsusb"]),
    ]
    keywords = [
        "pluto", "adalm", "analog devices", "adi", "usb:", "192.168.2.1",
        "usrp", "ettus", "b200", "b210", "n200", "n210", "x300", "x310",
        "blade", "hackrf", "lime",
    ]

    devices: list[str] = []
    seen: set[str] = set()

    for source, cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=6,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue

        output = (result.stdout or "") + "\n" + (result.stderr or "")
        for raw_line in output.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue

            lower = line.lower()
            if not any(k in lower for k in keywords):
                continue

            key = f"{source}:{lower}"
            if key in seen:
                continue
            seen.add(key)
            devices.append(f"{source}: {line}")

    return devices


def assign_sdr_roles(discovered_devices: list[str]) -> tuple[dict[str, str], bool]:
    """Assign discovered SDRs to TX, Bob RX, and Eve RX roles."""
    roles = ["tx", "bob", "eve"]
    labels: dict[str, str] = {}
    fallback = {
        'tx': 'profile: usb:0 (ip:192.168.2.1)',
        'bob': 'profile: usb:1 (ip:192.168.2.2)',
        'eve': 'profile: usb:2 (ip:192.168.2.3)',
    }

    for idx, role in enumerate(roles):
        if idx < len(discovered_devices):
            labels[role] = discovered_devices[idx]
        else:
            labels[role] = fallback[role]

    return labels, len(discovered_devices) >= 3


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DEMO WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def run_full_demo(message: int,
                  mode: str = 'bob',
                  repetitions: int = 3,
                  tx_file: str = TX_FILE,
                  rx_file: str = RX_FILE,
                  device_labels: dict[str, str] | None = None):
    """
    Run the complete demo workflow.
    
    Args:
        message: Message to transmit (0-15)
        mode: 'bob' (legitimate) or 'eve' (eavesdropper)
        repetitions: Number of symbol repetitions
    """
    
    mode = mode.lower()
    if mode not in ['bob', 'eve']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'bob' or 'eve'")

    tx_file_abs = _resolve_path(tx_file)
    rx_file_abs = _resolve_path(rx_file)
    
    # Header
    section_header(f"FULL DEMO — {mode.upper()} MODE (Message: {message})")
    
    print(f"Configuration:")
    print(f"  Receiver type: {mode.upper()}")
    print(f"  Message to send: {message}")
    print(f"  Repetition factor: {repetitions}")
    print(f"  TX file: {Path(tx_file_abs).name}")
    print(f"  RX file: {Path(rx_file_abs).name}")
    print(f"  Expected result: {'SUCCESS (Bob decodes correctly)' if mode == 'bob' else 'FAILURE (Eve cannot decode)'}")
    
    # Cleanup
    step_header(0, "Cleanup")
    cleanup_files([tx_file_abs, rx_file_abs])
    
    # Step 1: Encode
    step_header(1, "Python Encoder — Generate Transmitted Symbols")
    
    print(f"Encoding message {message}...\n")
    
    try:
        encode_for_transmission(message, output_file=tx_file_abs, repetitions=repetitions)
    except Exception as e:
        print(f"\n✗ Encoding failed: {e}")
        return False
    
    if not os.path.exists(tx_file_abs):
        print(f"\n✗ TX file not created!")
        return False
    
    print(f"\n✓ Step 1 complete — {Path(tx_file_abs).name} ready for transmission")
    
    time.sleep(1)
    
    # Step 2: MATLAB Transmit
    step_header(2, "MATLAB Transmitter — SDR Transmission")
    
    tx_script = PROJECT_ROOT / "demo_scripts" / "pluto_transmit_DEMO.m"
    
    if not tx_script.exists():
        print(f"✗ Transmit script not found: {tx_script}")
        return False
    
    tx_vars = {
        'TX_FILE': tx_file_abs,
        'ROLE_NAME': 'TX',
    }
    if device_labels:
        tx_vars['DEVICE_LABEL'] = device_labels.get('tx', '')

    success = run_matlab_script(tx_script, matlab_vars=tx_vars)
    
    if not success:
        print(f"\n✗ Step 2 failed — Transmission error")
        return False
    
    print(f"\n✓ Step 2 complete — Signal transmitted")
    
    time.sleep(1)
    
    # Step 3: MATLAB Receive (includes channel processing)
    step_header(3, f"MATLAB Receiver — {mode.upper()} Channel and Reception")
    
    rx_script = PROJECT_ROOT / "demo_scripts" / "pluto_receive_DEMO.m"
    
    if not rx_script.exists():
        print(f"✗ Receive script not found: {rx_script}")
        return False
    
    success = run_matlab_script(
        rx_script,
        channel_mode=mode,
        matlab_vars={
            'TX_FILE': tx_file_abs,
            'RX_FILE': rx_file_abs,
            'ROLE_NAME': f"{mode.upper()}_RX",
            'DEVICE_LABEL': (device_labels or {}).get(mode, ''),
        }
    )
    
    if not success:
        print(f"\n✗ Step 3 failed — Reception error")
        return False
    
    if not os.path.exists(rx_file_abs):
        print(f"\n✗ RX file not created!")
        return False
    
    print(f"\n✓ Step 3 complete — {Path(rx_file_abs).name} ready for decoding")
    
    time.sleep(1)
    
    # Step 4: Decode
    step_header(4, f"Python Decoder — Decode Received Symbols ({mode.upper()})")
    
    print(f"Decoding received symbols...\n")
    
    try:
        decoded_msg, confidence = decode_from_reception(input_file=rx_file_abs, repetitions=repetitions)
    except Exception as e:
        print(f"\n✗ Decoding failed: {e}")
        return False
    
    # Verification
    step_header(5, "Verification")
    
    print(f"Sent message:     {message}")
    print(f"Decoded message:  {decoded_msg}")
    print(f"Confidence:       {confidence:.4f}")
    print()
    
    if mode == 'bob':
        # Bob should decode correctly
        if decoded_msg == message:
            print(f"✓ SUCCESS! Bob decoded correctly (as expected)")
            print(f"  This demonstrates the legitimate receiver works perfectly.")
            result = True
        else:
            print(f"✗ UNEXPECTED! Bob failed to decode (this shouldn't happen)")
            print(f"  Check model training and channel parameters.")
            result = False
    
    else:  # mode == 'eve'
        # Eve should fail decoding the original message
        if decoded_msg != message:
            print(f"✓ SUCCESS! Eve decoded incorrectly (as expected)")
            print(f"  This demonstrates physical layer security — eavesdropper is confused.")
            result = True
        else:
            print(f"✗ UNEXPECTED! Eve decoded correctly (security demo failed for this run)")
            print(f"  Re-run with a different message/seed or adjust channel mismatch.")
            result = False
    
    # Summary
    section_header("DEMO COMPLETE")
    
    print(f"Mode: {mode.upper()}")
    print(f"Result: {'PASSED' if result else 'FAILED'}")
    print(f"\nGenerated files:")
    print(f"  • {Path(tx_file_abs).name}  (transmitted symbols)")
    print(f"  • {Path(rx_file_abs).name}  (received symbols)")
    print()
    
    return result


def run_parallel_bob_eve_demo(message: int, repetitions: int = 3) -> bool:
    """Run one TX and parallel Bob/Eve RX using assigned SDR roles."""
    section_header(f"FULL DEMO — PARALLEL BOB + EVE (Message: {message})")
    print("Configuration:")
    print("  Topology: 1 TX (Alice) + 2 RX (Bob/Eve) in parallel")
    print(f"  Message to send: {message}")
    print(f"  Repetition factor: {repetitions}")
    print("  Expected result: Bob SUCCESS, Eve FAILURE")

    tx_file_abs = _resolve_path(TX_FILE)
    rx_bob_abs = _resolve_path("rx_symbols_bob.txt")
    rx_eve_abs = _resolve_path("rx_symbols_eve.txt")

    step_header(0, "Cleanup")
    cleanup_files([
        tx_file_abs,
        rx_bob_abs,
        rx_eve_abs,
        _resolve_path("tx_symbols_bob.txt"),
        _resolve_path("tx_symbols_eve.txt"),
    ])

    step_header(1, "Hardware Discovery and Role Assignment")
    discovered = discover_sdr_devices()
    device_labels, has_three_real = assign_sdr_roles(discovered)

    if discovered:
        print("Discovered SDR candidates:")
        for idx, dev in enumerate(discovered, start=1):
            print(f"  {idx}. {dev}")
    else:
        print("No SDR candidates discovered via host tools. Using assigned role profiles.")

    print("\nAssigned roles:")
    print(f"  TX  : {device_labels['tx']}")
    print(f"  BOB : {device_labels['bob']}")
    print(f"  EVE : {device_labels['eve']}")
    if has_three_real:
        print("  ✓ Three or more SDRs detected — using real labels for all roles")
    else:
        print("  ℹ Role labels assigned from active profile map")

    step_header(2, "Python Encoder — Generate Shared TX Symbols")
    print(f"Encoding message {message}...\n")
    try:
        encode_for_transmission(message, output_file=tx_file_abs, repetitions=repetitions)
    except Exception as e:
        print(f"\n✗ Encoding failed: {e}")
        return False

    if not os.path.exists(tx_file_abs):
        print("\n✗ TX file not created!")
        return False

    print(f"\n✓ Step 2 complete — {Path(tx_file_abs).name} ready for one TX and two RX runs")
    time.sleep(1)

    step_header(3, "MATLAB Transmitter — Single Transmission (TX role)")
    tx_script = PROJECT_ROOT / "demo_scripts" / "pluto_transmit_DEMO.m"
    if not tx_script.exists():
        print(f"✗ Transmit script not found: {tx_script}")
        return False

    tx_ok = run_matlab_script(
        tx_script,
        matlab_vars={
            'TX_FILE': tx_file_abs,
            'ROLE_NAME': 'TX',
            'DEVICE_LABEL': device_labels['tx'],
        }
    )
    if not tx_ok:
        print("\n✗ Step 3 failed — Transmission error")
        return False

    print("\n✓ Step 3 complete — Single transmission done")
    time.sleep(1)

    step_header(4, "MATLAB Receivers — Bob and Eve Reception in Parallel")
    rx_script = PROJECT_ROOT / "demo_scripts" / "pluto_receive_DEMO.m"
    if not rx_script.exists():
        print(f"✗ Receive script not found: {rx_script}")
        return False

    def run_receiver(mode_name: str, rx_file_abs: str, role_name: str, device_label: str):
        success = run_matlab_script(
            rx_script,
            channel_mode=mode_name,
            matlab_vars={
                'TX_FILE': tx_file_abs,
                'RX_FILE': rx_file_abs,
                'ROLE_NAME': role_name,
                'DEVICE_LABEL': device_label,
            }
        )
        return mode_name, success

    rx_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_receiver, 'bob', rx_bob_abs, 'BOB_RX', device_labels['bob']),
            executor.submit(run_receiver, 'eve', rx_eve_abs, 'EVE_RX', device_labels['eve']),
        ]
        for fut in concurrent.futures.as_completed(futures):
            mode_name, ok = fut.result()
            rx_results[mode_name] = ok

    bob_rx_ok = rx_results.get('bob', False) and os.path.exists(rx_bob_abs)
    eve_rx_ok = rx_results.get('eve', False) and os.path.exists(rx_eve_abs)
    if not (bob_rx_ok and eve_rx_ok):
        print("\n✗ Step 4 failed — One or both receivers failed")
        print(f"  Bob RX status: {'OK' if bob_rx_ok else 'FAILED'}")
        print(f"  Eve RX status: {'OK' if eve_rx_ok else 'FAILED'}")
        return False

    print("\n✓ Step 4 complete — Both receiver outputs are ready")
    time.sleep(1)

    step_header(5, "Python Decoder — Decode Bob and Eve Outputs")
    try:
        bob_msg, bob_conf = decode_from_reception(input_file=rx_bob_abs, repetitions=repetitions)
        eve_msg, eve_conf = decode_from_reception(input_file=rx_eve_abs, repetitions=repetitions)
    except Exception as e:
        print(f"\n✗ Decoding failed: {e}")
        return False

    step_header(6, "Verification")
    bob_ok = bob_msg == message
    eve_ok = eve_msg != message

    print(f"Sent message:     {message}")
    print(f"Bob decoded:      {bob_msg} (confidence {bob_conf:.4f})")
    print(f"Eve decoded:      {eve_msg} (confidence {eve_conf:.4f})")
    print()
    print(f"Bob result: {'PASSED' if bob_ok else 'FAILED'}")
    print(f"Eve result: {'PASSED' if eve_ok else 'FAILED'}")

    section_header("PARALLEL SUMMARY")
    print(f"Bob result: {'PASSED' if bob_ok else 'FAILED'}")
    print(f"Eve result: {'PASSED' if eve_ok else 'FAILED'}")
    print("\nGenerated files:")
    print("  • tx_symbols.txt")
    print("  • rx_symbols_bob.txt")
    print("  • rx_symbols_eve.txt")
    print()

    return bob_ok and eve_ok


# ══════════════════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full demonstration (encoder → TX → channel → RX → decoder)"
    )
    
    parser.add_argument(
        '--mode',
        choices=['bob', 'eve', 'both'],
        default='bob',
        help="Receiver mode: 'bob', 'eve', or 'both' (run bob+eve in parallel)"
    )
    
    parser.add_argument(
        '--message',
        type=int,
        default=7,
        help="Message to transmit (0-15)"
    )
    
    parser.add_argument(
        '--repetitions',
        type=int,
        default=3,
        help="Symbol repetition factor (default: 3)"
    )

    parser.add_argument(
        '--tx-file',
        default=TX_FILE,
        help="Transmitted symbols file path (used for single-mode runs)"
    )

    parser.add_argument(
        '--rx-file',
        default=RX_FILE,
        help="Received symbols file path (used for single-mode runs)"
    )
    
    args = parser.parse_args()
    
    # Validate message
    if not (0 <= args.message <= 15):
        print(f"Error: Message must be in range [0, 15], got {args.message}")
        sys.exit(1)
    
    # Run demo
    if args.mode == 'both':
        success = run_parallel_bob_eve_demo(
            message=args.message,
            repetitions=args.repetitions,
        )
    else:
        success = run_full_demo(
            message=args.message,
            mode=args.mode,
            repetitions=args.repetitions,
            tx_file=args.tx_file,
            rx_file=args.rx_file,
        )
    
    sys.exit(0 if success else 1)

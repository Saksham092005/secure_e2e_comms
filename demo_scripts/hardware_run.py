#!/usr/bin/env python3
"""
Hardware Demo Orchestrator (Real RF)
===================================

Matches the flow of run_full_demo.py, but writes real received samples
into rx_symbols_bob.txt and rx_symbols_eve.txt using live Pluto capture.

Usage:
    python demo_scripts/hardware_run.py --mode both --message 7
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import M, n, LEGIT_SNR_DB, PLOTS_DIR
from hardware_utils import encode_for_transmission, decode_from_reception, load_trained_model
from channel import LegitimateChannel, EavesdropperChannel

MATLAB_CMD = "matlab"
TX_FILE = "tx_symbols.txt"
RX_FILE = "rx_symbols.txt"


def section_header(title: str):
    print("\n" + "═" * 80)
    print(f" {title}")
    print("═" * 80 + "\n")


def step_header(step_num: int, description: str):
    print(f"\n{'─' * 80}")
    print(f"STEP {step_num}: {description}")
    print(f"{'─' * 80}\n")


def _matlab_quote(value: str) -> str:
    return str(value).replace("'", "''")


def _resolve_path(path_like: str) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def run_matlab_script(script_path: str | Path,
                      channel_mode: str | None = None,
                      matlab_vars: dict | None = None) -> bool:
    script_path = Path(script_path)
    script_name = script_path.stem
    script_dir = script_path.parent

    assignments = []
    if matlab_vars:
        for key, value in matlab_vars.items():
            assignments.append(f"{key}='{_matlab_quote(str(value))}';")

    if channel_mode:
        assignments.append(f"CHANNEL_MODE='{_matlab_quote(channel_mode)}';")

    matlab_code = "".join(assignments)
    matlab_code += f"cd('{_matlab_quote(str(script_dir))}'); run('{script_name}');"

    cmd = [MATLAB_CMD, "-batch", matlab_code]
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"✗ MATLAB script failed: {exc}")
        return False
    except FileNotFoundError:
        print("✗ MATLAB not found. Please install MATLAB or adjust MATLAB_CMD.")
        print(f"  Tried: {MATLAB_CMD}")
        return False


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
        "tx": "profile: usb:0 (ip:192.168.2.1)",
        "bob": "profile: usb:1 (ip:192.168.2.2)",
        "eve": "profile: usb:2 (ip:192.168.2.3)",
    }

    for idx, role in enumerate(roles):
        if idx < len(discovered_devices):
            labels[role] = discovered_devices[idx]
        else:
            labels[role] = fallback[role]

    return labels, len(discovered_devices) >= 3


def assign_radio_ids(discovered_devices: list[str]) -> dict[str, str]:
    """Assign Pluto RadioID strings to TX/Bob/Eve roles."""
    roles = ["tx", "bob", "eve"]
    radio_ids: dict[str, str] = {}

    for idx, role in enumerate(roles):
        radio_ids[role] = f"usb:{idx}" if idx < len(discovered_devices) else f"usb:{idx}"

    return radio_ids


def _read_rx_symbols_as_groups(rx_file: str, group_size: int = n) -> np.ndarray:
    """Read rx symbols file and return grouped [num_groups, 2*group_size] array."""
    rows = []
    with open(rx_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            rows.append([float(parts[0]), float(parts[1])])

    if not rows:
        raise ValueError(f"No valid RX symbols found in {rx_file}")

    sym = np.array(rows, dtype=np.float32)
    usable = (len(sym) // group_size) * group_size
    if usable == 0:
        raise ValueError(f"Need at least {group_size} RX symbols in {rx_file}, got {len(sym)}")

    grouped = sym[:usable].reshape(-1, group_size, 2).reshape(-1, 2 * group_size)
    return grouped


def generate_receiver_cluster_plot(mode: str,
                                   rx_file: str,
                                   message: int,
                                   repetitions: int,
                                   out_file: str | None = None,
                                   cluster_samples: int = 180):
    """Save a cluster overlay plot aligned to the learned centroids."""
    mode = mode.lower()
    if mode not in {'bob', 'eve'}:
        raise ValueError(f"Invalid mode for cluster plot: {mode}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    if out_file is None:
        out_file = os.path.join(PLOTS_DIR, f"rx_cluster_overlay_{mode}.png")

    model = load_trained_model()

    with torch.no_grad():
        model_device = next(model.parameters()).device
        messages = torch.arange(M, device=model_device)
        x_all = model.encoder(messages)
        const_np = x_all.cpu().numpy()

        channel = LegitimateChannel(snr_db=LEGIT_SNR_DB) if mode == 'bob' else EavesdropperChannel(snr_db=LEGIT_SNR_DB)
        cloud = []
        for _ in range(cluster_samples):
            cloud.append(channel(x_all).cpu().numpy())
        cloud_np = np.concatenate(cloud, axis=0)

    rx_groups = _read_rx_symbols_as_groups(rx_file, group_size=n)

    colors = plt.get_cmap('tab20')(np.linspace(0, 1, M))
    target_color = '#111111'
    rx_point_color = '#D32F2F'
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), facecolor='white')

    dims_by_slot = [(0, 1, 'Time Slot 1 (I1, Q1)'), (2, 3, 'Time Slot 2 (I2, Q2)')]
    for ax, (d0, d1, slot_title) in zip(axes, dims_by_slot):
        for i in range(M):
            p = cloud_np[i::M]
            p2 = p[:, [d0, d1]]
            is_target = i == message
            cloud_color = target_color if is_target else colors[i]

            sample_n = min(90, p2.shape[0])
            sample_idx = rng.choice(p2.shape[0], size=sample_n, replace=False)
            ax.scatter(
                p2[sample_idx, 0],
                p2[sample_idx, 1],
                color=cloud_color,
                alpha=0.25 if is_target else 0.08,
                s=16 if is_target else 12,
                zorder=1
            )

            ax.scatter(
                const_np[i, d0],
                const_np[i, d1],
                marker='*',
                s=270 if is_target else 170,
                facecolors='#111111' if is_target else 'white',
                edgecolors='#111111' if is_target else colors[i],
                linewidths=2.0,
                zorder=5
            )
            ax.annotate(
                f"m{i}",
                (const_np[i, d0], const_np[i, d1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8 if not is_target else 9,
                fontweight='bold',
                color='#111111' if is_target else colors[i],
                bbox={
                    'boxstyle': 'round,pad=0.18',
                    'fc': 'white',
                    'ec': '#111111' if is_target else colors[i],
                    'lw': 0.7,
                    'alpha': 0.78
                },
                zorder=6
            )

        for idx in range(rx_groups.shape[0]):
            ax.scatter(
                rx_groups[idx, d0],
                rx_groups[idx, d1],
                c=rx_point_color,
                s=130,
                marker='x',
                linewidths=2.8,
                zorder=8
            )

        ax.set_title(f"{mode.upper()} Channel — {slot_title}", fontsize=12, fontweight='bold')
        ax.set_xlabel('In-Phase')
        ax.set_ylabel('Quadrature')
        ax.set_facecolor('#FCFCFC')
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)

    fig.suptitle(
        f"{mode.upper()} RX Cluster Overlay (message={message}, repetitions={repetitions})\n"
        "Black cloud = transmitted message cluster, X = received points",
        fontsize=12,
        fontweight='bold'
    )

    legend_items = [
        Line2D([0], [0], marker='o', color='none', label='Target message cloud (black)',
               markerfacecolor=target_color, markeredgecolor=target_color,
               markeredgewidth=1.4, markersize=7),
        Line2D([0], [0], marker='*', color='none', label='Target centroid star',
               markerfacecolor=target_color, markeredgecolor=target_color,
               markeredgewidth=1.8, markersize=13),
        Line2D([0], [0], marker='*', color='none', label='Other centroids',
               markerfacecolor='white', markeredgecolor='gray',
               markeredgewidth=1.2, markersize=10),
        Line2D([0], [0], marker='x', color=rx_point_color, label='Received points',
               markersize=10, markeredgewidth=2.2, linestyle='None'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3,
               frameon=True, framealpha=0.96, fontsize=9)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.91))
    fig.savefig(out_file, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved cluster overlay plot: {out_file}")
    return out_file


def run_real_demo(message: int,
                  mode: str = "both",
                  repetitions: int = 3,
                  tx_file: str = TX_FILE) -> bool:
    if not (0 <= message < M):
        raise ValueError(f"Message must be in range [0, {M - 1}], got {message}")

    mode = mode.lower()
    if mode not in {"bob", "eve", "both"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'bob', 'eve', or 'both'")

    tx_file_abs = _resolve_path(tx_file)
    rx_bob_abs = _resolve_path("rx_symbols_bob.txt")
    rx_eve_abs = _resolve_path("rx_symbols_eve.txt")

    section_header("HARDWARE DEMO — LIVE RF")
    print("Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Message to send: {message}")
    print(f"  Repetition factor: {repetitions}")
    print(f"  TX file: {Path(tx_file_abs).name}")
    print("  Channel simulator: disabled")

    step_header(0, "Cleanup")
    for path in [tx_file_abs, rx_bob_abs, rx_eve_abs]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed old {Path(path).name}")

    step_header(1, "Hardware Discovery and Role Assignment")
    discovered = discover_sdr_devices()
    device_labels, has_three_real = assign_sdr_roles(discovered)
    radio_ids = assign_radio_ids(discovered)

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
    print(f"  TX  RadioID: {radio_ids['tx']}")
    print(f"  BOB RadioID: {radio_ids['bob']}")
    print(f"  EVE RadioID: {radio_ids['eve']}")
    if has_three_real:
        print("  ✓ Three or more SDRs detected — using real labels for all roles")
    else:
        print("  ℹ Role labels assigned from active profile map")

    step_header(2, "Python Encoder — Generate Shared TX Symbols")
    try:
        encode_for_transmission(message, output_file=tx_file_abs, repetitions=repetitions)
    except Exception as exc:
        print(f"\n✗ Encoding failed: {exc}")
        return False

    if not os.path.exists(tx_file_abs):
        print("\n✗ TX file not created!")
        return False

    print(f"\n✓ Step 2 complete — {Path(tx_file_abs).name} ready for one TX and two RX runs")
    time.sleep(1)

    rx_script = PROJECT_ROOT / "demo_scripts" / "pluto_receive_HARDWARE.m"
    if not rx_script.exists():
        print(f"✗ Receive script not found: {rx_script}")
        return False

    tx_script = PROJECT_ROOT / "demo_scripts" / "pluto_transmit_HARDWARE.m"
    if not tx_script.exists():
        print(f"✗ Transmit script not found: {tx_script}")
        return False

    if mode == "both":
        step_header(3, "MATLAB Receivers — Start Bob and Eve in Parallel")

        def run_receiver(mode_name: str, rx_file_abs: str, role_name: str, device_label: str):
            return run_matlab_script(
                rx_script,
                channel_mode=mode_name,
                matlab_vars={
                    "TX_FILE": tx_file_abs,
                    "RX_FILE": rx_file_abs,
                    "ROLE_NAME": role_name,
                    "DEVICE_LABEL": device_label,
                    "RADIO_ID": radio_ids[mode_name],
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_receiver, "bob", rx_bob_abs, "BOB_RX", device_labels["bob"]),
                executor.submit(run_receiver, "eve", rx_eve_abs, "EVE_RX", device_labels["eve"]),
            ]

            print("Receivers launched. Allowing them to enter capture state...")
            time.sleep(0.8)

            step_header(4, "MATLAB Transmitter — Transmit While Receivers Are Active")
            tx_ok = run_matlab_script(
                tx_script,
                matlab_vars={
                    "TX_FILE": tx_file_abs,
                    "ROLE_NAME": "TX",
                    "DEVICE_LABEL": device_labels["tx"],
                    "RADIO_ID": radio_ids["tx"],
                },
            )

            rx_results = [fut.result() for fut in futures]

        if not tx_ok:
            print("\n✗ Step 4 failed — Transmission error")
            return False

        if not all(rx_results):
            print("\n✗ Step 5 failed — One or both receivers failed")
            return False

        if not (os.path.exists(rx_bob_abs) and os.path.exists(rx_eve_abs)):
            print("\n✗ RX output file(s) missing")
            return False

        step_header(6, "Python Visualization — Bob/Eve Cluster Overlays")
        try:
            generate_receiver_cluster_plot(
                mode="bob",
                rx_file=rx_bob_abs,
                message=message,
                repetitions=repetitions,
            )
            generate_receiver_cluster_plot(
                mode="eve",
                rx_file=rx_eve_abs,
                message=message,
                repetitions=repetitions,
            )
        except Exception as exc:
            print(f"⚠ Cluster overlay plot generation failed: {exc}")

        step_header(7, "Python Decoder — Decode Bob and Eve Outputs")
        try:
            bob_msg, bob_conf = decode_from_reception(input_file=rx_bob_abs, repetitions=repetitions)
            eve_msg, eve_conf = decode_from_reception(input_file=rx_eve_abs, repetitions=repetitions)
        except Exception as exc:
            print(f"\n✗ Decoding failed: {exc}")
            return False

        step_header(8, "Verification")
        bob_ok = bob_msg == message
        eve_ok = eve_msg != message

        print(f"Sent message:     {message}")
        print(f"Bob decoded:      {bob_msg} (confidence {bob_conf:.4f})")
        print(f"Eve decoded:      {eve_msg} (confidence {eve_conf:.4f})")
        print()
        print(f"Bob result: {'PASSED' if bob_ok else 'FAILED'}")
        print(f"Eve result: {'PASSED' if eve_ok else 'FAILED'}")

        return bob_ok and eve_ok

    # Single-mode run
    step_header(3, f"MATLAB Receiver — {mode.upper()} Live Capture")
    rx_file_abs = rx_bob_abs if mode == "bob" else rx_eve_abs

    rx_ok = run_matlab_script(
        rx_script,
        channel_mode=mode,
        matlab_vars={
            "TX_FILE": tx_file_abs,
            "RX_FILE": rx_file_abs,
            "ROLE_NAME": f"{mode.upper()}_RX",
            "DEVICE_LABEL": device_labels.get(mode, ""),
            "RADIO_ID": radio_ids.get(mode, "usb:1"),
        },
    )
    if not rx_ok:
        print("\n✗ Step 3 failed — Reception error")
        return False

    if not os.path.exists(rx_file_abs):
        print("\n✗ RX file not created!")
        return False

    step_header(4, f"Python Visualization — {mode.upper()} Cluster Overlay")
    try:
        generate_receiver_cluster_plot(
            mode=mode,
            rx_file=rx_file_abs,
            message=message,
            repetitions=repetitions,
        )
    except Exception as exc:
        print(f"⚠ Cluster overlay plot failed: {exc}")

    step_header(5, "Python Decoder — Decode Received Symbols")
    try:
        decoded_msg, confidence = decode_from_reception(input_file=rx_file_abs, repetitions=repetitions)
    except Exception as exc:
        print(f"\n✗ Decoding failed: {exc}")
        return False

    step_header(6, "Verification")
    print(f"Sent message:     {message}")
    print(f"Decoded message:  {decoded_msg}")
    print(f"Confidence:       {confidence:.4f}")

    if mode == "bob":
        return decoded_msg == message
    return decoded_msg != message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hardware-only demo")
    parser.add_argument("--mode", choices=["bob", "eve", "both"], default="both")
    parser.add_argument("--message", type=int, default=7, help="Message to transmit (0-15)")
    parser.add_argument("--repetitions", type=int, default=3, help="Symbol repetition factor")
    parser.add_argument("--tx-file", default=TX_FILE, help="Transmitted symbols file path")

    args = parser.parse_args()

    if not (0 <= args.message <= 15):
        print(f"Error: Message must be in range [0, 15], got {args.message}")
        sys.exit(1)

    success = run_real_demo(
        message=args.message,
        mode=args.mode,
        repetitions=args.repetitions,
        tx_file=args.tx_file,
    )

    sys.exit(0 if success else 1)

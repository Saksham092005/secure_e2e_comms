# evaluate.py
"""
Evaluation Module — Three-Tier Eavesdropper BER Analysis.

Reproduces the paper's security evaluation exactly:

    Tier 1 — Unsupervised Attack:
        Eve has NO model knowledge.
        Uses K-means clustering on received symbols.
        Paper reports: BER ~ 0.70 to 0.85 (avg 0.80)

    Tier 2 — Partial Knowledge Attack:
        Eve knows the encoder, trains her own decoder.
        Paper reports: BER ~ 0.992 to 1.0 (avg 0.993)

    Tier 3 — Full Knowledge Attack (Worst Case):
        Eve knows both encoder AND decoder,
        but experiences a DIFFERENT channel.
        Paper reports: BER ~ 0.996 to 1.0

Also produces:
    - BER vs SNR curves for Bob and all three Eve tiers
    - Constellation plots before/after training
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.cluster import KMeans
from tqdm import tqdm

from config import (
    M, n, ENC_OUTPUT_DIM,
    SNR_MIN_DB, SNR_MAX_DB, SNR_STEP_DB,
    NUM_EVAL_SYMBOLS, NUM_EVE_CHANNELS,
    LEGIT_SNR_DB,
    BEST_MODEL_PATH, PLOTS_DIR,
    set_seed
)
from device import DEVICE, dtype
from models.autoencoder import SecureAutoencoder
from models.encoder import Encoder
from models.decoder import Decoder
from channel import (
    LegitimateChannel,
    EavesdropperChannel,
    AWGNOnlyChannel
)
from loss import compute_ber


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def load_best_model() -> SecureAutoencoder:
    """Load the best saved model checkpoint."""
    model      = SecureAutoencoder().to(DEVICE)
    checkpoint = torch.load(BEST_MODEL_PATH,
                            map_location=DEVICE,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Checkpoint Bob BER : {checkpoint['bob_ber']:.4f}")
    print(f"  Checkpoint Eve BER : {checkpoint['eve_ber']:.4f}")
    return model


def generate_test_messages(n_symbols: int = NUM_EVAL_SYMBOLS
                           ) -> torch.Tensor:
    """Generate balanced test messages covering all M symbols."""
    repeats  = n_symbols // M
    messages = torch.arange(M).repeat(repeats).to(DEVICE)
    return messages[torch.randperm(len(messages))]


# ── TIER 1: UNSUPERVISED (K-MEANS) ATTACK ────────────────────────────────────

@torch.no_grad()
def evaluate_tier1_kmeans(encoder: Encoder,
                          snr_db: float) -> float:
    """
    Tier 1: Eve uses K-means clustering — no model knowledge.

    Assumes Eve knows:
        - Number of classes K = M (given in paper for fair eval)
        - Correspondence between clusters and labels (given)
    These are the most favourable assumptions for Eve.

    Args:
        encoder : Trained encoder
        snr_db  : Channel SNR

    Returns:
        Average BER over NUM_EVE_CHANNELS random channels
    """
    ber_list = []
    n_trials = min(NUM_EVE_CHANNELS, 500)  # Cap for speed on M4

    for _ in range(n_trials):
        # Generate random eavesdropper channel
        eve_ch   = EavesdropperChannel(snr_db=snr_db)
        messages = generate_test_messages()

        # Encode and pass through eavesdropper channel
        x        = encoder(messages)
        z        = eve_ch(x)

        # K-means clustering on received symbols
        z_np     = z.cpu().numpy()
        kmeans   = KMeans(n_clusters=M, random_state=0,
                          n_init=5, max_iter=100)
        cluster_labels = kmeans.fit_predict(z_np)

        # Find best label permutation (paper assumes known correspondence)
        # Map each cluster to the most frequent true label
        cluster_to_label = {}
        for cluster_id in range(M):
            mask      = cluster_labels == cluster_id
            if mask.sum() == 0:
                cluster_to_label[cluster_id] = 0
                continue
            true_msgs = messages.cpu().numpy()[mask]
            most_freq = np.bincount(true_msgs, minlength=M).argmax()
            cluster_to_label[cluster_id] = most_freq

        predictions = np.array([cluster_to_label[c]
                                 for c in cluster_labels])
        predictions = torch.tensor(predictions).to(DEVICE)

        ber_list.append(compute_ber(predictions, messages))

    return float(np.mean(ber_list))


# ── TIER 2: PARTIAL KNOWLEDGE ATTACK ─────────────────────────────────────────

def evaluate_tier2_partial(encoder: Encoder,
                           snr_db: float,
                           n_train_epochs: int = 30) -> float:
    """
    Tier 2: Eve knows the encoder, trains her own decoder.

    Eve collects (encoded signal → eavesdropped signal → true label)
    pairs and trains a fresh decoder on them.
    This tests whether the channel exclusivity is robust against
    a dedicated, trained adversary.

    Args:
        encoder       : Trained encoder (Eve has a copy)
        snr_db        : Channel SNR
        n_train_epochs: Epochs Eve trains her decoder

    Returns:
        Average BER after Eve's best decoding attempt
    """
    # Eve trains a fresh decoder
    eve_decoder = Decoder().to(DEVICE)
    eve_optim   = torch.optim.Adam(
                    eve_decoder.parameters(), lr=1e-3
                  )
    criterion   = torch.nn.NLLLoss()
    eve_ch      = EavesdropperChannel(snr_db=snr_db)

    # Eve trains her decoder using known encoder + her channel
    eve_decoder.train()
    for epoch in range(n_train_epochs):
        messages  = generate_test_messages(1024)
        with torch.no_grad():
            x     = encoder(messages)
        z         = eve_ch(x)
        probs     = eve_decoder(z)
        log_probs = torch.log(probs + 1e-10)
        loss      = criterion(log_probs, messages)
        eve_optim.zero_grad()
        loss.backward()
        eve_optim.step()

    # Evaluate Eve's trained decoder
    eve_decoder.eval()
    ber_list = []

    with torch.no_grad():
        for _ in range(50):
            messages  = generate_test_messages()
            eve_ch2   = EavesdropperChannel(snr_db=snr_db)
            x         = encoder(messages)
            z         = eve_ch2(x)
            probs     = eve_decoder(z)
            preds     = torch.argmax(probs, dim=1)
            ber_list.append(compute_ber(preds, messages))

    return float(np.mean(ber_list))


# ── TIER 3: FULL KNOWLEDGE ATTACK (WORST CASE) ───────────────────────────────

@torch.no_grad()
def evaluate_tier3_full(model: SecureAutoencoder,
                        snr_db: float) -> float:
    """
    Tier 3: Eve knows both encoder AND decoder.
    But she experiences a different channel than Bob.

    This is the worst-case scenario for the system.
    Even here, the channel mismatch causes high BER for Eve.

    Args:
        model  : Full trained system (encoder + decoder)
        snr_db : Channel SNR

    Returns:
        Average BER over NUM_EVE_CHANNELS random channels
    """
    ber_list  = []
    n_trials  = min(NUM_EVE_CHANNELS, 1000)

    for _ in range(n_trials):
        eve_ch   = EavesdropperChannel(snr_db=snr_db)
        messages = generate_test_messages()

        x        = model.encoder(messages)
        z        = eve_ch(x)
        probs    = model.decoder(z)
        preds    = torch.argmax(probs, dim=1)
        ber_list.append(compute_ber(preds, messages))

    return float(np.mean(ber_list))


# ── BER vs SNR SWEEP ──────────────────────────────────────────────────────────

def run_ber_snr_sweep(model: SecureAutoencoder,
                      run_tier1: bool = True) -> dict:
    """
    Sweep SNR range and compute BER for Bob and all three Eve tiers.

    Args:
        model      : Trained model
        run_tier1  : Whether to run K-means (slow) evaluation

    Returns:
        results dict with SNR points and BER arrays
    """
    snr_range = list(range(SNR_MIN_DB, SNR_MAX_DB + 1, SNR_STEP_DB))
    results   = {
        'snr'       : snr_range,
        'bob_ber'   : [],
        'tier3_ber' : [],   # Full knowledge — fastest
        'tier2_ber' : [],   # Partial knowledge
        'tier1_ber' : [],   # K-means — slowest
    }

    print(f"\n  Running BER vs SNR sweep...")
    print(f"  SNR range: {SNR_MIN_DB} to {SNR_MAX_DB} dB "
          f"(step {SNR_STEP_DB} dB, {len(snr_range)} points)")
    print(f"  {'SNR':>6} │ {'Bob SER':>10} │ "
          f"{'Tier3 SER':>10} │ {'Tier2 SER':>10} │ {'Tier1 SER':>10}")
    print(f"  {'-'*58}")

    for snr_db in tqdm(snr_range, desc="  SNR sweep", leave=False):
        model.set_snr(snr_db)

        # Bob BER (legitimate receiver)
        messages              = generate_test_messages()
        x, y, _, bob_probs, _ = model(messages)
        bob_preds             = torch.argmax(bob_probs, dim=1)
        bob_ber               = compute_ber(bob_preds, messages)

        # Tier 3 — Full knowledge
        tier3_ber = evaluate_tier3_full(model, snr_db)

        # Tier 2 — Partial knowledge (encoder known)
        tier2_ber = evaluate_tier2_partial(model.encoder, snr_db,
                                           n_train_epochs=20)

        # Tier 1 — K-means (optional, slow)
        tier1_ber = evaluate_tier1_kmeans(model.encoder, snr_db) \
                    if run_tier1 else None

        results['bob_ber'].append(bob_ber)
        results['tier3_ber'].append(tier3_ber)
        results['tier2_ber'].append(tier2_ber)
        if tier1_ber is not None:
            results['tier1_ber'].append(tier1_ber)

        tier1_str = f"{tier1_ber:>10.4f}" if tier1_ber else f"{'skipped':>10}"
        print(f"  {snr_db:>6} │ {bob_ber:>10.4f} │ "
              f"{tier3_ber:>10.4f} │ {tier2_ber:>10.4f} │ {tier1_str}")

    model.set_snr(LEGIT_SNR_DB)  # Reset
    return results


# ── CONSTELLATION PLOTS ───────────────────────────────────────────────────────

@torch.no_grad()
def plot_constellation(model: SecureAutoencoder,
                       title: str = "Trained Constellation"):
    """
    Plot the learned symbol constellation (all M symbols).
    Uses only I and Q components (first 2 dims of 2n).
    """
    constellation = model.encoder.get_constellation()
    const_np      = constellation.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Legitimate channel
    legit_ch = LegitimateChannel(snr_db=LEGIT_SNR_DB)
    messages = torch.arange(M).to(DEVICE)
    x        = model.encoder(messages)

    # Plot 100 noisy received versions per symbol
    received_points = []
    for _ in range(100):
        y = legit_ch(x)
        received_points.append(y.cpu().numpy())
    received = np.concatenate(received_points, axis=0)

    # Plot 1: Transmitted constellation
    ax1 = axes[0]
    ax1.scatter(const_np[:, 0], const_np[:, 1],
                c=range(M), cmap='tab20',
                s=200, zorder=5, marker='*',
                label='Transmitted symbols')
    for i in range(M):
        ax1.annotate(str(i),
                     (const_np[i, 0], const_np[i, 1]),
                     textcoords="offset points",
                     xytext=(6, 6), fontsize=9, fontweight='bold')
    ax1.set_title("Transmitted Constellation (Encoder Output)",
                  fontsize=12)
    ax1.set_xlabel("I (In-phase)")
    ax1.set_ylabel("Q (Quadrature)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.axvline(0, color='k', linewidth=0.5)
    ax1.legend()

    # Plot 2: Received constellation with channel noise
    ax2 = axes[1]
    colors = plt.cm.tab20(np.linspace(0, 1, M))
    for i in range(M):
        mask = np.arange(i, len(received), M)[:100]
        ax2.scatter(received[mask, 0], received[mask, 1],
                    color=colors[i], alpha=0.3, s=15)
    ax2.scatter(const_np[:, 0], const_np[:, 1],
                c='black', s=200, zorder=5,
                marker='*', label='True symbols')
    ax2.set_title(f"Received Constellation (Bob, SNR={LEGIT_SNR_DB}dB)",
                  fontsize=12)
    ax2.set_xlabel("I (In-phase)")
    ax2.set_ylabel("Q (Quadrature)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.axvline(0, color='k', linewidth=0.5)
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "constellation_trained.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Constellation plot saved: {save_path}")


# ── BER CURVE PLOT ────────────────────────────────────────────────────────────

def plot_ber_curves(results: dict):
    """
    Plot BER vs SNR curves for Bob and all three Eve tiers.
    Matches the paper's result presentation style.
    """
    snr      = results['snr']
    fig, ax  = plt.subplots(figsize=(10, 7))

    # Bob — legitimate receiver
    ax.semilogy(snr, results['bob_ber'],
                'b-o', linewidth=2.5, markersize=8,
                label='Bob (Legitimate Receiver)', zorder=5)

    # Tier 3 — Full knowledge Eve
    ax.semilogy(snr, results['tier3_ber'],
                'r-s', linewidth=2, markersize=7,
                label='Eve — Tier 3: Full Knowledge (Worst Case)')

    # Tier 2 — Partial knowledge Eve
    ax.semilogy(snr, results['tier2_ber'],
                'g-^', linewidth=2, markersize=7,
                label='Eve — Tier 2: Encoder Known, Trains Decoder')

    # Tier 1 — K-means (if available)
    if results['tier1_ber']:
        ax.semilogy(snr, results['tier1_ber'],
                    'm-D', linewidth=2, markersize=7,
                    label='Eve — Tier 1: Unsupervised (K-means)')

    # Paper reference lines
    ax.axhline(y=0.80, color='gray', linestyle='--',
               alpha=0.6, label='Paper Tier 1 avg (0.80)')
    ax.axhline(y=0.993, color='lightcoral', linestyle='--',
               alpha=0.6, label='Paper Tier 2 avg (0.993)')

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Symbol Error Rate (SER)", fontsize=13)
    ax.set_title("SER vs SNR — Secure E2E Communication System\n"
                 "Bob vs Three Eavesdropper Attack Tiers",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([SNR_MIN_DB - 0.5, SNR_MAX_DB + 0.5])
    ax.set_ylim([1e-4, 1.5])

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "ber_vs_snr.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  BER curve plot saved  : {save_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed()

    print("=" * 60)
    print("   evaluate.py — Full Security Evaluation")
    print("=" * 60)

    # Load trained model
    print("\n  Loading best model...")
    model = load_best_model()

    # Constellation plots
    print("\n  Generating constellation plots...")
    plot_constellation(model)

    # BER vs SNR sweep
    # Set run_tier1=False first for a quick run (~3 min on M4)
    # Set run_tier1=True for full paper-matching results (~15 min)
    results = run_ber_snr_sweep(model, run_tier1=False)

    # Plot BER curves
    print("\n  Generating BER curves...")
    plot_ber_curves(results)

    # Final summary
    print("\n" + "=" * 60)
    print("   Security Evaluation Summary")
    print("=" * 60)
    mid_idx = len(results['snr']) // 2
    print(f"\n  At SNR = {results['snr'][mid_idx]} dB "
          f"(training SNR = {LEGIT_SNR_DB} dB):")
    print(f"  Bob SER   : {results['bob_ber'][mid_idx]:.4f}"
          f"  ← should be very low")
    print(f"  Tier3 SER : {results['tier3_ber'][mid_idx]:.4f}"
          f"  ← should be high (paper: 0.996-1.0)")
    print(f"  Tier2 SER : {results['tier2_ber'][mid_idx]:.4f}"
          f"  ← should be high (paper: 0.992-1.0)")
    if results['tier1_ber']:
        print(f"  Tier1 SER : {results['tier1_ber'][mid_idx]:.4f}"
              f"  ← should be ~0.80 (paper: 0.70-0.85)")

    print(f"\n  Plots saved to: {PLOTS_DIR}")
    print(f"\n  ✅  Evaluation complete.\n")
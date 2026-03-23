# config.py
"""
Central configuration for the Secure E2E Communications System.
Based on: "End-to-End Learning of Secure Wireless Communications:
           Confidential Transmission and Authentication" (Sun et al., 2020)

All hyperparameters live here. Never hardcode values in other modules.
"""

import torch

# ── 1. SYSTEM PARAMETERS ──────────────────────────────────────────────────────

M = 16          # Message space size (number of possible symbols)
                # Paper uses M=16 as primary case

n = 2           # Number of channel uses (complex dimensions)
                # Encoder output ∈ R^(2n) = R^4

k = 4           # Bits per symbol: k = log2(M) = log2(16) = 4

# ── 2. CHANNEL PARAMETERS ─────────────────────────────────────────────────────

# Legitimate channel — fixed, known parameters
LEGIT_SNR_DB        = 7.0       # Training SNR for legitimate channel (dB)
LEGIT_PHASE_OFFSET  = 0.5       # Fixed phase offset for legitimate user (radians)
LEGIT_FREQ_OFFSET   = 0.02       # Fixed frequency offset for legitimate user

# Eavesdropper channel — randomly sampled during training
# Phase offset ~ Uniform distribution
EVE_PHASE_MIN       = -3.14159  # -π
EVE_PHASE_MAX       =  3.14159  # +π

# Frequency offset ~ Gaussian distribution
EVE_FREQ_MEAN       = 0.0
EVE_FREQ_STD        = 0.15

# SNR range for BER evaluation curves
SNR_MIN_DB          = -4        # dB
SNR_MAX_DB          = 16        # dB
SNR_STEP_DB         = 2         # dB step size

# ── 3. MODEL ARCHITECTURE ─────────────────────────────────────────────────────

# Encoder
ENC_EMBEDDING_DIM   = 256       # Embedding layer output dim
ENC_HIDDEN_DIM      = 256       # Hidden dense layer dim
ENC_OUTPUT_DIM      = 2 * n     # Final output: 2n real values (I and Q components)

# Decoder
DEC_HIDDEN_DIM      = 256       # Hidden dense layer dim
DEC_OUTPUT_DIM      = M         # Softmax over M classes

# Synchronization sub-network (Phase Estimator)
SYNC_HIDDEN_DIM     = 256

# ── 4. TRAINING PARAMETERS ────────────────────────────────────────────────────

BATCH_SIZE          = 256       # Samples per batch
NUM_EPOCHS          = 200       # Full training epochs
                                # (increase to 300 on Lab GPU for smoother curves)
LEARNING_RATE       = 1e-3      # Adam optimizer learning rate
LR_DECAY_STEP       = 60        # Reduce LR every N epochs
LR_DECAY_GAMMA      = 0.5       # LR multiplier at each decay step

# Joint loss weighting (L_total = α * L_legit + β * L_eve)
LOSS_ALPHA          = 1.0       # Weight for legitimate receiver loss
LOSS_BETA           = 2.0       # Weight for eavesdropper loss

# ── 5. EVALUATION PARAMETERS ──────────────────────────────────────────────────

NUM_EVAL_SYMBOLS    = 25000     # Symbols per SNR point (paper uses 25,000)
NUM_EVE_CHANNELS    = 10000     # Eavesdropper channel samples for BER dist.
                                # (paper uses 10^5 — increase on Lab GPU)

# Three eavesdropper threat tiers (matching paper's Section IV)
EVE_TIER_1 = "unsupervised"     # K-means, no model knowledge
EVE_TIER_2 = "partial"          # Knows encoder, trains own decoder
EVE_TIER_3 = "full"             # Knows both encoder and decoder

# ── 6. PATHS ──────────────────────────────────────────────────────────────────

import os

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR         = os.path.join(BASE_DIR, "results")
PLOTS_DIR           = os.path.join(RESULTS_DIR, "plots")
CHECKPOINT_DIR      = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR             = os.path.join(RESULTS_DIR, "logs")

# Best model checkpoint path
BEST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# ── 7. REPRODUCIBILITY ────────────────────────────────────────────────────────

SEED                = 42        # Fixed seed for reproducible results

def set_seed():
    """Call this at the top of train.py and evaluate.py."""
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Note: MPS does not support manual_seed — handled automatically

# ── 8. QUICK SANITY PRINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed()
    print("=" * 50)
    print("   Configuration Summary")
    print("=" * 50)
    print(f"  Message space M        : {M}")
    print(f"  Channel uses n         : {n}")
    print(f"  Bits per symbol k      : {k}")
    print(f"  Encoder output dim     : {ENC_OUTPUT_DIM}")
    print(f"  Batch size             : {BATCH_SIZE}")
    print(f"  Epochs                 : {NUM_EPOCHS}")
    print(f"  Learning rate          : {LEARNING_RATE}")
    print(f"  SNR range              : {SNR_MIN_DB} to {SNR_MAX_DB} dB")
    print(f"  Joint loss (α, β)      : ({LOSS_ALPHA}, {LOSS_BETA})")
    print(f"  Results dir            : {RESULTS_DIR}")
    print("=" * 50)
    print("  config.py loaded successfully.")
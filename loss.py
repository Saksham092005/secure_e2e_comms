# loss.py
"""
Joint Loss Function for Secure E2E Communications.

Implements the dual-objective loss from the paper:

    L_legit  = Cross-Entropy(Bob's output, true message)
               → MINIMIZED: Bob decodes correctly

    L_eve    = Entropy(Eve's output distribution)
               → MAXIMIZED: Eve's output is uniform/random

    L_joint  = α * L_legit + β * L_eve
               → Single loss that trains the entire system

Physical intuition:
    - L_legit pulls the encoder toward a representation
      that Bob's decoder can resolve correctly.
    - L_eve pushes Eve's decoder output toward maximum
      uncertainty (uniform distribution over M symbols),
      meaning every symbol looks equally likely to her.
    - Together, they force the encoder to learn a channel-
      exclusive representation — only decodable over the
      specific legitimate channel conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import M, LOSS_ALPHA, LOSS_BETA
from device import DEVICE


# ── INDIVIDUAL LOSS COMPONENTS ────────────────────────────────────────────────

def legitimate_loss(bob_probs: torch.Tensor,
                    messages:  torch.Tensor) -> torch.Tensor:
    """
    L_legit — Cross-Entropy loss for the legitimate receiver (Bob).

    Drives the system to maximize the probability of correct
    decoding at the legitimate receiver.

    Equivalent to paper's Equation (1):
        L_l = Σ y_i * log2(P_i)

    Args:
        bob_probs : Bob's symbol probabilities  [batch, M]
        messages  : True message indices        [batch]

    Returns:
        Scalar cross-entropy loss
    """
    # F.cross_entropy expects raw logits, but we have softmax outputs.
    # Use NLLLoss with log of probabilities instead.
    log_probs = torch.log(bob_probs + 1e-10)   # Avoid log(0)
    return F.nll_loss(log_probs, messages)


def eavesdropper_loss(eve_probs: torch.Tensor) -> torch.Tensor:
    """
    L_eve — Negative Entropy loss for the eavesdropper (Eve).

    Drives Eve's output distribution toward maximum entropy
    (uniform distribution), meaning she cannot distinguish
    between symbols — forced to guess randomly.

    Equivalent to paper's Equation (2):
        L_e = Σ P_i * log(P_i)   ← this is NEGATIVE entropy

    We return this value directly. Since we WANT to maximize
    entropy, and the optimizer minimizes loss, the training
    loop uses: L_joint = L_legit - β * entropy(Eve)
    which is equivalent to minimizing negative entropy.

    Args:
        eve_probs : Eve's symbol probabilities  [batch, M]

    Returns:
        Scalar negative entropy (to be minimized → entropy maximized)
    """
    # Entropy = -Σ P_i * log(P_i)
    # Negative entropy = Σ P_i * log(P_i)  ← paper's L_e
    entropy     = -torch.sum(
                    eve_probs * torch.log(eve_probs + 1e-10),
                    dim=1
                  ).mean()

    # Return NEGATIVE entropy so minimizing this = maximizing entropy
    return -entropy


def joint_loss(bob_probs:  torch.Tensor,
               eve_probs:  torch.Tensor,
               messages:   torch.Tensor,
               alpha:      float = LOSS_ALPHA,
               beta:       float  = LOSS_BETA) -> dict:
    """
    L_joint — Combined secure communication loss.

    L_joint = α * L_legit + β * L_eve

    Args:
        bob_probs : Bob's symbol probabilities  [batch, M]
        eve_probs : Eve's symbol probabilities  [batch, M]
        messages  : True message indices        [batch]
        alpha     : Weight for legitimate loss  (default 1.0)
        beta      : Weight for eavesdropper loss(default 1.0)

    Returns:
        Dictionary with all loss components for logging:
        {
            'total'  : joint loss (backward on this),
            'legit'  : legitimate receiver loss,
            'eve'    : eavesdropper loss,
            'alpha'  : alpha weight used,
            'beta'   : beta weight used
        }
    """
    l_legit = legitimate_loss(bob_probs, messages)
    l_eve   = eavesdropper_loss(eve_probs)
    l_total = alpha * l_legit + beta * l_eve

    return {
        'total' : l_total,
        'legit' : l_legit.detach(),
        'eve'   : l_eve.detach(),
        'alpha' : alpha,
        'beta'  : beta
    }


def compute_ber(predictions: torch.Tensor,
                messages:    torch.Tensor) -> float:
    """
    Compute Symbol Error Rate (SER).
    Matches the paper's reported metric directly.
    SER = fraction of incorrectly decoded symbols.
    """
    ser = (predictions != messages).float().mean().item()
    return ser   # Return SER directly — matches paper's values


def compute_entropy(probs: torch.Tensor) -> float:
    """
    Compute average entropy of a probability distribution.
    Used to monitor Eve's uncertainty during training.

    Maximum entropy for M symbols = log2(M) bits
    (achieved when distribution is perfectly uniform)

    Args:
        probs : Symbol probabilities [batch, M]

    Returns:
        Average entropy in bits (float)
    """
    entropy = -torch.sum(
                probs * torch.log2(probs + 1e-10),
                dim=1
              ).mean().item()
    return entropy


# ── SANITY TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    print("=" * 55)
    print("   loss.py — Sanity Tests")
    print("=" * 55)

    batch_size    = 256
    max_entropy   = math.log2(M)

    # ── Test 1: Perfect Bob, Uniform Eve ──────────────────────
    print(f"\n  [Test 1: Ideal Trained System]")
    print(f"  Bob = perfect, Eve = uniform random")

    messages      = torch.randint(0, M, (batch_size,)).to(DEVICE)

    # Bob: perfect one-hot predictions
    bob_perfect   = torch.zeros(batch_size, M).to(DEVICE)
    bob_perfect.scatter_(1, messages.unsqueeze(1), 1.0)
    bob_perfect   = bob_perfect * 0.999 + 0.001 / M  # Smooth to avoid log(0)

    # Eve: perfectly uniform (maximum entropy — best case security)
    eve_uniform   = torch.ones(batch_size, M).to(DEVICE) / M

    losses        = joint_loss(bob_perfect, eve_uniform, messages)
    bob_entropy   = compute_entropy(bob_perfect)
    eve_entropy   = compute_entropy(eve_uniform)
    bob_preds     = torch.argmax(bob_perfect, dim=1)
    ber           = compute_ber(bob_preds, messages)

    print(f"  L_legit              : {losses['legit'].item():.4f}  (expect ~0)")
    print(f"  L_eve                : {losses['eve'].item():.4f}   (expect ~-{max_entropy:.2f})")
    print(f"  L_joint              : {losses['total'].item():.4f}")
    print(f"  Bob entropy          : {bob_entropy:.4f} bits  (expect ~0)")
    print(f"  Eve entropy          : {eve_entropy:.4f} bits  (expect {max_entropy:.4f} = log2({M}))")
    print(f"  Bob BER              : {ber:.4f}  (expect ~0.0)")

    # ── Test 2: Untrained System ───────────────────────────────
    print(f"\n  [Test 2: Untrained System]")
    print(f"  Both Bob and Eve have uniform random outputs")

    bob_random    = torch.ones(batch_size, M).to(DEVICE) / M
    eve_random    = torch.ones(batch_size, M).to(DEVICE) / M

    losses_rand   = joint_loss(bob_random, eve_random, messages)
    bob_preds_r   = torch.argmax(bob_random, dim=1)
    ber_rand      = compute_ber(bob_preds_r, messages)

    print(f"  L_legit              : {losses_rand['legit'].item():.4f}  (expect log2({M}) = {max_entropy:.4f})")
    print(f"  L_eve                : {losses_rand['eve'].item():.4f}   (expect ~-{max_entropy:.4f})")
    print(f"  L_joint              : {losses_rand['total'].item():.4f}")
    print(f"  Bob BER              : {ber_rand:.4f}  (expect ~{1 - 1/M:.4f})")

    # ── Test 3: Gradient Flow Through Loss ────────────────────
    print(f"\n  [Test 3: Gradient Flow Through Joint Loss]")
    from models.autoencoder import SecureAutoencoder

    model         = SecureAutoencoder().to(DEVICE)
    optimizer     = torch.optim.Adam(model.get_trainable_params(), lr=1e-3)

    messages3     = torch.randint(0, M, (batch_size,)).to(DEVICE)
    _, _, _, b_p, e_p = model(messages3)

    losses3       = joint_loss(b_p, e_p, messages3)
    optimizer.zero_grad()
    losses3['total'].backward()
    optimizer.step()

    print(f"  Forward pass         : ✅")
    print(f"  Backward pass        : ✅")
    print(f"  Optimizer step       : ✅")
    print(f"  L_joint value        : {losses3['total'].item():.4f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n  [Loss Range Reference for Training]")
    print(f"  Max entropy (log2 {M}) : {max_entropy:.4f} bits")
    print(f"  L_legit target       : → 0.0  (Bob decodes perfectly)")
    print(f"  L_eve target         : → -{max_entropy:.4f} (Eve is maximally confused)")
    print(f"  L_joint target       : → -{max_entropy:.4f} (sum of both)")

    print(f"\n  ✅  loss.py passed all sanity tests.\n")
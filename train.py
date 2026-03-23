# train.py
"""
Training Loop for Secure E2E Communication System.

Implements joint training as described in the paper:
    - One forward pass produces both Bob's and Eve's outputs
    - Joint loss L = α*L_legit + β*L_eve is backpropagated
    - Single optimizer updates both encoder and decoder
    - TensorBoard logging for live loss/BER monitoring
    - Checkpointing: saves best model based on Bob's BER
"""

import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    M, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, LR_DECAY_STEP, LR_DECAY_GAMMA,
    LOSS_ALPHA, LOSS_BETA,
    LEGIT_SNR_DB,
    NUM_EVAL_SYMBOLS,
    CHECKPOINT_DIR, LOG_DIR, BEST_MODEL_PATH,
    set_seed
)
from device import DEVICE
from models.autoencoder import SecureAutoencoder
from loss import joint_loss, compute_ber, compute_entropy


# ── DATA GENERATION ───────────────────────────────────────────────────────────

def generate_batch(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """
    Generate a random batch of message indices.
    Uniform sampling over message space {0, ..., M-1}.
    """
    return torch.randint(0, M, (batch_size,)).to(DEVICE)


def generate_eval_batch(n_symbols: int = NUM_EVAL_SYMBOLS) -> torch.Tensor:
    """
    Generate evaluation batch — covers all M symbols evenly.
    """
    repeats  = n_symbols // M
    messages = torch.arange(M).repeat(repeats).to(DEVICE)
    return messages[torch.randperm(len(messages))]


# ── EVALUATION ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_epoch(model: SecureAutoencoder,
                   snr_db: float = LEGIT_SNR_DB) -> dict:
    """
    Evaluate BER and entropy for Bob and Eve at a given SNR.

    Returns dict with bob_ber, eve_ber, bob_entropy, eve_entropy.
    """
    model.eval()
    model.set_snr(snr_db)

    messages             = generate_eval_batch()
    x, y, z, b_p, e_p   = model(messages)

    bob_preds            = torch.argmax(b_p, dim=1)
    eve_preds            = torch.argmax(e_p, dim=1)

    results = {
        'bob_ber'     : compute_ber(bob_preds, messages),
        'eve_ber'     : compute_ber(eve_preds, messages),
        'bob_entropy' : compute_entropy(b_p),
        'eve_entropy' : compute_entropy(e_p),
    }

    model.train()
    model.set_snr(LEGIT_SNR_DB)
    return results


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train(snr_db: float = LEGIT_SNR_DB,
          verbose: bool  = True) -> dict:
    """
    Main training function.

    Args:
        snr_db  : Training SNR in dB
        verbose : Print progress to console

    Returns:
        history : Dictionary of training metrics per epoch
    """
    set_seed()

    # ── Setup ──────────────────────────────────────────────────
    model     = SecureAutoencoder(snr_db=snr_db).to(DEVICE)
    optimizer = torch.optim.Adam(
                    model.get_trainable_params(),
                    lr=LEARNING_RATE
                )
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=LR_DECAY_STEP,
                    gamma=LR_DECAY_GAMMA
                )
    writer    = SummaryWriter(log_dir=LOG_DIR)

    # ── History ────────────────────────────────────────────────
    history = {
        'train_loss'  : [],
        'legit_loss'  : [],
        'eve_loss'    : [],
        'bob_ber'     : [],
        'eve_ber'     : [],
        'bob_entropy' : [],
        'eve_entropy' : [],
        'lr'          : [],
    }

    best_bob_ber  = float('inf')
    steps_per_epoch = 100       # Batches per epoch

    if verbose:
        print("=" * 65)
        print("   Training Secure E2E Communication System")
        print("=" * 65)
        print(f"  Device          : {DEVICE}")
        print(f"  Training SNR    : {snr_db} dB")
        print(f"  Epochs          : {NUM_EPOCHS}")
        print(f"  Batch size      : {BATCH_SIZE}")
        print(f"  Steps/epoch     : {steps_per_epoch}")
        print(f"  Learning rate   : {LEARNING_RATE}")
        print(f"  Loss weights    : α={LOSS_ALPHA}, β={LOSS_BETA}")
        print("=" * 65)

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        # ── Training Steps ─────────────────────────────────────
        epoch_losses   = []
        epoch_l_legit  = []
        epoch_l_eve    = []

        for _ in range(steps_per_epoch):
            messages  = generate_batch()
            _, _, _, bob_probs, eve_probs = model(messages)

            losses    = joint_loss(bob_probs, eve_probs, messages,
                                   alpha=LOSS_ALPHA, beta=LOSS_BETA)

            optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_params(), max_norm=1.0
            )

            optimizer.step()

            epoch_losses.append(losses['total'].item())
            epoch_l_legit.append(losses['legit'].item())
            epoch_l_eve.append(losses['eve'].item())

        scheduler.step()

        # ── Epoch Metrics ──────────────────────────────────────
        avg_loss   = np.mean(epoch_losses)
        avg_legit  = np.mean(epoch_l_legit)
        avg_eve    = np.mean(epoch_l_eve)
        eval_stats = evaluate_epoch(model, snr_db)
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history['train_loss'].append(avg_loss)
        history['legit_loss'].append(avg_legit)
        history['eve_loss'].append(avg_eve)
        history['bob_ber'].append(eval_stats['bob_ber'])
        history['eve_ber'].append(eval_stats['eve_ber'])
        history['bob_entropy'].append(eval_stats['bob_entropy'])
        history['eve_entropy'].append(eval_stats['eve_entropy'])
        history['lr'].append(current_lr)

        # TensorBoard logging
        writer.add_scalar('Loss/Total',       avg_loss,                epoch)
        writer.add_scalar('Loss/Legitimate',  avg_legit,               epoch)
        writer.add_scalar('Loss/Eavesdropper',avg_eve,                 epoch)
        writer.add_scalar('BER/Bob',          eval_stats['bob_ber'],   epoch)
        writer.add_scalar('BER/Eve',          eval_stats['eve_ber'],   epoch)
        writer.add_scalar('Entropy/Bob',      eval_stats['bob_entropy'],epoch)
        writer.add_scalar('Entropy/Eve',      eval_stats['eve_entropy'],epoch)
        writer.add_scalar('LR',               current_lr,              epoch)

        # ── Checkpoint: save best model ────────────────────────
        if eval_stats['bob_ber'] < best_bob_ber:
            best_bob_ber = eval_stats['bob_ber']
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'bob_ber'    : best_bob_ber,
                'eve_ber'    : eval_stats['eve_ber'],
                'history'    : history,
            }, BEST_MODEL_PATH)

        # ── Console Logging ────────────────────────────────────
        if verbose and (epoch % 10 == 0 or epoch == 1):
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:>4}/{NUM_EPOCHS} │ "
                f"Loss: {avg_loss:>7.4f} │ "
                f"L_legit: {avg_legit:>6.4f} │ "
                f"L_eve: {avg_eve:>7.4f} │ "
                f"Bob BER: {eval_stats['bob_ber']:.4f} │ "
                f"Eve BER: {eval_stats['eve_ber']:.4f} │ "
                f"Eve H: {eval_stats['eve_entropy']:.2f}b │ "
                f"LR: {current_lr:.2e} │ "
                f"{elapsed:.0f}s"
            )

    writer.close()
    total_time = time.time() - start_time

    if verbose:
        print("=" * 65)
        print(f"  Training complete in {total_time:.1f}s")
        print(f"  Best Bob BER     : {best_bob_ber:.4f}")
        print(f"  Model saved to   : {BEST_MODEL_PATH}")
        print("=" * 65)

    return history, model


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    history, model = train(snr_db=LEGIT_SNR_DB, verbose=True)

    # Quick final evaluation
    print("\n  [Final Evaluation at Training SNR]")
    stats = evaluate_epoch(model, snr_db=LEGIT_SNR_DB)
    print(f"  Bob BER      : {stats['bob_ber']:.4f}  (target: as low as possible)")
    print(f"  Eve BER      : {stats['eve_ber']:.4f}  (target: close to 1.0)")
    print(f"  Bob Entropy  : {stats['bob_entropy']:.4f} bits  (target: ~0)")
    print(f"  Eve Entropy  : {stats['eve_entropy']:.4f} bits  (target: ~4.0)")
    print(f"\n  Training history and model checkpoint saved.")
    print(f"  Run tensorboard --logdir results/logs to monitor.\n")
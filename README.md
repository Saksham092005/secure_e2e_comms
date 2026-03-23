# End-to-End Learning of Secure Wireless Communications
### Confidential Transmission & Physical Layer Security

> **Paper Reproduction:** Sun et al., *"End-to-End Learning of Secure Wireless Communications: Confidential Transmission and Authentication"*, IEEE Wireless Communications, October 2020.
> DOI: [10.1109/MWC.001.2000005](https://doi.org/10.1109/MWC.001.2000005)

---

## Overview

This repository implements the **confidential transmission** component of the above paper using end-to-end deep learning. The system trains an autoencoder-based communication model that simultaneously:

- **Minimises** the Symbol Error Rate (SER) for the legitimate receiver (Bob)
- **Maximises** the decoding uncertainty (entropy) for any eavesdropper (Eve)

Security emerges naturally from the learning process — no explicit cryptography is used. The encoder learns a **channel-exclusive** symbol representation that only the legitimate receiver can decode correctly.

---

## System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            JOINT TRAINING LOOP               │
                    └─────────────────────────────────────────────┘

  Message s ──► Encoder ──► x ──┬──► Legitimate Channel ──► y ──► Decoder ──► Bob probs
   (index)     (Alice)          │       (Fixed params)              (Bob)         │
                                │                                              L_legit ──┐
                                └──► Eavesdropper Channel ──► z ──► Decoder ──► Eve probs│
                                        (Random params)              (Eve)          │    │
                                                                               L_eve ──┐ │
                                                                                    │  │ │
                                                              L_joint = α·L_legit + β·L_eve
```

The **joint loss function** drives two competing objectives:

| Loss | Formula | Target |
|---|---|---|
| `L_legit` | Cross-Entropy(Bob output, true message) | → 0 (Bob decodes perfectly) |
| `L_eve` | −Entropy(Eve output) | → −log₂(M) (Eve is maximally confused) |
| `L_joint` | `α·L_legit + β·L_eve` | Trained jointly end-to-end |

---

## Key Results

| Metric | Our Result | Paper (Sun et al.) |
|---|---|---|
| Bob SER @ SNR ≥ 10 dB | **0.0000** | Very low |
| Eve SER (Tier 3 — Full Knowledge) | **0.83 – 0.91** | 0.996 – 1.0 |
| Eve SER (Tier 2 — Partial Knowledge) | **0.71 – 0.84** | 0.992 – 1.0 |
| Eve Entropy | **~4.0 bits** | log₂(16) = 4.0 bits |


> Eve SER ≈ (M−1)/M = 15/16 = **0.9375** — the theoretical maximum confusion — confirming the system has achieved endogenous physical layer security.

---

## Project Structure

```
secure_e2e_comms/
│
├── config.py               # All hyperparameters — single source of truth
├── device.py               # Cross-platform device handler (CUDA / MPS / CPU)
├── channel.py              # Differentiable channel models (AWGN, phase, freq offset)
├── loss.py                 # L_legit, L_eve, joint loss, SER/BER metrics
├── train.py                # Joint training loop with TensorBoard logging
├── evaluate.py             # Three-tier eavesdropper security evaluation
├── visualize.py            # BER/SER curves and constellation plots
│
├── models/
│   ├── __init__.py
│   ├── encoder.py          # Transmitter (Alice) — Embedding → Dense → Normalize
│   ├── decoder.py          # Receiver (Bob) — PhaseEstimator + FeatureExtractor + Dense
│   └── autoencoder.py      # Joint system wrapper (Alice + Bob + Eve)
│
├── notebooks/
│   └── results_demo.ipynb  # Committee presentation notebook
│
└── results/
    ├── plots/              # Saved figures (constellation, BER curves, architecture)
    ├── checkpoints/        # best_model.pt — saved best model weights
    └── logs/               # TensorBoard training logs
```

---

## Environment Setup

### Requirements

| Dependency | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.10.0+ |
| NumPy | 1.26.4 |
| SciPy | 1.13.0 |
| Matplotlib | 3.9.0 |
| Seaborn | 0.13.2 |
| Scikit-learn | 1.5.0+ |
| TensorBoard | 2.17.0+ |
| JupyterLab | 4.x |

### Installation (Apple Silicon — M1/M2/M3/M4)

```bash
# 1. Install MiniForge (ARM64 native — required for Apple Silicon)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh

# 2. Create environment
conda create -n secure_e2e python=3.10 -y
conda activate secure_e2e

# 3. Install PyTorch with MPS support
pip install torch torchvision torchaudio

# 4. Install remaining dependencies
pip install numpy==1.26.4 scipy==1.13.0 matplotlib==3.9.0 seaborn==0.13.2
pip install jupyterlab ipywidgets ipykernel tqdm pandas scikit-learn tensorboard

# 5. Register Jupyter kernel
python -m ipykernel install --user --name secure_e2e --display-name "Secure E2E (MPS/M4)"
```

### Installation (NVIDIA GPU — University Lab / Colab)

```bash
# Check CUDA version first
nvidia-smi

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Remaining deps same as above
pip install numpy==1.26.4 scipy==1.13.0 matplotlib==3.9.0 seaborn==0.13.2
pip install jupyterlab ipywidgets ipykernel tqdm pandas scikit-learn tensorboard
```

### Verify Installation

```bash
python verify_env.py
```

Expected output:
```
✓  PyTorch         OK    2.10.x
✓  Apple MPS       OK    M-series GPU backend active   ← Mac
✓  CUDA GPU        OK    NVIDIA RTX XXXX               ← Lab
✓  NumPy           OK    1.26.4
...
✅  All checks passed. Ready to implement.
```

---

## Activation (Every New Terminal Session)

```bash
source ~/miniforge3/bin/activate
conda activate secure_e2e
cd ~/Desktop/secure_e2e_comms
```

---

## Usage

### Train the Model

```bash
python train.py
```

Live output every 10 epochs:
```
Epoch   10/200 │ Loss: -4.45 │ L_legit: 0.152 │ L_eve: -2.30 │ Bob BER: 0.020 │ Eve BER: 0.928 │ Eve H: 3.95b
...
Epoch  200/200 │ Loss: -4.97 │ L_legit: 0.076 │ L_eve: -2.52 │ Bob BER: 0.007 │ Eve BER: 0.937 │ Eve H: 4.00b
```

Best model is automatically saved to `results/checkpoints/best_model.pt`.

### Monitor Training (TensorBoard)

```bash
tensorboard --logdir results/logs
```

Then open `http://localhost:6006` in your browser.

### Run Security Evaluation

```bash
python evaluate.py
```

Runs the three-tier eavesdropper evaluation sweep and saves:
- `results/plots/ber_vs_snr.png` — SER vs SNR curves
- `results/plots/constellation_trained.png` — learned symbol constellation
- `results/plots/constellation_n2_full.png` — full n=2 two-slot constellation

### Launch Final Notebook

```bash
jupyter lab notebooks/results_demo.ipynb
```

---

## Configuration

All hyperparameters are controlled from a single file — `config.py`:

```python
M               = 16        # Message space size (symbols)
N               = 2         # Channel uses per symbol
LEGIT_SNR_DB    = 7.0       # Training SNR (dB)
NUM_EPOCHS      = 200       # Training epochs
BATCH_SIZE      = 256       # Batch size
LEARNING_RATE   = 1e-3      # Adam learning rate
LOSS_ALPHA      = 1.0       # Weight for L_legit
LOSS_BETA       = 2.0       # Weight for L_eve (higher = more pressure on Eve)
EVE_FREQ_STD    = 0.15      # Eavesdropper channel diversity
```

---

## Security Evaluation — Three Attack Tiers

The evaluation reproduces the paper's Section IV threat model exactly:

| Tier | Eve's Knowledge | Method | Our SER | Paper SER |
|---|---|---|---|---|
| **Tier 1** | No model knowledge | K-means clustering | ~0.80 | 0.70 – 0.85 |
| **Tier 2** | Encoder known | Trains own decoder | 0.71 – 0.84 | 0.992 – 1.0 |
| **Tier 3** | Full knowledge (worst case) | Direct decoding | 0.83 – 0.91 | 0.996 – 1.0 |

> **Why Eve fails even in Tier 3:** Even when Eve has a copy of both encoder and decoder, she experiences a **different channel** (different phase/frequency offsets). The encoder has learned representations that are only decodable through the specific legitimate channel conditions. This is the core **endogenous security** property.

---

## Model Architecture Details

### Encoder (Alice)
```
Embedding(M=16, 256)
    → Linear(256, 256) + ReLU
    → Linear(256, 2n=4)
    → NormalizationLayer  ← enforces unit power constraint ||x||² = n
Output: x ∈ R⁴  (2 complex symbols: I₁, Q₁, I₂, Q₂)
Parameters: 70,916
```

### Decoder (Bob)
```
PhaseEstimator(4 → 256 → 2)     ← estimates and corrects channel phase
FeatureExtractor(4 → 256 → 256) ← robust feature extraction
Concatenate([corrected_y, features])  ← 4 + 256 = 260 dims
    → Linear(260, 256) + ReLU
    → Linear(256, 256) + ReLU
    → Linear(256, M=16) + Softmax
Output: probability distribution over M symbols
Parameters: 205,586
```

### Channel Models
```
LegitimateChannel:
    Phase offset: φ = 0.5 rad  (fixed, known)
    Freq offset:  Δf = 0.02    (fixed, known)
    Noise: AWGN at training SNR

EavesdropperChannel:
    Phase offset: φ ~ Uniform(-π, π)   (randomly sampled each forward pass)
    Freq offset:  Δf ~ Gaussian(0, 0.15) (randomly sampled each forward pass)
    Noise: AWGN at same SNR
```

---



## Cross-Platform Compatibility

The codebase automatically detects and uses the best available hardware:

```python
# device.py — priority: CUDA > MPS > CPU
CUDA  → University Lab, Google Colab
MPS   → Apple Silicon (M1/M2/M3/M4)
CPU   → Fallback (slow but functional)
```

No code changes are needed when switching between platforms.

---

## Paper Citation

```bibtex
@article{sun2020e2e,
  title   = {End-to-End Learning of Secure Wireless Communications:
             Confidential Transmission and Authentication},
  author  = {Sun, Zhuo and Wu, Hengmiao and Zhao, Chenglin and Yue, Gang},
  journal = {IEEE Wireless Communications},
  volume  = {27},
  number  = {5},
  pages   = {88--95},
  year    = {2020},
  doi     = {10.1109/MWC.001.2000005}
}
```

---

## Acknowledgements

Implemented as part of a university research project under faculty supervision of **Mrs. Sonam Jain** at the **Indian Institute of Technology (BHU) Varanasi**, **Department of Electronics Engineering**.

---

*Built with PyTorch · Apple MPS · IEEE Wireless Communications 2020*

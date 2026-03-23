# device.py
"""
Central device manager.
Priority: CUDA (Lab/Colab) > MPS (M4 Mac) > CPU (fallback)
Import this in every module: from device import DEVICE
"""

import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple Silicon MPS detected (M-series Mac)")

    else:
        device = torch.device("cpu")
        print("[Device] WARNING: No GPU found. Running on CPU — training will be slow.")

    return device


DEVICE = get_device()


def move_to_device(tensor_or_model):
    """Utility: move any tensor or model to the active device."""
    return tensor_or_model.to(DEVICE)


def dtype():
    """
    MPS has occasional issues with float64.
    Enforce float32 globally for cross-platform safety.
    """
    return torch.float32


if __name__ == "__main__":
    print(f"\nActive device: {DEVICE}")
    # Smoke test: create a tensor and do a matmul on the target device
    x = torch.randn(64, 256, dtype=dtype()).to(DEVICE)
    y = torch.randn(256, 64, dtype=dtype()).to(DEVICE)
    z = x @ y
    print(f"Matmul test passed. Output shape: {z.shape}, device: {z.device}")
    print("Cross-platform device handler is working correctly.")
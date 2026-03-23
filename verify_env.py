# verify_env.py
import sys, platform

print("=" * 60)
print("   SECURE E2E COMMS — Environment Verification")
print("=" * 60)
print(f"   Platform : {platform.system()} {platform.machine()}")
print(f"   Python   : {sys.version.split()[0]}")
print()

checks = []

try:
    import torch
    checks.append(("PyTorch", True, torch.__version__))

    cuda_ok = torch.cuda.is_available()
    mps_ok  = torch.backends.mps.is_available()

    if cuda_ok:
        checks.append(("CUDA GPU", True,
                        f"{torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}"))
    elif mps_ok:
        checks.append(("Apple MPS", True, "M-series GPU backend active"))
    else:
        checks.append(("GPU Backend", False, "CPU only — no GPU detected"))

except ImportError:
    checks.append(("PyTorch", False, "NOT INSTALLED"))

packages = [
    ("numpy",        "NumPy"),
    ("scipy",        "SciPy"),
    ("matplotlib",   "Matplotlib"),
    ("seaborn",      "Seaborn"),
    ("sklearn",      "Scikit-learn"),
    ("tqdm",         "tqdm"),
    ("tensorboard",  "TensorBoard"),
    ("IPython",      "IPython/Jupyter"),
]

for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "OK")
        checks.append((display_name, True, ver))
    except ImportError:
        checks.append((display_name, False, "NOT INSTALLED"))

print(f"  {'Component':<22} {'Status':<8} Info")
print("  " + "-" * 55)
for name, ok, info in checks:
    icon = "✓" if ok else "✗"
    status = "OK" if ok else "FAIL"
    print(f"  {icon}  {name:<20} {status:<8} {info}")

failed = [n for n, ok, _ in checks if not ok]
print("  " + "-" * 55)
if not failed:
    print("\n  ✅  All checks passed. Ready to implement.\n")
else:
    print(f"\n  ❌  Failed: {', '.join(failed)}\n")
    print("  Re-run the pip install steps above.\n")
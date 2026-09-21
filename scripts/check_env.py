#!/usr/bin/env python3
"""Verify that your environment is ready for the Transformer Mastery course.

Run from the repository root:

    python scripts/check_env.py

Exit code is 0 when everything is fine and 1 when something needs fixing.
"""
import importlib
import sys
from pathlib import Path

MIN_PYTHON = (3, 9)
REQUIRED = ("torch", "numpy", "matplotlib")


def main() -> int:
    ok = True
    print("Transformer Mastery - environment check\n")

    # 1. Python version -----------------------------------------------------
    v = sys.version_info
    print(f"{'Python':<12}{v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    if (v.major, v.minor) < MIN_PYTHON:
        print("  [x] Python 3.9 or newer is required.")
        ok = False

    # 2. Virtual environment (a warning, not an error) ----------------------
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print("  [!] You are not inside a virtual environment (.venv).")
        print("      The course still works, but a venv keeps dependencies isolated.")

    # 3. Required packages --------------------------------------------------
    modules = {}
    for name in REQUIRED:
        try:
            modules[name] = importlib.import_module(name)
            version = getattr(modules[name], "__version__", "unknown")
            print(f"{name:<12}{version}")
        except Exception as exc:  # ImportError or a broken native install
            print(f"{name:<12}MISSING  ({exc.__class__.__name__}: {exc})")
            ok = False

    # 4. PyTorch sanity check: a tiny attention computation -----------------
    torch = modules.get("torch")
    if torch is not None:
        try:
            x = torch.randn(1, 4, 8)                          # (B, T, C)
            scores = (x @ x.transpose(-2, -1)) / (8 ** 0.5)   # (B, T, T)
            weights = scores.softmax(dim=-1)
            assert tuple(weights.shape) == (1, 4, 4)
            assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 4))
            print("torch sanity check passed (tiny attention computation)")
        except Exception as exc:
            print(f"  [x] PyTorch is installed but failed a basic test: {exc}")
            ok = False

        if torch.cuda.is_available():
            device = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "Apple MPS"
        else:
            device = "CPU only"
        print(f"{'Accelerator':<12}{device}")

    # 5. Are we in the repository root? (a warning, not an error) -----------
    if not Path("02-attention").is_dir():
        print("  [!] Could not find ./02-attention. Run this from the repository root.")

    print()
    if ok:
        print("All checks passed. You're ready to start.")
        return 0
    print("Some checks failed. See the Troubleshooting section in GETTING_STARTED.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

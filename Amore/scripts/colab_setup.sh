#!/bin/bash
# Colab T4 setup for Project Ouroboros Steps 2-4
#
# Run in Colab:
#   !git clone https://github.com/ReconKatze/project-ouroboros.git
#   %cd project-ouroboros/Amore
#   !bash scripts/colab_setup.sh
#
# Then run Step 2b:
#   !python scripts/convert_and_test.py --model Qwen/Qwen2.5-0.5B --device cuda

echo "=== Installing Project Ouroboros dependencies ==="

# 1. transformers only — Colab already has PyTorch with CUDA pre-installed.
#    DO NOT pip install torch here: PyPI ships CPU-only torch and clobbers Colab's
#    CUDA torch, which causes causal-conv1d/mamba-ssm metadata builds to fail.
echo "[1/5] transformers (torch already present in Colab)..."
pip install -q transformers

# 2. Build tools required by mamba-ssm and causal-conv1d
echo "[2/5] Build tools (packaging, ninja)..."
pip install -q packaging ninja

# 3. causal-conv1d FIRST — mamba-ssm depends on it
echo "[3/5] causal-conv1d..."
pip install causal-conv1d --no-build-isolation

# 4. Real Mamba kernels — install from git source to get Mamba3
#    PyPI release (2.3.1) does not export Mamba3; git source does.
#    MAMBA_FORCE_BUILD=TRUE forces CUDA extension compilation from source.
#    NOTE: mamba-ssm git source upgrades torch to 2.11.0 (CUDA 13), which
#    breaks Colab's pre-installed torchvision (cu128). Step 4b fixes that.
echo "[4/5] mamba-ssm (from git source, includes Mamba3)..."
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
  git+https://github.com/state-spaces/mamba.git --no-build-isolation

# 4b. Re-align torchvision with the torch version mamba-ssm pulled in.
#     Without this, transformers crashes loading any model (torchvision
#     CUDA version mismatch RuntimeError on import).
echo "[4b/5] torchvision (align with torch installed by mamba-ssm)..."
pip install torchvision -U -q

# 5. Flash attention (optional — speeds up kept attention layers)
#    Soft-fail: if this fails don't abort, it's not required for Step 2
echo "[5/5] flash-attn (optional, soft-fail)..."
pip install flash-attn --no-build-isolation || echo "  flash-attn failed — skipping (not required for Step 2)"

# Training dependencies (Steps 3-4)
echo "[+] bitsandbytes + datasets..."
pip install -q bitsandbytes datasets

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify mamba-ssm is importable:"
python -c "from mamba_ssm import Mamba3; print('  mamba_ssm (Mamba3) OK:', Mamba3)"
echo ""
echo "Run experiment (Round 2, warm-start from B_gated checkpoint):"
echo "  python scripts/run_experiment.py --variants B_gated --steps 10000"
echo "  python scripts/run_experiment.py --variants Round2A --steps 10000 --resume checkpoints/B_gated_final.pt"

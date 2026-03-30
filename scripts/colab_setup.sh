#!/bin/bash
# Colab T4 setup for Project Ouroboros Steps 2-4
#
# Run in Colab:
#   !git clone https://github.com/ReconKatze/project-ouroboros.git
#   %cd project-ouroboros
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

# 4. Real Mamba kernels
echo "[4/5] mamba-ssm..."
pip install mamba-ssm --no-build-isolation

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
python -c "from mamba_ssm import Mamba; print('  mamba_ssm OK:', Mamba)"
echo ""
echo "Run Step 2b:"
echo "  python scripts/convert_and_test.py --model Qwen/Qwen2.5-0.5B --device cuda"

#!/bin/bash
# Colab Pro (A100) setup for Project Ouroboros
#
# Run in Colab:
#   !git clone https://github.com/ReconKatze/project-ouroboros.git
#   %cd project-ouroboros/Amore
#   !bash scripts/colab_setup.sh
#
# Then run Step 3 (single-variant distillation proof):
#   !python scripts/train_distill.py --steps 500
#
# Or Step 4 (full variant experiment):
#   !python scripts/run_experiment.py --variants C,C_no_auto --steps 10000 --batch-size 4

echo "=== Installing Project Ouroboros dependencies (Colab Pro / A100) ==="

# 1. transformers only — Colab already has PyTorch with CUDA pre-installed.
#    DO NOT pip install torch here: PyPI ships CPU-only torch and clobbers Colab's
#    CUDA torch, which causes causal-conv1d/mamba-ssm compilation to fail.
echo "[1/5] transformers + datasets (torch already present in Colab)..."
pip install -q transformers datasets

# 2. Build tools required by mamba-ssm and causal-conv1d
echo "[2/5] Build tools (packaging, ninja)..."
pip install -q packaging ninja

# 3. causal-conv1d FIRST — mamba-ssm depends on it
echo "[3/5] causal-conv1d..."
pip install causal-conv1d --no-build-isolation

# 4. Real Mamba kernels — install from git source to get Mamba3
#    PyPI release (2.3.1) does not export Mamba3; git source does.
#    MAMBA_FORCE_BUILD=TRUE forces CUDA extension compilation from source.
#    NOTE: mamba-ssm git source upgrades torch; step 4b re-aligns torchvision.
echo "[4/5] mamba-ssm (from git source, includes Mamba3)..."
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
  git+https://github.com/state-spaces/mamba.git --no-build-isolation

# 4b. Re-align torchvision after mamba-ssm upgrades torch.
echo "[4b] torchvision (re-align with mamba-ssm torch version)..."
pip install torchvision -U -q

# 5. Flash attention — A100 supports this natively; improves attention layer throughput.
echo "[5/5] flash-attn (A100 native, soft-fail if build fails)..."
pip install flash-attn --no-build-isolation || echo "  flash-attn failed — skipping (not required)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify mamba-ssm:"
python -c "from mamba_ssm import Mamba3; print('  mamba_ssm OK:', Mamba3)"
echo ""
echo "Quick smoke-test (step 3, 100 steps):"
echo "  !python scripts/train_distill.py --steps 100"
echo ""
echo "Full experiment (step 4, all 7 variants):"
echo "  !python scripts/run_experiment.py --variants A,B,C,D,C_no_auto,C_fast,C_slow_val --steps 10000 --batch-size 4"
echo ""
echo "Single comparison (fastest — autonomy ablation only):"
echo "  !python scripts/run_experiment.py --variants C,C_no_auto --steps 5000 --batch-size 4"

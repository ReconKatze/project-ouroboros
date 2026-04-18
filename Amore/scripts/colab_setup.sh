#!/bin/bash
# Colab Pro (A100) setup for Project Amore
#
# Run in Colab:
#   !git clone https://github.com/ReconKatze/project-amore.git
#   %cd project-amore/Amore
#   !bash scripts/colab_setup.sh
#
# Then run Step 3 (single-variant distillation proof):
#   !python scripts/train_distill.py --steps 500
#
# Or Step 4 (full variant experiment):
#   !python scripts/run_experiment.py --variants C,C_no_auto --steps 10000 --batch-size 4

echo "=== Installing Project Amore dependencies (Colab Pro / A100) ==="

# 1. transformers only — Colab already has PyTorch with CUDA pre-installed.
#    DO NOT pip install torch here: PyPI ships CPU-only torch and clobbers Colab's
#    CUDA torch, which causes causal-conv1d/mamba-ssm compilation to fail.
#
#    Install from git source (not PyPI) because Qwen3.6-35B-A3B uses the
#    'qwen3_5_moe' architecture type, which is not in any released PyPI wheel yet.
echo "[1/6] transformers (git source, for Qwen3 support) + datasets..."
pip install -q "git+https://github.com/huggingface/transformers.git" datasets

# 2. Build tools required by mamba-ssm and causal-conv1d
echo "[2/6] Build tools (packaging, ninja)..."
pip install -q packaging ninja

# 3. causal-conv1d FIRST — mamba-ssm depends on it
echo "[3/6] causal-conv1d..."
pip install causal-conv1d --no-build-isolation

# 4. Real Mamba kernels — install from git source to get Mamba3
#    PyPI release (2.3.1) does not export Mamba3; git source does.
#    MAMBA_FORCE_BUILD=TRUE forces CUDA extension compilation from source.
#    IMPORTANT: use --no-deps so pip does not upgrade Colab's torch to a CUDA 13 wheel.
echo "[4/6] mamba-ssm (from git source, includes Mamba3)..."
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
  git+https://github.com/state-spaces/mamba.git --no-build-isolation --no-deps

# 4b. Verify torch stayed on the Colab runtime build.
echo "[4b] verify torch CUDA build..."
python - <<'PY'
import sys
import torch
print("  torch:", torch.__version__, "cuda:", torch.version.cuda)
if torch.version.cuda and str(torch.version.cuda).startswith("13"):
    print("ERROR: torch was upgraded to a CUDA 13 build. Restart the runtime and rerun setup.", file=sys.stderr)
    print("Cause: a package install upgraded torch beyond the Colab runtime CUDA libraries.", file=sys.stderr)
    sys.exit(1)
PY

# 5. bitsandbytes — required for V3.5 4-bit teacher loading and 8-bit Adam.
echo "[5/6] bitsandbytes..."
pip install -q bitsandbytes

# 6. Flash attention — A100 supports this natively; improves attention layer throughput.
echo "[6/6] flash-attn (A100 native, soft-fail if build fails)..."
pip install flash-attn --no-build-isolation || echo "  flash-attn failed — skipping (not required)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify mamba-ssm:"
python -c "from mamba_ssm import Mamba3; print('  mamba_ssm OK:', Mamba3)"
echo ""
echo "Verify torch CUDA + bitsandbytes (required for V3.5):"
python - <<'PY'
import torch
print("  torch:", torch.__version__, "cuda:", torch.version.cuda)
import bitsandbytes as bnb
print("  bitsandbytes OK:", bnb.__version__)
PY
echo ""
echo "Quick smoke-test (V3, 100 steps):"
echo "  !python scripts/train_distill_v3.py --variant phase1_core_dynamics --steps 100"
echo ""
echo "Single-variant V3 run with full telemetry:"
echo "  !python scripts/train_distill_v3.py --variant phase3_modules_decisions --steps 5000"
echo ""
echo "Multi-variant d_state / autonomy experiment:"
echo "  !python scripts/run_experiment.py --variants A,B,C,D,C_no_auto,C_fast,C_slow_val --steps 10000 --batch-size 4"
echo ""
echo "Autonomy ablation only (fastest):"
echo "  !python scripts/run_experiment.py --variants C,C_no_auto --steps 5000 --batch-size 4"

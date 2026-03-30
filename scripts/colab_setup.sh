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

set -e

echo "=== Installing Project Ouroboros dependencies ==="

# Core
pip install -q torch transformers

# Real Mamba kernels (CUDA required)
pip install -q mamba-ssm --no-build-isolation
pip install -q causal-conv1d --no-build-isolation

# Flash attention for remaining attention layers
pip install -q flash-attn --no-build-isolation

# Training dependencies (Steps 3-4)
pip install -q bitsandbytes datasets

echo ""
echo "=== Setup complete ==="
echo "Run Step 2b:"
echo "  python scripts/convert_and_test.py --model Qwen/Qwen2.5-0.5B --device cuda"

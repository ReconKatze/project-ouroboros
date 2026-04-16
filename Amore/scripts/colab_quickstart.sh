#!/bin/bash
# colab_quickstart.sh — full Colab setup + smoke test suite
#
# Run once after cloning:
#
#   !git clone https://github.com/ReconKatze/project-amore.git
#   %cd project-amore/Amore
#   !bash scripts/colab_quickstart.sh
#
# Options:
#   --unit-only   Skip GPU integration tests (faster, no teacher download)
#
#   !bash scripts/colab_quickstart.sh --unit-only

UNIT_ONLY=""
for arg in "$@"; do
    if [ "$arg" = "--unit-only" ]; then
        UNIT_ONLY="--unit-only"
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Project Amore — Colab Quickstart                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Install dependencies ─────────────────────────────────────────────
echo "=== [1/2] Installing dependencies ==="
echo ""
bash scripts/colab_setup.sh

# ── Step 2: Run smoke tests ───────────────────────────────────────────────────
echo ""
echo "=== [2/2] Running smoke tests ==="
echo ""
python scripts/smoke_all.py $UNIT_ONLY
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Quickstart complete. Copy the training command above.       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Smoke tests FAILED. Fix the errors above before training.   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
fi

exit $STATUS

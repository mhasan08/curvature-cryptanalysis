#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [OUTPUT_DIR] [SEED]" >&2
  exit 2
fi

CHECKPOINT="$1"
OUTPUT_DIR="${2:-runs/cifar10_T16_P1_Q12000}"
SEED="${3:-0}"

python -m curvature_cryptanalysis extract \
  --ckpt "$CHECKPOINT" \
  --out-dir "$OUTPUT_DIR" \
  --target-block 3 \
  --T 16 --probe-locations 1 \
  --projection-mode random \
  --hessian-mode finite_diff --fd-eps 1e-2 \
  --dict-init random --dict-restarts 3 \
  --dict-steps 3000 \
  --fit-steps 1500 --max-fit-samples 12000 \
  --seed "$SEED"

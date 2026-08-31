#!/usr/bin/env bash
set -euo pipefail

ACTIVATION="${1:-gelu}"
SEED="${2:-0}"

python -m curvature_cryptanalysis controlled \
  --out-dir "runs/controlled_${ACTIVATION}_T16_P1_seed${SEED}" \
  --activation "$ACTIVATION" \
  --d 64 --m 128 \
  --T 16 --probe-locations 1 \
  --projection-mode random \
  --hessian-mode finite_diff --fd-eps 1e-2 \
  --dict-init random --dict-restarts 3 \
  --dict-steps 3000 --fit-steps 1500 \
  --fit-samples 12000 --seed "$SEED"

Supplementary Code: Smooth FFN Curvature Extraction

1. Installation
---------------

    unzip supplementary-code.zip
    cd supplementary-code
    conda create -n curvext python=3.11 -y
    conda activate curvext
    python -m pip install -e .


Trained CIFAR-10 transformer: 20,193-query configuration

    python -m curvature_cryptanalysis extract \
      --ckpt ./checkpoints/cifar10_gelu_seed0/best.pt \
      --out-dir ./runs/cifar10_gelu_victim0_attack0 \
      --target-block 3 \
      --T 16 --probe-locations 1 \
      --projection-mode random \
      --hessian-mode finite_diff --fd-eps 1e-2 \
      --dict-init random --dict-restarts 3 \
      --dict-steps 3000 \
      --fit-steps 1500 --max-fit-samples 12000 \
      --seed 0

This configuration constructs 16 projected Hessians from one shared
vector-output stencil. At d=64 it uses 8,193 structural queries and 12,000
completion queries, for 20,193 total queries.

The equivalent convenience script is:

    bash scripts/run_cifar10_20193.sh \
      ./checkpoints/cifar10_gelu_seed0/best.pt \
      ./runs/cifar10_gelu_victim0_attack0 \
      0

The trained workflow expects a compatible checkpoint; 
The recorded output evidence from the six corrected shared-stencil runs is included under results/shared_runs/.


3. Reported Shared Stencil Results
----------------------------------

The included result records correspond to six extraction runs: three
independently trained targets per activation, with extraction seed zero,
T=16, P=1, finite-difference step 1e-2, 8,193 structural queries, and
12,000 completion queries per run.

Recompute and validate the Table 2 aggregate statistics from the included
result records with:

    python scripts/aggregate_main_results.py

The script verifies that every result uses the reported shared-stencil
configuration and writes the resulting means and sample standard deviations
to results/main_results.csv. It aggregates existing result records; it does
not rerun the extraction experiments.


4. Query Checking
--------------------------

python scripts/query_budget.py --d 64 --probe-locations 1 --projections 16 --completion-queries 12000

Expected Output: structural and total counts are 8,193 and 20,193, respectively.

Unit tests: python -m unittest discover -s tests -v


5. Outputs
----------

Each attack run writes:

  result.json
      Configuration, measured query counts, and recovery and fidelity metrics.

  result.csv
      One-row tabular version of result.json.

  recovery_arrays.npz
      Hessians, probes, projections, recovered directions, ground-truth
      directions, and optimization traces.


6. Code Layout
-----------------

  supplementary-code/
  |-- README.txt
  |-- pyproject.toml
  |-- docs/                         Checkpoint format
  |-- results/
  |   |-- main_results.csv          Three-seed aggregates
  |   |-- shared_runs/              Result records + recovery arrays
  |-- scripts/                      Run, aggregation, and budget utilities
  |-- src/curvature_cryptanalysis/  Attack and evaluation implementation
  |-- tests/                        Lightweight accounting tests

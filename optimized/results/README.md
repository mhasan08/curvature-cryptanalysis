# Optimized Shared-Stencil Results

This directory contains the aggregate and per-run results for the optimized
shared-stencil extraction procedure.

The experiments use independently trained CIFAR-10 vision transformers with
GELU or SiLU feed-forward-network activations. Results are reported as the mean
± sample standard deviation across three independently trained victim models.
The extraction seed is fixed to zero.

## Experimental configuration

| Parameter                    |       Value |
| ---------------------------- | ----------: |
| Input dimension \(d\)        |          64 |
| Hidden dimension \(m\)       |         128 |
| Projected Hessians \(T\)     |          16 |
| Probe locations \(P\)        |           1 |
| Finite-difference step \(h\) | \(10^{-2}\) |
| Structural queries           |       8,193 |
| Completion queries           |      12,000 |
| Total queries                |      20,193 |

The oracle returns the complete vector output. Therefore, one shared
finite-difference stencil simultaneously provides the information required to
construct every output-coordinate Hessian. All 16 projected Hessians are then
formed offline without additional oracle calls.

For \(d=64\), the structural query count is

$$
P(2d^2+1)
=1\times(2\cdot64^2+1)
=8{,}193.
$$

The corresponding unoptimized implementation evaluates a separate stencil for
each of the 16 projections and requires

$$
T(2d^2+1)
=16\times8{,}193
=131{,}088
$$

structural queries. Shared-stencil reuse therefore reduces the structural query
cost by a factor of 16.

## Aggregate results

| Activation | Direction recovery | Victim accuracy | Surrogate accuracy | Target–surrogate agreement |
| ---------- | -----------------: | --------------: | -----------------: | -------------------------: |
| GELU       |    0.9187 ± 0.0138 |   79.18 ± 0.36% |      77.94 ± 0.47% |              92.81 ± 0.22% |
| SiLU       |    0.9402 ± 0.0119 |   78.18 ± 0.49% |      77.61 ± 0.49% |              94.97 ± 0.36% |

## Additional metrics

| Metric                            |            GELU |            SiLU |
| --------------------------------- | --------------: | --------------: |
| Fraction above 0.90 alignment     | 0.8672 ± 0.0234 | 0.9115 ± 0.0325 |
| Accuracy drop (percentage points) |     1.24 ± 0.23 |     0.57 ± 0.09 |
| KL divergence                     | 0.0432 ± 0.0030 | 0.0209 ± 0.0031 |
| Centered-logit MSE                | 2.6383 ± 0.1761 | 1.2743 ± 0.1840 |
| FFN CKA                           | 0.7979 ± 0.0340 | 0.9034 ± 0.0129 |
| Matched feature correlation       | 0.7128 ± 0.0095 | 0.8107 ± 0.0015 |

## Included files

* [`main_results.csv`](main_results.csv) contains the aggregate results in
  machine-readable form.
* [`shared_runs/`](shared_runs/) contains the result records and recovery
  arrays for the six individual extraction runs.
* Each `result.json` records the complete configuration, query accounting,
  structural-recovery metrics, and functional-replacement metrics.
* Each `recovery_arrays.npz` contains the recovered directions, Hessian
  measurements, projections, and optimization traces.

The six included runs consist of three independently trained target models for
each activation:

```text
shared_runs/
├── gelu_victim0_attack0/
├── gelu_victim1_attack0/
├── gelu_victim2_attack0/
├── silu_victim0_attack0/
├── silu_victim1_attack0/
└── silu_victim2_attack0/
```

## Regenerating the aggregate table

From the `optimized/` directory, run:

```bash
python scripts/aggregate_main_results.py
```

This reads the six records under `results/shared_runs/` and regenerates
`results/main_results.csv`.

## Reproducing an extraction run

The corresponding trained victim checkpoints are included under
[`../checkpoints/`](../checkpoints/). For example, from the `optimized/`
directory:

```bash
bash scripts/run_cifar10_20193.sh \
  ./checkpoints/cifar10_gelu_seed0/best.pt \
  ./runs/cifar10_gelu_victim0_attack0 \
  0
```

The final argument is the extraction seed. This command performs one complete
optimized extraction using 8,193 structural queries and 12,000 completion
queries.

## Interpretation

The optimized attack reduces the structural query budget from 131,088 to 8,193
while preserving strong structural recovery and functional replacement.
Across the six trained targets, mean direction recovery remains above 0.91,
and target–surrogate top-1 agreement remains above 92%.

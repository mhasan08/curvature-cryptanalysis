# Original Separate-Stencil Results

This directory contains the aggregate results for the original, unoptimized
separate-stencil extraction procedure.

The experiments use independently trained CIFAR-10 vision transformers with
GELU or SiLU feed-forward-network activations. Results are reported as the mean
± sample standard deviation across three independently trained victim models.

## Experimental configuration

| Parameter                    |       Value |
| ---------------------------- | ----------: |
| Input dimension \(d\)        |          64 |
| Hidden dimension \(m\)       |         128 |
| Projected Hessians \(T\)     |          16 |
| Finite-difference step \(h\) | \(10^{-2}\) |
| Completion queries           |      12,000 |

In the original implementation, each projected Hessian is evaluated using a
separate finite-difference stencil. Each stencil requires

$$
2d^2+1 = 8{,}193
$$

vector-output queries. Consequently, the structural query count is

$$
T(2d^2+1)
=16\times8{,}193
=131{,}088.
$$

Including 12,000 completion queries, each extraction uses 143,088 total
queries.

## Aggregate results

| Activation | Direction recovery | Victim accuracy | Surrogate accuracy | Target–surrogate agreement |
| ---------- | -----------------: | --------------: | -----------------: | -------------------------: |
| GELU       |    0.9644 ± 0.0026 |   79.18 ± 0.36% |      78.28 ± 0.35% |              93.24 ± 0.45% |
| SiLU       |    0.9476 ± 0.0022 |   78.18 ± 0.49% |      77.57 ± 0.41% |              94.89 ± 0.40% |

## Additional metrics

| Metric                            |            GELU |            SiLU |
| --------------------------------- | --------------: | --------------: |
| Fraction above 0.90 alignment     | 0.9505 ± 0.0119 | 0.9193 ± 0.0090 |
| Accuracy drop (percentage points) |     0.90 ± 0.05 |     0.62 ± 0.12 |
| KL divergence                     | 0.0353 ± 0.0042 | 0.0199 ± 0.0053 |
| Centered-logit MSE                | 2.1961 ± 0.2163 | 1.1991 ± 0.2922 |
| FFN CKA                           | 0.8021 ± 0.0291 | 0.9071 ± 0.0035 |
| Matched feature correlation       | 0.7355 ± 0.0055 | 0.8124 ± 0.0130 |

## Files

* [`main_results.csv`](main_results.csv) contains the aggregate numerical
  results in machine-readable form.
* The implementation and reproduction scripts corresponding to these results
  are located in the repository root.
* Results for the query-optimized shared-stencil implementation are maintained
  separately under [`../optimized/results/`](../optimized/results/).

## Interpretation

The original separate-stencil attack achieves strong structural recovery and
high-fidelity functional replacement for both GELU and SiLU victim models.
Its main limitation is query cost: evaluating an independent stencil for each
of the 16 projected Hessians requires 131,088 structural oracle queries.

# Results

## Primary result

| Checkpoint | no-KD AP50-95 | FD `sample_sum` AP50-95 | Delta |
|---|---:|---:|---:|
| Last | 0.507141 | 0.507415 | +0.000274 |
| Best | **0.509587** | **0.509225** | **-0.000362** |

The last-three-epoch training-metric means were `0.507210` for no-KD and `0.507073` for FD, a
delta of `-0.000137`. The seed-0 performance gate failed and seeds 1-2 were not run.

![Training curves](../experiments/figures/map_curves_seed0.png)

## Size-specific AP

| Metric | no-KD best | FD best | Delta |
|---|---:|---:|---:|
| AP Small | 0.323336 | 0.321367 | -0.001970 |
| AP Medium | 0.563444 | 0.562991 | -0.000453 |
| AP Large | 0.666738 | 0.665870 | -0.000869 |

## TIDE auxiliary analysis

| Error | no-KD | FD | FD - no-KD |
|---|---:|---:|---:|
| Cls | 2.284175 | 2.322259 | +0.038084 |
| Loc | 5.417643 | 5.382084 | -0.035560 |
| Both | 1.117997 | 1.115526 | -0.002471 |
| Dupe | 0.603451 | 0.591723 | -0.011728 |
| Bkg | 4.125355 | 4.106109 | -0.019246 |
| Miss | 4.528051 | 4.566586 | +0.038535 |
| False positive | 18.092136 | 17.862208 | -0.229928 |
| False negative | 9.691544 | 9.860338 | +0.168794 |

FD slightly reduced localization, duplicate, background, and false-positive components while
increasing classification, miss, and false-negative components. These diagnostics are descriptive;
one seed and a near-zero AP delta do not support a causal class-level conclusion.

## Runtime

| Arm | Runtime | Relative |
|---|---:|---:|
| no-KD | 23.41 h | 1.00x |
| FD `sample_sum` | 49.12 h | 2.10x |

Under this setup, FD more than doubled training time without improving the primary endpoint.

## What this result establishes

- The `sample_sum` code path was active.
- FD loss and P3/P4/P5 gradients were finite and non-zero.
- The intervention raised the measured feature-gradient ratio above the signal threshold.
- No accuracy improvement was observed against the matched seed-0 no-KD control.

## What it does not establish

- It does not evaluate upstream `mean` under the exact 50-epoch/batch-16 README recipe.
- It does not estimate a multi-seed mean or confidence interval.
- It does not evaluate a symmetric frequency mask.
- It does not reproduce the referenced paper's full cross-modal objective.

Machine-readable source data are in [`experiments/results`](../experiments/results).

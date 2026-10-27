# Origin-strict FD-CMKD baseline

## Overview

This record captures the completed seed-0 comparison at clean upstream commit `548f2e2`. It used
the upstream `mean` feature reduction, original frequency mask, `KD_W=0.1`, `HEAD_W=0.2`, original
P3/P4/P5 and low/high weights, and the same baseline checkpoint for both arms.

Batch 16 completed for no-KD but FD-CMKD immediately ran out of memory on the 16 GB GPU. That
partial result was preserved but excluded. Both arms were then restarted from the identical
baseline at batch 12 and trained for 50 epochs.

## Result

| Metric | no-KD | FD-CMKD | Delta |
|---|---:|---:|---:|
| Official best AP50-95 | 0.509661648 | 0.509905356 | +0.000243709 |
| AP50 | 0.681031091 | 0.680987174 | -0.000043917 |
| AP75 | 0.556566355 | 0.555429329 | -0.001137025 |
| AP-S | 0.326667609 | 0.327255238 | +0.000587629 |
| AP-M | 0.564727029 | 0.563853701 | -0.000873328 |
| AP-L | 0.676247211 | 0.677651110 | +0.001403898 |
| Last-five mean | 0.509530 | 0.509434 | -0.000096 |

The seed-0 gate failed because the best delta was below `+0.001` and the last-five delta was
negative. Seeds 1-2 were not run.

## Correctness and signal checks

- KD weight zero matched no-KD exactly for loss and the one-step update.
- All 405 compared gradients matched in the zero-weight audit.
- No NaN or Inf was observed.
- The original feature path was finite and non-zero.
- Median combined feature/detection gradient ratio was approximately `4.33e-9`.

These results point to an extremely weak upstream feature-KD signal rather than a broken training
connection. The next experiment—scale-normalized reduction only—is currently in progress.

## Archived evidence

- `results/official_coco_b12_seeds_0.csv`: official COCO best/last metrics and hashes
- `results/seed0_gate.json`: prespecified gate result
- `results/origin_mean_probe_summary.json`: gradient probe summary
- `results/origin_mean_level_summary.csv`: P3/P4/P5 probe breakdown
- `results/zero_weight_equivalence.json`: zero-weight correctness audit
- `manifests/provenance.json`: source, environment, dataset, and checkpoint provenance
- `status.json`: terminal pipeline state

Checkpoints, COCO prediction files, and full training logs are intentionally excluded because of
size. Their identifying hashes and summarized measurements are retained where applicable.

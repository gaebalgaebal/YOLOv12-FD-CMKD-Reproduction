# Methodology

## Question

Does the FD-CMKD feature term provide a meaningful optimization signal, and does increasing that
signal improve YOLOv12-M accuracy relative to an otherwise identical no-KD control?

## Upstream objective

For each P3/P4/P5 feature level, upstream applies channel alignment, a non-local teacher/student
block, spatial mean removal, global sample-wise L2 normalization, frequency decomposition, and a
second L2 normalization. Its FD loss is:

```text
L_fd = low_w  * mean((low_s - low_t)^2)
     + high_w * mean((signed_log1p(high_s) - signed_log1p(high_t))^2)
```

The full training loss also contains YOLO detection loss and head-output KD:

```text
L_total = L_detection
        + kd_w * feature_schedule * sum(level_w * L_fd)
        + head_w * head_schedule * L_head_KD
```

## Signal diagnosis

A no-optimizer-step FP32 probe measured loss contributions and gradients on 16 batches. Model and
checkpoint hashes were verified before and after the probe. With upstream `mean` reduction:

| Quantity | Median |
|---|---:|
| Detection loss | 3.586051 |
| Raw FD feature loss | 1.248970e-5 |
| Effective FD feature loss | 1.248970e-6 |
| Effective feature/detection loss ratio | 3.482856e-7 |
| Combined FD/detection gradient RMS ratio | 2.203983e-8 |

All values were finite and per-level gradients were non-zero, but the combined ratio missed the
predefined `1e-3` signal threshold by approximately 45,372x.

## Experimental reduction

The fork retains upstream behavior by default and adds:

```text
mean:       mean_BCHW((student - teacher)^2)
sample_sum: mean_B(sum_CHW((student - teacher)^2))
```

For a fixed tensor, `sample_sum = C * H * W * mean`. At epoch 5 the `sample_sum` probe reported a
combined gradient ratio of `0.007681`; at epoch 15 it reported `0.007718`. Both passed the signal
gate, establishing that the intervention changed the optimizer-scale signal as intended.

## Controlled comparison

Both arms used the same dataset, starting checkpoint lineage, optimizer family, schedule, batch,
seed, augmentation configuration, and total epoch budget. The treatment difference was the KD path.
The primary endpoint was official COCO AP50-95 at the best checkpoint. A seed-0 gate required:

1. last-three-epoch AP50-95 delta of at least `+0.001`; and
2. a positive official best-checkpoint AP50-95 delta.

Failure of either criterion stopped additional seeds. This avoided spending two more multi-day runs
after the treatment missed both prespecified criteria.

## Interpretation rule

Passing the signal gate means only that FD gradients are finite, non-zero, and large enough relative
to detection gradients. It does not imply those gradients improve the task objective. Accuracy is
judged only against the matched no-KD arm.

# Upstream, paper, and fork differences

## Scope labels

Use the following labels to prevent results from different objectives being conflated:

- **origin-strict**: clean upstream code, `mean` reduction, exact upstream README settings
- **origin + sample_sum**: this fork's loss-scale intervention
- **origin + mask fix**: upstream objective with only a conjugate-symmetric 2D mask
- **paper-faithful**: separate implementation of the paper's complete objective

The completed experiment in this repository is **origin + sample_sum with a low-LR schedule**.

## Paper versus upstream repository

| Topic | Referenced paper | Upstream YOLO repository |
|---|---|---|
| Setting | Cross-modal KD | Same-modal RGB YOLO X-to-M |
| Representation | Feature vector | Detection feature maps `B x C x H x W` |
| Frequency axis | Feature-vector frequency | Per-channel spatial `rfft2` |
| Alignment | Shared-classifier cross entropy | Non-local teacher/student block |
| Task supervision | Raw/low/high feature task terms | Standard YOLO detection loss |
| Extra KD | Not in the main equation | Classification BCE and DFL head KD |

The upstream repository should therefore be described as a YOLO adaptation inspired by FD-CMKD,
not as a literal implementation of the paper's complete method.

## Upstream versus this fork

| Item | Upstream | This fork |
|---|---|---|
| Default reduction | `mean` | `mean` |
| Optional reduction | none | `sample_sum` |
| Probe instrumentation | limited rounded logs | per-level low/high/effective losses and opt-in graph tensors |
| Frequency mask | top-left rectangular mask | unchanged |

## Frequency-mask concern

`torch.fft.rfft2` is one-sided only along the last dimension. The vertical frequency dimension still
contains positive frequencies at the beginning and negative frequencies at the end. Upstream keeps:

```python
mask[..., :keep_h, :keep_w] = 1
```

For height 80 and ratio 0.25, rows 0-19 are kept while DC-adjacent negative-frequency rows 61-79
are discarded. Low and high components still sum back to the input, but the low component is not a
vertically conjugate-symmetric real 2D low-pass decomposition. This remains a hypothesis requiring
the sinusoid tests and controlled mask-only run in the follow-up roadmap.

## Reduction concern

`sample_sum` resolves the measured vanishing-scale problem, but it also weights levels according to
their element counts. It should not be treated as a neutral implementation fix. A dimension-aware
alternative must be selected using fixed-tensor loss/gradient tests before further accuracy runs.

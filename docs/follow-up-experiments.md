# Follow-up experiment roadmap

## Status snapshot

| Priority | Experiment | Status |
|---|---|---|
| 1 | Scale-normalized feature reduction | **In progress** |
| 2 | Head-only KD | Planned |
| 3 | Conjugate-symmetric frequency mask only | Planned |
| 4 | Selected reduction + symmetric mask | Conditional |
| 5 | P3/P4/P5 scale ablations | Conditional |
| 6 | Seeds 1-2 confirmation | Only after the seed-0 gate passes |

The origin-strict seed-0 baseline is complete and failed its performance gate. The immediate
follow-up changes only feature-loss scaling; the original mask, KD weights, data, optimizer,
schedule, and evaluation policy remain fixed.

## P0: scale-normalized reduction (in progress)

Compare upstream `mean`, diagnostic `sample_sum`, and scale-normalized candidates on identical
fixed batches. Record per-level loss, detection/feature/head gradient norms, gradient cosine,
one-step parameter deltas, peak memory, and finite-value checks. Select a candidate only if it
produces a measurable feature signal without dominating detection gradients or unintentionally
weighting P3/P4/P5 by their raw element counts.

The selected candidate will first receive a 5-10 epoch seed-0 screen. A full 50-epoch matched A/B
run follows only if optimization remains stable and the curve is plausibly positive. Seeds 1-2
remain gated on both:

1. official best AP50-95 delta `>= +0.001`; and
2. positive last-five-epoch AP50-95 delta.

The remaining experiments are ordered to maximize information gained before committing to additional multi-day
COCO training.

## P0: correctness gates

### 1. Zero-weight equivalence

Run the KD code path with feature and head coefficients set to zero. Compare the first optimizer
step against no-KD from an identical checkpoint and batch. Parameters, optimizer state, and EMA
must match within a predefined numerical tolerance.

**Purpose:** exclude hidden changes caused by hooks, teacher execution, adapters, or resume state.

### 2. Frequency-mask unit tests

Use horizontal and vertical low/high-frequency sinusoids, impulses, and constant inputs. Compare the
current mask with a wrap-around conjugate-symmetric mask. Check retained energy, reconstruction,
orientation symmetry, and real-valued output.

**Purpose:** determine whether the current decomposition implements the intended 2D low/high split.

### 3. Fixed-tensor reduction benchmark

Compare `mean`, `sample_sum`, and dimension-aware candidates such as a reference-dimension-scaled
mean. Record loss, feature gradient RMS, adapter gradient RMS, and P3/P4/P5 balance on the exact same
tensors.

**Purpose:** select a scale correction without accidentally weighting levels by spatial size.

## P1: strict baseline

### 4. Origin-strict seed 0

From a clean upstream checkout, run matched no-KD and FD arms using 50 epochs, batch 16, upstream
optimizer selection, default LR/warmup, and `mean`. Capture starting validation AP, optimizer choice,
per-level FD signals, best/last official COCO metrics, and complete provenance.

**Decision:** continue only if the origin FD signal is non-negligible or accuracy shows a plausible
positive delta.

### 5. Single-change ladder

Use the strict baseline and change exactly one factor at a time:

1. upstream origin
2. symmetric mask only
3. selected reduction/scaling only
4. symmetric mask plus selected scaling

**Purpose:** attribute any improvement to a specific correction.

## P2: objective ablations

### 6. Feature versus head KD

Run no-KD, head-only, feature-only, and feature+head arms. The upstream feature signal was tiny while
the head contribution was much larger, so aggregate FD-CMKD results cannot identify the responsible
component.

### 7. Frequency ablations

Run low-only, high-only, and low+high after selecting a valid mask and scale. Keep total effective
gradient magnitude comparable across arms.

### 8. Pyramid-level ablations

Run P3-only, P4-only, P5-only, and balanced multi-level variants. Do not reuse raw `sample_sum`
coefficients because feature element counts differ by up to 8x between P3 and P5.

## P3: robust performance estimate

### 9. Multi-seed confirmation

For any configuration that passes the seed-0 gate, run seeds 0, 1, and 2 from equivalent starting
states. Report mean, standard deviation, paired deltas, best and last checkpoints, and bootstrap or
paired confidence intervals.

### 10. Per-class and error analysis

Perform class AP and TIDE analysis only after a repeatable global gain is established. Apply
multiple-comparison caution; class-level fluctuations from one seed are exploratory.

## P4: paper-faithful track

Create a separate branch for the referenced paper's shared-classifier alignment and raw/low/high
task supervision. Do not mix these changes into origin-reproduction commits.

## Reporting checklist

Every run should preserve:

- repository commit and dirty-state patch
- checkpoint and prediction SHA-256 hashes
- full resolved training arguments and KD environment variables
- Python, PyTorch, CUDA, cuDNN, GPU, and driver versions
- dataset identity and annotation hashes
- starting AP before optimization
- epoch curves, official COCO metrics, and wall-clock/GPU-memory cost
- prespecified continuation and success gates

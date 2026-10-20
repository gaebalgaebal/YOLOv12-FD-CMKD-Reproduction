# Experiment artifacts

This directory contains a curated, small-footprint record of the completed seed-0 experiment.

- `configs/`: resolved Ultralytics arguments with machine-specific paths replaced by placeholders
- `results/`: official COCO metrics, training summaries, gates, and no-step signal probes
- `figures/`: training curves, official AP comparison, and upstream-mean loss contribution plot
- `provenance.json`: upstream commit, environment, checkpoint hashes, and continuation policy

Model weights, COCO data, and prediction JSON files are excluded. Hashes in the result artifacts
identify the files used during evaluation. The archived `args.yaml` files describe the stage-15 to
epoch-45 continuation; the methodology document records the full interpretation boundary.

# TAAC-2026-Tencent

Single-model solution for the Tencent Advertising Algorithm Competition 2026
academic track.

![Leaderboard result](assets/leaderboard_result.png)

## Result

| Track | Rank | Best Score | Submission Time |
|---|---:|---:|---|
| Academic Track | 360 | 0.827503 | 2026-05-18 12:04:56 |

This repository open-sources my best stable single-model version from the
competition. The released code is based on the `v43.1` / `v37.2` line, which was
the most reliable version after many feature, architecture, and training
experiments.

No training data, checkpoints, or private competition artifacts are included.

## Problem

The task is a large-scale advertising conversion prediction problem. For each
candidate ad impression, the model receives heterogeneous information:

- user profile features;
- item/ad features;
- dense numerical features;
- four-domain behavior sequences;
- timestamps and time-bucket features;
- binary conversion labels.

The model predicts a conversion probability, and the leaderboard evaluates AUC.
In a real advertising system, this is the ranking model that decides which ad is
more likely to convert for a user at a given moment.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── assets/
│   ├── leaderboard_result.png
│   └── leaderboard_result.svg
├── docs/
│   └── VERSION_NOTES.md
├── eval/
│   ├── dataset.py
│   ├── infer.py
│   └── model.py
├── scripts/
│   └── run_v43_1.sh
└── src/
    ├── dataset.py
    ├── infer.py
    ├── model.py
    ├── ns_groups.json
    ├── train.py
    ├── trainer.py
    └── utils.py
```

## Evaluation Package

The `eval/` folder contains the three files expected by the competition
evaluation upload:

```text
eval/dataset.py
eval/model.py
eval/infer.py
```

The folder is intentionally minimal and does not include training-only files.

## Disclaimer

This is a competition research implementation. The official TAAC dataset is not
redistributed here. Please follow the competition rules and dataset license when
using this code.

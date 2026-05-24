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

## Method

The final submitted model keeps a compact unified ranking backbone and focuses
on stable user-time and dense-pair modeling.

Main components:

- **PCVRHyFormer backbone** for user, item, dense, and sequence features.
- **RankMixer NS tokenizer** for non-sequential user/item features.
- **DensePair compressor** for aligned dense/int field pairs:
  `62,63,64,65,66,89,90,91`.
- **Exposure time context** for impression time.
- **Multi-resolution exposure time** for coarse and fine time patterns.
- **Calendar time embeddings** for local calendar structure.
- **Calendar user activity cross** for user activity at different time periods.
- **User field coverage time context** for profile completeness and time-aware
  user-side signals.
- **Raw-AUC checkpoint selection** with validation diagnostics.

The strongest lesson from the competition was that adding more modules is not
automatically useful. Many larger branches improved local validation but hurt
online AUC. The stable solution keeps the main representation geometry intact
and adds only signals that consistently helped online ranking.

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

## Environment

The code was developed for the TAAC platform PyTorch runtime. A local
environment can be prepared with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The platform provides the actual training data path through environment
variables such as `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, and `TRAIN_LOG_PATH`.

## Training

Use the released best-version training script:

```bash
bash scripts/run_v43_1.sh
```

The key hyperparameters are:

```bash
--d_model 80
--num_heads 5
--num_queries 2
--dense_pair_compressor
--exposure_time_context
--multi_res_exposure_time
--calendar_time_embeddings
--cross_calendar_time_context
--calendar_user_activity_cross
--user_field_coverage_time_context
--checkpoint_selection auc
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

## Notes From Experiments

Useful directions:

- user-side feature coverage;
- exposure and calendar time modeling;
- DensePair modeling for aligned dense/int fields;
- careful checkpoint/export consistency.

Directions that were unstable in my experiments:

- large extra context branches;
- direct final-logit fusion branches;
- full tokenizer replacement;
- heavy query-memory retrieval;
- aggressive sequence reservoir sampling;
- large DIN/SMoE branches;
- raw high-order interaction over large dense fields.

## Disclaimer

This is a competition research implementation. The official TAAC dataset is not
redistributed here. Please follow the competition rules and dataset license when
using this code.

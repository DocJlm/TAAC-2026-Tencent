#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"

# ---- v43.1: v37.2 exact re-export + checkpoint/latency audit ----
python3 -u "${REPO_DIR}/src/train.py" \
    --d_model 80 \
    --num_heads 5 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 7 \
    --item_ns_tokens 4 \
    --num_queries 2 \
    --dense_pair_compressor \
    --dense_pair_fids "62,63,64,65,66,89,90,91" \
    --dense_pair_high_dim_threshold 128 \
    --dense_pair_weight 0.12 \
    --sparse_lr 0.05 \
    --dropout_rate 0.01 \
    --reinit_cardinality_threshold 0 \
    --exposure_time_context \
    --exposure_time_weight 0.026 \
    --multi_res_exposure_time \
    --calendar_time_embeddings \
    --cross_calendar_time_context \
    --cross_calendar_time_weight 0.028 \
    --calendar_user_activity_cross \
    --calendar_user_activity_weight 0.014 \
    --user_field_coverage_time_context \
    --user_field_coverage_time_weight 0.006 \
    --checkpoint_selection auc \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

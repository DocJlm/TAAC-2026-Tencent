"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int, int]]:
    """Build feature_specs of the form ``[(fid, vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((fid, vs, offset, length))
    return specs


def parse_fids(fid_str: str) -> List[int]:
    return [int(x.strip()) for x in str(fid_str).split(',') if x.strip()]


def _feature_role(
    fid: int,
    vocab_size: int,
    length: int,
    pair_fids: List[int],
    low_card_threshold: int,
    high_card_threshold: int,
) -> str:
    if fid in pair_fids:
        return 'dense_aligned'
    if length > 1:
        return 'multi_hot'
    if vocab_size > high_card_threshold:
        return 'high_card'
    if 0 < vocab_size <= low_card_threshold:
        return 'low_card'
    return 'mid_card'


def log_feature_profile(
    user_specs: List[Tuple[int, int, int, int]],
    item_specs: List[Tuple[int, int, int, int]],
    dense_entries: List[Tuple[int, int, int]],
    seq_feature_ids: Dict[str, List[int]],
    seq_vocab_sizes: Dict[str, List[int]],
    pair_fids: List[int],
    low_card_threshold: int,
    high_card_threshold: int,
    seq_low_card_threshold: int,
    seq_id_threshold: int,
) -> None:
    def summarize(name: str, specs: List[Tuple[int, int, int, int]]) -> None:
        counts: Dict[str, int] = {}
        for fid, vs, _, length in specs:
            role = _feature_role(fid, vs, length, pair_fids, low_card_threshold, high_card_threshold)
            counts[role] = counts.get(role, 0) + 1
        logging.info(f"FEATURE_PROFILE {name}: counts={counts}")
        for fid, vs, _, length in specs:
            role = _feature_role(fid, vs, length, pair_fids, low_card_threshold, high_card_threshold)
            logging.info(f"FEATURE_PROFILE {name}: fid={fid} dim={length} vocab={vs} role={role}")

    summarize('user_int', user_specs)
    summarize('item_int', item_specs)
    dense_summary = [
        {'fid': int(fid), 'dim': int(length), 'pair': int(fid) in pair_fids}
        for fid, _, length in dense_entries
    ]
    logging.info(f"FEATURE_PROFILE user_dense: {dense_summary}")
    for domain, vocab_sizes in seq_vocab_sizes.items():
        fids = seq_feature_ids.get(domain, list(range(len(vocab_sizes))))
        low = sum(1 for v in vocab_sizes if 0 < int(v) <= seq_low_card_threshold)
        stat = sum(1 for v in vocab_sizes if seq_low_card_threshold < int(v) <= seq_id_threshold)
        high = sum(1 for v in vocab_sizes if int(v) > seq_id_threshold)
        logging.info(
            f"FEATURE_PROFILE {domain}: fids={fids}, low={low}, stat={stat}, high_card={high}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (v28 keeps 0 as full sparse cold restart: '
                             'reset every Embedding with vocab_size > 0)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer', 'role_rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable), '
                             'role_rankmixer = deterministic role-stratified chunks')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--role_pair_fids', type=str,
                        default='62,63,64,65,66,89,90,91',
                        help='Fids treated as dense/int aligned for role_rankmixer')
    parser.add_argument('--role_low_card_threshold', type=int, default=1000,
                        help='Vocab <= this value is low-card in role_rankmixer')
    parser.add_argument('--role_high_card_threshold', type=int, default=100000,
                        help='Vocab > this value is high-card in role_rankmixer')

    # Dense/int feature partition.
    parser.add_argument('--dense_pair_compressor', action='store_true', default=False,
                        help='Use a single dense token with high-dim, scalar, and dense/int pair paths')
    parser.add_argument('--dense_pair_fids', type=str,
                        default='62,63,64,65,66,89,90,91',
                        help='Dense fids with element-wise aligned user_int fields')
    parser.add_argument('--dense_pair_high_dim_threshold', type=int, default=128,
                        help='Dense feature dim >= this threshold is treated as high-dimensional semantic')
    parser.add_argument('--dense_pair_weight', type=float, default=0.10,
                        help='Residual weight for DenseIntPairCompressor delta path')

    # Sequence event tokenizer.
    parser.add_argument('--semantic_seq_tokenizer', action='store_true', default=False,
                        help='Split sequence event embeddings into low-card/stat/high-card roles')
    parser.add_argument('--seq_low_card_threshold', type=int, default=1000,
                        help='Sequence vocab <= this value is treated as low-card action/context')
    parser.add_argument('--seq_high_card_residual_weight', type=float, default=0.08,
                        help='Bounded residual weight for high-card sequence id-like features')

    # Query context for validated combo.
    parser.add_argument('--use_prequery_role_context', action='store_true', default=False,
                        help='Add one role-aware context token into query generation')
    parser.add_argument('--prequery_context_weight', type=float, default=0.05,
                        help='Scale for the role-aware pre-query context token')

    # v28 controlled query/attention interactions.
    parser.add_argument('--item_crossnet', action='store_true', default=False,
                        help='Add item-conditioned DensePair CrossNet context into query generation')
    parser.add_argument('--item_cross_rank', type=int, default=16,
                        help='Low-rank bottleneck for item-conditioned CrossNet')
    parser.add_argument('--item_cross_weight', type=float, default=0.025,
                        help='Residual scale for item-conditioned CrossNet query context')
    parser.add_argument('--time_attention_bias', action='store_true', default=False,
                        help='Add learned per-domain recency bias to query-to-sequence attention')
    parser.add_argument('--time_attention_clip', type=float, default=0.10,
                        help='Absolute clip for learned attention bias')
    parser.add_argument('--exposure_time_context', action='store_true', default=False,
                        help='Add exposure timestamp/user context into query generation')
    parser.add_argument('--exposure_time_weight', type=float, default=0.02,
                        help='Residual scale for exposure user-time query context')
    parser.add_argument('--multi_res_exposure_time', action='store_true', default=False,
                        help='Use UTC + China-local multi-resolution exposure time features')
    parser.add_argument('--calendar_time_embeddings', action='store_true', default=False,
                        help='Add small learned embeddings for hour/weekday/period exposure buckets')
    parser.add_argument('--cross_calendar_time_context', action='store_true', default=False,
                        help='Add crossed local-hour/weekday exposure-time embeddings')
    parser.add_argument('--cross_calendar_time_weight', type=float, default=0.028,
                        help='Residual scale for crossed calendar-time context')
    parser.add_argument('--user_time_segment_experts', action='store_true', default=False,
                        help='Add small workday/weekend day/night user-time experts')
    parser.add_argument('--segment_expert_weight', type=float, default=0.018,
                        help='Residual scale for user-time segment experts')
    parser.add_argument('--calendar_domain_router', action='store_true', default=False,
                        help='Add calendar/user-conditioned domain query deltas')
    parser.add_argument('--calendar_domain_router_weight', type=float, default=0.010,
                        help='Residual scale for calendar-conditioned domain router')
    parser.add_argument('--query_ranklift_regularizer', action='store_true', default=False,
                        help='Add training-only query diversity/effective-rank regularizer')
    parser.add_argument('--query_ranklift_weight', type=float, default=1.0e-4,
                        help='Weight for query RankLift regularization loss')
    parser.add_argument('--query_ranklift_warmup_epoch', type=int, default=2,
                        help='First epoch that applies query RankLift regularization')
    parser.add_argument('--user_time_film', action='store_true', default=False,
                        help='Apply tiny exposure-time FiLM to user-side tokens')
    parser.add_argument('--user_time_film_weight', type=float, default=0.015,
                        help='Residual scale for user-time FiLM token modulation')
    parser.add_argument('--time_delta_sidecar', action='store_true', default=False,
                        help='Add query context from per-domain sequence time-delta histograms')
    parser.add_argument('--time_delta_sidecar_weight', type=float, default=0.015,
                        help='Residual scale for time-delta histogram sidecar')
    parser.add_argument('--user_time_evidence_block', action='store_true', default=False,
                        help='Add tiny MetaFormer-style user/time evidence RecBlock query context')
    parser.add_argument('--user_time_evidence_weight', type=float, default=0.015,
                        help='Residual scale for unified user-time evidence RecBlock')
    parser.add_argument('--target_lite_domain_router', action='store_true', default=False,
                        help='Add low-card target/user-time/domain-evidence context into query generation')
    parser.add_argument('--target_lite_low_card_threshold', type=int, default=1000,
                        help='Max vocab size for target-lite low-card item features')
    parser.add_argument('--target_lite_weight', type=float, default=0.012,
                        help='Residual scale for target-lite domain router context')
    parser.add_argument('--low_card_temporal_content_sidecar', action='store_true', default=False,
                        help='Add low-card action/stat sequence content sidecar into evidence block')
    parser.add_argument('--low_card_content_weight', type=float, default=0.012,
                        help='Residual scale for low-card temporal content sidecar')
    parser.add_argument('--query_conditioned_time_attention', action='store_true', default=False,
                        help='Condition time attention bias on generated q tokens')
    parser.add_argument('--query_time_rank', type=int, default=8,
                        help='Low-rank size for query-conditioned time attention')
    parser.add_argument('--user_time_item_mixer', action='store_true', default=False,
                        help='Add low-rank user/time/item evidence mixer into query generation')
    parser.add_argument('--user_time_item_rank', type=int, default=8,
                        help='Low-rank bottleneck for user/time/item evidence mixer')
    parser.add_argument('--user_time_item_weight', type=float, default=0.015,
                        help='Residual scale for user/time/item evidence mixer')
    parser.add_argument('--calendar_user_activity_cross', action='store_true', default=False,
                        help='Cross exposure calendar buckets with stable user-activity summaries')
    parser.add_argument('--calendar_user_activity_weight', type=float, default=0.014,
                        help='Residual scale for calendar x user-activity query context')
    parser.add_argument('--head_recent_activity_cross', action='store_true', default=False,
                        help='Cross head-window recent user activity with China-local period/weekend buckets')
    parser.add_argument('--head_recent_activity_weight', type=float, default=0.006,
                        help='Residual scale for head-recent activity query context')
    parser.add_argument('--head_recent_activity_k', type=int, default=64,
                        help='Head sequence window size for recent activity summaries')
    parser.add_argument('--user_field_coverage_time_context', action='store_true', default=False,
                        help='Cross low-card user field coverage summaries with China-local time')
    parser.add_argument('--user_field_coverage_time_weight', type=float, default=0.006,
                        help='Residual scale for user field coverage x time context')
    parser.add_argument('--dense_semantic_time_bilinear', action='store_true', default=False,
                        help='Cross DensePair semantic/scalar/pair summaries with exposure calendar time')
    parser.add_argument('--dense_semantic_time_rank', type=int, default=8,
                        help='Low-rank size for dense semantic time-bilinear context')
    parser.add_argument('--dense_semantic_time_weight', type=float, default=0.012,
                        help='Residual scale for dense semantic time-bilinear query context')
    parser.add_argument('--calendar_balanced_loss', action='store_true', default=False,
                        help='Use calendar-bucket sample weights for train BCE/focal loss')
    parser.add_argument('--calendar_balanced_clip_min', type=float, default=0.90,
                        help='Minimum calendar-bucket loss weight')
    parser.add_argument('--calendar_balanced_clip_max', type=float, default=1.10,
                        help='Maximum calendar-bucket loss weight')
    parser.add_argument('--checkpoint_selection', type=str, default='auc',
                        choices=['auc', 'robust_slices'],
                        help='Checkpoint selector: raw AUC or raw-AUC-priority robust slice selector')
    parser.add_argument('--robust_slice_penalty', type=float, default=0.002,
                        help='Penalty applied to validation slice AUC std for robust_slices logging')
    parser.add_argument('--robust_auc_margin', type=float, default=0.00025,
                        help='Raw AUC tie margin before robust slice stability can select a checkpoint')
    parser.add_argument('--robust_logloss_tolerance', type=float, default=0.0002,
                        help='Maximum LogLoss rebound allowed for robust_slices tie-breaks')
    parser.add_argument('--hrrm_dq_adapter', action='store_true', default=False,
                        help='Enable minimal HRRM/domain-quality query adapter')
    parser.add_argument('--hrrm_query_weight', type=float, default=0.02,
                        help='Query residual scale for minimal HRRM-DQ adapter')
    parser.add_argument('--hrrm_memory_weight', type=float, default=0.03,
                        help='Memory residual scale for minimal HRRM-DQ adapter')
    parser.add_argument('--hrrm_recent_k', type=int, default=64,
                        help='Recent window size for HRRM-DQ recent memory')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    dense_pair_fids = parse_fids(args.dense_pair_fids)
    log_feature_profile(
        user_specs=user_int_feature_specs,
        item_specs=item_int_feature_specs,
        dense_entries=pcvr_dataset.user_dense_schema.entries,
        seq_feature_ids=pcvr_dataset.seq_feature_ids,
        seq_vocab_sizes=pcvr_dataset.seq_domain_vocab_sizes,
        pair_fids=dense_pair_fids,
        low_card_threshold=args.role_low_card_threshold,
        high_card_threshold=args.role_high_card_threshold,
        seq_low_card_threshold=args.seq_low_card_threshold,
        seq_id_threshold=args.seq_id_threshold,
    )

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "role_pair_fids": args.role_pair_fids,
        "role_low_card_threshold": args.role_low_card_threshold,
        "role_high_card_threshold": args.role_high_card_threshold,
        "dense_pair_compressor": args.dense_pair_compressor,
        "dense_pair_fids": args.dense_pair_fids,
        "dense_pair_high_dim_threshold": args.dense_pair_high_dim_threshold,
        "dense_pair_weight": args.dense_pair_weight,
        "semantic_seq_tokenizer": args.semantic_seq_tokenizer,
        "seq_low_card_threshold": args.seq_low_card_threshold,
        "seq_high_card_residual_weight": args.seq_high_card_residual_weight,
        "use_prequery_role_context": args.use_prequery_role_context,
        "prequery_context_weight": args.prequery_context_weight,
        "item_crossnet": args.item_crossnet,
        "item_cross_rank": args.item_cross_rank,
        "item_cross_weight": args.item_cross_weight,
        "time_attention_bias": args.time_attention_bias,
        "time_attention_clip": args.time_attention_clip,
        "exposure_time_context": args.exposure_time_context,
        "exposure_time_weight": args.exposure_time_weight,
        "multi_res_exposure_time": args.multi_res_exposure_time,
        "calendar_time_embeddings": args.calendar_time_embeddings,
        "cross_calendar_time_context": args.cross_calendar_time_context,
        "cross_calendar_time_weight": args.cross_calendar_time_weight,
        "user_time_segment_experts": args.user_time_segment_experts,
        "segment_expert_weight": args.segment_expert_weight,
        "calendar_domain_router": args.calendar_domain_router,
        "calendar_domain_router_weight": args.calendar_domain_router_weight,
        "query_ranklift_regularizer": args.query_ranklift_regularizer,
        "query_ranklift_weight": args.query_ranklift_weight,
        "query_ranklift_warmup_epoch": args.query_ranklift_warmup_epoch,
        "user_time_film": args.user_time_film,
        "user_time_film_weight": args.user_time_film_weight,
        "time_delta_sidecar": args.time_delta_sidecar,
        "time_delta_sidecar_weight": args.time_delta_sidecar_weight,
        "user_time_evidence_block": args.user_time_evidence_block,
        "user_time_evidence_weight": args.user_time_evidence_weight,
        "target_lite_domain_router": args.target_lite_domain_router,
        "target_lite_low_card_threshold": args.target_lite_low_card_threshold,
        "target_lite_weight": args.target_lite_weight,
        "low_card_temporal_content_sidecar": args.low_card_temporal_content_sidecar,
        "low_card_content_weight": args.low_card_content_weight,
        "query_conditioned_time_attention": args.query_conditioned_time_attention,
        "query_time_rank": args.query_time_rank,
        "user_time_item_mixer": args.user_time_item_mixer,
        "user_time_item_rank": args.user_time_item_rank,
        "user_time_item_weight": args.user_time_item_weight,
        "calendar_user_activity_cross": args.calendar_user_activity_cross,
        "calendar_user_activity_weight": args.calendar_user_activity_weight,
        "head_recent_activity_cross": args.head_recent_activity_cross,
        "head_recent_activity_weight": args.head_recent_activity_weight,
        "head_recent_activity_k": args.head_recent_activity_k,
        "user_field_coverage_time_context": args.user_field_coverage_time_context,
        "user_field_coverage_time_weight": args.user_field_coverage_time_weight,
        "dense_semantic_time_bilinear": args.dense_semantic_time_bilinear,
        "dense_semantic_time_rank": args.dense_semantic_time_rank,
        "dense_semantic_time_weight": args.dense_semantic_time_weight,
        "hrrm_dq_adapter": args.hrrm_dq_adapter,
        "hrrm_query_weight": args.hrrm_query_weight,
        "hrrm_memory_weight": args.hrrm_memory_weight,
        "hrrm_recent_k": args.hrrm_recent_k,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        calendar_balanced_loss=args.calendar_balanced_loss,
        calendar_balanced_clip_min=args.calendar_balanced_clip_min,
        calendar_balanced_clip_max=args.calendar_balanced_clip_max,
        checkpoint_selection=args.checkpoint_selection,
        robust_slice_penalty=args.robust_slice_penalty,
        robust_auc_margin=args.robust_auc_margin,
        robust_logloss_tolerance=args.robust_logloss_tolerance,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()

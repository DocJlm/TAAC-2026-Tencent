"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        calendar_balanced_loss: bool = False,
        calendar_balanced_clip_min: float = 0.90,
        calendar_balanced_clip_max: float = 1.10,
        checkpoint_selection: str = 'auc',
        robust_slice_penalty: float = 0.002,
        robust_auc_margin: float = 0.00025,
        robust_logloss_tolerance: float = 0.0002,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.calendar_balanced_loss: bool = bool(calendar_balanced_loss)
        self.calendar_balanced_clip_min: float = float(calendar_balanced_clip_min)
        self.calendar_balanced_clip_max: float = float(calendar_balanced_clip_max)
        self.calendar_bucket_weights: Optional[torch.Tensor] = None
        self._calendar_weighted_loss_sum: float = 0.0
        self._calendar_weighted_loss_count: int = 0
        self.checkpoint_selection: str = str(checkpoint_selection)
        self.robust_slice_penalty: float = float(robust_slice_penalty)
        self.robust_auc_margin: float = float(robust_auc_margin)
        self.robust_logloss_tolerance: float = float(robust_logloss_tolerance)
        self._last_robust_score: Optional[float] = None
        self._last_slice_std: Optional[float] = None
        self._last_slice_metrics: Dict[str, float] = {}
        self._best_selector_raw_auc: Optional[float] = None
        self._best_selector_robust_score: Optional[float] = None
        self._best_selector_logloss: Optional[float] = None
        self._top_ckpt_stats: List[Dict[str, float]] = []
        self._dense_pair_fids: Tuple[int, ...] = (62, 63, 64, 65, 66, 89, 90, 91)
        self._user_dense_pair_slices: List[Tuple[int, int]] = []
        self._user_int_pair_slices: List[Tuple[int, int]] = []
        self._init_validation_slice_layout()

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")
        if self.calendar_balanced_loss:
            logging.info(
                "Calendar-balanced loss enabled: "
                f"clip=[{self.calendar_balanced_clip_min}, {self.calendar_balanced_clip_max}]"
            )
        if self.checkpoint_selection == 'robust_slices':
            logging.info(
                "Robust checkpoint selector enabled: "
                f"raw_auc_margin={self.robust_auc_margin}, "
                f"slice_penalty={self.robust_slice_penalty}, "
                f"logloss_tolerance={self.robust_logloss_tolerance}"
            )

    def _init_validation_slice_layout(self) -> None:
        dataset = getattr(self.valid_loader, 'dataset', None)
        dense_schema = getattr(dataset, 'user_dense_schema', None)
        int_schema = getattr(dataset, 'user_int_schema', None)
        dense_entries = getattr(dense_schema, 'entries', []) or []
        int_entries = getattr(int_schema, 'entries', []) or []
        self._user_dense_pair_slices = [
            (int(offset), int(length))
            for fid, offset, length in dense_entries
            if int(fid) in self._dense_pair_fids
        ]
        self._user_int_pair_slices = [
            (int(offset), int(length))
            for fid, offset, length in int_entries
            if int(fid) in self._dense_pair_fids
        ]

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    @staticmethod
    def _safe_slice_auc(labels: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if mask.sum() < 32:
            return None
        y = labels[mask]
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, probs[mask]))

    def _validation_slice_features(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        seq_domains = batch.get('_seq_domains', [])
        label = batch['label']
        B = int(label.shape[0])
        device = label.device

        total_seq_len = torch.zeros(B, dtype=torch.float32, device=device)
        head_recent_ratios: List[torch.Tensor] = []
        domain_empty = torch.zeros(B, dtype=torch.bool, device=device)
        for domain in seq_domains:
            lens = batch.get(f'{domain}_len')
            if lens is not None:
                total_seq_len = total_seq_len + lens.float()
            tb = batch.get(f'{domain}_time_bucket')
            seq = batch.get(domain)
            if tb is not None:
                valid = tb > 0
            elif seq is not None:
                valid = torch.ones(seq.shape[0], seq.shape[2], dtype=torch.bool, device=device)
            else:
                continue
            head_k = min(64, int(valid.shape[1]))
            if head_k > 0:
                head_valid = valid[:, :head_k]
                head_recent_ratios.append(head_valid.float().mean(dim=1))
            domain_empty = domain_empty | (valid.sum(dim=1) <= 0)

        if head_recent_ratios:
            recent_ratio = torch.stack(head_recent_ratios, dim=1).mean(dim=1)
        else:
            recent_ratio = torch.zeros(B, dtype=torch.float32, device=device)

        dense_feats = batch.get('user_dense_feats')
        if dense_feats is not None and dense_feats.numel() > 0:
            dense_coverage = (dense_feats.abs() > 1.0e-12).float().mean(dim=1)
        else:
            dense_coverage = torch.zeros(B, dtype=torch.float32, device=device)

        dense_pair_present = torch.zeros(B, dtype=torch.bool, device=device)
        if dense_feats is not None:
            for offset, length in self._user_dense_pair_slices:
                dense_pair_present = dense_pair_present | (
                    dense_feats[:, offset:offset + length].abs().sum(dim=1) > 1.0e-12
                )
        int_feats = batch.get('user_int_feats')
        if int_feats is not None:
            for offset, length in self._user_int_pair_slices:
                dense_pair_present = dense_pair_present | (
                    int_feats[:, offset:offset + length].long().ne(0).sum(dim=1) > 0
                )

        timestamp = batch.get('timestamp')
        if timestamp is None:
            hour = torch.zeros(B, dtype=torch.long, device=device)
            weekday = torch.zeros(B, dtype=torch.long, device=device)
        else:
            ts_cn = timestamp.float() + 8.0 * 3600.0
            hour = torch.remainder(torch.floor(ts_cn / 3600.0), 24.0).long()
            weekday = torch.remainder(torch.floor(ts_cn / 86400.0) + 4.0, 7.0).long()
        period3 = torch.floor(hour.float() / 3.0).long().clamp(0, 7)
        weekend = weekday >= 5

        return {
            'hour': hour.detach().cpu(),
            'period3': period3.detach().cpu(),
            'weekend': weekend.detach().cpu(),
            'total_seq_len': total_seq_len.detach().cpu(),
            'domain_empty': domain_empty.detach().cpu(),
            'recent_ratio': recent_ratio.detach().cpu(),
            'dense_coverage': dense_coverage.detach().cpu(),
            'dense_pair_present': dense_pair_present.detach().cpu(),
        }

    def _log_validation_slices(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        slice_features: Dict[str, np.ndarray],
    ) -> Tuple[float, float, Dict[str, float]]:
        slice_metrics: Dict[str, float] = {}

        def add(name: str, mask: np.ndarray) -> None:
            auc = self._safe_slice_auc(labels, probs, mask.astype(bool))
            if auc is not None:
                slice_metrics[name] = auc

        if 'hour' in slice_features:
            hour = slice_features['hour']
            for h in range(24):
                add(f'hour={h:02d}', hour == h)
        if 'period3' in slice_features:
            period3 = slice_features['period3']
            for p in range(8):
                add(f'period3={p}', period3 == p)
        if 'weekend' in slice_features:
            weekend = slice_features['weekend'].astype(bool)
            add('weekday', ~weekend)
            add('weekend', weekend)
        if 'total_seq_len' in slice_features:
            total_len = slice_features['total_seq_len']
            add('seq_short_p40', total_len <= np.quantile(total_len, 0.40))
            add('seq_long_p60', total_len >= np.quantile(total_len, 0.60))
        if 'domain_empty' in slice_features:
            domain_empty = slice_features['domain_empty'].astype(bool)
            add('domain_empty', domain_empty)
            add('domain_nonempty', ~domain_empty)
        if 'recent_ratio' in slice_features:
            recent = slice_features['recent_ratio']
            add('recent_heavy_p70', recent >= np.quantile(recent, 0.70))
            add('recent_light_p30', recent <= np.quantile(recent, 0.30))
        if 'dense_coverage' in slice_features:
            dense_cov = slice_features['dense_coverage']
            add('dense_low_p30', dense_cov <= np.quantile(dense_cov, 0.30))
            add('dense_high_p70', dense_cov >= np.quantile(dense_cov, 0.70))
        if 'dense_pair_present' in slice_features:
            pair_present = slice_features['dense_pair_present'].astype(bool)
            add('dense_pair_present', pair_present)
            add('dense_pair_absent', ~pair_present)

        auc_values = list(slice_metrics.values())
        slice_std = float(np.std(auc_values)) if len(auc_values) >= 2 else 0.0
        if len(np.unique(labels)) < 2:
            overall_auc = 0.0
        else:
            overall_auc = float(roc_auc_score(labels, probs))
        robust_score = overall_auc - self.robust_slice_penalty * slice_std
        ordered = ", ".join(
            f"{name}:{value:.6f}" for name, value in sorted(slice_metrics.items())
        )
        logging.info(
            f"Validation robust_slices | {ordered} | "
            f"slice_std={slice_std:.6f}, robust_score={robust_score:.6f}"
        )
        return robust_score, slice_std, slice_metrics

    def _validation_selection_score(self, val_auc: float, val_logloss: float) -> float:
        if self.checkpoint_selection != 'robust_slices':
            return val_auc
        old_best = self.early_stopping.best_score
        robust_score = self._last_robust_score if self._last_robust_score is not None else val_auc
        best_raw = self._best_selector_raw_auc
        best_robust = self._best_selector_robust_score
        best_logloss = self._best_selector_logloss
        if best_raw is None or old_best is None:
            return val_auc
        if val_auc > best_raw + self.robust_auc_margin:
            return old_best + self.early_stopping.delta + 1.0e-6
        raw_tied = val_auc >= best_raw - self.robust_auc_margin
        robust_better = best_robust is None or robust_score > best_robust + 1.0e-12
        logloss_ok = best_logloss is None or val_logloss <= best_logloss + self.robust_logloss_tolerance
        if raw_tied and robust_better and logloss_ok:
            return old_best + self.early_stopping.delta + 1.0e-6
        return old_best

    def _log_top_ckpt_stats(self, total_step: int, val_auc: float, val_logloss: float) -> None:
        stat = {
            "step": float(total_step),
            "auc": float(val_auc),
            "logloss": float(val_logloss),
            "slice_std": float(self._last_slice_std or 0.0),
            "robust_score": float(self._last_robust_score if self._last_robust_score is not None else val_auc),
        }
        self._top_ckpt_stats.append(stat)
        self._top_ckpt_stats = sorted(
            self._top_ckpt_stats, key=lambda x: (x["auc"], -x["logloss"]), reverse=True
        )[:3]
        best_auc = self._top_ckpt_stats[0]["auc"] if self._top_ckpt_stats else float(val_auc)
        desc = []
        for rank, item in enumerate(self._top_ckpt_stats, start=1):
            desc.append(
                "#{rank}:step={step:.0f},auc={auc:.6f},logloss={logloss:.6f},"
                "delta={delta:+.6f},slice_std={slice_std:.6f}".format(
                    rank=rank,
                    step=item["step"],
                    auc=item["auc"],
                    logloss=item["logloss"],
                    delta=item["auc"] - best_auc,
                    slice_std=item["slice_std"],
                )
            )
        logging.info("TOP3_CHECKPOINT_STATS | " + " | ".join(desc))

    def _enabled_flag_summary(self) -> str:
        cfg = self.train_config or {}
        interesting = [
            "dense_pair_compressor",
            "exposure_time_context",
            "multi_res_exposure_time",
            "calendar_time_embeddings",
            "cross_calendar_time_context",
            "calendar_user_activity_cross",
            "user_field_coverage_time_context",
            "time_attention_bias",
            "query_conditioned_time_attention",
            "query_conditioned_time_bias_v2",
            "densepair_internal_alignment_v2",
            "v43_optimizer_split_only",
        ]
        enabled = [name for name in interesting if bool(cfg.get(name, False))]
        return ",".join(enabled) if enabled else "none"

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        selector_score = self._validation_selection_score(val_auc, val_logloss)
        self._log_top_ckpt_stats(total_step, val_auc, val_logloss)
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or selector_score > old_best + self.early_stopping.delta
        )
        extra_metrics = {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
            "checkpoint_score": selector_score,
            "checkpoint_selection": self.checkpoint_selection,
            "valid_robust_score": self._last_robust_score,
            "valid_slice_std": self._last_slice_std,
            "valid_raw_auc_margin": self.robust_auc_margin,
        }
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(selector_score, self.model, extra_metrics)
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(selector_score, self.model, extra_metrics)

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._best_selector_raw_auc = val_auc
            self._best_selector_robust_score = (
                self._last_robust_score if self._last_robust_score is not None else val_auc
            )
            self._best_selector_logloss = val_logloss
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        if self.calendar_balanced_loss and self.calendar_bucket_weights is None:
            self._fit_calendar_bucket_weights()
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch, epoch=epoch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                    if (
                        getattr(self.model, 'query_ranklift_regularizer', False)
                        and total_step % 100 == 0
                        and hasattr(self.model, 'query_ranklift_metrics')
                    ):
                        for name, value in self.model.query_ranklift_metrics().items():
                            self.writer.add_scalar(f'RankLift/{name}', value, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")
            if self._calendar_weighted_loss_count > 0:
                logging.info(
                    "Calendar-balanced weighted loss mean: "
                    f"{self._calendar_weighted_loss_sum / self._calendar_weighted_loss_count:.6f}"
                )
                self._calendar_weighted_loss_sum = 0.0
                self._calendar_weighted_loss_count = 0

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize sparse Embeddings whose
            # vocab exceeds the threshold. threshold=0 is intentionally kept for
            # v28 because it reproduces the v25.2 full sparse cold-restart
            # regularization that beat the baseline.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    @staticmethod
    def _calendar_bucket_ids(timestamp: torch.Tensor) -> torch.Tensor:
        ts_cn = timestamp.float() + 8.0 * 3600.0
        hour_cn = torch.remainder(torch.floor(ts_cn / 3600.0), 24.0).long()
        weekday_cn = torch.remainder(torch.floor(ts_cn / 86400.0) + 4.0, 7.0).long()
        period3 = torch.floor(hour_cn.float() / 3.0).long().clamp(0, 7)
        return (period3 * 7 + weekday_cn).clamp(0, 55)

    def _fit_calendar_bucket_weights(self) -> None:
        """Estimate 3-hour x weekday bucket frequencies from train row groups."""
        counts = np.zeros(56, dtype=np.float64)
        dataset = getattr(self.train_loader, 'dataset', None)
        rg_list = getattr(dataset, '_rg_list', None)
        if not rg_list:
            logging.warning("Calendar-balanced loss could not inspect train row groups; using uniform weights.")
            self.calendar_bucket_weights = torch.ones(56, dtype=torch.float32)
            return
        try:
            import pyarrow.parquet as pq
            parquet_cache: Dict[str, Any] = {}
            for file_path, rg_idx, _ in rg_list:
                pf = parquet_cache.get(file_path)
                if pf is None:
                    pf = pq.ParquetFile(file_path)
                    parquet_cache[file_path] = pf
                table = pf.read_row_group(rg_idx, columns=['timestamp'])
                timestamps = table.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
                ts = torch.from_numpy(timestamps)
                bucket = self._calendar_bucket_ids(ts).numpy()
                counts += np.bincount(bucket, minlength=56)
        except Exception as exc:  # pragma: no cover - platform data dependent
            logging.warning(f"Calendar-balanced loss frequency scan failed: {exc}; using uniform weights.")
            self.calendar_bucket_weights = torch.ones(56, dtype=torch.float32)
            return

        nonzero = counts > 0
        weights = np.ones(56, dtype=np.float64)
        if nonzero.any():
            mean_count = counts[nonzero].mean()
            weights[nonzero] = np.sqrt(mean_count / counts[nonzero])
        weights = np.clip(weights, self.calendar_balanced_clip_min, self.calendar_balanced_clip_max)
        self.calendar_bucket_weights = torch.tensor(weights, dtype=torch.float32)
        bucket_stats = [
            (int(i), float(counts[i] / max(counts.sum(), 1.0)), float(weights[i]))
            for i in range(56)
        ]
        top = sorted([(int(i), int(c)) for i, c in enumerate(counts)], key=lambda x: x[1], reverse=True)[:8]
        logging.info(
            "Calendar-balanced loss fitted 56 buckets: "
            f"total={int(counts.sum())}, nonzero={int(nonzero.sum())}, "
            f"weight_min={weights.min():.4f}, weight_max={weights.max():.4f}, "
            f"top_counts={top}, bucket_share_weight={bucket_stats}"
        )

    def _calendar_sample_weights(self, timestamp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.calendar_balanced_loss or self.calendar_bucket_weights is None or timestamp is None:
            return None
        bucket = self._calendar_bucket_ids(timestamp.detach().to('cpu'))
        weights = self.calendar_bucket_weights[bucket].to(device=timestamp.device, dtype=torch.float32)
        return weights

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            timestamp=device_batch.get('timestamp', None),
        )

    def _train_step(self, batch: Dict[str, Any], epoch: int = 1) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        if self.loss_type == 'focal':
            loss_vec = sigmoid_focal_loss(
                logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')
        else:
            loss_vec = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
        sample_weights = self._calendar_sample_weights(device_batch.get('timestamp', None))
        if sample_weights is not None:
            loss = (loss_vec * sample_weights).sum() / sample_weights.sum().clamp(min=1.0)
            self._calendar_weighted_loss_sum += float((loss_vec.detach() * sample_weights).mean().cpu().item())
            self._calendar_weighted_loss_count += 1
        else:
            loss = loss_vec.mean()
        if (
            getattr(self.model, 'query_ranklift_regularizer', False)
            and epoch >= getattr(self.model, 'query_ranklift_warmup_epoch', 2)
            and hasattr(self.model, 'query_ranklift_regularization_loss')
        ):
            ranklift_loss = self.model.query_ranklift_regularization_loss()
            if ranklift_loss is not None:
                loss = loss + getattr(self.model, 'query_ranklift_weight', 0.0) * ranklift_loss
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        slice_feature_lists: Dict[str, List[torch.Tensor]] = {}
        data_times: List[float] = []
        forward_times: List[float] = []
        batch_sizes: List[int] = []
        last_batch_end = time.perf_counter()

        with torch.no_grad():
            for step, batch in pbar:
                batch_ready = time.perf_counter()
                data_times.append(batch_ready - last_batch_end)
                fwd_start = time.perf_counter()
                logits, labels = self._evaluate_step(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fwd_end = time.perf_counter()
                forward_times.append(fwd_end - fwd_start)
                batch_sizes.append(int(labels.numel()))
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                for name, values in self._validation_slice_features(batch).items():
                    slice_feature_lists.setdefault(name, []).append(values)
                last_batch_end = time.perf_counter()

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        self._last_robust_score = auc
        self._last_slice_std = 0.0
        self._last_slice_metrics = {}
        if slice_feature_lists:
            slice_features_np: Dict[str, np.ndarray] = {}
            for name, chunks in slice_feature_lists.items():
                values = torch.cat(chunks, dim=0).numpy()
                if nan_mask.any():
                    values = values[~nan_mask]
                slice_features_np[name] = values
            robust_score, slice_std, slice_metrics = self._log_validation_slices(
                labels_np, probs, slice_features_np)
            self._last_robust_score = robust_score
            self._last_slice_std = slice_std
            self._last_slice_metrics = slice_metrics

        if forward_times:
            avg_forward_ms = 1000.0 * float(np.mean(forward_times))
            avg_data_ms = 1000.0 * float(np.mean(data_times)) if data_times else 0.0
            p95_forward_ms = 1000.0 * float(np.percentile(forward_times, 95))
            mean_batch = float(np.mean(batch_sizes)) if batch_sizes else 0.0
            logging.info(
                "VALIDATION_PROFILER | "
                f"batch_size_mean={mean_batch:.1f}, num_workers={getattr(self.valid_loader, 'num_workers', 'na')}, "
                f"forward_ms_per_batch={avg_forward_ms:.3f}, forward_p95_ms={p95_forward_ms:.3f}, "
                f"dataloader_ms_per_batch={avg_data_ms:.3f}, enabled_flags={self._enabled_flag_summary()}"
            )

        if hasattr(self.model, "runtime_debug_metrics"):
            try:
                metrics = self.model.runtime_debug_metrics(reset=True)
            except TypeError:
                metrics = self.model.runtime_debug_metrics()
            if metrics:
                metric_str = ", ".join(f"{k}={v:.6f}" for k, v in sorted(metrics.items()))
                logging.info(f"MODEL_RUNTIME_METRICS | {metric_str}")

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label

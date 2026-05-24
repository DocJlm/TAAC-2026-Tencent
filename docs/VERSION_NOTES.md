# v43.1 - v37.2 Exact Re-export + CKPT/Latency Audit

This version intentionally keeps the v37.2 model and recipe unchanged.

It exists to answer a practical question from v42: whether the large online
latency regression came from model ideas or from export/eval package drift.
The only additions are diagnostics:

- top-3 validation checkpoint stats by raw AUC;
- validation profiler with dataloader and forward milliseconds per batch;
- inference profiler in `infer.py`;
- minimal eval model code copied from v37.2, without v42 modules.

Selector remains raw validation AUC. No new features, loss, reinit, context, or
architecture branches are enabled.

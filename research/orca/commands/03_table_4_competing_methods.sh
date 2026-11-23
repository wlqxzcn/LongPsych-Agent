#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_competing_methods_local.py" \
  --data "$DATA" \
  --models_root "$MODELS_ROOT" \
  --models gte,bge \
  --gte_path "$GTE_MODEL" \
  --bge_path "$BGE_MODEL" \
  --gte_batch_size "$GTE_BATCH_SIZE" \
  --bge_batch_size "$BGE_BATCH_SIZE" \
  --k_list 1,2,3,4,8,16,32,64 \
  --top_k 10 \
  --fusion_depth 1000 \
  --selection_metric R@5 \
  --random_trials 5 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache_dir "$ROOT/cache/competing_gte_bge" \
  --output_json "$ROOT/results/table_4_competing_methods.json" \
  --output_md "$ROOT/results/table_4_competing_methods.md" \
  2>&1 | tee "$ROOT/logs/03_table_4_competing_methods.log"

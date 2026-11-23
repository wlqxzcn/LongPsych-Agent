#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_full_experiments_local.py" \
  --data "$DATA" \
  --models_root "$MODELS_ROOT" \
  --models gte,bge \
  --gte_path "$GTE_MODEL" \
  --bge_path "$BGE_MODEL" \
  --gte_batch_size "$GTE_BATCH_SIZE" \
  --bge_batch_size "$BGE_BATCH_SIZE" \
  --k_list 1,2,3,4,8,16,32,64 \
  --top_k 10 \
  --selection_metric R@5 \
  --device "$DEVICE" \
  --cache_dir "$ROOT/cache/main_gte_bge" \
  --output_json "$ROOT/results/tables_2_3_main.json" \
  --output_md "$ROOT/results/tables_2_3_main.md" \
  2>&1 | tee "$ROOT/logs/02_tables_2_3_main.log"

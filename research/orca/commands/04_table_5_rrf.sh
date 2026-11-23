#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_orca_rrf_fusion_local.py" \
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
  --rrf_k_list 10,30,60,100 \
  --dense_weight_list 0.5,1.0,1.5,2.0 \
  --orca_weight_list 0.5,1.0,1.5,2.0 \
  --bm25_weight_list 0.5,1.0,1.5,2.0 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache_dir "$ROOT/cache/rrf_gte_bge" \
  --output_json "$ROOT/results/table_5_rrf.json" \
  --output_md "$ROOT/results/table_5_rrf.md" \
  2>&1 | tee "$ROOT/logs/04_table_5_rrf.log"

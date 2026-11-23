#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/export_dense_orca_qualitative_cases.py" \
  --benchmark_json "$DATA" \
  --encoder_name "$GTE_MODEL" \
  --split test \
  --batch_size "$GTE_BATCH_SIZE" \
  --device "$DEVICE" \
  --cache_dir "$ROOT/cache/qualitative_gte" \
  --out_dir "$ROOT/results/table_14_qualitative_gte" \
  --orca_k 2 \
  --topk 10 \
  --max_cases 12 \
  --seed "$SEED" \
  --max_chars 80 \
  2>&1 | tee "$ROOT/logs/13_table_14_qualitative.log"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_cross_pool_paired_significance.py" \
  --data "$DATA" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --source-key emotion_corpus \
  --k 2 \
  --bootstrap-samples 5000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/cross_pool" \
  --output-prefix "$ROOT/results/table_12_cross_pool" \
  2>&1 | tee "$ROOT/logs/11_table_12_cross_pool.log"

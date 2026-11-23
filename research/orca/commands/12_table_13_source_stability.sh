#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_source_stability_enhanced.py" \
  --data "$DATA" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --split dev \
  --source-key emotion_corpus \
  --k 2 \
  --sizes 250,500,1000,2000,4000,5000 \
  --repeats 10 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/source_stability" \
  --output-prefix "$ROOT/results/table_13_source_stability" \
  2>&1 | tee "$ROOT/logs/12_table_13_source_stability.log"

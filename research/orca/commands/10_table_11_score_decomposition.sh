#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_score_decomposition.py" \
  --data "$DATA" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --split test \
  --source-key emotion_corpus \
  --k 2 \
  --group-k 5 \
  --ranking-depth 101 \
  --permutations 10000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/score_decomposition" \
  --output-prefix "$ROOT/results/table_11_score_decomposition" \
  2>&1 | tee "$ROOT/logs/10_table_11_score_decomposition.log"

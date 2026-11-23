#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_mean_direction_baselines.py" \
  --data "$DATA" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --split dev \
  --source-key emotion_corpus \
  --k 2 \
  --random-trials 5 \
  --bootstrap-samples 5000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/mean_direction" \
  --output-prefix "$ROOT/results/table_10_mean_direction" \
  2>&1 | tee "$ROOT/logs/09_table_10_mean_direction.log"

# Frozen-test confirmation used by the mechanism discussion.
python -u "$ROOT/src/run_test_mean_direction_ablation.py" \
  --data "$DATA" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --device "$DEVICE" \
  --source-key emotion_corpus \
  --bootstrap-samples 5000 \
  --cluster-permutations 10000 \
  --seed "$SEED" \
  --cache-dir "$ROOT/cache/test_mean_direction" \
  --output-prefix "$ROOT/results/test_mean_direction" \
  2>&1 | tee "$ROOT/logs/09_test_mean_direction.log"

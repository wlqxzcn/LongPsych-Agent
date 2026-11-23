#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

python -u "$ROOT/src/run_cross_pool_cluster_significance.py" \
  --data "$DATA" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --source-key emotion_corpus \
  --k 2 \
  --bootstrap-samples 5000 \
  --cluster-permutations 10000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/table_12_candidate_cluster" \
  --output-prefix "$ROOT/results/table_12_candidate" \
  2>&1 | tee "$ROOT/logs/17_table_12_candidate_cluster.log"

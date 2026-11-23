#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

python -u "$ROOT/src/run_test_mean_direction_ablation.py" \
  --data "$DATA" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --device "$DEVICE" \
  --source-key emotion_corpus \
  --bootstrap-samples 5000 \
  --cluster-permutations 10000 \
  --seed "$SEED" \
  --cache-dir "$ROOT/cache/bge_pmain_cluster" \
  --output-prefix "$ROOT/results/bge_pmain_cluster" \
  2>&1 | tee "$ROOT/logs/15_bge_pmain_cluster.log"

python -u "$ROOT/src/summarize_pmain_dialogue_counts.py" \
  --data "$DATA" \
  --per-query-npz "$ROOT/results/bge_pmain_cluster_per_query.npz" \
  --model bge \
  --method-a ORCA-k2 \
  --method-b Dense \
  --output-prefix "$ROOT/results/bge_pmain_dialogue_counts" \
  2>&1 | tee -a "$ROOT/logs/15_bge_pmain_cluster.log"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_var PSYDT_TEST_SINGLE
require_var PSYDT_TRAIN_MULTI
require_file "$PSYDT_TEST_SINGLE"
require_file "$PSYDT_TRAIN_MULTI"
for SPEC in "gte:$GTE_MODEL:$GTE_BATCH_SIZE" "bge:$BGE_MODEL:$BGE_BATCH_SIZE"; do
  IFS=: read -r TAG MODEL BATCH <<< "$SPEC"
  mkdir -p "$ROOT/results/runtime/$TAG" "$ROOT/cache/scaling/$TAG"
  python -u "$ROOT/src/run_runtime_repeats.py" \
    --script "$ROOT/src/orca_bigdata_final_safe.py" \
    --tag "$TAG" \
    --test_pairs_file "$PSYDT_TEST_SINGLE" \
    --train_responses_file "$PSYDT_TRAIN_MULTI" \
    --train_user_file "$PSYDT_TRAIN_MULTI" \
    --encoder_name "$MODEL" \
    --pool_sizes 5000 10000 20000 50000 full \
    --orca_k 2 \
    --topk 5 \
    --max_queries 2000 \
    --source_limit 5000 \
    --source_mode affective_user \
    --batch_size "$BATCH" \
    --cache_dir "$ROOT/cache/scaling/$TAG" \
    --out_dir "$ROOT/results/runtime/$TAG" \
    --runs 5 \
    --warmup_runs 1 \
    --seed "$SEED" \
    --device "$DEVICE" \
    2>&1 | tee "$ROOT/logs/07_table_8_runtime_${TAG}.log"
done

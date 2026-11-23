#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_var RAW_PSYDT
require_file "$RAW_PSYDT"

# Fixed-benchmark repeated source sampling for both Chinese encoders.
python -u "$ROOT/src/run_user_source_ablation_repeated.py" \
  --data "$DATA" \
  --psydt-path "$RAW_PSYDT" \
  --model "gte=$GTE_MODEL" \
  --model "bge=$BGE_MODEL" \
  --model-batch-size "gte=$GTE_BATCH_SIZE" \
  --model-batch-size "bge=$BGE_BATCH_SIZE" \
  --split test \
  --source-size 5000 \
  --repeats 5 \
  --k 2 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/user_sources" \
  --output-prefix "$ROOT/results/table_9_user_sources" \
  2>&1 | tee "$ROOT/logs/08_table_9_user_sources.log"

# The paper's GTE row uses the 5k expansion protocol. Set these two variables
# to reproduce that exact row; otherwise the fixed-benchmark result above is complete.
if [[ -n "${PSYDT_TEST_SINGLE:-}" && -n "${PSYDT_TRAIN_MULTI:-}" ]]; then
  for MODE in affective_user all_user non_affective_user first_person_non_affective_user; do
    mkdir -p "$ROOT/results/table_9_gte_scaling_sources/$MODE" "$ROOT/cache/table_9_gte_scaling_sources/$MODE"
    python -u "$ROOT/src/orca_bigdata_final_safe.py" \
      --test_pairs_file "$PSYDT_TEST_SINGLE" \
      --train_responses_file "$PSYDT_TRAIN_MULTI" \
      --train_user_file "$PSYDT_TRAIN_MULTI" \
      --encoder_name "$GTE_MODEL" \
      --pool_sizes 5000 \
      --orca_k 2 \
      --topk 5 \
      --max_queries 2000 \
      --source_limit 5000 \
      --source_mode "$MODE" \
      --batch_size "$GTE_BATCH_SIZE" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --cache_dir "$ROOT/cache/table_9_gte_scaling_sources/$MODE" \
      --out_scaling "$ROOT/results/table_9_gte_scaling_sources/$MODE/scaling.csv" \
      --out_runtime "$ROOT/results/table_9_gte_scaling_sources/$MODE/runtime.csv" \
      --out_metadata "$ROOT/results/table_9_gte_scaling_sources/$MODE/metadata.json"
  done
fi

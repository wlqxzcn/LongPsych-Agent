#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

RAW_PSYDT="${RAW_PSYDT:-$ROOT/data/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json}"
require_file "$RAW_PSYDT"

python -u "$ROOT/src/run_user_source_ablation_cluster.py" \
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
  --sources "main_affective_user,all_train_user,non_affective_user,first_person_non_affective_user,train_counselor" \
  --bootstrap-samples 5000 \
  --cluster-permutations 10000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/table_9_source_cluster" \
  --output-prefix "$ROOT/results/table_9_source" \
  2>&1 | tee "$ROOT/logs/16_table_9_source_cluster.log"

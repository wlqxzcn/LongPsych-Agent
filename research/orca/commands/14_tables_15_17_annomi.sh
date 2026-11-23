#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_var ANNOMI_CSV
require_var MPNET_MODEL
require_var BGE_EN_MODEL
require_file "$ANNOMI_CSV"
require_dir "$MPNET_MODEL"
require_dir "$BGE_EN_MODEL"
python -u "$ROOT/src/run_annomi_fivefold.py" \
  --csv "$ANNOMI_CSV" \
  --model "mpnet=$MPNET_MODEL" \
  --model "bge_en=$BGE_EN_MODEL" \
  --model-batch-size "mpnet=32" \
  --model-batch-size "bge_en=32" \
  --folds 5 \
  --k 2 \
  --bootstrap-samples 5000 \
  --cluster-permutations 10000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cache-dir "$ROOT/cache/annomi" \
  --output-prefix "$ROOT/results/tables_15_17_annomi" \
  2>&1 | tee "$ROOT/logs/14_tables_15_17_annomi.log"

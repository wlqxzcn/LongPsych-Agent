#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_var RAW_PSYDT
require_file "$RAW_PSYDT"
python -u "$ROOT/src/build_psydt_final_dataset.py" \
  --psydt_path "$RAW_PSYDT" \
  --output_dir "$ROOT/data/cleaned" \
  --output_name psydt_task_retrieval_final.json \
  --seed "$SEED" \
  2>&1 | tee "$ROOT/logs/01_build_psydt.log"

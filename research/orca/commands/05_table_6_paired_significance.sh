#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
python -u "$ROOT/src/run_supplementary_experiments_local.py" \
  --data "$DATA" \
  --models_root "$MODELS_ROOT" \
  --models gte,bge \
  --gte_path "$GTE_MODEL" \
  --bge_path "$BGE_MODEL" \
  --gte_batch_size "$GTE_BATCH_SIZE" \
  --bge_batch_size "$BGE_BATCH_SIZE" \
  --k_list 1,2,3,4,8,16,32,64 \
  --n_bootstrap 5000 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --skip_energy_threshold \
  --skip_noise \
  --skip_svd_norm_ablation \
  --skip_emotion_examples \
  --cache_dir "$ROOT/cache/significance_gte_bge" \
  --output_json "$ROOT/results/table_6_paired_significance.json" \
  --output_md "$ROOT/results/table_6_paired_significance.md" \
  2>&1 | tee "$ROOT/logs/05_table_6_paired_significance.log"

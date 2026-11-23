#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for script in \
  02_tables_2_3_main.sh \
  03_table_4_competing_methods.sh \
  04_table_5_rrf.sh \
  05_table_6_paired_significance.sh \
  06_table_7_candidate_scaling.sh \
  07_table_8_runtime_repeats.sh \
  08_table_9_user_sources.sh \
  09_table_10_mean_direction.sh \
  10_table_11_score_decomposition.sh \
  11_table_12_cross_pool.sh \
  12_table_13_source_stability.sh \
  13_table_14_qualitative.sh \
  14_tables_15_17_annomi.sh
do
  echo "================================================================"
  echo "Running $script"
  echo "================================================================"
  bash "$HERE/$script"
done

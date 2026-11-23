#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-$ROOT/data/cleaned/psydt_task_retrieval_final.json}"
MODELS_ROOT="${MODELS_ROOT:-$ROOT/models}"
GTE_MODEL="${GTE_MODEL:-$MODELS_ROOT/thenlper/gte-large-zh}"
BGE_MODEL="${BGE_MODEL:-$MODELS_ROOT/BAAI/bge-large-zh-v1.5}"
DEVICE="${DEVICE:-cuda}"
GTE_BATCH_SIZE="${GTE_BATCH_SIZE:-16}"
BGE_BATCH_SIZE="${BGE_BATCH_SIZE:-32}"
SEED="${SEED:-42}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/results" "$ROOT/cache" "$ROOT/logs"

require_file() {
  [[ -f "$1" ]] || { echo "[error] file not found: $1" >&2; exit 1; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "[error] directory not found: $1" >&2; exit 1; }
}
require_var() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "[error] environment variable $name is required" >&2; exit 1; }
}

require_file "$DATA"
require_dir "$GTE_MODEL"
require_dir "$BGE_MODEL"

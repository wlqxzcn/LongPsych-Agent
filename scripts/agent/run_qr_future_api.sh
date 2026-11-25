#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="agent:."
export LONGPSYCH_DEVICE="${LONGPSYCH_DEVICE:-cuda}"
export LONGPSYCH_BATCH_SIZE="${LONGPSYCH_BATCH_SIZE:-16}"
export LONGPSYCH_CANDIDATE_K="${LONGPSYCH_CANDIDATE_K:-100}"
export LONGPSYCH_VERIFY_HASHES="${LONGPSYCH_VERIFY_HASHES:-true}"

# One worker is intentional: each worker would load a separate BGE encoder,
# five QR-Future models, and response embedding assets into GPU/host memory.
exec python -m uvicorn longpsych_agent.api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --log-level info

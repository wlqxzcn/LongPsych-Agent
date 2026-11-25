#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MCP_PYTHON="${MCP_PYTHON:-$PROJECT_ROOT/.venv-mcp/bin/python}"
if [[ ! -x "$MCP_PYTHON" ]]; then
  echo "MCP Python environment not found: $MCP_PYTHON" >&2
  echo "Run: bash scripts/agent/setup_mcp_env.sh" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/agent:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MCP_PYTHON" != /* ]]; then
  MCP_PYTHON="$PROJECT_ROOT/$MCP_PYTHON"
fi

exec "$MCP_PYTHON" -m longpsych_agent.mcp_server.server

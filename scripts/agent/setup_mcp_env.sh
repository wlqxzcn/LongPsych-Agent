#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MCP_ENV_DIR="$PROJECT_ROOT/.venv-mcp"
python -m venv "$MCP_ENV_DIR"
"$MCP_ENV_DIR/bin/python" -m pip install --upgrade pip
"$MCP_ENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-mcp.txt"

echo
echo "MCP environment installed."
echo "Start with: bash scripts/agent/run_qr_future_mcp.sh"

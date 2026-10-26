#!/usr/bin/env bash
#
# bvbrc-agents setup script
#
# Creates the orchestrator virtual environment and installs its dependencies,
# including the bvbrc-mcp package (the BV-BRC tools the agents use).
#
# Usage:
#   ./setup.sh
#
#   BVBRC_MCP_SOURCE=<pip spec or path> ./setup.sh
#       Where bvbrc-mcp comes from. Default: a bvbrc-mcp checkout next to this
#       repo (../bvbrc-mcp), installed editable, for development. For a
#       deployment use a pinned release, e.g.
#       "bvbrc-mcp @ git+https://github.com/cucinellclark/bvbrc-mcp@vX.Y.Z".
#
# Prerequisites:
#   - Python 3.11

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ORCH_DIR="$REPO_ROOT/orchestrator"
BVBRC_MCP_SOURCE="${BVBRC_MCP_SOURCE:-$REPO_ROOT/../bvbrc-mcp}"

echo "=== BV-BRC Agents Setup ==="
echo "Repo root: $REPO_ROOT"
echo

# ---------------------------------------------------------------
# 1. Orchestrator venv
# ---------------------------------------------------------------
echo "[1/2] Setting up Orchestrator virtual environment..."
if [ ! -f "$ORCH_DIR/orchestrator_env/bin/activate" ]; then
    python3.11 -m venv "$ORCH_DIR/orchestrator_env"
fi
(
    source "$ORCH_DIR/orchestrator_env/bin/activate"
    pip install --upgrade pip -q
    if [ -f "$ORCH_DIR/pyproject.toml" ]; then
        pip install -e "$ORCH_DIR" -q
    fi
    echo "  Orchestrator deps installed."
)
echo

# ---------------------------------------------------------------
# 2. bvbrc-mcp (the BV-BRC tools)
# ---------------------------------------------------------------
echo "[2/2] Installing bvbrc-mcp from: $BVBRC_MCP_SOURCE"
(
    source "$ORCH_DIR/orchestrator_env/bin/activate"
    if [ -d "$BVBRC_MCP_SOURCE" ]; then
        pip install -e "$BVBRC_MCP_SOURCE[pdf]" -q
    else
        pip install "$BVBRC_MCP_SOURCE" -q
    fi
    python -c "import bvbrc_mcp; print('  bvbrc-mcp', bvbrc_mcp.__version__, '-', len(bvbrc_mcp.TOOLS), 'tools')"
)
echo

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
echo "=== Setup complete ==="
echo
echo "To start the orchestrator:"
echo "  cd orchestrator && ./scripts/start_orchestrator.sh --port 9100"
echo
echo "The external MCP server is part of bvbrc-mcp (bvbrc-mcp-http / bvbrc-mcp-stdio)."
echo "See STARTUP.md for details."

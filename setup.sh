#!/usr/bin/env bash
#
# bvbrc-agents setup script
#
# Clones the MCP server repo, creates virtual environments, and installs
# all dependencies needed to run the system.
#
# Usage:
#   ./setup.sh
#
# Prerequisites:
#   - Python 3.11
#   - git
#   - SSH access to github.com/cucinellclark repos

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MCP_DIR="$REPO_ROOT/mcp_server"
ORCH_DIR="$REPO_ROOT/orchestrator"

echo "=== BV-BRC Agents Setup ==="
echo "Repo root: $REPO_ROOT"
echo

# ---------------------------------------------------------------
# 1. Clone MCP server
# ---------------------------------------------------------------
if [ -d "$MCP_DIR/.git" ]; then
    echo "[1/4] MCP server already cloned, pulling latest..."
    git -C "$MCP_DIR" pull --ff-only
else
    echo "[1/4] Cloning MCP server..."
    if [ -d "$MCP_DIR" ]; then
        echo "  Warning: $MCP_DIR exists but is not a git repo. Removing..."
        rm -rf "$MCP_DIR"
    fi
    git clone git@github.com:cucinellclark/bvbrc-mcp-server.git "$MCP_DIR"
fi
echo

# ---------------------------------------------------------------
# 2. Clone bvbrc-python-api inside MCP server
# ---------------------------------------------------------------
if [ -d "$MCP_DIR/bvbrc-python-api/.git" ]; then
    echo "[2/4] bvbrc-python-api already cloned, pulling latest..."
    git -C "$MCP_DIR/bvbrc-python-api" pull --ff-only
else
    echo "[2/4] Cloning bvbrc-python-api..."
    git clone git@github.com:cucinellclark/bvbrc-python-api.git "$MCP_DIR/bvbrc-python-api"
fi
echo

# ---------------------------------------------------------------
# 3. Set up MCP server venv
# ---------------------------------------------------------------
echo "[3/4] Setting up MCP server virtual environment..."
if [ ! -f "$MCP_DIR/mcp_env/bin/activate" ]; then
    python3.11 -m venv "$MCP_DIR/mcp_env"
fi
(
    source "$MCP_DIR/mcp_env/bin/activate"
    pip install --upgrade pip -q
    pip install -r "$MCP_DIR/requirements.txt" -q
    pip install "openai>=1.0.0" -q
    pip install -e "$MCP_DIR/bvbrc-python-api/" -q
    pip install "pydantic[email]>=2.11.7" -q
    echo "  MCP server deps installed."
)
echo

# ---------------------------------------------------------------
# 4. Set up Orchestrator venv
# ---------------------------------------------------------------
echo "[4/4] Setting up Orchestrator virtual environment..."
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
# Done
# ---------------------------------------------------------------
echo "=== Setup complete ==="
echo
echo "To start the system:"
echo "  Terminal 1:  cd mcp_server && source mcp_env/bin/activate && python3 http_server.py"
echo "  Terminal 2:  cd orchestrator && ./scripts/start_orchestrator.sh"
echo
echo "See STARTUP.md for details."

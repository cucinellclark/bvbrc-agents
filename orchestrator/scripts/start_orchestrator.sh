#!/usr/bin/env bash
# Start the BV-BRC Copilot Orchestrator server.
#
# Usage:
#   ./scripts/start_orchestrator.sh              # foreground (default)
#   ./scripts/start_orchestrator.sh --background # daemonize with log file
#   ./scripts/start_orchestrator.sh --port 9001  # custom port
#   ./scripts/start_orchestrator.sh --stop       # stop a backgrounded instance
#
# Prerequisites:
#   - Python venv at Orchestrator/orchestrator_env/
#   - Auth token at Orchestrator/auth_token.txt (or BV_BRC_AUTH_TOKEN env var)
#   - Agents already running (use scripts/start_agents.py)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/orchestrator_env"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/orchestrator.pid"
LOG_FILE="$LOG_DIR/orchestrator.log"
CONFIG_FILE="$PROJECT_DIR/config/agents.yaml"
AUTH_TOKEN_FILE="$PROJECT_DIR/auth_token.txt"

# ── Defaults ──────────────────────────────────────────────────────────
HOST="0.0.0.0"
PORT=9000
LOG_LEVEL="info"
BACKGROUND=false
STOP=false

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --stop)
            STOP=true
            shift
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --background, -b   Run in background (logs to $LOG_DIR/)"
            echo "  --stop             Stop a backgrounded orchestrator"
            echo "  --port, -p PORT    Port to listen on (default: 9000)"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --log-level LEVEL  debug|info|warning|error (default: info)"
            echo "  --config PATH      Path to agents.yaml"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Stop mode ─────────────────────────────────────────────────────────
if $STOP; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping orchestrator (PID: $PID)..."
            kill "$PID"
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Still running, sending SIGKILL..."
                kill -9 "$PID"
            fi
            echo "Orchestrator stopped."
        else
            echo "Orchestrator not running (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found. Orchestrator may not be running in background mode."
    fi
    exit 0
fi

# ── Preflight checks ─────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Create it with:  python -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -e $PROJECT_DIR"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Set auth token if available ───────────────────────────────────────
if [[ -z "${BV_BRC_AUTH_TOKEN:-}" ]] && [[ -f "$AUTH_TOKEN_FILE" ]]; then
    export BV_BRC_AUTH_TOKEN
    BV_BRC_AUTH_TOKEN="$(cat "$AUTH_TOKEN_FILE")"
    echo "Loaded auth token from $AUTH_TOKEN_FILE"
fi

# ── Ensure log directory exists ───────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Banner ────────────────────────────────────────────────────────────
echo "========================================"
echo "  BV-BRC Copilot Orchestrator"
echo "========================================"
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo "  Config:    $CONFIG_FILE"
echo "  Log level: $LOG_LEVEL"
echo "  Venv:      $VENV_DIR"
echo "  Auth:      ${BV_BRC_AUTH_TOKEN:+set (${#BV_BRC_AUTH_TOKEN} chars)}"
echo "  Mode:      $( $BACKGROUND && echo 'background' || echo 'foreground' )"
echo "========================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────
CMD=(
    python -m orchestrator
    --host "$HOST"
    --port "$PORT"
    --config "$CONFIG_FILE"
    --log-level "$LOG_LEVEL"
)

if $BACKGROUND; then
    echo "Starting in background..."
    echo "Log file: $LOG_FILE"
    echo ""

    nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "$PID_FILE"

    # Wait a moment and check it's still alive
    sleep 3
    if ps -p "$BG_PID" > /dev/null 2>&1; then
        echo "Orchestrator started (PID: $BG_PID)"
        echo ""
        echo "  Health:  curl http://$HOST:$PORT/health"
        echo "  Logs:    tail -f $LOG_FILE"
        echo "  Stop:    $0 --stop"
    else
        echo "ERROR: Orchestrator failed to start. Check logs:"
        echo "  tail -20 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
else
    # Foreground — exec replaces this shell so signals propagate cleanly
    exec "${CMD[@]}"
fi

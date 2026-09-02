#!/usr/bin/env bash
# Start the LLM admission proxy on holly.
#
# Usage:
#   ./start_admission_proxy.sh              # foreground (default)
#   ./start_admission_proxy.sh --background # daemonize with log file
#   ./start_admission_proxy.sh --port 8005  # custom port (overrides ADMISSION_PORT)
#   ./start_admission_proxy.sh --stop       # stop a backgrounded instance
#
# Environment variables (see llm_admission/config.py):
#   LLM_UPSTREAM_URL        — real vLLM endpoint (default: http://mango.cels.anl.gov:8004/v1)
#   ADMISSION_MAX_IN_FLIGHT — max concurrent upstream requests (default: 4)
#   ADMISSION_MAX_WAITING   — max queued requests before 429 (default: 16)
#   ADMISSION_PORT          — listen port (default: 8005)
#
# Prerequisites:
#   - Python venv at ../orchestrator/orchestrator_env/ (shared with orchestrator)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/orchestrator/orchestrator_env"
LOG_DIR="$(cd "$PROJECT_DIR/../.." && pwd)/DevEnvironment/logs/agents"
PID_FILE="$SCRIPT_DIR/admission_proxy.pid"
LOG_FILE="$LOG_DIR/admission_proxy.log"

# ── Defaults ──────────────────────────────────────────────────────────
HOST="${ADMISSION_HOST:-0.0.0.0}"
PORT="${ADMISSION_PORT:-8005}"
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
            export ADMISSION_PORT="$PORT"
            shift 2
            ;;
        --host)
            HOST="$2"
            export ADMISSION_HOST="$HOST"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --background, -b   Run in background (logs to $LOG_DIR/)"
            echo "  --stop             Stop a backgrounded proxy"
            echo "  --port, -p PORT    Port to listen on (default: 8005)"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --log-level LEVEL  debug|info|warning|error (default: info)"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Environment variables:"
            echo "  LLM_UPSTREAM_URL        Real vLLM endpoint"
            echo "  ADMISSION_MAX_IN_FLIGHT Max concurrent upstream requests (default: 4)"
            echo "  ADMISSION_MAX_WAITING   Max queued requests before 429 (default: 16)"
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
            echo "Stopping admission proxy (PID: $PID)..."
            kill "$PID"
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Still running, sending SIGKILL..."
                kill -9 "$PID"
            fi
            echo "Admission proxy stopped."
        else
            echo "Admission proxy not running (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found. Proxy may not be running in background mode."
    fi
    exit 0
fi

# ── Preflight checks ─────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "The admission proxy shares the orchestrator venv."
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Ensure log directory exists ───────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Banner ────────────────────────────────────────────────────────────
echo "========================================"
echo "  LLM Admission Proxy"
echo "========================================"
echo "  Host:           $HOST"
echo "  Port:           $PORT"
echo "  Upstream:       ${LLM_UPSTREAM_URL:-http://mango.cels.anl.gov:8004/v1}"
echo "  Max in-flight:  ${ADMISSION_MAX_IN_FLIGHT:-4}"
echo "  Max waiting:    ${ADMISSION_MAX_WAITING:-16}"
echo "  Log level:      $LOG_LEVEL"
echo "  Venv:           $VENV_DIR"
echo "  Mode:           $( $BACKGROUND && echo 'background' || echo 'foreground' )"
echo "========================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────
# Must run from the bvbrc-agents repo root so `llm_admission` is importable
# as a Python package.
cd "$PROJECT_DIR"

CMD=(
    python -m uvicorn llm_admission.proxy:app
    --host "$HOST"
    --port "$PORT"
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
        echo "Admission proxy started (PID: $BG_PID)"
        echo ""
        echo "  Health:  curl http://$HOST:$PORT/health"
        echo "  Logs:    tail -f $LOG_FILE"
        echo "  Stop:    $0 --stop"
    else
        echo "ERROR: Admission proxy failed to start. Check logs:"
        echo "  tail -20 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
else
    # Foreground — exec replaces this shell so signals propagate cleanly
    exec "${CMD[@]}"
fi

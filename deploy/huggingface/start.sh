#!/usr/bin/env bash
# ── Hugging Face Spaces entrypoint ──────────────────────────────────────────
#
# Launches the FastAPI service in the background on localhost:8000, waits
# for it to pass /health, then starts Streamlit in the foreground on $PORT
# (7860 by default). If Streamlit exits the API is killed so the container
# terminates cleanly.

set -euo pipefail

cd /home/user/app

cleanup() {
    echo "[start.sh] shutting down API (pid=${API_PID:-?})"
    if [[ -n "${API_PID:-}" ]]; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "[start.sh] starting FastAPI on localhost:8000"
uvicorn api.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info &
API_PID=$!

echo "[start.sh] waiting for API /health (timeout 60s)"
READY=0
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "[start.sh] API is healthy after ${i}s"
        READY=1
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[start.sh] API process died before becoming healthy"
        exit 1
    fi
    sleep 1
done

if [[ "$READY" -ne 1 ]]; then
    echo "[start.sh] API did not become healthy within 60s — aborting"
    kill "$API_PID" 2>/dev/null || true
    exit 1
fi

echo "[start.sh] starting Streamlit on 0.0.0.0:${PORT:-7860}"
exec streamlit run streamlit_app/app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection false

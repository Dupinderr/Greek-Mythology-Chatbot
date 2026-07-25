#!/usr/bin/env bash
# Runs the API and the UI in one container: uvicorn in the background,
# Streamlit in the foreground so the container's lifetime tracks the UI.
set -euo pipefail

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "FATAL: GROQ_API_KEY is not set. Add it as a secret in your host's settings." >&2
  exit 1
fi

uvicorn api:app --host 0.0.0.0 --port 8000 --log-level warning &
API_PID=$!

# Stop the whole container if the API dies, rather than serving a UI whose
# every request will fail.
trap 'kill $API_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if python -c "import requests,sys; sys.exit(0 if requests.get('http://localhost:8000/health',timeout=2).ok else 1)" 2>/dev/null; then
    echo "API is up."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FATAL: API exited during startup." >&2
    exit 1
  fi
  sleep 2
done

exec streamlit run streamlit_app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true

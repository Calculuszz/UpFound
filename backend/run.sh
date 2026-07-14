#!/usr/bin/env bash
# UpFound backend — start the API + frontend server.
#   ./run.sh              # http://<spark-ip>:8000  (serves Web_dev + /api)
#   PORT=9000 ./run.sh    # different port
set -euo pipefail
cd "$(dirname "$0")"

VENV="${UPFOUND_VENV:-$HOME/upfound-env}"
if [ -f "$VENV/bin/activate" ]; then set +u; source "$VENV/bin/activate"; set -u; fi

echo "▶  UpFound backend on http://0.0.0.0:${PORT:-8000}  (frontend: Web_dev)"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"

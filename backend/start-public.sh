#!/usr/bin/env bash
# Start UpFound backend + Cloudflare named tunnel (idempotent — safe to re-run).
#   ใช้ตอนรีบูต (cron @reboot) หรือรันมือเพื่อกู้ระบบให้กลับมา online
set -u
cd "$(dirname "$0")"                 # ~/UpFound/backend
LOGDIR="$(pwd)"

# 1) backend บน :8000 — สตาร์ตถ้ายังไม่ตอบ
if ! curl -sf -m 3 http://localhost:8000/ >/dev/null 2>&1; then
  nohup ./run.sh > "$LOGDIR/.tunnel-backend.log" 2>&1 &
  echo "started backend"
else
  echo "backend already up"
fi

# 2) named tunnel (URL ถาวร up-found.com) — สตาร์ตถ้ายังไม่รัน
if ! pgrep -f "cloudflared tunnel run upfound" >/dev/null 2>&1; then
  nohup "$HOME/cloudflared" tunnel run upfound > "$LOGDIR/.tunnel-named.log" 2>&1 &
  echo "started tunnel"
else
  echo "tunnel already up"
fi

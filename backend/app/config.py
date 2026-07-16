"""Central config for the UpFound backend (local-first; maps to AWS later).

Everything is env-overridable so the same code runs on the Spark now and on
AWS (Aurora/S3/Cognito) later without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # ~/UpFound/backend
DATA_DIR = Path(os.getenv("UPFOUND_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"                          # user-uploaded lost-item photos
DB_PATH = os.getenv("UPFOUND_DB", str(DATA_DIR / "upfound.db"))

# Existing static frontend (Web_dev) — served by this same app so it's same-origin.
WEB_DIR = Path(os.getenv("UPFOUND_WEB_DIR", BASE_DIR.parent / "Web_dev"))

# Process 1 (EdgeAI) output — the detected-item events + crops we ingest.
EDGEAI_DIR = Path(os.getenv("UPFOUND_EDGEAI_DIR", BASE_DIR.parent / "EdgeAI"))
EVENTS_JSONL = Path(os.getenv("UPFOUND_EVENTS_JSONL", EDGEAI_DIR / "out" / "events.jsonl"))
EDGE_OUT_DIR = EDGEAI_DIR / "out"                          # crop_ref is "./out/crops/..." under here

# Auth (JWT). On AWS use Cognito instead.
JWT_ALG = "HS256"
JWT_TTL_HOURS = int(os.getenv("UPFOUND_JWT_TTL_HOURS", "24"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_jwt_secret() -> str:
    """Prefer an explicit secret; otherwise generate a strong random one and
    persist it (so tokens survive restarts and we never ship a weak default)."""
    env = os.getenv("UPFOUND_JWT_SECRET")
    if env:
        return env
    import secrets

    path = DATA_DIR / ".jwt_secret"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    path.write_text(secret, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return secret


JWT_SECRET = _resolve_jwt_secret()

# Auto-ingest EdgeAI events every N seconds (0 disables the background poller).
INGEST_INTERVAL_SECONDS = int(os.getenv("UPFOUND_INGEST_INTERVAL", "30"))

# Upload limits / validation.
MAX_UPLOAD_MB = int(os.getenv("UPFOUND_MAX_UPLOAD_MB", "10"))
ALLOWED_IMAGE_TYPES = (
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
)

# Demo account — seeded on startup and shown on the login page for quick trials.
DEMO_ENABLED = os.getenv("UPFOUND_DEMO_ENABLED", "1") not in ("0", "false", "False")
DEMO_EMAIL = os.getenv("UPFOUND_DEMO_EMAIL", "demo@upfound.co")
DEMO_PASSWORD = os.getenv("UPFOUND_DEMO_PASSWORD", "demo1234")
DEMO_NAME = os.getenv("UPFOUND_DEMO_NAME", "ผู้ใช้ทดลอง")

# Matching
MATCH_TOP_K = int(os.getenv("UPFOUND_MATCH_TOP_K", "5"))

# CLIP cosine sits on two scales that no single floor can serve. Measured against
# real crops: text↔image runs ~0.19-0.28 (keyboard mash already reaches 0.22, so
# a floor under ~0.25 admits pure noise), while image↔image runs ~0.45-1.0 (even
# unrelated photos score ~0.55, so the same floor filters nothing at all).
MATCH_MIN_SCORE_TEXT = float(os.getenv("UPFOUND_MATCH_MIN_SCORE_TEXT", "0.25"))
MATCH_MIN_SCORE_IMAGE = float(os.getenv("UPFOUND_MATCH_MIN_SCORE_IMAGE", "0.65"))

# Raw cosine reads as a terrible number to a user (a strong text match is only
# ~0.28), so matches also carry a 0-1 confidence rescaled from floor to ceiling.
MATCH_FULL_SCORE_TEXT = float(os.getenv("UPFOUND_MATCH_FULL_SCORE_TEXT", "0.32"))
MATCH_FULL_SCORE_IMAGE = float(os.getenv("UPFOUND_MATCH_FULL_SCORE_IMAGE", "0.95"))

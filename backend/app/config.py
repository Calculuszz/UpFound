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

# Auth (JWT). CHANGE the secret in production / on AWS use Cognito instead.
JWT_SECRET = os.getenv("UPFOUND_JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_TTL_HOURS = int(os.getenv("UPFOUND_JWT_TTL_HOURS", "24"))

# Matching
MATCH_TOP_K = int(os.getenv("UPFOUND_MATCH_TOP_K", "5"))
MATCH_MIN_SCORE = float(os.getenv("UPFOUND_MATCH_MIN_SCORE", "0.20"))  # cosine floor

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

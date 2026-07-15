"""SQLite storage (local-first). Maps 1:1 to Aurora Postgres + pgvector later —
embeddings are stored as JSON now; on Postgres they become a `vector` column.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name          TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lost_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),    -- nullable: found reports are public
    kind        TEXT NOT NULL DEFAULT 'lost',     -- 'lost' (owner) | 'found' (finder)
    item_name   TEXT NOT NULL,
    color       TEXT,
    qty         INTEGER DEFAULT 1,
    item_type   TEXT,
    contact     TEXT,                             -- finder's contact (found reports)
    location    TEXT,
    lost_date   TEXT,
    detail      TEXT,
    image_paths TEXT,          -- json array of stored upload paths
    embedding   TEXT,          -- json array (query vector for matching)
    status      TEXT DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),   -- nullable: person reports are public
    kind        TEXT NOT NULL DEFAULT 'lost',    -- 'lost' (missing) | 'found' (spotted)
    full_name   TEXT,                            -- lost side (finder may not know it)
    gender      TEXT,
    age         INTEGER,
    height_cm   INTEGER,
    contact     TEXT,                            -- found side (how to reach the finder)
    location    TEXT,
    report_date TEXT,
    detail      TEXT,
    image_paths TEXT,          -- json array of stored upload paths
    embedding   TEXT,          -- json CLIP image vector (mean of photos) for matching
    status      TEXT DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detected_events (
    event_id      TEXT PRIMARY KEY,        -- from EdgeAI Event Contract
    object_class  TEXT,
    zone          TEXT,
    capture_ts    TEXT,
    crop_ref      TEXT,
    bbox          TEXT,                     -- json [x,y,w,h]
    model_version TEXT,
    embedding     TEXT,                     -- json array (CLIP 512-d)
    source        TEXT,
    ingested_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  INTEGER NOT NULL REFERENCES lost_reports(id),
    event_id   TEXT NOT NULL REFERENCES detected_events(event_id),
    score      REAL NOT NULL,
    status     TEXT DEFAULT 'suggested',   -- suggested | confirmed | rejected
    created_at TEXT NOT NULL,
    UNIQUE(report_id, event_id)
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(_SCHEMA)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    # decode json-ish columns for callers that want them as python objects
    for k in ("image_paths", "embedding", "bbox"):
        if k in d and isinstance(d[k], str) and d[k]:
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d

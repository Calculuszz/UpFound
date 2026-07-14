"""Ingest Process 1 (EdgeAI) events → detected_events.

Reads the Event Contract JSONL that EdgeAI emits. If an event has no embedding
(e.g. EdgeAI ran with --no-embed), we re-embed its crop image here with the same
CLIP model, so matching always has a vector to compare against.

On AWS this becomes a Lambda triggered by SQS/Kinesis instead of a file read.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import EDGE_OUT_DIR, EVENTS_JSONL
from .db import db


def _resolve_crop(crop_ref: str | None) -> Path | None:
    """Map a stored crop_ref ('./out/crops/xxx.jpg') to a real file under EdgeAI."""
    if not crop_ref:
        return None
    rel = crop_ref.lstrip("./")                  # -> 'out/crops/xxx.jpg'
    cand = EDGE_OUT_DIR.parent / rel             # EdgeAI/ + out/crops/xxx.jpg
    return cand if cand.exists() else None


def ingest_events(reembed_missing: bool = True) -> dict:
    if not EVENTS_JSONL.exists():
        return {"processed": 0, "reason": f"no events file at {EVENTS_JSONL}"}

    now = datetime.now(timezone.utc).isoformat()
    processed = embedded = skipped = 0

    for line in EVENTS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        eid = e.get("event_id")
        if not eid:
            skipped += 1
            continue

        emb = e.get("embedding")
        if emb is None and reembed_missing:
            crop = _resolve_crop(e.get("crop_ref"))
            if crop is not None:
                from . import embeddings  # lazy — only load CLIP if we must embed
                try:
                    emb = embeddings.embed_image(str(crop)).tolist()
                    embedded += 1
                except Exception:  # noqa: BLE001 — a bad crop shouldn't stop ingest
                    emb = None

        with db() as conn:
            conn.execute(
                """INSERT INTO detected_events
                   (event_id, object_class, zone, capture_ts, crop_ref, bbox,
                    model_version, embedding, source, ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET
                       embedding = COALESCE(excluded.embedding, detected_events.embedding)""",
                (
                    eid, e.get("object_class"), e.get("zone"), e.get("capture_ts"),
                    e.get("crop_ref"), json.dumps(e.get("bbox")), e.get("model_version"),
                    json.dumps(emb) if emb is not None else None, e.get("source"), now,
                ),
            )
        processed += 1

    return {
        "processed": processed,
        "embedded_from_crop": embedded,
        "skipped": skipped,
        "events_file": str(EVENTS_JSONL),
    }

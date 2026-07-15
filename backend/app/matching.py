"""Matching — cosine similarity between a report's query vector and detected
events' CLIP embeddings. In-memory numpy now (fine for demo scale); becomes a
pgvector / OpenSearch k-NN query at scale — same math, just pushed into the DB.
"""
from __future__ import annotations

import json

import numpy as np

from . import config
from .db import db


def cosine_matches(
    query_vec: np.ndarray, top_k: int | None = None, min_score: float | None = None
) -> list[dict]:
    top_k = top_k or config.MATCH_TOP_K
    min_score = config.MATCH_MIN_SCORE if min_score is None else min_score

    with db() as conn:
        rows = conn.execute(
            "SELECT event_id, object_class, zone, capture_ts, crop_ref, "
            "model_version, embedding FROM detected_events "
            "WHERE embedding IS NOT NULL AND embedding != ''"
        ).fetchall()
    if not rows:
        return []

    mats, meta = [], []
    for r in rows:
        try:
            v = np.asarray(json.loads(r["embedding"]), dtype="float32")
        except (json.JSONDecodeError, TypeError):
            continue
        if v.ndim != 1 or v.shape[0] != query_vec.shape[0]:
            continue
        mats.append(v)
        meta.append(dict(r))
    if not mats:
        return []

    M = np.vstack(mats)                          # (N, D), already L2-normalized
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scores = M @ q                               # cosine (both normalized)

    order = np.argsort(-scores)[:top_k]
    out = []
    for i in order:
        s = float(scores[i])
        if s < min_score:
            continue
        m = meta[i]
        out.append({
            "event_id": m["event_id"],
            "object_class": m["object_class"],
            "zone": m["zone"],
            "capture_ts": m["capture_ts"],
            "crop_ref": m["crop_ref"],
            "model_version": m["model_version"],
            "score": round(s, 4),
        })
    return out


def cosine_lost_item_matches(
    query_vec: np.ndarray, top_k: int | None = None, min_score: float | None = None,
) -> list[dict]:
    """Match a FOUND item's photo/description against LOST item reports (owners),
    so a finder's report surfaces the people looking for it."""
    top_k = top_k or config.MATCH_TOP_K
    min_score = config.MATCH_MIN_SCORE if min_score is None else min_score

    with db() as conn:
        rows = conn.execute(
            "SELECT id, item_name, color, location, detail, image_paths, embedding "
            "FROM lost_reports WHERE kind='lost' AND embedding IS NOT NULL AND embedding != ''"
        ).fetchall()
    if not rows:
        return []

    mats, meta = [], []
    for r in rows:
        try:
            v = np.asarray(json.loads(r["embedding"]), dtype="float32")
        except (json.JSONDecodeError, TypeError):
            continue
        if v.ndim != 1 or v.shape[0] != query_vec.shape[0]:
            continue
        mats.append(v)
        meta.append(dict(r))
    if not mats:
        return []

    M = np.vstack(mats)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scores = M @ q
    out = []
    for i in np.argsort(-scores)[:top_k]:
        s = float(scores[i])
        if s < min_score:
            continue
        m = meta[i]
        try:
            arr = json.loads(m["image_paths"]) if m["image_paths"] else []
            img = arr[0] if arr else None
        except (json.JSONDecodeError, TypeError):
            img = None
        out.append({
            "report_id": m["id"],
            "name": m["item_name"],
            "color": m["color"],
            "location": m["location"],
            "detail": m["detail"],
            "image_path": img,
            "score": round(s, 4),
        })
    return out


def cosine_person_matches(
    query_vec: np.ndarray, kind: str, top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """Match a person photo against stored person_reports of the given `kind`
    (image↔image CLIP cosine). Used to link a lost person to found sightings."""
    top_k = top_k or config.MATCH_TOP_K
    min_score = config.MATCH_MIN_SCORE if min_score is None else min_score

    with db() as conn:
        rows = conn.execute(
            "SELECT id, kind, full_name, location, report_date, detail, contact, "
            "image_paths, embedding FROM person_reports "
            "WHERE kind = ? AND embedding IS NOT NULL AND embedding != ''",
            (kind,),
        ).fetchall()
    if not rows:
        return []

    mats, meta = [], []
    for r in rows:
        try:
            v = np.asarray(json.loads(r["embedding"]), dtype="float32")
        except (json.JSONDecodeError, TypeError):
            continue
        if v.ndim != 1 or v.shape[0] != query_vec.shape[0]:
            continue
        mats.append(v)
        meta.append(dict(r))
    if not mats:
        return []

    M = np.vstack(mats)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scores = M @ q
    order = np.argsort(-scores)[:top_k]
    out = []
    for i in order:
        s = float(scores[i])
        if s < min_score:
            continue
        m = meta[i]
        img = None
        try:
            arr = json.loads(m["image_paths"]) if m["image_paths"] else []
            img = arr[0] if arr else None
        except (json.JSONDecodeError, TypeError):
            pass
        out.append({
            "person_report_id": m["id"],
            "kind": m["kind"],
            "name": m["full_name"],
            "location": m["location"],
            "report_date": m["report_date"],
            "detail": m["detail"],
            "contact": m["contact"],
            "image_path": img,
            "score": round(s, 4),
        })
    return out

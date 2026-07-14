"""UpFound backend — FastAPI. Serves the Web_dev frontend (same-origin) and the
API: auth (register/login), lost-item reports, EdgeAI event ingest, matching.
"""
from __future__ import annotations

import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, matching, security
from .db import db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="UpFound backend", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _crop_url(crop_ref: str | None) -> str | None:
    """EdgeAI crop_ref ('./out/crops/x.jpg') → served URL under /edgeout."""
    if not crop_ref:
        return None
    rel = crop_ref.lstrip("./")                  # 'out/crops/x.jpg'
    return "/edgeout/" + rel[len("out/"):] if rel.startswith("out/") else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
class RegisterIn(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginIn(BaseModel):
    email: str
    password: str


@app.post("/api/register")
def register(body: RegisterIn):
    if not body.email or not body.password:
        raise HTTPException(400, "email and password are required")
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email=?", (body.email,)).fetchone():
            raise HTTPException(409, "email already registered")
        cur = conn.execute(
            "INSERT INTO users(email, password_hash, name, created_at) VALUES(?,?,?,?)",
            (body.email, security.hash_password(body.password), body.name, _now()),
        )
        uid = cur.lastrowid
    return {
        "token": security.make_token(uid, body.email),
        "user": {"id": uid, "email": body.email, "name": body.name},
    }


@app.post("/api/login")
def login(body: LoginIn):
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (body.email,)).fetchone()
    if row is None or not security.verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "invalid email or password")
    return {
        "token": security.make_token(row["id"], row["email"]),
        "user": {"id": row["id"], "email": row["email"], "name": row["name"]},
    }


@app.get("/api/me")
def me(user=Depends(security.current_user)):
    return user


# --------------------------------------------------------------------------- #
# lost-item reports  (+ immediate matching)
# --------------------------------------------------------------------------- #
@app.post("/api/reports")
async def create_report(
    itemName: str = Form(...),
    itemColor: str = Form(""),
    itemQty: int = Form(1),
    itemType: str = Form(""),
    itemLocation: str = Form(""),
    itemDate: str = Form(""),
    itemDetail: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    user=Depends(security.current_user),
):
    from . import embeddings  # lazy — first report warms up CLIP

    saved: list[str] = []
    for up in images or []:
        if not up.filename:
            continue
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(up.filename).suffix or '.jpg'}"
        with dest.open("wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append(str(dest))

    # query vector: average of uploaded photos, else CLIP text of the description
    if saved:
        vecs = np.vstack([embeddings.embed_image(p) for p in saved])
        qvec = vecs.mean(axis=0)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
    else:
        text = " ".join(x for x in (itemName, itemColor, itemDetail) if x) or itemName
        qvec = embeddings.embed_text(text)

    now = _now()
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO lost_reports
               (user_id, item_name, color, qty, item_type, location, lost_date,
                detail, image_paths, embedding, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (user["id"], itemName, itemColor, itemQty, itemType, itemLocation, itemDate,
             itemDetail, json.dumps(saved), json.dumps(qvec.tolist()), now),
        )
        rid = cur.lastrowid

    results = matching.cosine_matches(qvec)
    with db() as conn:
        for r in results:
            conn.execute(
                "INSERT OR IGNORE INTO matches(report_id, event_id, score, created_at) "
                "VALUES(?,?,?,?)",
                (rid, r["event_id"], r["score"], now),
            )

    return {
        "report_id": rid,
        "used": "images" if saved else "text",
        "matches": [{**r, "crop_url": _crop_url(r.get("crop_ref"))} for r in results],
    }


@app.get("/api/reports")
def list_reports(user=Depends(security.current_user)):
    with db() as conn:
        rows = conn.execute(
            "SELECT id, item_name, color, item_type, location, lost_date, status, "
            "created_at FROM lost_reports WHERE user_id=? ORDER BY id DESC",
            (user["id"],),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/reports/{rid}/matches")
def report_matches(rid: int, user=Depends(security.current_user)):
    with db() as conn:
        owner = conn.execute(
            "SELECT user_id FROM lost_reports WHERE id=?", (rid,)
        ).fetchone()
        if owner is None:
            raise HTTPException(404, "report not found")
        if owner["user_id"] != user["id"]:
            raise HTTPException(403, "not your report")
        rows = conn.execute(
            """SELECT m.event_id, m.score, m.status, e.object_class, e.zone,
                      e.capture_ts, e.crop_ref
               FROM matches m JOIN detected_events e ON e.event_id = m.event_id
               WHERE m.report_id=? ORDER BY m.score DESC""",
            (rid,),
        ).fetchall()
    return [{**dict(r), "crop_url": _crop_url(r["crop_ref"])} for r in rows]


# --------------------------------------------------------------------------- #
# EdgeAI events / ingest
# --------------------------------------------------------------------------- #
@app.post("/api/ingest")
def ingest():
    from . import ingest as ing
    return ing.ingest_events()


@app.get("/api/events")
def events(limit: int = 50):
    with db() as conn:
        rows = conn.execute(
            "SELECT event_id, object_class, zone, capture_ts, crop_ref, model_version, "
            "(embedding IS NOT NULL) AS has_embedding FROM detected_events "
            "ORDER BY capture_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{**dict(r), "crop_url": _crop_url(r["crop_ref"])} for r in rows]


@app.get("/api/health")
def health():
    return {"ok": True, "clip_model": None}  # clip_model filled lazily elsewhere


# --------------------------------------------------------------------------- #
# static — frontend + images (same-origin, no CORS). Mounted LAST so /api wins.
# --------------------------------------------------------------------------- #
app.mount("/edgeout", StaticFiles(directory=str(config.EDGE_OUT_DIR)), name="edgeout")
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")

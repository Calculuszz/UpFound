"""UpFound backend — FastAPI. Serves the Web_dev frontend (same-origin) and the
API: auth (register/login), lost-item reports, EdgeAI event ingest, matching.
"""
from __future__ import annotations

import asyncio
import json
import logging
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

log = logging.getLogger("uvicorn.error")  # share uvicorn's handler so logs show


async def _auto_ingest_loop():
    """Poll EdgeAI's events file and ingest new detections automatically, so the
    camera → backend loop needs no manual /api/ingest call. Only re-ingests when
    the file actually changed (mtime), and runs the blocking work off the loop."""
    from . import ingest as ing

    last_mtime = None
    while True:
        try:
            await asyncio.sleep(config.INGEST_INTERVAL_SECONDS)
            p = config.EVENTS_JSONL
            if not p.exists():
                continue
            mtime = p.stat().st_mtime
            if mtime == last_mtime:
                continue
            res = await asyncio.to_thread(ing.ingest_events)
            last_mtime = mtime
            log.info("auto-ingest: %s", res)
        except asyncio.CancelledError:
            break
        except Exception as e:  # noqa: BLE001 — a bad cycle shouldn't kill the poller
            log.warning("auto-ingest error: %r", e)


def _seed_demo():
    """Create the demo account if missing so people can try the app without
    registering. Credentials are shown on the login page via /api/demo-account."""
    if not config.DEMO_ENABLED:
        return
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email=?", (config.DEMO_EMAIL,)).fetchone():
            return
        conn.execute(
            "INSERT INTO users(email, password_hash, name, created_at) VALUES(?,?,?,?)",
            (config.DEMO_EMAIL, security.hash_password(config.DEMO_PASSWORD),
             config.DEMO_NAME, _now()),
        )
    log.info("seeded demo account: %s", config.DEMO_EMAIL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _seed_demo()
    task = None
    if config.INGEST_INTERVAL_SECONDS > 0:
        task = asyncio.create_task(_auto_ingest_loop())
        log.info("auto-ingest every %ss", config.INGEST_INTERVAL_SECONDS)
    yield
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


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


def _upload_url(path: str | None) -> str | None:
    """A stored upload path → served URL under /uploads."""
    if not path:
        return None
    return "/uploads/" + Path(path).name


def _first_upload_url(image_paths_json: str | None) -> str | None:
    if not image_paths_json:
        return None
    try:
        arr = json.loads(image_paths_json)
    except (json.JSONDecodeError, TypeError):
        return None
    return _upload_url(arr[0]) if arr else None


def _save_uploads(images: list[UploadFile]) -> list[str]:
    """Persist uploaded images after validating type + size. Rejects non-images
    and oversized files with 400 rather than letting them hit disk / CLIP."""
    saved: list[str] = []
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    for up in images or []:
        if not up.filename:
            continue
        ct = (up.content_type or "").lower()
        if ct and ct not in config.ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, f"ไฟล์ไม่รองรับ: {ct} (รับเฉพาะรูปภาพ)")
        data = up.file.read()
        if len(data) > max_bytes:
            raise HTTPException(400, f"ไฟล์ใหญ่เกิน {config.MAX_UPLOAD_MB} MB")
        if not data:
            continue
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(up.filename).suffix or '.jpg'}"
        dest.write_bytes(data)
        saved.append(str(dest))
    return saved


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


@app.get("/api/demo-account")
def demo_account():
    """Public — lets the login page offer a one-click demo sign-in."""
    if not config.DEMO_ENABLED:
        return {"enabled": False}
    return {"enabled": True, "email": config.DEMO_EMAIL, "password": config.DEMO_PASSWORD}


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

    saved = _save_uploads(images)

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

    results = matching.cosine_matches(qvec, matching.IMAGE if saved else matching.TEXT)
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


def _person_query_vec(saved: list[str]):
    """Mean CLIP image embedding of the uploaded person photos (None if none)."""
    if not saved:
        return None
    from . import embeddings

    vecs = []
    for p in saved:
        try:
            vecs.append(embeddings.embed_image(p))
        except Exception:  # noqa: BLE001 — skip an unreadable image
            pass
    if not vecs:
        return None
    v = np.vstack(vecs).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def _insert_person(kind, user, full_name, gender, age, height_cm, contact,
                   location, report_date, detail, saved, qvec):
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO person_reports
               (user_id, kind, full_name, gender, age, height_cm, contact,
                location, report_date, detail, image_paths, embedding, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (user["id"] if user else None, kind, full_name, gender, age, height_cm,
             contact, location, report_date, detail, json.dumps(saved),
             json.dumps(qvec.tolist()) if qvec is not None else None, _now()),
        )
        return cur.lastrowid


def _person_matches(qvec, opposite_kind):
    if qvec is None:
        return []
    return [
        {**m, "image_url": _upload_url(m.get("image_path"))}
        for m in matching.cosine_person_matches(qvec, opposite_kind)
    ]


@app.post("/api/person-reports")
async def create_person_report(
    personName: str = Form(...),
    personGender: str = Form(""),
    personAge: int = Form(0),
    personHeight: int = Form(0),
    personLocation: str = Form(""),
    personDate: str = Form(""),
    personDetail: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    user=Depends(security.optional_user),  # public — reporting a person needs no login
):
    # Lost person → match its photo against FOUND person sightings (image↔image).
    saved = _save_uploads(images)
    qvec = _person_query_vec(saved)
    pid = _insert_person("lost", user, personName, personGender, personAge,
                         personHeight, None, personLocation, personDate,
                         personDetail, saved, qvec)
    return {"person_report_id": pid, "used": "images" if saved else "none",
            "matches": _person_matches(qvec, "found")}


@app.post("/api/found-persons")
async def create_found_person(
    foundLocation: str = Form(""),
    foundDate: str = Form(""),
    foundDetail: str = Form(""),
    foundContact: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    user=Depends(security.optional_user),  # public
):
    # Found person → match its photo against LOST person reports.
    saved = _save_uploads(images)
    qvec = _person_query_vec(saved)
    pid = _insert_person("found", user, None, None, 0, 0, foundContact,
                         foundLocation, foundDate, foundDetail, saved, qvec)
    return {"found_report_id": pid, "used": "images" if saved else "none",
            "matches": _person_matches(qvec, "lost")}


@app.post("/api/found-items")
async def create_found_item(
    foundItemName: str = Form(...),
    foundItemColor: str = Form(""),
    foundItemLocation: str = Form(""),
    foundItemDate: str = Form(""),
    foundItemDetail: str = Form(""),
    foundItemContact: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    user=Depends(security.optional_user),  # public — a finder needs no account
):
    # Found item → embed photo/description, match against LOST item reports so
    # the owner looking for it is surfaced to the finder.
    from . import embeddings

    saved = _save_uploads(images)
    if saved:
        vecs = np.vstack([embeddings.embed_image(p) for p in saved])
        qvec = vecs.mean(axis=0)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
    else:
        text = " ".join(x for x in (foundItemName, foundItemColor, foundItemDetail) if x) or foundItemName
        qvec = embeddings.embed_text(text)

    with db() as conn:
        cur = conn.execute(
            """INSERT INTO lost_reports
               (user_id, kind, item_name, color, contact, location, lost_date,
                detail, image_paths, embedding, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (user["id"] if user else None, "found", foundItemName, foundItemColor,
             foundItemContact, foundItemLocation, foundItemDate, foundItemDetail,
             json.dumps(saved), json.dumps(qvec.tolist()), _now()),
        )
        rid = cur.lastrowid

    matches = matching.cosine_lost_item_matches(
        qvec, matching.IMAGE if saved else matching.TEXT)
    return {"found_report_id": rid, "used": "images" if saved else "text",
            "matches": [{**m, "image_url": _upload_url(m.get("image_path"))} for m in matches]}


@app.get("/api/feed")
def feed(limit: int = 60):
    """Public gallery for data.html — lost items + lost persons, newest first."""
    out = []
    with db() as conn:
        for r in conn.execute(
            "SELECT id, kind, item_name, color, detail, location, status, image_paths, "
            "created_at FROM lost_reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall():
            out.append({
                "kind": "item-found" if r["kind"] == "found" else "item",
                "id": r["id"],
                "name": r["item_name"],
                "detail": " ".join(x for x in (r["color"], r["detail"]) if x),
                "location": r["location"],
                "status": r["status"],
                "image_url": _first_upload_url(r["image_paths"]),
                "created_at": r["created_at"],
            })
        for r in conn.execute(
            "SELECT id, kind, full_name, detail, location, status, image_paths, "
            "created_at FROM person_reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall():
            name = r["full_name"] or ("พบบุคคล" + (" @ " + r["location"] if r["location"] else ""))
            out.append({
                "kind": "person-found" if r["kind"] == "found" else "person",
                "id": r["id"],
                "name": name,
                "detail": r["detail"],
                "location": r["location"],
                "status": r["status"],
                "image_url": _first_upload_url(r["image_paths"]),
                "created_at": r["created_at"],
            })
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out[:limit]


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
class _RevalidatingStatic(StaticFiles):
    """Frontend assets that must never be served stale. Cloudflare caches .js/.css
    by extension and tells browsers to hold them for 4h, so without this a deploy
    stays invisible to anyone who loaded the page before it. "no-cache" still
    allows caching — it just forces a revalidate, so unchanged files cost a 304.
    Uploaded/crop images keep the default (their filenames are unique, so a stale
    hit is impossible and caching them is a win)."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/edgeout", StaticFiles(directory=str(config.EDGE_OUT_DIR)), name="edgeout")
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")
app.mount("/", _RevalidatingStatic(directory=str(config.WEB_DIR), html=True), name="web")

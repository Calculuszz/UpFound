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
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
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
# abuse controls — every write path is reachable without an account (a finder
# needs none) and the DB holds people's phone numbers, so throttle by client IP:
# slows brute-forcing logins, mass-registering, harvesting contacts through the
# public demo login, and burning the Gemini quota / disk with spammed uploads.
# In-memory is enough for a single-process booth deployment.
# --------------------------------------------------------------------------- #
import time
from collections import defaultdict, deque

# path → (max POSTs, window seconds) per IP
_RL_RULES: dict[str, tuple[int, int]] = {
    "/api/login": (10, 60),
    "/api/register": (5, 300),
    "/api/reports": (20, 60),
    "/api/found-items": (20, 60),
    "/api/found-persons": (20, 60),
    "/api/person-reports": (20, 60),
}
_rl_hits: dict[tuple[str, str], deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # The socket peer is always the tunnel (127.0.0.1); the real client is in the
    # header Cloudflare adds. Fall back to the peer for direct/local access.
    return (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "?")
    )


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    rule = _RL_RULES.get(request.url.path)
    if rule and request.method == "POST":
        limit, window = rule
        hits = _rl_hits[(_client_ip(request), request.url.path)]
        now = time.monotonic()
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= limit:
            return JSONResponse({"detail": "คำขอถี่เกินไป กรุณาลองใหม่อีกสักครู่"}, status_code=429)
        hits.append(now)
    return await call_next(request)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    # Defence in depth for the upload path: even if something non-image slips
    # through, the browser must not sniff it into an executable type.
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    return resp


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _crop_url(crop_ref: str | None) -> str | None:
    """EdgeAI crop_ref ('./out/crops/x.jpg') → served URL under /edgeout."""
    if not crop_ref:
        return None
    rel = crop_ref.lstrip("./")                  # 'out/crops/x.jpg'
    return "/edgeout/" + rel[len("out/"):] if rel.startswith("out/") else None


def _crop_path(crop_ref: str | None) -> Path | None:
    """EdgeAI crop_ref ('./out/crops/x.jpg') → the file on disk."""
    if not crop_ref:
        return None
    rel = crop_ref.lstrip("./")
    return config.EDGE_OUT_DIR / rel[len("out/"):] if rel.startswith("out/") else None


def _rerank_crops(description: str, results: list[dict]) -> list[dict]:
    """Let Gemini look at the crops CLIP shortlisted and drop the ones that
    aren't the item at all (cosine put a tablet top for "black wallet").

    Any failure — no key, no wifi, unparsable reply — falls back to CLIP's own
    ranking, so this can only improve results, never withhold them. When the
    model does answer, its verdict replaces the cosine-derived confidence,
    because "the judge looked at it" means more than a rescaled dot product.
    """
    from . import llm

    trimmed = results[: config.MATCH_TOP_K]
    if not results or not description.strip() or not llm.enabled():
        return trimmed

    from PIL import Image

    blobs: list[tuple[str, bytes]] = []
    for r in results:
        p = _crop_path(r.get("crop_ref"))
        if not (p and p.exists()):
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            if w * h < config.MIN_CROP_PIXELS:
                continue  # unreadable; judging it would only invent confidence
            blobs.append((str(r["event_id"]), p.read_bytes()))
        except (OSError, ValueError):
            pass
    scores = llm.rerank(description, blobs)
    if not scores:
        return trimmed

    kept = [
        {**r, "llm_score": s, "confidence": round(s / 100, 4)}
        for r in results
        if (s := scores.get(str(r["event_id"]), 0)) >= config.RERANK_MIN_SCORE
    ]
    kept.sort(key=lambda r: r["llm_score"], reverse=True)
    return kept[: config.MATCH_TOP_K]


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


# Real image format (from decoding the bytes) → the extension we save under. The
# client's filename and content-type are attacker-controlled and were the stored
# XSS vector: an .html uploaded with a fake image/jpeg content-type got saved as
# .html and served from our own origin as text/html, so any script in it ran with
# access to the visitor's token in localStorage. We now ignore both labels and
# trust only what PIL can actually decode.
_IMAGE_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


def _save_uploads(images: list[UploadFile]) -> list[str]:
    """Persist genuine images only. A file that PIL cannot decode into a known
    image format is rejected before it touches disk, and the on-disk name uses a
    server-chosen extension so nothing user-named (.html, .svg, .js) can ever be
    served back."""
    import io

    from PIL import Image, UnidentifiedImageError

    saved: list[str] = []
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    for up in images or []:
        if not up.filename:
            continue
        data = up.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(400, f"ไฟล์ใหญ่เกิน {config.MAX_UPLOAD_MB} MB")
        if not data:
            continue
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()               # structural check; consumes the stream
            fmt = Image.open(io.BytesIO(data)).format   # re-open to read the format
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise HTTPException(400, "ไฟล์ไม่ใช่รูปภาพที่ถูกต้อง")
        ext = _IMAGE_EXT.get(fmt or "")
        if ext is None:
            raise HTTPException(400, f"ชนิดรูปไม่รองรับ: {fmt}")
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
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
    text = ""
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

    modality = matching.IMAGE if saved else matching.TEXT
    if saved:
        results = matching.cosine_matches(qvec, matching.IMAGE)
    else:
        # cosine recalls the right crop but ranks it badly, so shortlist wide and
        # let the judge decide what actually gets shown
        results = _rerank_crops(text, matching.cosine_matches(
            qvec, matching.TEXT, top_k=config.RERANK_CANDIDATES))

    # A camera sighting only says where the item was; someone who already handed
    # it in can hand it back, so search their reports too. Without this the two
    # directions were asymmetric — reporting a find searched for owners, but
    # reporting a loss only ever looked at crops, so whether a pair matched came
    # down to which form the user happened to fill in first.
    found_reports = matching.cosine_item_report_matches(qvec, modality, "found")
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
        "found_matches": [
            {**m, "image_url": _upload_url(m.get("image_path"))} for m in found_reports
        ],
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

    matches = matching.cosine_item_report_matches(
        qvec, matching.IMAGE if saved else matching.TEXT, "lost")
    return {"found_report_id": rid, "used": "images" if saved else "text",
            "matches": [{**m, "image_url": _upload_url(m.get("image_path"))} for m in matches]}


@app.get("/api/feed")
def feed(limit: int = 60):
    """Public gallery for data.html — user reports + items the camera detected,
    newest first. Camera detections were previously invisible here (they only
    surfaced as matches when someone reported a loss), so a freshly-abandoned
    item never appeared on the site; include them as their own 'camera' kind."""
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
        for r in conn.execute(
            "SELECT event_id, object_class, zone, capture_ts, crop_ref "
            "FROM detected_events ORDER BY rowid DESC LIMIT ?", (limit,)
        ).fetchall():
            out.append({
                "kind": "camera",
                "id": r["event_id"],
                "name": r["object_class"],
                "detail": "กล้องตรวจพบของที่ถูกทิ้งไว้",
                "location": r["zone"],
                "status": "camera",
                "image_url": _crop_url(r["crop_ref"]),
                "created_at": r["capture_ts"],
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

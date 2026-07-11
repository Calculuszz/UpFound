"""§5 Emit Event Contract.

Builds the exact same event schema as the replay producer (clip_to_events.py).
The ONLY difference is `source="cctv"`. The backend must not be able to tell a
replay event from a cctv event apart otherwise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any

from . import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP embedding — 512-d L2-normalized vector for the object crop.
# ---------------------------------------------------------------------------
class ClipEmbedder:
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL_NAME,
        pretrained: str = config.CLIP_PRETRAINED,
    ) -> None:
        import open_clip  # lazy import (heavy)
        import torch

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)

    def embed(self, crop_bgr) -> list[float]:
        from PIL import Image

        rgb = crop_bgr[:, :, ::-1]  # BGR (opencv) → RGB
        img = Image.fromarray(rgb)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
        return feats.squeeze(0).cpu().tolist()


# ---------------------------------------------------------------------------
# Persistence of crops / person frames
# ---------------------------------------------------------------------------
def save_image(img, directory: str, name: str) -> str | None:
    if img is None:
        return None
    import cv2

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    cv2.imwrite(path, img)
    return path.replace("\\", "/")  # normalize to forward slash for cross-platform


# ---------------------------------------------------------------------------
# Event construction (pure — unit testable without models)
# ---------------------------------------------------------------------------
def make_event_id(camera_id: str, track_id: int, ts: datetime) -> str:
    raw = f"{camera_id}|{track_id}|{ts.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_event(
    *,
    camera_id: str,
    zone: str,
    track_id: int,
    cls: int,
    bbox_xyxy: tuple[int, int, int, int],
    ts: datetime,
    crop_ref: str | None,
    person_ref: str | None,
    embedding: list[float] | None,
    item_classes: dict[int, str] | None = None,
) -> dict[str, Any]:
    item_classes = item_classes or config.ITEM_CLASSES
    x1, y1, x2, y2 = bbox_xyxy
    return {
        "schema_version": config.SCHEMA_VERSION,
        "model_version": config.ACTIVE_MODEL_VERSION,
        "event_id": make_event_id(camera_id, track_id, ts),
        "source": "cctv",  # ← the only field that differs from replay
        "camera_id": camera_id,
        "zone": zone,
        "capture_ts": ts.isoformat(),
        "detect_type": "abandoned_object",
        "object_class": item_classes[cls],
        "track_id": track_id,
        "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
        "crop_ref": crop_ref,
        "person_ref": person_ref,  # nullable (design v3)
        "embedding": embedding,  # 512-d normalized
    }


# ---------------------------------------------------------------------------
# Publishers
# ---------------------------------------------------------------------------
class JsonlPublisher:
    def __init__(self, path: str = config.EVENTS_JSONL) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def publish(self, event: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class RedisPublisher:
    def __init__(self, url: str = config.REDIS_URL, stream: str = config.REDIS_STREAM):
        import redis  # lazy

        self.client = redis.Redis.from_url(url)
        self.stream = stream

    def publish(self, event: dict) -> None:
        # Redis stream fields must be flat strings → JSON-encode the payload.
        self.client.xadd(self.stream, {"event": json.dumps(event, ensure_ascii=False)})


def make_publisher(backend: str = config.PUBLISH_BACKEND):
    if backend == "redis":
        return RedisPublisher()
    if backend == "kafka":
        raise NotImplementedError("kafka publisher not yet implemented")
    return JsonlPublisher()


# ---------------------------------------------------------------------------
# High-level emit — wires crop → embed → build → publish
# ---------------------------------------------------------------------------
class Emitter:
    def __init__(self, camera, embedder: ClipEmbedder | None = None, publisher=None):
        self.camera = camera
        self.embedder = embedder
        self.publisher = publisher or make_publisher()

    def emit_event(self, frame, bbox_xyxy, track_id, cls, ts, person_frame=None) -> dict:
        x1, y1, x2, y2 = bbox_xyxy
        crop = frame[y1:y2, x1:x2]
        eid = make_event_id(self.camera.camera_id, track_id, ts)
        crop_ref = save_image(crop, config.CROP_DIR, f"{eid}.jpg")
        person_ref = (
            save_image(person_frame, config.PERSON_DIR, f"{eid}_person.jpg")
            if person_frame is not None
            else None
        )
        embedding = self.embedder.embed(crop) if self.embedder is not None else None
        event = build_event(
            camera_id=self.camera.camera_id,
            zone=self.camera.zone,
            track_id=track_id,
            cls=cls,
            bbox_xyxy=bbox_xyxy,
            ts=ts,
            crop_ref=crop_ref,
            person_ref=person_ref,
            embedding=embedding,
        )
        self.publisher.publish(event)
        log.info("emitted event %s (source=cctv, person_ref=%s)", event["event_id"], person_ref)
        return event

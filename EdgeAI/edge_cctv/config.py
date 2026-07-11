"""§6 Config — central settings for the CCTV Edge AI pipeline.

Secrets (RTSP password) are read from the environment, never hardcoded.
See §2 of PROCESS_1_cctv_edge_spec.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model / contract
# ---------------------------------------------------------------------------
# MUST match Process 2 (matching) — the backend keys off this string.
# NOTE: changing the YOLO model changes this contract string, so the replay
# producer (clip_to_events.py) and Process 2 MUST be updated to the same value.
ACTIVE_MODEL_VERSION = "yolo26m_clip-vitb32"
SCHEMA_VERSION = "1.0"

# YOLO weights + CLIP model. Kept together so model_version stays coherent.
# yolo26m: YOLO26 medium — better accuracy from overhead CCTV (mAP 53.1).
YOLO_WEIGHTS = os.getenv("EDGE_YOLO_WEIGHTS", "yolo26m.pt")
CLIP_MODEL_NAME = os.getenv("EDGE_CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("EDGE_CLIP_PRETRAINED", "openai")

# Input resolution for YOLO inference. Larger = better for small/distant objects
# but slower. 640 is default; 1280 recommended for elevated CCTV cameras.
IMGSZ = int(os.getenv("EDGE_IMGSZ", "1280"))


# ---------------------------------------------------------------------------
# Detection / dwell — same values as the replay producer (clip_to_events.py)
# ---------------------------------------------------------------------------
SAMPLE_EVERY = int(os.getenv("EDGE_SAMPLE_EVERY", "3"))   # CCTV fps often < clip fps
DWELL_SECONDS = float(os.getenv("EDGE_DWELL_SECONDS", "8.0"))
MOVE_TOL = int(os.getenv("EDGE_MOVE_TOL", "25"))          # px centroid movement tolerance

# COCO class ids we treat as "abandonable" objects — personal belongings that
# get left behind. Kept focused on items with low false-positive risk in public
# spaces (bags/luggage + high-value forgotten items). Avoids bottle/cup/etc.
# which appear everywhere and would spam events.
# NOTE: this set MUST match the replay producer (clip_to_events.py) and Process 2
# so the Event Contract's object_class values stay in sync.
ITEM_CLASSES: dict[int, str] = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    67: "cell phone",
    73: "book",
}

# How close (px) a person centroid must be to an object to count as "placer".
PERSON_CLASS_ID = 0
PERSON_NEAR_TOL = int(os.getenv("EDGE_PERSON_NEAR_TOL", "150"))

TRACKER_CFG = os.getenv("EDGE_TRACKER", "bytetrack.yaml")


# ---------------------------------------------------------------------------
# Accuracy filters — cut false positives before an event is fired.
# ---------------------------------------------------------------------------
# Default minimum YOLO confidence to accept an item detection.
CONF_MIN = float(os.getenv("EDGE_CONF_MIN", "0.10"))

# Per-class confidence overrides. Small / error-prone classes need a higher bar.
# (Real data showed a 39x50 "cell phone" false positive firing an event.)
CONF_BY_CLASS: dict[int, float] = {
    24: 0.10,  # backpack — ต่ำสุด เพื่อจับจากมุมเพดานให้ได้
    26: 0.10,  # handbag
    28: 0.10,  # suitcase
    63: 0.10,  # laptop
    67: 0.55,  # cell phone — คงสูงไว้ กัน false positive
    73: 0.45,  # book
    25: 0.40,  # umbrella
}

# Confidence for person detections (used only as placer context, not fired).
PERSON_CONF = float(os.getenv("EDGE_PERSON_CONF", "0.35"))

# Reject item boxes whose shorter side is below this many pixels. Kills tiny
# spurious detections that YOLO occasionally emits on background texture.
# Set very low (10) for overhead CCTV where objects appear small.
MIN_BBOX_SIDE = int(os.getenv("EDGE_MIN_BBOX_SIDE", "10"))

# Only fire "abandoned" if the object was seen MOVING before it went still.
# A genuinely abandoned item is carried in then set down; permanent fixtures /
# background clutter are still from the very first frame and never move.
REQUIRE_MOVEMENT = os.getenv("EDGE_REQUIRE_MOVEMENT", "1") not in ("0", "false", "False")


def conf_threshold_for(cls: int) -> float:
    """Per-class confidence floor, falling back to the global CONF_MIN."""
    return CONF_BY_CLASS.get(cls, CONF_MIN)


# ---------------------------------------------------------------------------
# Stillness robustness — anchor-based, scale-aware (reduces false resets).
# ---------------------------------------------------------------------------
# "Still" = centroid stays within tol of the settled anchor. tol is the larger
# of MOVE_TOL (px floor) and MOVE_TOL_FRAC * shorter-box-side, so a big close
# object and a small far object are judged fairly.
MOVE_TOL_FRAC = float(os.getenv("EDGE_MOVE_TOL_FRAC", "0.15"))

# ---------------------------------------------------------------------------
# Track persistence — inherit dwell state when ByteTrack changes the id.
# ---------------------------------------------------------------------------
# When a new track_id appears where a just-lost track (same class) was, and the
# boxes overlap at least this IoU, the new id inherits the old dwell state.
ADOPT_IOU = float(os.getenv("EDGE_ADOPT_IOU", "0.3"))
# Drop track state that hasn't been seen for this long (also caps adoption age).
TRACK_STALE_SECONDS = float(os.getenv("EDGE_TRACK_STALE_SECONDS", "5.0"))

# ---------------------------------------------------------------------------
# Owner-left gating — the real "abandoned" signal.
# ---------------------------------------------------------------------------
# A person within this radius (px) of an object counts as its owner/placer.
OWNER_RADIUS = int(os.getenv("EDGE_OWNER_RADIUS", str(PERSON_NEAR_TOL)))
# After a placer was seen, no person may be near for this long before firing.
OWNER_LEFT_SECONDS = float(os.getenv("EDGE_OWNER_LEFT_SECONDS", "3.0"))
# Master switch for the owner-left requirement (needs person capture enabled).
REQUIRE_OWNER_LEFT = os.getenv("EDGE_REQUIRE_OWNER_LEFT", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# RTSP / reconnect
# ---------------------------------------------------------------------------
RTSP_TRANSPORT = os.getenv("EDGE_RTSP_TRANSPORT", "tcp")
RECONNECT_MAX_BACKOFF = int(os.getenv("EDGE_RECONNECT_MAX_BACKOFF", "30"))
CHANNEL_DEV, CHANNEL_PROD = "102", "101"
# Background reader thread keeps only the newest frame → bounded latency.
# Strongly recommended for live camera use; disable only for debugging.
READER_THREAD = os.getenv("EDGE_READER_THREAD", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Output / publish
# ---------------------------------------------------------------------------
CROP_DIR = os.getenv("EDGE_CROP_DIR", "./out/crops")
PERSON_DIR = os.getenv("EDGE_PERSON_DIR", "./out/persons")
EVENTS_JSONL = os.getenv("EDGE_EVENTS_JSONL", "./out/events.jsonl")
# publish backend: "jsonl" (dev), "redis", "kafka"
PUBLISH_BACKEND = os.getenv("EDGE_PUBLISH_BACKEND", "jsonl")
REDIS_URL = os.getenv("EDGE_REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM = os.getenv("EDGE_REDIS_STREAM", "events")


@dataclass
class CameraConfig:
    """One camera == one zone (§2). Zone mapping is external."""

    camera_id: str = "cam-01"
    zone: str = "fl2-zoneA"
    ip: str = "192.168.1.64"
    username: str = "admin"
    # NEVER hardcode — pulled from env/secret manager.
    password: str = field(default_factory=lambda: os.getenv("EDGE_RTSP_PASSWORD", ""))
    channel: str = CHANNEL_PROD
    rtsp_tmpl: str = "rtsp://{u}:{p}@{ip}:554/Streaming/Channels/{ch}"

    def rtsp_url(self) -> str:
        if not self.password:
            raise RuntimeError(
                "RTSP password not set. Export EDGE_RTSP_PASSWORD (never hardcode)."
            )
        return self.rtsp_tmpl.format(
            u=self.username, p=self.password, ip=self.ip, ch=self.channel
        )

    def rtsp_url_redacted(self) -> str:
        """URL safe for logging — password masked."""
        return self.rtsp_tmpl.format(
            u=self.username, p="****", ip=self.ip, ch=self.channel
        )


def load_camera(use_dev_channel: bool = False) -> CameraConfig:
    cam = CameraConfig(
        camera_id=os.getenv("EDGE_CAMERA_ID", "cam-01"),
        zone=os.getenv("EDGE_ZONE", "fl2-zoneA"),
        ip=os.getenv("EDGE_CAMERA_IP", "192.168.1.64"),
        username=os.getenv("EDGE_CAMERA_USER", "admin"),
    )
    cam.channel = CHANNEL_DEV if use_dev_channel else CHANNEL_PROD
    return cam

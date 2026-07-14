"""§4 Detect + Track + Dwell.

Reuses the same dwell logic as the replay producer (clip_to_events.py):
an object that stays still (centroid moves < MOVE_TOL) for >= DWELL_SECONDS
is "abandoned" and fires exactly once.

The DwellTracker is pure Python (no YOLO / no cv2) so it is unit-testable.
The YoloTracker wraps ultralytics and is imported lazily.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

from . import config

log = logging.getLogger(__name__)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Detection(NamedTuple):
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int
    cls: int
    conf: float = 1.0  # detection confidence (default 1.0 keeps old callers/tests valid)


def centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def item_passes_filters(det: Detection) -> bool:
    """Accuracy gate for item detections: per-class confidence + min box size.

    Filters the tiny / low-confidence spurious boxes that would otherwise dwell
    and fire false abandoned-object events (e.g. a 39x50 'cell phone').
    """
    if det.conf < config.conf_threshold_for(det.cls):
        return False
    x1, y1, x2, y2 = det.bbox
    if min(x2 - x1, y2 - y1) < config.MIN_BBOX_SIDE:
        return False
    return True


@dataclass
class TrackState:
    still_since: datetime | None = None
    anchor: tuple[float, float] | None = None  # settled centroid; stillness reference
    fired: bool = False
    last_bbox: tuple[int, int, int, int] | None = None
    last_seen: datetime | None = None
    cls: int | None = None
    class_votes: Counter = field(default_factory=Counter)  # majority-vote class (#5)
    has_moved: bool = False  # seen moving at least once (placement signal)
    person_seen: bool = False  # a person was near it at some point (placer signal)
    last_person_near_ts: datetime | None = None  # last time a person was near (owner)
    person_frame: object | None = None  # snapshot of the placer (nullable)

    def voted_class(self) -> int | None:
        if self.class_votes:
            return self.class_votes.most_common(1)[0][0]
        return self.cls


@dataclass
class DwellTracker:
    """Tracks per-object stillness and decides when to fire an abandoned event.

    Robustness upgrades over a naive frame-to-frame check:
      - stillness is measured against a settled *anchor* with a scale-aware
        tolerance, so bbox jitter doesn't keep resetting the clock (#2);
      - dwell state is inherited across ByteTrack id changes via IoU (#1);
      - firing requires the placer/owner to have LEFT for a while (#3);
      - object class is a majority vote over the track's life (#5).
    """

    dwell_seconds: float = config.DWELL_SECONDS
    move_tol: int = config.MOVE_TOL
    require_movement: bool = config.REQUIRE_MOVEMENT
    require_owner_left: bool = config.REQUIRE_OWNER_LEFT
    owner_left_seconds: float = config.OWNER_LEFT_SECONDS
    owner_radius: int = config.OWNER_RADIUS
    tracks: dict[int, TrackState] = field(default_factory=dict)

    # -- stillness (single detection) -------------------------------------
    def _tol_for(self, bbox: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return max(self.move_tol, config.MOVE_TOL_FRAC * min(x2 - x1, y2 - y1))

    def update(self, det: Detection, now: datetime) -> None:
        """Update one detection's stillness + class vote (anchor-based).

        .get() คืน None ถ้าไม่เจอ key (ต่างจาก self.tracks[id] ที่จะพัง KeyError).
        """
        st = self.tracks.get(det.track_id)
        c = centroid(det.bbox)
        if st is None:
            st = TrackState(still_since=now, anchor=c)
            self.tracks[det.track_id] = st
        elif _dist(c, st.anchor) > self._tol_for(det.bbox):
            # drifted beyond tolerance from the settled position → real move.
            st.anchor = c
            st.still_since = now
            st.fired = False
            st.has_moved = True
        st.last_bbox = det.bbox
        st.last_seen = now
        st.cls = det.cls
        st.class_votes[det.cls] += 1

    # -- full frame (adoption + persons + prune) --------------------------
    def update_frame(self, items, persons, now: datetime, frame=None) -> None:
        """Process one frame's item + person detections end to end."""
        self._adopt_lost_tracks(items)
        for det in items:
            self.update(det, now)
        self._note_persons(items, persons or [], now, frame)
        self._prune(now)

    def _adopt_lost_tracks(self, items) -> None:
        """Inherit dwell state when ByteTrack assigns a new id to the same object
        at the same location (occlusion / brief disappearance)."""
        seen = {d.track_id for d in items}
        lost = [tid for tid in self.tracks if tid not in seen]
        if not lost:
            return
        for det in items:
            if det.track_id in self.tracks:  # not a new id
                continue
            best_id, best_iou = None, config.ADOPT_IOU
            for tid in lost:
                st = self.tracks[tid]
                if st.last_bbox is None or st.voted_class() != det.cls:
                    continue
                iou = _iou(det.bbox, st.last_bbox)
                if iou >= best_iou:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                self.tracks[det.track_id] = self.tracks.pop(best_id)
                lost.remove(best_id)

    def _note_persons(self, items, persons, now: datetime, frame) -> None:
        """Record owner proximity + capture the placer frame (face compare)."""
        if not persons:
            return
        for det in items:
            st = self.tracks.get(det.track_id)
            if st is None or st.fired:
                continue
            ic = centroid(det.bbox)
            if any(_dist(ic, centroid(p.bbox)) <= self.owner_radius for p in persons):
                st.person_seen = True
                st.last_person_near_ts = now
                if frame is not None:
                    st.person_frame = frame.copy() if hasattr(frame, "copy") else frame

    def _prune(self, now: datetime) -> None:
        stale = [
            tid for tid, st in self.tracks.items()
            if st.last_seen is not None
            and (now - st.last_seen).total_seconds() > config.TRACK_STALE_SECONDS
        ]
        for tid in stale:
            del self.tracks[tid]

    # -- queries ----------------------------------------------------------
    def dwell_seconds_of(self, track_id: int, now: datetime) -> float:
        st = self.tracks.get(track_id)
        if st is None or st.still_since is None:
            return 0.0
        return (now - st.still_since).total_seconds()

    def voted_class(self, track_id: int) -> int | None:
        st = self.tracks.get(track_id)
        return st.voted_class() if st is not None else None

    def owner_present(self, track_id: int, now: datetime) -> bool:
        """True while a placer was seen and a person is still nearby recently."""
        st = self.tracks.get(track_id)
        if st is None or not st.person_seen or st.last_person_near_ts is None:
            return False
        return (now - st.last_person_near_ts).total_seconds() < self.owner_left_seconds

    def should_fire(self, track_id: int, now: datetime) -> bool:
        st = self.tracks.get(track_id)
        if st is None or st.fired:
            return False
        # Placement gate: carried in (moved) OR a placer was seen near it.
        if self.require_movement and not (st.has_moved or st.person_seen):
            return False
        # Stillness gate.
        if self.dwell_seconds_of(track_id, now) < self.dwell_seconds:
            return False
        # Owner-left gate: if a placer was seen, they must have left for a while.
        if self.require_owner_left and self.owner_present(track_id, now):
            return False
        return True

    def mark_fired(self, track_id: int) -> None:
        st = self.tracks.get(track_id)
        if st is not None:
            st.fired = True


class YoloTracker:
    """Thin wrapper over ultralytics `.track` (persist + bytetrack).

    Supports two detectors, selected by ``config.DETECTOR``:
      * ``yolo``  — ultralytics YOLO with fixed COCO classes (yolo26x default).
      * ``yoloe`` — YOLOE open-vocabulary; the class vocabulary is the text
        prompts in ``config.ITEM_CLASSES.values()`` + ``"person"``. Class ids
        are the prompt indices, so ``config.PERSON_CLASS_ID`` == len(prompts).

    Both return the same ``Detection`` list, so dwell/emit code is identical.
    """

    def __init__(
        self,
        weights: str = config.YOLO_WEIGHTS,
        item_classes: dict[int, str] | None = None,
        tracker_cfg: str = config.TRACKER_CFG,
    ) -> None:
        import torch  # lazy import (heavy)

        self.item_classes = item_classes or config.ITEM_CLASSES
        self.tracker_cfg = tracker_cfg
        self.is_yoloe = config.DETECTOR == "yoloe"

        if self.is_yoloe:
            from ultralytics import YOLOE

            self.model = YOLOE(weights)
            # Vocabulary = item prompts (in id order) + person appended last, so
            # the person index matches config.PERSON_CLASS_ID.
            prompts = list(self.item_classes.values()) + ["person"]
            self.model.set_classes(prompts, self.model.get_text_pe(prompts))
            log.info("YOLOE open-vocab prompts: %s", prompts)
        else:
            from ultralytics import YOLO

            self.model = YOLO(weights)

        # FP16 only makes sense on GPU; on CPU it would silently slow things down.
        # ultralytics >= 8.4 takes quantize=16 (FP16) / None (FP32) instead of half=.
        use_fp16 = config.HALF and torch.cuda.is_available()
        self.quantize = 16 if use_fp16 else None
        if config.HALF and not use_fp16:
            log.info("EDGE_HALF requested but CUDA unavailable — using FP32")
        # Ask the model for anything at/above the lowest threshold we might keep,
        # then apply the exact per-class thresholds in Python (conf is global).
        self._conf_floor = min(
            [config.CONF_MIN, config.PERSON_CONF, *config.CONF_BY_CLASS.values()]
        )

    def _track(self, frame, classes):
        """Run one `.track` pass. YOLOE's vocabulary is fixed by set_classes(),
        so it ignores the `classes=` filter — pass it only for plain YOLO."""
        kw = dict(
            persist=True,
            conf=self._conf_floor,
            imgsz=config.IMGSZ,
            quantize=self.quantize,
            tracker=self.tracker_cfg,
            verbose=False,
        )
        if not self.is_yoloe:
            kw["classes"] = classes
        return self.model.track(frame, **kw)

    def track_items(self, frame) -> list[Detection]:
        """Return filtered item-class detections with stable track ids."""
        results = self._track(frame, classes=list(self.item_classes.keys()))
        raw = self._parse(results)
        if raw:
            log.debug("track_items raw: %d dets %s", len(raw),
                      [(d.cls, d.conf, d.bbox) for d in raw[:5]])
        # For YOLOE the person class is in the vocab too — keep item classes only.
        filtered = [
            d for d in raw if d.cls in self.item_classes and item_passes_filters(d)
        ]
        if raw and not filtered:
            log.debug("all %d dets filtered out", len(raw))
        return filtered

    def track_items_and_persons(self, frame) -> tuple[list[Detection], list[Detection]]:
        """Single pass covering items + persons, split into two lists.

        Avoids running inference twice per frame when person capture / preview
        is on — a major throughput win.
        """
        classes = list(self.item_classes.keys()) + [config.PERSON_CLASS_ID]
        results = self._track(frame, classes=classes)
        dets = self._parse(results)
        if dets:
            log.debug("track_items_and_persons raw: %d dets", len(dets))
        items = [
            d for d in dets if d.cls in self.item_classes and item_passes_filters(d)
        ]
        persons = [
            d for d in dets
            if d.cls == config.PERSON_CLASS_ID and d.conf >= config.PERSON_CONF
        ]
        return items, persons

    @staticmethod
    def _parse(results) -> list[Detection]:
        dets: list[Detection] = []
        if not results:
            return dets
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.id is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(ids)
        for (x1, y1, x2, y2), tid, cls, cf in zip(xyxy, ids, clss, confs):
            dets.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    track_id=int(tid),
                    cls=int(cls),
                    conf=float(cf),
                )
            )
        return dets


class FrameSampler:
    """`should_skip()` → True for all but every Nth frame (§4)."""

    def __init__(self, sample_every: int = config.SAMPLE_EVERY) -> None:
        self.sample_every = max(1, sample_every)
        self._i = -1

    def should_skip(self) -> bool:
        self._i = (self._i + 1) % self.sample_every
        return self._i != 0

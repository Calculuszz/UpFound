"""Pipeline tests — T3 (fire once), T4 (schema/source/model), T6 (same contract), T7 (null person).

Pure-logic tests: no YOLO / CLIP / cv2 required.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from UpFound.EdgeAI.edge_cctv import config
from UpFound.EdgeAI.edge_cctv.detector import (
    Detection,
    DwellTracker,
    FrameSampler,
    item_passes_filters,
)
from UpFound.EdgeAI.edge_cctv.emitter import Emitter, build_event, make_event_id
from UpFound.EdgeAI.edge_cctv.person_capture import find_person_detections


UTC = timezone.utc


# --- T3: dwell fires exactly once after DWELL_SECONDS of stillness -----------
def test_t3_fires_once_after_dwell():
    # require_movement=False isolates the pure dwell-timing logic here; the
    # movement gate is covered by test_requires_movement_before_firing below.
    dwell = DwellTracker(dwell_seconds=8.0, move_tol=25, require_movement=False)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    det = Detection(bbox=(100, 100, 140, 160), track_id=7, cls=24)

    # object present but not yet still long enough
    dwell.update(det, t0)
    assert not dwell.should_fire(7, t0)
    assert not dwell.should_fire(7, t0 + timedelta(seconds=5))

    # past the dwell threshold → should fire
    t_fire = t0 + timedelta(seconds=8.1)
    dwell.update(det, t_fire)  # same centroid → still
    assert dwell.should_fire(7, t_fire)

    # fire it, then it must not fire again on subsequent frames
    dwell.mark_fired(7)
    assert not dwell.should_fire(7, t_fire + timedelta(seconds=3))


def test_t3_movement_resets_dwell():
    dwell = DwellTracker(dwell_seconds=8.0, move_tol=25)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    dwell.update(Detection((100, 100, 140, 160), 7, 24), t0)
    # moves a lot at t+7 → clock resets
    moved = Detection((300, 300, 340, 360), 7, 24)
    dwell.update(moved, t0 + timedelta(seconds=7))
    assert not dwell.should_fire(7, t0 + timedelta(seconds=9))
    # now stays still 8s from the reset point
    assert dwell.should_fire(7, t0 + timedelta(seconds=7 + 8.1))


def test_frame_sampler_skips_all_but_every_n():
    s = FrameSampler(sample_every=3)
    skips = [s.should_skip() for _ in range(6)]
    # process frame 0 and 3 (skip=False), skip the rest
    assert skips == [False, True, True, False, True, True]


# --- #1 track-id adoption: dwell survives a ByteTrack id change --------------
def test_track_id_adoption_inherits_dwell_state():
    dwell = DwellTracker(dwell_seconds=8.0, require_movement=False)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    box = (100, 100, 200, 220)

    dwell.update_frame([Detection(box, 1, 28)], [], t0)
    dwell.update_frame([Detection(box, 1, 28)], [], t0 + timedelta(seconds=5))

    # id 1 vanishes; a near-identical box reappears as id 2 (tracker relabel)
    dwell.update_frame([Detection((102, 101, 201, 219), 2, 28)], [], t0 + timedelta(seconds=6))

    assert 1 not in dwell.tracks and 2 in dwell.tracks  # inherited, not duplicated
    # dwell clock was NOT reset by the id change → fires on schedule
    assert dwell.should_fire(2, t0 + timedelta(seconds=8.1))


def test_adoption_rejects_far_or_wrong_class():
    dwell = DwellTracker(require_movement=False)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    dwell.update_frame([Detection((100, 100, 200, 220), 1, 28)], [], t0)
    # new id, same class but no overlap → fresh track (no adoption)
    dwell.update_frame([Detection((600, 600, 700, 720), 2, 28)], [], t0 + timedelta(seconds=1))
    assert 1 in dwell.tracks and 2 in dwell.tracks


# --- #2 scale-aware stillness: sub-tolerance jitter must not reset the clock --
def test_scaled_tol_ignores_jitter():
    dwell = DwellTracker(dwell_seconds=8.0, move_tol=25, require_movement=False)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    # big box (min side 200) → scaled tol = max(25, 0.15*200) = 30
    dwell.update(Detection((100, 100, 300, 400), 1, 28), t0)
    # centroid jitters ~28px — over the old fixed 25 floor but under scaled 30
    dwell.update(Detection((120, 120, 320, 420), 1, 28), t0 + timedelta(seconds=3))
    assert dwell.tracks[1].still_since == t0            # clock NOT reset
    assert dwell.should_fire(1, t0 + timedelta(seconds=8.1))


# --- #5 majority-vote object class over the track's life ---------------------
def test_majority_vote_class():
    dwell = DwellTracker(require_movement=False)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    box = (100, 100, 200, 220)
    for i, cls in enumerate([24, 24, 26, 24]):  # backpack x3, handbag x1
        dwell.update(Detection(box, 1, cls), t0 + timedelta(seconds=i * 0.1))
    assert dwell.voted_class(1) == 24


# --- accuracy filters: per-class confidence + min bbox size ------------------
def test_item_passes_filters_confidence_and_size():
    # a comfortably large, high-conf suitcase passes
    ok = Detection((100, 100, 300, 260), 1, 28, conf=0.9)
    assert item_passes_filters(ok)

    # low confidence for its class → rejected (suitcase floor = CONF_MIN 0.40)
    low_conf = Detection((100, 100, 300, 260), 1, 28, conf=0.30)
    assert not item_passes_filters(low_conf)

    # tiny box (< MIN_BBOX_SIDE) → rejected even at high conf.
    # This is exactly the 39x50 "cell phone" false positive from real data.
    tiny = Detection((341, 53, 380, 103), 2, 67, conf=0.99)  # 39x50 px
    assert min(380 - 341, 103 - 53) < config.MIN_BBOX_SIDE
    assert not item_passes_filters(tiny)

    # cell phone needs the higher per-class floor (0.55): 0.45 fails though big
    weak_phone = Detection((100, 100, 200, 260), 3, 67, conf=0.45)
    assert not item_passes_filters(weak_phone)
    strong_phone = Detection((100, 100, 200, 260), 3, 67, conf=0.60)
    assert item_passes_filters(strong_phone)


# --- abandonment gate: object must have moved before it can fire -------------
def test_requires_movement_before_firing():
    dwell = DwellTracker(dwell_seconds=8.0, move_tol=25, require_movement=True)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    static = Detection((100, 100, 160, 180), 5, 28)

    # a never-moving object (background/fixture) stays still forever but must
    # NOT fire because it was never carried in.
    dwell.update(static, t0)
    dwell.update(static, t0 + timedelta(seconds=20))
    assert dwell.tracks[5].has_moved is False
    assert not dwell.should_fire(5, t0 + timedelta(seconds=20))


def test_moved_then_still_fires():
    dwell = DwellTracker(dwell_seconds=8.0, move_tol=25, require_movement=True)
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    # carried in (moves), then set down and held still
    dwell.update(Detection((100, 100, 160, 180), 5, 28), t0)
    dwell.update(Detection((300, 300, 360, 380), 5, 28), t0 + timedelta(seconds=1))
    assert dwell.tracks[5].has_moved is True
    # still from the settle point → fires after dwell threshold
    assert not dwell.should_fire(5, t0 + timedelta(seconds=5))
    assert dwell.should_fire(5, t0 + timedelta(seconds=1 + 8.1))


def test_owner_left_gating():
    # An object that never "moved" (YOLO locked on only after placement) still
    # fires if a person placed it — but ONLY after that owner has left (#3).
    dwell = DwellTracker(
        dwell_seconds=8.0, move_tol=25, require_movement=True,
        require_owner_left=True, owner_left_seconds=3.0, owner_radius=150,
    )
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    item = Detection((100, 100, 200, 260), 9, 24)
    person = Detection((150, 150, 220, 300), -1, config.PERSON_CLASS_ID)

    # placer standing next to the object
    dwell.update_frame([item], [person], t0, FakeFrame())
    assert dwell.tracks[9].has_moved is False
    assert dwell.tracks[9].person_seen is True

    # dwell satisfied but owner is STILL next to it → must not fire (it's theirs)
    dwell.update_frame([item], [person], t0 + timedelta(seconds=8.1), FakeFrame())
    assert dwell.owner_present(9, t0 + timedelta(seconds=8.1))
    assert not dwell.should_fire(9, t0 + timedelta(seconds=8.1))

    # owner walks away (no persons near) → after owner_left_seconds it fires
    dwell.update_frame([item], [], t0 + timedelta(seconds=9.0))
    assert not dwell.should_fire(9, t0 + timedelta(seconds=10.0))   # only 1s gone
    assert dwell.should_fire(9, t0 + timedelta(seconds=12.5))       # >3s gone → 🔔


# --- T4: emitted event matches schema, source=cctv, model_version correct ----
class FakeFrame:
    """Minimal ndarray-like supporting crop slicing and .copy()."""

    def __getitem__(self, _slices):
        return self

    def copy(self):
        return self


class MemPublisher:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


class FakeCamera:
    camera_id = "cam-01"
    zone = "fl2-zoneA"


def test_t4_event_schema(monkeypatch):
    # avoid touching disk / cv2 in save_image
    monkeypatch.setattr("edge_cctv.emitter.save_image", lambda img, d, n: f"{d}/{n}")
    pub = MemPublisher()
    em = Emitter(camera=FakeCamera(), embedder=None, publisher=pub)
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)

    ev = em.emit_event(FakeFrame(), (10, 20, 50, 80), track_id=7, cls=24, ts=ts)

    assert ev["schema_version"] == config.SCHEMA_VERSION
    # Contract pin: model_version must match config AND the expected default
    # (yolo detector). Bumping the model = a deliberate Event Contract change,
    # so this literal must be updated in lockstep with Process 2 + replay.
    assert ev["model_version"] == config.ACTIVE_MODEL_VERSION == "yolo26x_clip-vitb32"
    assert ev["source"] == "cctv"
    assert ev["detect_type"] == "abandoned_object"
    assert ev["object_class"] == "backpack"
    assert ev["bbox"] == [10, 20, 40, 60]  # x, y, w, h
    assert ev["capture_ts"] == ts.isoformat()
    assert pub.events == [ev]


def test_event_id_is_deterministic():
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    a = make_event_id("cam-01", 7, ts)
    b = make_event_id("cam-01", 7, ts)
    c = make_event_id("cam-01", 8, ts)
    assert a == b and a != c


# --- T6: same contract as replay (only `source` differs) ---------------------
def test_t6_contract_matches_replay_except_source():
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    kwargs = dict(
        camera_id="cam-01",
        zone="fl2-zoneA",
        track_id=7,
        cls=24,
        bbox_xyxy=(10, 20, 50, 80),
        ts=ts,
        crop_ref="crops/x.jpg",
        person_ref=None,
        embedding=[0.0] * 512,
    )
    cctv = build_event(**kwargs)

    # simulate the replay event: identical builder, source patched to "replay"
    replay = dict(cctv)
    replay["source"] = "replay"

    # every key/value identical except `source`
    assert set(cctv.keys()) == set(replay.keys())
    diffs = {k for k in cctv if cctv[k] != replay[k]}
    assert diffs == {"source"}
    assert len(cctv["embedding"]) == 512


# --- T7: person_ref null when no placer frame captured -----------------------
def test_t7_person_ref_null_when_no_capture(monkeypatch):
    monkeypatch.setattr("edge_cctv.emitter.save_image", lambda img, d, n: (None if img is None else f"{d}/{n}"))
    pub = MemPublisher()
    em = Emitter(camera=FakeCamera(), embedder=None, publisher=pub)
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)

    ev = em.emit_event(FakeFrame(), (10, 20, 50, 80), 7, 24, ts, person_frame=None)
    assert ev["person_ref"] is None


def test_person_capture_stores_frame_when_person_near():
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    item = Detection((100, 100, 140, 160), 7, 24)
    frame = FakeFrame()

    # person near → placer frame captured (for Process 3 face compare)
    dwell = DwellTracker(owner_radius=150)
    persons = find_person_detections(
        [Detection((110, 110, 130, 200), -1, config.PERSON_CLASS_ID)]
    )
    dwell.update_frame([item], persons, ts, frame)
    assert dwell.tracks[7].person_frame is not None
    assert dwell.tracks[7].person_seen is True

    # far-away person → not captured
    dwell2 = DwellTracker(owner_radius=150)
    far = find_person_detections([Detection((900, 900, 950, 1000), -1, 0)])
    dwell2.update_frame([item], far, ts, frame)
    assert dwell2.tracks[7].person_frame is None
    assert dwell2.tracks[7].person_seen is False

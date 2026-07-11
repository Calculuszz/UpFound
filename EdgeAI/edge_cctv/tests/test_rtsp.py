"""RTSP source tests — T1 (continuous frames), T2 (reconnect, no crash), T5 (latency).

Uses a fake cv2/VideoCapture so tests run without opencv or a real camera.
"""
from __future__ import annotations

import types

import pytest

from UpFound.EdgeAI.edge_cctv import rtsp_source


class FakeCap:
    """Simulates cv2.VideoCapture with a scripted sequence of read() results."""

    def __init__(self, script):
        # script: list of (ok, frame) tuples, consumed per read().
        self._script = list(script)
        self._i = 0
        self.props = {}
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, val):
        self.props[prop] = val
        return True

    def read(self):
        if self._i < len(self._script):
            r = self._script[self._i]
            self._i += 1
            return r
        return (True, "FRAME")  # steady stream afterwards

    def release(self):
        self.released = True


def make_fake_cv2(cap_factory):
    fake = types.SimpleNamespace()
    fake.CAP_FFMPEG = 0
    fake.CAP_PROP_BUFFERSIZE = 38
    fake.VideoCapture = lambda url, backend=0: cap_factory()
    return fake


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # reconnect() sleeps for backoff — skip it in tests.
    monkeypatch.setattr(rtsp_source.time, "sleep", lambda *_: None)


def test_t1_continuous_frames(monkeypatch):
    cv2 = make_fake_cv2(lambda: FakeCap([(True, "F1"), (True, "F2")]))
    monkeypatch.setattr(rtsp_source, "cv2", cv2)

    src = rtsp_source.RtspSource("rtsp://x", use_tcp=True)
    frames = [src.read_latest() for _ in range(5)]
    assert all(f is not None for f in frames)


def test_t1_forces_tcp_and_buffersize(monkeypatch):
    caps = []

    def factory():
        c = FakeCap([(True, "F")])
        caps.append(c)
        return c

    cv2 = make_fake_cv2(factory)
    monkeypatch.setattr(rtsp_source, "cv2", cv2)

    rtsp_source.RtspSource("rtsp://x", use_tcp=True, buffersize=1)
    import os

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"
    # BUFFERSIZE prop set to 1 (drop stale frames / no latency pileup — T5).
    assert caps[0].props[cv2.CAP_PROP_BUFFERSIZE] == 1


def test_t2_reconnect_on_read_failure_no_crash(monkeypatch):
    # First cap fails its read → reconnect swaps in a healthy cap → recovers.
    caps = iter([FakeCap([(False, None)]), FakeCap([(True, "F1"), (True, "F2")])])
    cv2 = make_fake_cv2(lambda: next(caps))
    monkeypatch.setattr(rtsp_source, "cv2", cv2)

    src = rtsp_source.RtspSource("rtsp://x")
    first = src.read_latest()   # bad cap → triggers reconnect
    assert first is None
    second = src.read_latest()  # healthy cap after reconnect
    assert second is not None   # did not crash, recovered


def test_t2_backoff_grows_then_caps(monkeypatch):
    cv2 = make_fake_cv2(lambda: FakeCap([(False, None)]))
    monkeypatch.setattr(rtsp_source, "cv2", cv2)

    src = rtsp_source.RtspSource("rtsp://x", max_backoff=8)
    backoffs = []
    for _ in range(6):
        backoffs.append(src._backoff)
        src.reconnect()
    # 1,2,4,8,8,8 — exponential then capped at max_backoff.
    assert backoffs[0] == 1
    assert max(backoffs) <= 8
    assert backoffs[-1] == 8

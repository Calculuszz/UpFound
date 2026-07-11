"""§3 RtspSource — resilient RTSP capture that never accumulates latency.

Two fixes over the naive `while cap.read()` preview code:
  1. Force TCP transport (Hikvision drops packets badly on UDP).
  2. A background reader thread that continuously reads and keeps ONLY the
     newest frame. The consumer always gets the freshest frame, so latency
     stays bounded even when inference is slower than the camera fps.

Why a thread? `CAP_PROP_BUFFERSIZE=1` is unreliable for RTSP+FFMPEG, and a
single-threaded `cap.read()` returns frames in FIFO order — if the consumer is
slower than the camera, frames pile up in the socket/ffmpeg buffer and you fall
progressively behind real time (growing delay). Draining in a dedicated thread
and overwriting the latest frame fixes this.

On read failure we reconnect with exponential backoff instead of breaking.
"""
from __future__ import annotations

import logging
import os
import threading
import time

try:
    import cv2
except ImportError:  # pragma: no cover - import guard for envs without opencv
    cv2 = None

log = logging.getLogger(__name__)


class RtspSource:
    def __init__(
        self,
        url: str,
        use_tcp: bool = True,
        max_backoff: int = 30,
        buffersize: int = 1,
        redacted_url: str | None = None,
        reader_thread: bool = False,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python (cv2) is required for RtspSource")
        # Must be set BEFORE VideoCapture is created for FFMPEG to pick it up.
        if use_tcp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.url = url
        self._log_url = redacted_url or url
        self.max_backoff = max_backoff
        self.buffersize = buffersize
        self.cap = None
        self._backoff = 1

        # threaded latest-frame plumbing
        self.reader_thread = reader_thread
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._open()

        if reader_thread:
            self._thread = threading.Thread(
                target=self._reader_loop, name="rtsp-reader", daemon=True
            )
            self._thread.start()

    # -- lifecycle ---------------------------------------------------------
    def _open(self) -> None:
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Best-effort small buffer; not reliably honored on RTSP/FFMPEG, which
        # is exactly why the reader thread exists.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
        if self.cap.isOpened():
            log.info("RTSP opened: %s", self._log_url)
        else:
            log.warning("RTSP open failed: %s", self._log_url)

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # -- background reader (drop-frame) ------------------------------------
    def _reader_loop(self) -> None:
        """Continuously read; keep only the newest frame (older ones dropped)."""
        while not self._stop.is_set():
            cap = self.cap
            if cap is None:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("read failed → reconnecting: %s", self._log_url)
                self.reconnect()
                continue
            self._backoff = 1
            with self._lock:
                self._latest = frame  # overwrite → we only ever keep the latest

    # -- reading -----------------------------------------------------------
    def read_latest(self):
        """Return the freshest frame, or None if none is currently available.

        Threaded mode: returns the newest frame captured by the reader thread
        (None if no new frame since the last call). Direct mode: reads one frame
        inline and reconnects on failure.
        """
        if self.reader_thread:
            with self._lock:
                frame = self._latest
                self._latest = None  # consume so we never reprocess the same frame
            return frame

        if self.cap is None:
            self.reconnect()
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            log.warning("read failed → reconnecting: %s", self._log_url)
            self.reconnect()
            return None
        # Reset backoff only after a genuinely good frame — a stream can open
        # yet immediately fail reads, which should keep growing the backoff.
        self._backoff = 1
        return frame

    def reconnect(self) -> None:
        """Exponential backoff reconnect: 1,2,4,...,max_backoff (never break)."""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:  # pragma: no cover - release best-effort
            pass
        backoff = self._backoff
        log.info("reconnect in %ss: %s", backoff, self._log_url)
        time.sleep(backoff)
        self._backoff = min(self._backoff * 2, self.max_backoff)
        self._open()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "RtspSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

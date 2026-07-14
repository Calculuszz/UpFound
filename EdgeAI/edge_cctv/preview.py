"""Debug preview — draws detections + dwell state.

Two output modes share the *same* drawing logic (`annotate_frame`):

  * ``PreviewWindow`` — live GUI window via ``cv2.imshow`` (dev/debug, needs a
    display). If no display is available the window disables itself gracefully.
  * ``PreviewWriter`` — writes annotated frames to an ``.mp4`` file via
    ``cv2.VideoWriter`` (headless-safe, no GUI calls). Handy on remote/SSH
    boxes with no screen — ``scp`` the file back and watch it locally.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime

from . import config
from .detector import centroid

log = logging.getLogger(__name__)


def _color(fired: bool):
    # BGR: red when abandoned/fired, green while still counting down.
    return (0, 0, 255) if fired else (0, 200, 0)


def _draw_owner_links(cv2, canvas, item_dets, persons, dwell) -> None:
    """Connect each item to nearby person(s) within the owner radius.

    The link is yellow when the person is the *nearest* (most likely placer)
    and dim yellow for other people also inside the radius. The distance in
    pixels is printed on the link — handy for tuning OWNER_RADIUS.
    """
    radius = getattr(dwell, "owner_radius", config.OWNER_RADIUS)
    for det in item_dets:
        ix, iy = centroid(det.bbox)
        ipt = (int(ix), int(iy))
        # people inside the radius, nearest first
        near = sorted(
            (
                (math.hypot(ix - px, iy - py), (int(px), int(py)))
                for p in (persons or [])
                for (px, py) in [centroid(p.bbox)]
                if math.hypot(ix - px, iy - py) <= radius
            ),
            key=lambda t: t[0],
        )
        for rank, (dist, ppt) in enumerate(near):
            bright = (0, 255, 255)          # nearest → bright yellow
            dim = (60, 170, 170)            # others → dim yellow
            color = bright if rank == 0 else dim
            thickness = 2 if rank == 0 else 1
            cv2.line(canvas, ipt, ppt, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, ppt, 4, color, -1)
            mid = ((ipt[0] + ppt[0]) // 2, (ipt[1] + ppt[1]) // 2)
            cv2.putText(
                canvas, f"{dist:0.0f}px", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
        if near:
            cv2.circle(canvas, ipt, 4, (0, 255, 255), -1)  # item end marker


def annotate_frame(cv2, frame, item_dets, dwell, now: datetime, persons=None):
    """Return an annotated *copy* of ``frame`` (does not mutate the input).

    Draws person boxes, owner links, item boxes and the dwell readout — the
    single source of truth for both the live window and the video writer.
    """
    canvas = frame.copy()

    # person boxes (thin, blue) — context for who might have placed an item
    for p in persons or []:
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 1)

    # owner links: draw a line from each item to every person within
    # OWNER_RADIUS (the same threshold the detector uses to decide a placer),
    # so it's obvious at a glance that someone is standing near the object.
    # Drawn first → stays under the boxes/labels.
    _draw_owner_links(cv2, canvas, item_dets, persons, dwell)

    # item boxes with dwell readout
    for det in item_dets:
        x1, y1, x2, y2 = det.bbox
        st = dwell.tracks.get(det.track_id)
        fired = bool(st and st.fired)
        dwell_s = dwell.dwell_seconds_of(det.track_id, now)
        color = _color(fired)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        voted = dwell.voted_class(det.track_id)
        cls_id = voted if voted is not None else det.cls
        cls_name = config.ITEM_CLASSES.get(cls_id, str(cls_id))
        status = "ABANDONED" if fired else f"{dwell_s:0.1f}/{dwell.dwell_seconds:0.0f}s"
        # Explain why it may not fire: placement gate not met, or owner still near.
        if not (st and (st.has_moved or st.person_seen)):
            gate = " static"
        elif dwell.owner_present(det.track_id, now):
            gate = " owner"
        else:
            gate = ""
        label = f"{cls_name} #{det.track_id} {det.conf:0.2f} {status}{gate}"
        cv2.putText(
            canvas, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )

    return canvas


class PreviewWindow:
    def __init__(self, window_name: str = "edge_cctv preview"):
        import cv2  # local import so headless envs that never preview don't need GUI libs

        self._cv2 = cv2
        self.window_name = window_name
        self.enabled = True

    def draw_and_show(self, frame, item_dets, dwell, now: datetime, persons=None) -> bool:
        """Annotate a copy of the frame and display it.

        Returns False if the user pressed 'q' (quit) or preview got disabled.
        """
        if not self.enabled:
            return True
        cv2 = self._cv2
        canvas = annotate_frame(cv2, frame, item_dets, dwell, now, persons)

        cv2.putText(
            canvas, "press 'q' to quit", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        try:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
        except Exception as e:  # no display / GUI backend missing
            log.warning("preview disabled (no display?): %r", e)
            self.enabled = False
            return True
        return key != ord("q")

    def close(self):
        if not self.enabled:
            return
        try:
            self._cv2.destroyAllWindows()
        except Exception:
            pass


class PreviewWriter:
    """Write annotated frames to a video file — headless-safe (no GUI calls).

    The writer opens lazily on the first frame so the output matches the actual
    drawn frame size. On ARM64 some codecs are missing from the OpenCV build, so
    we try a small list of codecs and, as a last resort, fall back to ``.avi``
    (XVID). The path that was actually opened is exposed as ``self.path`` and
    logged, so a silent/empty-file failure never goes unnoticed.
    """

    # (codec fourcc, file extension) tried in order.
    _MP4_CODECS = (("mp4v", ".mp4"), ("avc1", ".mp4"))
    _FALLBACK = ("XVID", ".avi")

    def __init__(self, path: str, fps: float = 25.0):
        import cv2  # local import — only needed when saving

        self._cv2 = cv2
        self.requested_path = path
        self.path: str | None = None
        self.fps = fps if fps and fps > 0 else 25.0
        self._writer = None
        self._frames_written = 0

    def _open(self, size):
        cv2 = self._cv2
        base, ext = os.path.splitext(self.requested_path)
        d = os.path.dirname(self.requested_path)
        if d:
            os.makedirs(d, exist_ok=True)

        # Prefer the extension the user asked for; if it's .mp4 (or unknown) try
        # the mp4 codecs first, then fall back to .avi/XVID.
        candidates = []
        if ext.lower() == ".avi":
            candidates.append((self._FALLBACK[0], self.requested_path))
        else:
            for fourcc, cext in self._MP4_CODECS:
                candidates.append((fourcc, base + cext))
        candidates.append((self._FALLBACK[0], base + self._FALLBACK[1]))

        for fourcc, out_path in candidates:
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*fourcc), self.fps, size
            )
            if writer.isOpened():
                self._writer = writer
                self.path = out_path
                log.info(
                    "preview writer: %s (codec=%s, %dx%d @ %.1ffps)",
                    out_path, fourcc, size[0], size[1], self.fps,
                )
                if out_path != self.requested_path:
                    log.warning(
                        "requested %s but wrote %s (codec fallback on this platform)",
                        self.requested_path, out_path,
                    )
                return
            writer.release()

        raise RuntimeError(
            f"cv2.VideoWriter could not open any codec for {self.requested_path} "
            f"(tried mp4v/avc1/XVID). No video preview will be written."
        )

    def write(self, frame, item_dets, dwell, now: datetime, persons=None) -> None:
        canvas = annotate_frame(self._cv2, frame, item_dets, dwell, now, persons)
        if self._writer is None:
            h, w = canvas.shape[:2]
            self._open((w, h))
        self._writer.write(canvas)
        self._frames_written += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info(
                "preview writer closed: %s (%d frames)",
                self.path, self._frames_written,
            )

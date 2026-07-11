"""Debug preview window — draws detections + dwell state and shows a live window.

Dev/debug only. Never used in headless production. If a display is not
available (e.g. server), the window creation fails gracefully and preview
disables itself after logging a warning.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime

from . import config
from .detector import centroid

log = logging.getLogger(__name__)


class PreviewWindow:
    def __init__(self, window_name: str = "edge_cctv preview"):
        import cv2  # local import so headless envs that never preview don't need GUI libs

        self._cv2 = cv2
        self.window_name = window_name
        self.enabled = True

    def _color(self, fired: bool):
        # BGR: red when abandoned/fired, green while still counting down.
        return (0, 0, 255) if fired else (0, 200, 0)

    def _draw_owner_links(self, canvas, item_dets, persons, dwell) -> None:
        """Connect each item to nearby person(s) within the owner radius.

        The link is yellow when the person is the *nearest* (most likely placer)
        and dim yellow for other people also inside the radius. The distance in
        pixels is printed on the link — handy for tuning OWNER_RADIUS.
        """
        cv2 = self._cv2
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

    def draw_and_show(self, frame, item_dets, dwell, now: datetime, persons=None) -> bool:
        """Annotate a copy of the frame and display it.

        Returns False if the user pressed 'q' (quit) or preview got disabled.
        """
        if not self.enabled:
            return True
        cv2 = self._cv2
        canvas = frame.copy()

        # person boxes (thin, blue) — context for who might have placed an item
        for p in persons or []:
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 1)

        # owner links: draw a line from each item to every person within
        # OWNER_RADIUS (the same threshold the detector uses to decide a placer),
        # so it's obvious at a glance that someone is standing near the object.
        # Drawn first → stays under the boxes/labels.
        self._draw_owner_links(canvas, item_dets, persons, dwell)

        # item boxes with dwell readout
        for det in item_dets:
            x1, y1, x2, y2 = det.bbox
            st = dwell.tracks.get(det.track_id)
            fired = bool(st and st.fired)
            dwell_s = dwell.dwell_seconds_of(det.track_id, now)
            color = self._color(fired)
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

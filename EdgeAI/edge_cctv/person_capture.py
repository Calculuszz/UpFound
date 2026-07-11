"""§4 Person capture helpers.

Design v3: the frame of the person who placed an object is kept for face
comparison in Process 3 — for face compare ONLY, never appearance/attribute
analysis (design decision).

The owner-proximity timing and the placer-frame snapshot now live in
DwellTracker.update_frame (see detector.py `_note_persons`), so this module
keeps only the small class-filter helper.
"""
from __future__ import annotations

import logging

from . import config
from .detector import Detection

log = logging.getLogger(__name__)


def find_person_detections(raw_detections) -> list[Detection]:
    """Filter a mixed detection list down to person-class detections."""
    return [d for d in raw_detections if d.cls == config.PERSON_CLASS_ID]

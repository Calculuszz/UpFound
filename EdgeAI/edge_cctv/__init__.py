"""UpFound Process 1 (Edge AI): CCTV stream → Event Contract.

Emits the same Event Contract as the replay producer (clip_to_events.py),
differing only by source="cctv".
"""

__all__ = [
    "config",
    "rtsp_source",
    "detector",
    "person_capture",
    "emitter",
]

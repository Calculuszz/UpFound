"""Entrypoint — wires RTSP/video → detect/track/dwell → person capture → emit.

Run modes:
    python -m edge_cctv.run                        # production RTSP (ch 101)
    python -m edge_cctv.run --dev                  # sub stream (ch 102)
    python -m edge_cctv.run --source video.mp4     # from video file (integration test)

timestamp for CCTV/RTSP uses wall-clock now(UTC) (§3).
timestamp for video-file mode uses frame_idx / fps (reproducible).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

from . import config
from .detector import DwellTracker, FrameSampler, YoloTracker
from .emitter import ClipEmbedder, Emitter
from .rtsp_source import RtspSource

log = logging.getLogger("edge_cctv.run")


class VideoSource:
    """Read frames from a video file (mp4) instead of RTSP.

    Useful for integration tests — deterministic, no camera needed, reproducible.
    Timestamps are derived from frame_idx / fps (not wall-clock).
    """

    def __init__(self, path: str):
        import cv2

        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_idx = 0
        log.info("VideoSource opened: %s (fps=%.1f)", path, self.fps)

    def read_latest(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._frame_idx += 1
        return frame

    @property
    def frame_ts_offset(self) -> float:
        """Seconds since start of video (for timestamp calculation)."""
        return self._frame_idx / self.fps

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def build_source(camera) -> RtspSource:
    return RtspSource(
        url=camera.rtsp_url(),
        use_tcp=(config.RTSP_TRANSPORT == "tcp"),
        max_backoff=config.RECONNECT_MAX_BACKOFF,
        redacted_url=camera.rtsp_url_redacted(),
        reader_thread=config.READER_THREAD,
    )


def run(
    source: RtspSource,
    camera,
    tracker: YoloTracker,
    emitter: Emitter,
    dwell: DwellTracker | None = None,
    sampler: FrameSampler | None = None,
    capture_person: bool = True,
    max_iters: int | None = None,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    heartbeat_every: int = 30,
    preview: bool = False,
    save_preview: str | None = None,
) -> None:
    """Main loop.

    Stops when any configured bound is hit; runs forever if all are None.
      max_iters    — total loop iterations (incl. reconnect/skip) — used by tests
      max_frames   — processed frames (after sampling) — handy for camera smoke tests
      max_seconds  — wall-clock duration
      preview      — draw detections + dwell and show a live window (dev/debug)
      save_preview — path to write annotated frames to a video file (headless-safe)
    """
    dwell = dwell or DwellTracker()
    sampler = sampler or FrameSampler()

    # Both preview modes need the same annotated frames; person capture drives
    # the owner links, so treat either as "we want persons this pass".
    want_preview = preview or bool(save_preview)

    window = None
    if preview:
        from .preview import PreviewWindow
        window = PreviewWindow()

    writer = None
    if save_preview:
        from .preview import PreviewWriter
        # We write one frame per *processed* frame, i.e. one of every
        # sampler.sample_every source frames — so the playback fps must be the
        # source fps divided by the sample rate, else the video plays sped up.
        src_fps = getattr(source, "fps", 25.0) or 25.0
        out_fps = max(1.0, src_fps / max(1, sampler.sample_every))
        writer = PreviewWriter(save_preview, fps=out_fps)

    it = 0
    processed = 0
    fired = 0
    start = time.monotonic()
    try:
        while max_iters is None or it < max_iters:
            it += 1
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                break

            frame = source.read_latest()
            if frame is None:
                if isinstance(source, VideoSource):
                    log.info("video ended")
                    break
                # threaded reader has no new frame yet (or reconnecting) — yield
                # briefly instead of busy-spinning.
                time.sleep(0.005)
                continue
            if sampler.should_skip():
                continue

            # Timestamp: wall-clock for RTSP, frame_idx/fps for video (reproducible).
            if isinstance(source, VideoSource):
                now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
                    seconds=source.frame_ts_offset
                )
            else:
                now = datetime.now(timezone.utc)

            # One YOLO pass. When we need persons (capture / preview) we get both
            # items and persons from the SAME inference instead of running twice.
            if capture_person or want_preview:
                item_dets, persons = tracker.track_items_and_persons(frame)
            else:
                item_dets = tracker.track_items(frame)
                persons = []
            processed += 1

            if heartbeat_every and processed % heartbeat_every == 0:
                log.info(
                    "heartbeat: processed=%d items_in_frame=%d tracks=%d fired=%d",
                    processed, len(item_dets), len(dwell.tracks), fired,
                )

            # One call: id-adoption + stillness + owner tracking + prune.
            dwell.update_frame(item_dets, persons, now, frame if capture_person else None)

            for det in item_dets:
                if dwell.should_fire(det.track_id, now):
                    st = dwell.tracks.get(det.track_id)
                    emitter.emit_event(
                        frame=frame,
                        bbox_xyxy=det.bbox,
                        track_id=det.track_id,
                        cls=dwell.voted_class(det.track_id),  # majority-vote class (#5)
                        ts=now,
                        person_frame=st.person_frame if st else None,
                    )
                    dwell.mark_fired(det.track_id)
                    fired += 1

            # Headless-safe video output: annotate + write the same frame.
            if writer is not None:
                writer.write(frame, item_dets, dwell, now, persons)

            if window is not None:
                if not window.draw_and_show(frame, item_dets, dwell, now, persons):
                    log.info("preview quit requested")
                    break

            if max_frames is not None and processed >= max_frames:
                break
    finally:
        # Release resources even on KeyboardInterrupt so the video is playable.
        if window is not None:
            window.close()
        if writer is not None:
            writer.close()

    log.info(
        "run finished: iters=%d processed=%d fired=%d elapsed=%.1fs",
        it, processed, fired, time.monotonic() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="UpFound Edge AI CCTV producer")
    parser.add_argument("--dev", action="store_true", help="use sub stream (ch 102)")
    parser.add_argument(
        "--no-embed", action="store_true", help="skip CLIP embedding (dev/preview)"
    )
    parser.add_argument(
        "--no-person", action="store_true", help="disable person-frame capture"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="stop after N seconds (smoke test); default runs forever",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="stop after N processed frames (smoke test)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="show a live window with detection boxes + dwell (dev/debug, needs a display)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="path to a video file (.mp4) to use instead of RTSP (integration test mode)",
    )
    parser.add_argument(
        "--save-preview", type=str, default=None, metavar="PATH",
        help="write annotated frames (boxes + dwell + owner links) to a video "
             "file instead of a GUI window — headless-safe (e.g. out/preview.mp4)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # One clear line up front: which detector/model/contract is active. Makes
    # accidental detector/model mismatches obvious in the logs.
    log.info(
        "detector=%s weights=%s imgsz=%d half=%s model_version=%s classes=%s",
        config.DETECTOR, config.YOLO_WEIGHTS, config.IMGSZ, config.HALF,
        config.ACTIVE_MODEL_VERSION, list(config.ITEM_CLASSES.values()),
    )

    camera = config.load_camera(use_dev_channel=args.dev)

    # Source: video file or RTSP camera
    if args.source:
        source = VideoSource(args.source)
        log.info("running from video: %s", args.source)
    else:
        log.info("starting edge_cctv for %s (%s)", camera.camera_id, camera.rtsp_url_redacted())
        source = build_source(camera)

    tracker = YoloTracker()
    embedder = None if args.no_embed else ClipEmbedder()
    emitter = Emitter(camera=camera, embedder=embedder)

    try:
        run(
            source, camera, tracker, emitter,
            capture_person=not args.no_person,
            max_frames=args.max_frames,
            max_seconds=args.duration,
            preview=args.preview,
            save_preview=args.save_preview,
        )
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
    finally:
        source.release()


if __name__ == "__main__":
    main()

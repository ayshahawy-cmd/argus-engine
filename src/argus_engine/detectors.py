"""Optional Ultralytics adapter. No weights or detector source are bundled."""

import math
import time

from PIL import Image

from .types import Detection, FramePacket


class YoloTracker:
    def __init__(
        self,
        source: str | int,
        *,
        model: str = "yolov8s.pt",
        tracker: str = "botsort.yaml",
        stride: int = 6,
        confidence: float = 0.25,
        max_seconds: float | None = None,
    ):
        if stride < 1:
            raise ValueError("stride must be positive")
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        self.source, self.model, self.tracker = source, model, tracker
        self.stride, self.confidence, self.max_seconds = stride, confidence, max_seconds

    def __iter__(self):
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        # Capture errors from OpenCV may otherwise print a credential-bearing RTSP URL.
        # These libraries use process-wide loggers; restore their state on exit.
        get_level = getattr(cv2, "getLogLevel", None)
        set_level = getattr(cv2, "setLogLevel", None)
        prior_cv_level = get_level() if callable(get_level) and callable(set_level) else None
        prior_yolo_disabled = LOGGER.disabled
        if prior_cv_level is not None:
            set_level(0)
        LOGGER.disabled = True
        capture = None
        try:
            model = YOLO(self.model)
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise ValueError("Unable to open video source")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else 30.0
            live = isinstance(self.source, int) or "://" in str(self.source)
            start, frame_index = time.monotonic(), 0
            while True:
                timestamp = time.monotonic() - start if live else frame_index / fps
                if self.max_seconds is not None and timestamp >= self.max_seconds:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                selected = frame_index % self.stride == 0
                frame_index += 1
                if not selected:
                    continue
                results = model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker,
                    classes=[0],
                    conf=self.confidence,
                    verbose=False,
                )
                boxes = results[0].boxes
                detections = ()
                if boxes is not None and boxes.id is not None:
                    detections = tuple(
                        Detection(int(tid), tuple(box), float(conf))
                        for tid, box, conf in zip(
                            boxes.id.int().tolist(),
                            boxes.xyxy.tolist(),
                            boxes.conf.tolist(),
                            strict=True,
                        )
                    )
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                yield FramePacket(timestamp, image, detections)
        finally:
            if capture is not None:
                capture.release()
            if prior_cv_level is not None:
                set_level(prior_cv_level)
            LOGGER.disabled = prior_yolo_disabled

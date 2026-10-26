import sys
from types import SimpleNamespace as NS

import pytest

from argus_engine.detectors import YoloTracker


def fake_libraries(monkeypatch, *, opened=True):
    class Tensor:
        def __init__(self, values):
            self.values = values

        def int(self):
            return self

        def tolist(self):
            return self.values

    class Capture:
        index = 0
        released = False

        def isOpened(self):
            return opened

        def get(self, _):
            return 10.0

        def read(self):
            self.index += 1
            return self.index <= 5, "fake-frame"

        def release(self):
            self.released = True

    capture, calls = Capture(), []
    levels = []
    cv = NS(
        VideoCapture=lambda _: capture,
        CAP_PROP_FPS=5,
        getLogLevel=lambda: 3,
        setLogLevel=levels.append,
        COLOR_BGR2RGB=4,
        cvtColor=lambda frame, _: frame,
    )
    logger = NS(disabled=False)

    class Model:
        def __init__(self, _):
            pass

        def track(self, frame, **kwargs):
            calls.append(kwargs)
            boxes = NS(id=Tensor([12]), xyxy=Tensor([[0, 0, 10, 20]]), conf=Tensor([0.8]))
            return [NS(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "cv2", cv)
    monkeypatch.setitem(sys.modules, "ultralytics", NS(YOLO=Model))
    monkeypatch.setitem(sys.modules, "ultralytics.utils", NS(LOGGER=logger))
    monkeypatch.setattr("argus_engine.detectors.Image.fromarray", lambda _: "pil-image")
    return capture, calls, levels, logger


def test_adapter_stride_timestamps_ids_and_cleanup(monkeypatch):
    capture, calls, levels, logger = fake_libraries(monkeypatch)
    frames = list(YoloTracker("video.mp4", stride=2))
    assert [frame.timestamp for frame in frames] == [0, 0.2, 0.4]
    assert frames[0].detections[0].track_id == 12
    assert all(call["persist"] for call in calls)
    assert capture.released
    assert levels == [0, 3]
    assert not logger.disabled


def test_adapter_stops_at_duration_limit(monkeypatch):
    fake_libraries(monkeypatch)
    frames = list(YoloTracker("video.mp4", stride=1, max_seconds=0.25))
    assert [frame.timestamp for frame in frames] == [0, 0.1, 0.2]


def test_source_open_failure_is_generic_and_releases_capture(monkeypatch):
    capture, _, levels, logger = fake_libraries(monkeypatch, opened=False)
    with pytest.raises(ValueError, match="Unable to open video source"):
        list(YoloTracker("rtsp://PRIVATE-CAMERA"))
    assert capture.released and levels[-1] == 3 and not logger.disabled


def test_adapter_supports_opencv_builds_without_log_level_api(monkeypatch):
    capture, _, _, logger = fake_libraries(monkeypatch)
    monkeypatch.delattr(sys.modules["cv2"], "getLogLevel")
    monkeypatch.delattr(sys.modules["cv2"], "setLogLevel")
    assert len(list(YoloTracker("video.mp4", stride=2))) == 3
    assert capture.released and not logger.disabled

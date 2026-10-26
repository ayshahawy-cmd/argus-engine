"""Synthetic demonstration. Detections and model answers are scripted, not measured."""

from PIL import Image, ImageDraw

from .config import EngineConfig
from .engine import Engine
from .types import Detection, FramePacket, VisionResponse


class ScriptedVision:
    def complete(self, prompt, images=(), *, max_tokens=300):
        if len(images) == 2:
            text = "SAME\nScripted demo: the two crops show the same blue outfit."
        elif images:
            text = "AREA: synthetic scene. PEOPLE: one blue rectangle. ACTIVITY: demo. FLAGS: none."
        else:
            text = "Scripted demo: two synthetic track IDs were joined into one estimated visit."
        return VisionResponse(text)


def synthetic_frames():
    for t in range(9):
        image = Image.new("RGB", (320, 240), (240, 242, 245))
        detections = ()
        if t <= 3 or t >= 5:
            tid = 41 if t <= 3 else 7  # IDs deliberately do not increase with time.
            x = 80 + t * 3
            box = (x, 40, x + 55, 180)
            ImageDraw.Draw(image).rectangle(box, fill=(30, 100, 220))
            detections = (Detection(tid, box),)
        yield FramePacket(float(t), image, detections)


def run_demo(output="output"):
    return Engine(
        ScriptedVision(),
        EngineConfig(scene_interval_seconds=3),
    ).run(synthetic_frames(), output)

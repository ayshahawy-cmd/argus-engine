"""Public configuration. Credentials deliberately have no config field."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    min_track_seconds: float = 2.0
    min_crop_height: int = 60
    reid_max_gap_seconds: float = 10.0
    reid_max_distance_pixels: float = 150.0
    color_similarity_threshold: float = 0.35
    scene_interval_seconds: float = 60.0
    first_scene_seconds: float = 2.0
    max_scene_frames: int = 60
    max_reid_calls: int = 50
    redact_heads: bool = True
    generate_summary: bool = True

    def __post_init__(self):
        for name in (
            "min_track_seconds",
            "reid_max_gap_seconds",
            "reid_max_distance_pixels",
            "first_scene_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.scene_interval_seconds) or self.scene_interval_seconds <= 0:
            raise ValueError("scene_interval_seconds must be finite and positive")
        if not math.isfinite(self.color_similarity_threshold) or not (
            -1 <= self.color_similarity_threshold <= 1
        ):
            raise ValueError("color_similarity_threshold must be between -1 and 1")
        for name in ("max_scene_frames", "max_reid_calls", "min_crop_height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("redact_heads", "generate_summary"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

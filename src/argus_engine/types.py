"""Small integration contracts; no detector or model SDK is required."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Detection:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class FramePacket:
    timestamp: float
    image: Image.Image
    detections: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class VisionResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class VisionProvider(Protocol):
    def complete(
        self,
        prompt: str,
        images: tuple[Path, ...] = (),
        *,
        max_tokens: int = 300,
    ) -> VisionResponse: ...


@dataclass
class Track:
    track_id: int
    first_seen: float
    last_seen: float
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    crop: str | None = None
    best_area: float = 0
    merged_into: int | None = None


@dataclass
class RunResult:
    run_id: str
    output_dir: Path
    raw_tracks: int
    kept_tracks: int
    estimated_visitors: int
    merges: int
    duration_seconds: float
    checks: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    visits: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        result = asdict(self)
        # Evidence paths are relative to the run directory; no camera URL or key is stored.
        result.pop("output_dir")
        return result

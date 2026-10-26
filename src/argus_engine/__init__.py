"""ARGUS Engine: local tracking with a pluggable visual reasoning layer."""

from .config import EngineConfig
from .engine import Engine
from .types import Detection, FramePacket, VisionResponse

__all__ = ["Detection", "Engine", "EngineConfig", "FramePacket", "VisionResponse"]
__version__ = "0.1.0"

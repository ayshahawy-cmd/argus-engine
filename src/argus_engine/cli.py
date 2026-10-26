"""Small command-line entry point. No cloud calls unless a provider is selected."""

import argparse
import json
import os
import sys
from pathlib import Path

from .config import EngineConfig
from .demo import run_demo
from .detectors import YoloTracker
from .engine import Engine
from .providers import make_provider


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARGUS Engine — YOLO sees. Your AI understands.")
    parser.add_argument("--version", action="version", version="argus-engine 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Synthetic demo; scripted tracks and AI, no network")
    demo.add_argument("--output", default="output")
    run = sub.add_parser("run", help="Track a video and optionally review screenshots with your AI")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Video path, camera index, or stream URL")
    source.add_argument(
        "--source-env",
        help="Read a private stream URL from an environment variable",
    )
    run.add_argument("--provider", choices=["none", "anthropic", "openai"], default="none")
    run.add_argument("--model", help="Vision model ID; alternatively use ARGUS_*_MODEL")
    run.add_argument("--yolo-model", default="yolov8s.pt")
    run.add_argument("--tracker", default="botsort.yaml")
    run.add_argument("--stride", type=int, default=6)
    run.add_argument("--confidence", type=float, default=0.25)
    run.add_argument("--max-seconds", type=float, help="Stop capture after this many seconds")
    run.add_argument("--config", type=Path, help="EngineConfig JSON; see examples/config.json")
    run.add_argument("--output", default="output")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            print("SYNTHETIC DEMO: scripted detections and model responses; no API calls.")
            result = run_demo(args.output)
        else:
            config = (
                EngineConfig(**json.loads(args.config.read_text()))
                if args.config
                else (EngineConfig())
            )
            source = os.environ.get(args.source_env) if args.source_env else args.source
            if not source:
                raise ValueError("Source environment variable is empty")
            source = int(source) if source.isdecimal() else source
            provider = make_provider(args.provider, args.model)
            frames = YoloTracker(
                source,
                model=args.yolo_model,
                tracker=args.tracker,
                stride=args.stride,
                confidence=args.confidence,
                max_seconds=args.max_seconds,
            )
            result = Engine(provider, config).run(frames, args.output)
        print(
            f"Tracks: {result.kept_tracks}; estimated visitors: {result.estimated_visitors}; "
            f"merges: {result.merges}; model calls: {result.usage['calls']}"
        )
        print(f"Results: {result.output_dir / 'result.json'}")
        if result.warnings:
            print(f"Model failures: {len(result.warnings)}; see result.json", file=sys.stderr)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # Never echo arbitrary SDK/OpenCV exceptions, configs, source URLs, or credentials.
        print(
            f"ARGUS stopped ({type(exc).__name__}). Check source, dependencies, model ID, "
            "environment, and configuration. No exception payload was logged.",
            file=sys.stderr,
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())

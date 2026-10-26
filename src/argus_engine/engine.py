"""The extracted two-layer tracking and screenshot reasoning pipeline."""

import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .config import EngineConfig
from .images import clamp_box, color_similarity, redact, save_jpeg
from .prompts import REID_PROMPT, SCENE_PROMPT, SUMMARY_PROMPT
from .types import FramePacket, RunResult, Track, VisionProvider


def find_switch_candidates(tracks: dict[int, Track], config: EngineConfig):
    """Gate crop comparisons by time and space, independently of numeric ID order."""
    pairs = []
    for old in tracks.values():
        for new in tracks.values():
            gap = new.first_seen - old.last_seen
            if old.track_id == new.track_id or not (0 < gap <= config.reid_max_gap_seconds):
                continue
            distance = math.dist(old.last_center, new.first_center)
            if distance <= config.reid_max_distance_pixels and old.crop and new.crop:
                pairs.append((old.track_id, new.track_id, gap, distance))
    return sorted(pairs, key=lambda p: (p[2], p[3], p[0], p[1]))


def resolve_root(tracks: dict[int, Track], track_id: int) -> int:
    seen = set()
    while tracks[track_id].merged_into is not None:
        if track_id in seen:
            raise ValueError("Track merge cycle")
        seen.add(track_id)
        track_id = tracks[track_id].merged_into
    return track_id


def parse_verdict(text: str) -> tuple[str, str]:
    lines = text.strip().splitlines()
    first = lines[0].strip().upper() if lines else ""
    verdict = first if first in {"SAME", "DIFFERENT", "UNSURE"} else "UNSURE"
    return verdict, " ".join(lines[1:])[:1000]


class Engine:
    """Consume tracked frames, save selected evidence, then apply optional AI review.

    Each invocation writes a separate run directory. No provider means no model
    calls. Custom providers can point at cloud services or local vision models.
    """

    def __init__(
        self,
        provider: VisionProvider | None = None,
        config: EngineConfig | None = None,
        *,
        reid_prompt: str = REID_PROMPT,
        scene_prompt: str = SCENE_PROMPT,
        summary_prompt: str = SUMMARY_PROMPT,
    ):
        self.provider = provider
        self.config = config or EngineConfig()
        self.reid_prompt = reid_prompt
        self.scene_prompt = scene_prompt
        self.summary_prompt = summary_prompt

    def run(self, frames: Iterable[FramePacket], output: str | Path = "output") -> RunResult:
        config = self.config
        run_id = uuid.uuid4().hex
        directory = Path(output) / run_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "crops").mkdir()
        (directory / "scenes").mkdir()
        tracks: dict[int, Track] = {}
        scenes = []
        next_scene = config.first_scene_seconds
        previous_time = -1.0
        duration = 0.0
        for packet in frames:
            t = packet.timestamp
            if not math.isfinite(t) or t < 0 or t < previous_time:
                raise ValueError("Frame timestamps must be finite, non-negative and ordered")
            previous_time = duration = t
            boxes = [d.box for d in packet.detections]
            image = redact(packet.image, boxes) if config.redact_heads else packet.image
            if t >= next_scene and len(scenes) < config.max_scene_frames:
                relative = f"scenes/{len(scenes):05d}.jpg"
                save_jpeg(image, directory / relative)
                scenes.append({"timestamp": t, "image": relative})
                next_scene = t + config.scene_interval_seconds
            for detection in packet.detections:
                box = clamp_box(detection.box, packet.image.size)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                tid = detection.track_id
                if not isinstance(tid, int) or isinstance(tid, bool) or tid < 0:
                    raise ValueError("Track IDs must be non-negative integers")
                if tid not in tracks:
                    tracks[tid] = Track(tid, t, t, center, center)
                track = tracks[tid]
                track.last_seen, track.last_center = t, center
                area = (x2 - x1) * (y2 - y1)
                if area > track.best_area and y2 - y1 >= config.min_crop_height:
                    track.crop = f"crops/{tid}.jpg"
                    save_jpeg(image.crop(box), directory / track.crop)
                    track.best_area = area

        kept = {
            tid: track
            for tid, track in tracks.items()
            if track.last_seen - track.first_seen >= config.min_track_seconds
        }
        result = RunResult(run_id, directory, len(tracks), len(kept), len(kept), 0, duration)
        result.usage = {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0}

        def ask(kind, prompt, images=(), max_tokens=300):
            result.usage["calls"] += 1
            try:
                reply = self.provider.complete(prompt, tuple(images), max_tokens=max_tokens)
                if not isinstance(reply.text, str) or not reply.text.strip():
                    raise ValueError("Empty model response")
                result.usage["input_tokens"] += reply.input_tokens
                result.usage["output_tokens"] += reply.output_tokens
                return reply.text.strip()
            except Exception:
                # SDK exceptions can contain credentials, request bodies or endpoint URLs.
                result.usage["failed_calls"] += 1
                result.warnings.append(f"{kind} request failed; no decision applied")
                return None

        if self.provider is not None:
            successors = set()
            reid_calls = 0
            for old_id, new_id, gap, distance in find_switch_candidates(kept, config):
                old, new = kept[old_id], kept[new_id]
                check = {
                    "old_id": old_id,
                    "new_id": new_id,
                    "gap_seconds": gap,
                    "distance_pixels": distance,
                }
                old_root, new_root = resolve_root(kept, old_id), resolve_root(kept, new_id)
                if old_id in successors or new.merged_into is not None or old_root == new_root:
                    check["verdict"] = "SKIPPED_CONFLICT"
                elif self._overlap(kept, old_root, new_root):
                    check["verdict"] = "SKIPPED_OVERLAP"
                elif reid_calls >= config.max_reid_calls:
                    check["verdict"] = "SKIPPED_BUDGET"
                elif color_similarity(directory / old.crop, directory / new.crop) < (
                    config.color_similarity_threshold
                ):
                    check["verdict"] = "SKIPPED_COLOR"
                else:
                    reid_calls += 1
                    prompt = (
                        self.reid_prompt + f"\nGap: {gap:.2f} seconds. "
                        f"Distance: {distance:.1f} pixels. Images decide."
                    )
                    text = ask("reid", prompt, [directory / old.crop, directory / new.crop], 200)
                    verdict, reason = parse_verdict(text or "")
                    check.update(verdict=verdict, reason=reason)
                    if text is None:
                        check["verdict"] = "ERROR"
                    if verdict == "SAME":
                        new.merged_into = old_id
                        successors.add(old_id)
                        result.merges += 1
                result.checks.append(check)

            for scene in scenes:
                text = ask("scene", self.scene_prompt, [directory / scene["image"]])
                result.observations.append(
                    {
                        **scene,
                        "observation": text,
                        "status": ("ok" if text is not None else "error"),
                    }
                )
        else:
            result.observations = [
                {**scene, "observation": None, "status": "not_requested"} for scene in scenes
            ]

        groups = {}
        for tid, track in kept.items():
            root = resolve_root(kept, tid)
            group = groups.setdefault(
                root,
                {"track_ids": [], "first_seen": track.first_seen, "last_seen": track.last_seen},
            )
            group["track_ids"].append(tid)
            group["first_seen"] = min(group["first_seen"], track.first_seen)
            group["last_seen"] = max(group["last_seen"], track.last_seen)
        result.visits = [
            {"person_id": root, **group, "span_seconds": group["last_seen"] - group["first_seen"]}
            for root, group in sorted(groups.items())
        ]
        result.estimated_visitors = len(groups)
        if self.provider is not None and config.generate_summary:
            evidence = {
                "estimated_visitors": len(groups),
                "duration_seconds": duration,
                "visits": result.visits,
                "observations": result.observations,
            }
            result.summary = (
                ask("summary", self.summary_prompt + "\n" + json.dumps(evidence), max_tokens=400)
                or ""
            )

        (directory / "tracks.json").write_text(
            json.dumps([asdict(t) for t in kept.values()], indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _overlap(tracks, first_root, second_root):
        first = [t for tid, t in tracks.items() if resolve_root(tracks, tid) == first_root]
        second = [t for tid, t in tracks.items() if resolve_root(tracks, tid) == second_root]
        return any(
            max(a.first_seen, b.first_seen) <= min(a.last_seen, b.last_seen)
            for a in first
            for b in second
        )

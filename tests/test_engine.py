import json

import pytest
from PIL import Image

from argus_engine import Detection, Engine, EngineConfig, FramePacket, VisionResponse
from argus_engine.demo import synthetic_frames
from argus_engine.engine import parse_verdict


class Provider:
    def __init__(self, text="SAME\nMatching blue jacket.", error=False):
        self.text, self.error, self.calls = text, error, []

    def complete(self, prompt, images=(), *, max_tokens=300):
        self.calls.append((prompt, images))
        if self.error:
            raise RuntimeError("PRIVATE_CREDENTIAL_MUST_NOT_APPEAR")
        return VisionResponse(self.text, 10, 5)


def config(**overrides):
    return EngineConfig(**{"max_scene_frames": 0, "generate_summary": False, **overrides})


def test_ai_can_join_decreasing_track_ids_and_persist_evidence(tmp_path):
    provider = Provider()
    result = Engine(provider, config()).run(synthetic_frames(), tmp_path)
    assert (result.raw_tracks, result.kept_tracks, result.estimated_visitors, result.merges) == (
        2,
        2,
        1,
        1,
    )
    assert result.visits[0]["track_ids"] == [41, 7]
    assert result.visits[0]["span_seconds"] == 8
    assert result.usage == {"calls": 1, "failed_calls": 0, "input_tokens": 10, "output_tokens": 5}
    saved = json.loads((result.output_dir / "result.json").read_text())
    assert saved["checks"][0]["verdict"] == "SAME"
    assert "output_dir" not in saved
    assert all(path.exists() for path in provider.calls[0][1])


@pytest.mark.parametrize(
    "reply",
    [
        "UNSURE\nBlurred",
        "DIFFERENT\nDifferent coat",
        "NOT SAME",
        "They are SAME",
        "SAME?",
        "",
        "```\nSAME\n```",
    ],
)
def test_only_exact_same_can_merge(tmp_path, reply):
    result = Engine(Provider(reply), config()).run(synthetic_frames(), tmp_path)
    assert result.estimated_visitors == 2
    assert result.merges == 0


def test_error_preserves_counts_and_does_not_log_private_exception(tmp_path):
    result = Engine(Provider(error=True), config()).run(synthetic_frames(), tmp_path)
    assert result.estimated_visitors == 2
    assert result.usage["failed_calls"] == 1
    assert "PRIVATE_CREDENTIAL" not in (result.output_dir / "result.json").read_text()
    assert result.checks[0]["verdict"] == "ERROR"


def test_no_provider_needs_no_credentials_and_keeps_tracks_separate(tmp_path):
    result = Engine(config=config()).run(synthetic_frames(), tmp_path)
    assert result.estimated_visitors == 2
    assert result.usage["calls"] == 0


def test_call_budget_is_enforced_before_model_invocation(tmp_path):
    provider = Provider()
    result = Engine(provider, config(max_reid_calls=0)).run(synthetic_frames(), tmp_path)
    assert not provider.calls
    assert result.checks[0]["verdict"] == "SKIPPED_BUDGET"


def packets(segments):
    image = Image.new("RGB", (200, 200), "blue")
    for t in range(15):
        detections = tuple(
            Detection(tid, (10, 10, 80, 150)) for tid, start, end in segments if start <= t <= end
        )
        yield FramePacket(float(t), image, detections)


def test_one_old_track_cannot_merge_into_two_concurrent_tracks(tmp_path):
    result = Engine(Provider(), config()).run(packets([(1, 0, 2), (2, 4, 6), (3, 4, 6)]), tmp_path)
    assert result.estimated_visitors == 2
    assert result.merges == 1


def test_merge_chains_resolve_to_one_visit(tmp_path):
    result = Engine(Provider(), config()).run(packets([(9, 0, 2), (4, 4, 6), (1, 8, 10)]), tmp_path)
    assert result.estimated_visitors == 1
    assert result.merges == 2


def test_overlapping_tracks_are_not_candidates(tmp_path):
    provider = Provider()
    result = Engine(provider, config()).run(packets([(1, 0, 6), (2, 4, 8)]), tmp_path)
    assert result.merges == 0
    assert not provider.calls


def test_short_ghost_tracks_are_filtered(tmp_path):
    result = Engine(config=config()).run(packets([(1, 0, 0), (2, 4, 6)]), tmp_path)
    assert (result.raw_tracks, result.kept_tracks, result.estimated_visitors) == (2, 1, 1)


def test_scene_sampling_and_summary_have_a_known_call_bound(tmp_path):
    provider = Provider()
    result = Engine(
        provider, config(max_scene_frames=2, scene_interval_seconds=2, generate_summary=True)
    ).run(synthetic_frames(), tmp_path)
    assert len(result.observations) == 2
    assert result.usage["calls"] == 4  # one pair, two scenes, one summary
    assert result.summary


def test_color_prefilter_avoids_a_paid_call(tmp_path):
    frames = list(packets([(1, 0, 2), (2, 4, 6)]))
    for index in range(4, len(frames)):
        frames[index] = FramePacket(
            frames[index].timestamp, Image.new("RGB", (200, 200), "red"), frames[index].detections
        )
    provider = Provider()
    result = Engine(provider, config()).run(frames, tmp_path)
    assert not provider.calls
    assert result.checks[0]["verdict"] == "SKIPPED_COLOR"


def test_runs_use_separate_evidence_directories(tmp_path):
    engine = Engine(config=config())
    first = engine.run(synthetic_frames(), tmp_path)
    second = engine.run(synthetic_frames(), tmp_path)
    assert first.output_dir != second.output_dir
    assert (first.output_dir / "result.json").exists()


@pytest.mark.parametrize("timestamp", [-1, float("nan"), float("inf")])
def test_bad_timestamps_are_rejected(tmp_path, timestamp):
    with pytest.raises(ValueError, match="timestamps"):
        Engine().run([FramePacket(timestamp, Image.new("RGB", (10, 10)))], tmp_path)


def test_backwards_time_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="timestamps"):
        Engine().run(reversed(list(synthetic_frames())), tmp_path)


def test_parse_verdict_allows_case_but_not_substrings():
    assert parse_verdict("same\nBlue coat") == ("SAME", "Blue coat")
    assert parse_verdict("NOT SAME")[0] == "UNSURE"

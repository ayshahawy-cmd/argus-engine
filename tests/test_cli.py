import pytest

from argus_engine.cli import main
from argus_engine.config import EngineConfig


def test_demo_cli_works_without_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["demo", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC DEMO" in output
    assert "estimated visitors: 1" in output


def test_cli_does_not_echo_private_config_values(tmp_path, capsys):
    path = tmp_path / "config.local.json"
    path.write_text('{"PRIVATE_SECRET_MUST_NOT_APPEAR": "sensitive"}')
    assert main(["run", "--source", "unused.mp4", "--config", str(path)]) == 1
    assert "PRIVATE_SECRET" not in capsys.readouterr().err


def test_empty_source_environment_fails_without_reading_camera(monkeypatch, capsys):
    monkeypatch.delenv("PRIVATE_CAMERA", raising=False)
    assert main(["run", "--source-env", "PRIVATE_CAMERA"]) == 1
    assert "ARGUS stopped" in capsys.readouterr().err


@pytest.mark.parametrize(
    "values",
    [
        {"scene_interval_seconds": 0},
        {"reid_max_gap_seconds": -1},
        {"min_track_seconds": float("nan")},
        {"max_reid_calls": -1},
        {"color_similarity_threshold": 2},
        {"max_scene_frames": 1.5},
        {"redact_heads": "false"},
    ],
)
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        EngineConfig(**values)

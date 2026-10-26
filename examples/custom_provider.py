"""Run with: python examples/custom_provider.py (after pip install -e .)."""

from argus_engine import Engine, EngineConfig, VisionResponse
from argus_engine.demo import synthetic_frames


class MyVisionModel:
    def complete(self, prompt, images=(), *, max_tokens=300):
        # Call your own vision model here. images contains local JPEG Paths.
        # This placeholder makes no network calls and never approves a merge.
        return VisionResponse("UNSURE\nConnect your model to evaluate these images.")


if __name__ == "__main__":
    result = Engine(MyVisionModel(), EngineConfig(generate_summary=False)).run(synthetic_frames())
    print(result.output_dir / "result.json")

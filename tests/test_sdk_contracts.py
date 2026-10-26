"""Exercise real SDK serialization through local mock transports. No network."""

import json

import pytest
from PIL import Image

from argus_engine.providers import AnthropicProvider, OpenAICompatibleProvider

httpx = pytest.importorskip("httpx")


def test_real_anthropic_sdk_accepts_request_and_response(tmp_path):
    anthropic = pytest.importorskip("anthropic")
    path = tmp_path / "crop.jpg"
    Image.new("RGB", (10, 10)).save(path)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "SAME\nBlue jacket."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 5},
            },
        )

    with anthropic.Anthropic(
        api_key="test-only",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as sdk:
        response = AnthropicProvider("test-model", client=sdk).complete("Compare", (path, path))
    assert response.text.startswith("SAME")
    assert response.input_tokens == 12
    assert len(requests[0]["messages"][0]["content"]) == 3


def test_real_openai_sdk_accepts_request_and_response(tmp_path):
    openai = pytest.importorskip("openai")
    path = tmp_path / "scene.jpg"
    Image.new("RGB", (10, 10)).save(path)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "AREA: synthetic scene."},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            },
        )

    with openai.OpenAI(
        api_key="test-only",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as sdk:
        response = OpenAICompatibleProvider("test-model", client=sdk).complete("Describe", (path,))
    assert response.text == "AREA: synthetic scene."
    assert requests[0]["max_completion_tokens"] == 300

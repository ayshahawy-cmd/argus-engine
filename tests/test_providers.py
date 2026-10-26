from types import SimpleNamespace as NS

from PIL import Image

from argus_engine.providers import AnthropicProvider, OpenAICompatibleProvider


def test_anthropic_adapter_sends_selected_jpegs_and_tracks_tokens(tmp_path):
    path = tmp_path / "crop.jpg"
    Image.new("RGB", (10, 10)).save(path)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(
            content=[NS(type="text", text="SAME")], usage=NS(input_tokens=12, output_tokens=3)
        )

    provider = AnthropicProvider("test-model", client=NS(messages=NS(create=create)))
    reply = provider.complete("Compare", (path, path), max_tokens=100)
    assert (reply.text, reply.input_tokens) == ("SAME", 12)
    content = calls[0]["messages"][0]["content"]
    assert len(content) == 3
    assert content[1]["source"]["media_type"] == "image/jpeg"
    assert calls[0]["max_tokens"] == 100


def test_compatible_adapter_sends_data_urls(tmp_path):
    path = tmp_path / "scene.jpg"
    Image.new("RGB", (10, 10)).save(path)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(
            choices=[NS(message=NS(content="Scene"))],
            usage=NS(prompt_tokens=5, completion_tokens=2),
        )

    client = NS(chat=NS(completions=NS(create=create)))
    reply = OpenAICompatibleProvider("test-model", client=client).complete("Describe", (path,))
    url = calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert (reply.text, reply.output_tokens) == ("Scene", 2)

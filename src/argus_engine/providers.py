"""Bring your own model. SDKs are imported only when their adapter is selected."""

import base64
import os
from pathlib import Path

from .types import VisionResponse


def _encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class AnthropicProvider:
    def __init__(self, model: str, *, client=None):
        if not model.strip():
            raise ValueError("A vision-capable model ID is required")
        self.model = model
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(timeout=60.0, max_retries=0)
        self.client = client

    def complete(self, prompt, images=(), *, max_tokens=300):
        content = [{"type": "text", "text": prompt}]
        content += [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _encoded(path)},
            }
            for path in images
        ]
        reply = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = "\n".join(block.text for block in reply.content if block.type == "text")
        return VisionResponse(text, reply.usage.input_tokens, reply.usage.output_tokens)


class OpenAICompatibleProvider:
    """Chat Completions adapter for a vision-capable OpenAI-compatible endpoint."""

    def __init__(self, model: str, *, client=None):
        if not model.strip():
            raise ValueError("A vision-capable model ID is required")
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=60.0, max_retries=0)
        self.client = client

    def complete(self, prompt, images=(), *, max_tokens=300):
        content = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encoded(path)}"}}
            for path in images
        ]
        reply = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        usage = reply.usage
        return VisionResponse(
            reply.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


def make_provider(name: str, model: str | None):
    if name == "none":
        return None
    env = "ARGUS_ANTHROPIC_MODEL" if name == "anthropic" else "ARGUS_OPENAI_MODEL"
    model = model or os.environ.get(env)
    if not model:
        raise ValueError(f"Pass --model or set {env}")
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "openai":
        return OpenAICompatibleProvider(model)
    raise ValueError("Unknown provider")

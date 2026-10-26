"""Crop selection, coarse head redaction, and a cheap color prefilter."""

import math
from pathlib import Path

from PIL import Image, ImageDraw


def clamp_box(box, size):
    width, height = size
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(width, int(x1))), max(0, min(height, int(y1)))
    x2, y2 = max(0, min(width, int(x2))), max(0, min(height, int(y2)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def redact(image, boxes):
    """Cover the upper 35% of each detected person box. This is a heuristic."""
    safe = image.convert("RGB").copy()
    draw = ImageDraw.Draw(safe)
    for box in boxes:
        bounded = clamp_box(box, safe.size)
        if bounded:
            x1, y1, x2, y2 = bounded
            bottom = min(y2, y1 + max(1, int((y2 - y1) * 0.35)))
            draw.rectangle((x1, y1, x2 - 1, bottom - 1), fill=(0, 0, 0))
    return safe


def save_jpeg(image, path: Path):
    image = image.convert("RGB").copy()
    image.thumbnail((1280, 1280))
    # A fresh image omits original EXIF and other source metadata.
    clean = Image.new("RGB", image.size)
    clean.paste(image)
    clean.save(path, "JPEG", quality=85)


def color_similarity(first: Path, second: Path) -> float:
    def histogram(path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize((64, 64))
            bins = [0] * 512
            pixels = image.load()
            for x in range(64):
                for y in range(64):
                    r, g, b = pixels[x, y]
                    bins[(r // 32) * 64 + (g // 32) * 8 + b // 32] += 1
        mean = sum(bins) / len(bins)
        return [v - mean for v in bins]

    a, b = histogram(first), histogram(second)
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denominator if denominator else 0.0

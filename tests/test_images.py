from PIL import Image

from argus_engine.images import clamp_box, redact, save_jpeg


def test_redaction_covers_heads_in_scene_and_crop_without_mutating_original():
    original = Image.new("RGB", (100, 100), "white")
    safe = redact(original, [(20, 10, 80, 90)])
    assert safe.getpixel((50, 15)) == (0, 0, 0)
    assert safe.crop((20, 10, 80, 90)).getpixel((30, 5)) == (0, 0, 0)
    assert safe.getpixel((50, 80)) == (255, 255, 255)
    assert original.getpixel((50, 15)) == (255, 255, 255)


def test_boxes_are_clamped_and_empty_or_nonfinite_boxes_rejected():
    assert clamp_box((-10, -20, 200, 200), (100, 100)) == (0, 0, 100, 100)
    assert clamp_box((20, 20, 10, 10), (100, 100)) is None
    assert clamp_box((float("nan"), 0, 10, 10), (100, 100)) is None


def test_jpeg_is_bounded_and_omits_source_metadata(tmp_path):
    image = Image.new("RGB", (2000, 1000), "blue")
    image.getexif()[270] = "PRIVATE CAMERA LOCATION"
    path = tmp_path / "image.jpg"
    save_jpeg(image, path)
    with Image.open(path) as saved:
        assert saved.size == (1280, 640)
        assert not saved.getexif()

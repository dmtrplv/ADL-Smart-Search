from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "adl_smart_search" / "assets"


def test_app_icon_png_is_square_512():
    with Image.open(ASSETS / "adl_search.png") as im:
        assert im.size == (512, 512)
        assert im.mode == "RGBA"
        # круглая подложка: углы прозрачные
        assert im.getpixel((0, 0))[3] == 0


def test_app_icon_ico_has_windows_sizes():
    with Image.open(ASSETS / "adl_search.ico") as im:
        sizes = set(im.info.get("sizes") or [])
        assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= sizes


def test_app_icon_svg_exists():
    svg = (ASSETS / "adl_search.svg").read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "<circle" in svg
    assert 'fill="#F5B301"' in svg or 'stroke="#F5B301"' in svg

"""Генератор иконки приложения: буква A в лупе.

Запуск: .venv\\Scripts\\python.exe tools\\make_icon.py
Создаёт SVG-исходник, PNG для вкладки браузера и ICO для exe.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "adl_smart_search" / "assets"

BACKGROUND = "#1B2430"
AMBER = "#F5B301"
WHITE = "#FFFFFF"

BASE = 1024
SCALE = 2
SIZE = BASE * SCALE


def _s(value: float) -> int:
    return int(round(value * SCALE))


def _polygon(draw: ImageDraw.ImageDraw, points, fill: str) -> None:
    draw.polygon([(_s(x), _s(y)) for x, y in points], fill=fill)


def _rounded_line(draw: ImageDraw.ImageDraw, start, end, width: int, fill: str) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line([(_s(x1), _s(y1)), (_s(x2), _s(y2))], fill=fill, width=_s(width))
    r = width / 2
    for x, y in (start, end):
        draw.ellipse(
            [(_s(x - r), _s(y - r)), (_s(x + r), _s(y + r))],
            fill=fill,
        )


def make_master() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # круглая подложка, как у стандартной иконки
    draw.ellipse([0, 0, SIZE - 1, SIZE - 1], fill=BACKGROUND)

    # ручка лупы
    _rounded_line(draw, (655, 655), (815, 815), 82, AMBER)

    # кольцо лупы
    ring = 64
    draw.ellipse(
        [(_s(440 - 265), _s(440 - 265)), (_s(440 + 265), _s(440 + 265))],
        outline=AMBER,
        width=_s(ring),
    )

    # буква A (белая, с вырезом внутри), по центру линзы (440, 440)
    outer = [
        (440, 270),
        (580, 610),
        (515, 610),
        (490, 540),
        (390, 540),
        (365, 610),
        (300, 610),
    ]
    inner = [(440, 360), (475, 470), (405, 470)]
    _polygon(draw, outer, WHITE)
    _polygon(draw, inner, BACKGROUND)

    return img.resize((BASE, BASE), Image.Resampling.LANCZOS)


def make_svg() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <circle cx="512" cy="512" r="512" fill="{BACKGROUND}"/>
  <line x1="655" y1="655" x2="815" y2="815" stroke="{AMBER}" stroke-width="82" stroke-linecap="round"/>
  <circle cx="440" cy="440" r="265" fill="none" stroke="{AMBER}" stroke-width="64"/>
  <path fill="{WHITE}" d="M440 270 L580 610 L515 610 L490 540 L390 540 L365 610 L300 610 Z"/>
  <path fill="{BACKGROUND}" d="M440 360 L475 470 L405 470 Z"/>
</svg>
"""


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    master = make_master()

    master.resize((512, 512), Image.Resampling.LANCZOS).save(
        ASSETS / "adl_search.png", format="PNG"
    )
    master.save(
        ASSETS / "adl_search.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    (ASSETS / "adl_search.svg").write_text(make_svg(), encoding="utf-8")
    print(f"written: {ASSETS}")


if __name__ == "__main__":
    main()

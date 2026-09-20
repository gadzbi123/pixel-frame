#!/usr/bin/env python3
"""Build an eight-FPS 1080p animated WebP from a four-panel keyframe sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def build(theme: str, steps: int = 6) -> Path:
    assets = Path(__file__).with_name("assets")
    source = assets / f"{theme}-animation-sheet.png"
    output = assets / f"{theme}-animated.webp"
    sheet = Image.open(source).convert("RGB")
    width, height = sheet.size
    half_width, half_height = width // 2, height // 2
    boxes = (
        (0, 0, half_width, half_height),
        (half_width, 0, width, half_height),
        (0, half_height, half_width, height),
        (half_width, half_height, width, height),
    )
    resampling = getattr(Image, "Resampling", Image)
    keyframes = [
        sheet.crop(box).resize((1920, 1080), resampling.LANCZOS)
        for box in boxes
    ]
    frames = []
    for index, current in enumerate(keyframes):
        following = keyframes[(index + 1) % len(keyframes)]
        for step in range(steps):
            amount = step / steps
            amount = amount * amount * (3 - 2 * amount)
            frames.append(Image.blend(current, following, amount))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=125,
        loop=0,
        quality=84,
        method=3,
    )
    print(f"{theme}: {len(frames)} frames -> {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("theme")
    parser.add_argument("--steps", type=int, default=6)
    arguments = parser.parse_args()
    build(arguments.theme, arguments.steps)

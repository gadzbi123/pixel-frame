#!/usr/bin/env python3
"""Build a scene's seamless 8 FPS 1080p loop, and its night-glow mask, from its keyframe.

The keyframe, assets/<scene>-keyframe.png, is the painting at 1920x1080; make
it with upscale_keyframes.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scene_motion import GLOW, glow_mask, scene_frames

ASSETS = Path(__file__).with_name("assets")
FPS = 8


def keyframe(theme: str) -> np.ndarray:
    base = np.array(Image.open(ASSETS/f"{theme}-keyframe.png").convert("RGB"))
    if base.shape[:2] != (1080, 1920):
        raise ValueError(f"{theme} keyframe must be 1920x1080, not {base.shape[1]}x{base.shape[0]}")
    return base


def write_glow(theme: str, base: np.ndarray) -> None:
    """The mask of windows and lamps that the player keeps lit at night."""
    if theme in GLOW:
        Image.fromarray(glow_mask(base, theme)).save(ASSETS/f"{theme}-glow.png", optimize=True)


def build(theme: str, beats: int = 4) -> Path:
    """A loop of `beats` six-second beats; the default 4 gives 24 seconds."""
    if beats < 1:
        raise ValueError("beats must be at least one")
    cv2.setNumThreads(4)
    base = keyframe(theme)
    write_glow(theme, base)
    frames = list(scene_frames(base, theme, 48*beats, beats=beats))
    output = ASSETS/f"{theme}-animated.webp"
    temporary = output.with_suffix(".building.webp")
    frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=1000//FPS,
                   loop=0, quality=84, method=3)
    temporary.replace(output)
    print(f"{theme}: {len(frames)} frames -> {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("themes", nargs="+")
    parser.add_argument("--beats", type=int, default=4, help="six-second beats per loop (default: 4)")
    parser.add_argument("--glow-only", action="store_true", help="only rewrite the night-glow masks")
    arguments = parser.parse_args()
    for theme in arguments.themes:
        if arguments.glow_only:
            write_glow(theme, keyframe(theme))
        else:
            build(theme, arguments.beats)

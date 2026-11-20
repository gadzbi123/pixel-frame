#!/usr/bin/env python3
"""Painted, animated landscapes for a Linux framebuffer picture frame.

Show the playlist on the display (the systemd service runs this at boot):
    python3 pixel_frame.py

Export a still or a shareable GIF, lit for a time of day:
    python3 pixel_frame.py --scene pond --at 23:30 --export pond.png
    python3 pixel_frame.py --scene pond --at 23:30 --export-gif pond.gif
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import fcntl
import math
import mmap
import os
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from sky import PRESETS, Moon, Sky, Weather

ROOT = Path(__file__).resolve().parent
PICTURES = ROOT / "pictures"  # every scene the frame shows, with its glow mask
SCENES = ("woodland", "meadow", "pond", "autumn", "winter",
          "christmas", "tornado", "tsunami", "balloons", "sakura")
# Months in which a scene joins the playlist. Unlisted scenes play all year.
SEASONS = {"sakura": {3, 4, 5}, "autumn": {9, 10, 11},
           "winter": {12, 1, 2}, "christmas": {12}}
# Scenes that are night scenes already. They are shown as painted after dark:
# moonlight would put out their lamplight.
PAINTED_AT_NIGHT = {"christmas"}
# Scenes with weather of their own, or half under water: they keep it whatever the sky outside does.
OWN_WEATHER = {"winter", "christmas", "tornado", "tsunami", "reef"}
MEDIA = {".webp", ".gif", ".png", ".jpg", ".jpeg"}
SCREEN = (1920, 1080)
SECONDS_PER_SCENE = 720
MAX_FRAMES = 480  # Longer drop-in animations are sampled to bound memory.

FBIOGET_VSCREENINFO, FBIOBLANK, FB_BLANK_UNBLANK = 0x4600, 0x4611, 0
KDSETMODE, KD_TEXT, KD_GRAPHICS = 0x4B3A, 0, 1
VT_SETMODE, VT_RELDISP, VT_AUTO, VT_PROCESS, VT_ACKACQ = 0x5602, 0x5605, 0, 1, 2


# Scenes and the playlist -------------------------------------------------

def scene_library(scenes_dir: Path = PICTURES) -> dict[str, Path]:
    """Built-in loops in playlist order, then every other still or animation in scenes_dir."""
    library = {name: PICTURES / f"{name}.webp" for name in SCENES}
    if scenes_dir.is_dir():
        for path in sorted(scenes_dir.iterdir()):
            if path.suffix.lower() not in MEDIA or path.stem.endswith("-glow"):
                continue
            try:
                with Image.open(path) as media:
                    media.verify()
            except Exception as error:
                print(f"Skipping {path.name}: {error}", file=sys.stderr)
                continue
            library.setdefault(path.stem, path)
    return library


def glow_path(name: str, path: Path) -> Path:
    """Optional mask of windows and lamps that stay lit at night."""
    return path.with_name(f"{name}-glow.png")


def playlist(names, month: int) -> list[str]:
    return [name for name in names if month in SEASONS.get(name, range(1, 13))] or list(names)


def scene_at(names, when: float, seconds_per_scene: float) -> tuple[str, float]:
    """The scene at a Unix time and the seconds left in its slot.

    Slots follow the clock, so a restart resumes the scene that was showing."""
    slot, elapsed = divmod(when, seconds_per_scene)
    month = time.localtime(slot*seconds_per_scene).tm_mon
    shown = playlist(names, month)
    return shown[int(slot) % len(shown)], seconds_per_scene - elapsed


# Night hours -------------------------------------------------------------

NIGHTFALL = 15  # minutes the picture takes to turn between day and night, inside the night hours


def parse_night(text: str) -> tuple[int, int]:
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"night hours look like 23:00-07:00, not {text}")
    return clock_minutes(parts[0]), clock_minutes(parts[1])


def parse_place(text: str) -> tuple[float, float]:
    try:
        latitude, longitude = (float(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"a location looks like 52.23,21.01 (latitude,longitude), not {text}") from None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"not a place on Earth: {text}")
    return latitude, longitude


def clock_minutes(text: str) -> int:
    hours, minutes = text.strip().split(":")
    if not (0 <= int(hours) < 24 and 0 <= int(minutes) < 60):
        raise ValueError(f"not a time of day: {text}")
    return int(hours)*60 + int(minutes)


# Lighting ----------------------------------------------------------------

# Scenes are painted in daylight. The frame relights them for a sun it sets
# on the clock: warm light while the sun is near the horizon, and under it a
# day-for-night grade, the way films shoot night scenes in sunshine.

TWILIGHT = 12.0  # degrees either side of the horizon; beyond them the light no longer changes
# What low sunlight keeps of each colour, and the colour of lamplight.
DAWN, DUSK = np.array([1, .86, .74], np.float32), np.array([1, .82, .62], np.float32)
LAMPLIGHT = np.array([255, 196, 120], np.float32)
MOONLIGHT = np.array([.62, .82, 1.18], np.float32)
STARLIGHT = np.array([225, 232, 255], np.float32)
SHADOW = np.array([2, 5, 13], np.float32)  # the darkest a night pixel gets
LUMA = np.array([.299, .587, .114], np.float32)
# Brightness as a red filter sees it: blue sky and water go dark, clouds and snow stay light.
RED_FILTER = np.array([.45, .65, -.10], np.float32)
OVERCAST = np.array([.97, 1, 1.05], np.float32)  # the cool grey of daylight under cloud
FOG = np.array([196, 202, 208], np.float32)
RAIN, SNOW = np.array([205, 215, 230], np.float32), np.array([245, 248, 252], np.float32)


class Light(NamedTuple):
    """How a picture is lit: the sun, and the moon and the weather over it.

    The defaults are a clear sky with the full moon up, the look the night
    grade was tuned for. Values come in steps, so lookup tables are reused."""
    elevation: float      # the sun's, in degrees
    rising: bool = False
    moon: float = 1.      # moonlight, 0 (none) to 1 (a full moon up)
    cloud: float = 0.     # how grey the sky is, 0 to 1
    fog: float = 0.


def step(amount: float) -> float:
    """An amount from 0 to 1 in steps of 1/20, the steps the lighting moves in."""
    return round(amount*20)/20


def moonshine(moon: Moon | None) -> float:
    """How much the moon lights the night: the lit share of its disc while it is up. Unknown: full."""
    if moon is None:
        return 1.
    return step(moon.illumination*min(1, max(0, (moon.altitude + 1)/8)))


def night_amount(elevation: float) -> float:
    """0 until the sun touches the horizon, 1 once it is 12 degrees under it.

    The eye judges brightness by ratios, so the mix moves fastest at first:
    each degree then darkens the picture by about the same share."""
    dusk = min(1, max(0, (1 - elevation)/(1 + TWILIGHT)))
    return (1 - .2**dusk)/.8


def star_strength(light: tuple) -> float:
    """Stars come out once the sun is 5 degrees down, the brightest first.

    Cloud and fog hide them; without the moon they shine brighter."""
    light = Light(*light)
    dark = min(1, max(0, (-light.elevation - 5)/(TWILIGHT - 5)))**2
    return dark*(1 - light.cloud)*(1 - light.fog)*(1 + .3*(1 - light.moon))


def moonlight(rgb: np.ndarray, moon: float = 1.) -> np.ndarray:
    """Daylight colours as a moonlit night: dark, blue, with a trace of their colour left.

    Without the moon the night is darker still."""
    x = rgb/255
    luma = x @ LUMA
    # Night vision hardly sees red: poppies and tulips are the first things to go dark.
    red = np.clip(x[..., 0] - np.maximum(x[..., 1], x[..., 2]) - .25, 0, 1)
    seen = np.clip(x @ RED_FILTER - .6*red, 0, 1)
    # Whites are held back at the top, or daisies and blossom look lit from inside.
    tone = .55*seen**1.4*(1 - .25*seen**4)*(.62 + .38*moon)
    colour = (x - luma[..., None])*(tone/np.maximum(luma, .02))[..., None]
    return (tone[..., None] + .15*colour)*MOONLIGHT*255 + SHADOW


def overcast(rgb: np.ndarray, cloud: float) -> np.ndarray:
    """Daylight under grey cloud: flatter, greyer and a little darker.

    Sky blue, and water that mirrors it, turns to the pale grey of cloud."""
    luma = rgb @ LUMA
    blue = np.clip((rgb[..., 2] - (rgb[..., 0] + rgb[..., 1])/2)/80, 0, 1)*np.clip((luma - 60)/60, 0, 1)
    grey = (luma + 45*blue)[..., None]*OVERCAST
    return (rgb + (grey - rgb)*((.35 + .5*blue)*cloud)[..., None])*(1 - .18*cloud)


def sun_low(elevation: float) -> float:
    """How close the sun is to the horizon: 1 at a degree up, 0 outside twilight."""
    return max(0, 1 - ((elevation-1)/11)**2)**3


def as_painted(light: tuple, moonlit: bool = True) -> bool:
    """Whether relight leaves every colour, lamps included, exactly as painted.

    True in clear full daylight, and all night for a scene painted at night."""
    light = Light(*light)
    return (sun_low(light.elevation) == 0 and not light.cloud and not light.fog
            and not (moonlit and night_amount(light.elevation)))


def relight(rgb, light: tuple, glow: bool = False, moonlit: bool = True) -> np.ndarray:
    """Light 8-bit colours for the sun, the moon and the weather; floats on the same scale, not yet clipped.

    Glowing pixels (windows, lanterns, lava) keep their brightness after dark
    and warm up instead of turning to moonlight. A scene that is not moonlit
    keeps its painted colours all night. Cloud greys the day and hides the
    moon; fog veils everything, lamps a little less."""
    elevation, rising, moon, cloud, fog = Light(*light)
    rgb = np.asarray(rgb, np.float32)
    low = sun_low(elevation)*(1 - cloud)  # cloud hides the low sun's colour too
    day = overcast(rgb, cloud) if cloud else rgb
    lit = day*(1 - .15*low)*(1 - low*(1 - (DAWN if rising else DUSK)))
    night = night_amount(elevation)
    if night and moonlit:
        lit += ((rgb*.946 + LAMPLIGHT*.14 if glow else moonlight(rgb, moon*(1 - cloud))) - lit)*night
    if fog:
        mist = relight(FOG, Light(elevation, rising, moon, cloud), moonlit=moonlit)
        lit += (mist - lit)*fog*(.25 if glow else .5)
    return lit


_CODES = np.arange(65536, dtype=np.uint32)
_R5, _G6, _B5 = _CODES >> 11, _CODES >> 5 & 63, _CODES & 31
_LEVELS = np.array([31, 63, 31], np.float32)
_RGB8 = np.stack([_R5, _G6, _B5], axis=1)*(255/_LEVELS)  # every RGB565 colour, on the 8-bit scale
# 16-bit colour has few dark levels, and a night sky would show them as bands.
# Each cell of a 2x2 pattern rounds a little differently, which averages them out.
DITHER = (-.375, .125, .375, -.125)  # by cell: row*2 + column
_LUT565: dict[tuple, np.ndarray] = {}


def lut565(light: tuple, glow: bool = False, moonlit: bool = True) -> np.ndarray:
    """Relight an RGB565 pixel with one lookup: four tables of 65536, one per dither cell."""
    key = (Light(*light), glow, moonlit)
    tables = _LUT565.get(key)
    if tables is None:
        if len(_LUT565) > 32:
            _LUT565.clear()
        levels = np.clip(relight(_RGB8, light, glow, moonlit), 0, 255)*(_LEVELS/255)
        tables = np.empty((4, 65536), np.uint16)
        for cell, offset in enumerate(DITHER):
            r, g, b = np.clip(np.floor(levels + .5 + offset), 0, _LEVELS).astype(np.uint16).T
            tables[cell] = r << 11 | g << 5 | b
        _LUT565[key] = tables
    return tables


def brighter565(pixels: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """RGB565 pixels raised to at least the given 8-bit colours."""
    r, g, b = (colours*(_LEVELS/255)).astype(np.uint16).T
    return (np.maximum(pixels >> 11, r) << 11 | np.maximum(pixels >> 5 & 63, g) << 5
            | np.maximum(pixels & 31, b))


def to565(rgb: np.ndarray, dither: np.ndarray | float = 0.) -> np.ndarray:
    """8-bit colours as floats, rounded to RGB565 codes, nudged by a dither offset per pixel if given."""
    levels = np.asarray(rgb)*(_LEVELS/255) + .5 + np.asarray(dither)[..., None]
    r, g, b = np.moveaxis(np.clip(np.floor(levels), 0, _LEVELS).astype(np.uint16), -1, 0)
    return r << 11 | g << 5 | b


_BLENDS: dict[tuple, np.ndarray] = {}


def blend565(colour, alpha: float) -> np.ndarray:
    """A table that mixes any RGB565 pixel with a colour: how raindrops and snowflakes are drawn."""
    key = (tuple(np.round(colour).astype(int)), alpha)
    table = _BLENDS.get(key)
    if table is None:
        if len(_BLENDS) > 16:
            _BLENDS.clear()
        table = _BLENDS[key] = to565(_RGB8 + (np.asarray(colour, np.float32) - _RGB8)*alpha)
    return table


_FLASH: list[np.ndarray] = []


def flash565() -> np.ndarray:
    """A table that lights any RGB565 pixel as a lightning flash would."""
    if not _FLASH:
        _FLASH.append(to565(_RGB8*1.15 + 35))
    return _FLASH[0]


def same_colour565(pixels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Where RGB565 pixels are within a couple of levels of a reference."""
    same = np.ones(pixels.shape, bool)
    for shift, mask, slack in ((11, 31, 2), (5, 63, 4), (0, 31, 2)):
        same &= np.abs((pixels >> shift & mask).astype(np.int16)
                       - (reference >> shift & mask).astype(np.int16)) <= slack
    return same


@dataclass
class Stars:
    """Points of light in a scene's clear sky, for the night."""
    index: np.ndarray   # flat pixel indices
    level: np.ndarray   # brightness of each, 0 to 1
    pace: np.ndarray    # twinkle, in radians per second
    offset: np.ndarray  # where in its twinkle each one starts

    def light(self, seconds: float, strength: float) -> np.ndarray:
        """Their 8-bit colours at a moment."""
        twinkle = .8 + .2*np.sin(seconds*self.pace + self.offset)
        return np.minimum(self.level*twinkle*strength, 1)[:, None]*STARLIGHT


def clear_sky(rgb: np.ndarray) -> np.ndarray:
    """Where a daylight picture shows smooth, clear blue sky: room for stars and the moon.

    Sky is what a column reaches from near the top through blue and cloud. Only
    smooth blue counts, and only above the middle of the picture, which keeps
    stars off clouds and peaks and, where sea meets sky, off the water."""
    height, width = rgb.shape[:2]
    value = rgb.astype(np.int32)
    r, g, b = value[..., 0], value[..., 1], value[..., 2]
    blue = (b > r + 20) & (b >= g) & (b > 110)
    through = blue | (value @ np.array([77, 150, 29]) > 140*256)  # blue, or bright like cloud
    started = np.logical_or.accumulate(through, axis=0)
    ended = np.logical_or.accumulate(started & ~through, axis=0)
    clear = started & ~ended & (np.argmax(through, axis=0) < height//4) & blue
    for shift in ((0, 3), (3, 0), (0, -3), (-3, 0)):  # smooth: no texture and no edge
        clear &= np.abs(b - np.roll(b, shift, (0, 1))) < 12
    for shift in ((0, 2), (2, 0), (0, -2), (-2, 0)):  # and a little way inside it
        clear &= np.roll(clear, shift, (0, 1))
    clear[int(.55*height):] = clear[:, :2] = clear[:, -2:] = False
    return clear


def find_stars(rgb: np.ndarray, clear: np.ndarray | None = None) -> Stars | None:
    """Scatter stars over the clear sky of a daylight picture, thinning out towards the horizon.

    None if the picture shows little sky."""
    height, width = rgb.shape[:2]
    clear = clear_sky(rgb) if clear is None else clear
    horizon = int(.55*height)
    rows, columns = np.nonzero(clear)
    if len(rows) < 2000:
        return None
    rng = np.random.default_rng(7)
    chance = (1 - rows/horizon)**1.5
    count = min(900, len(rows)//700)
    picks = rng.choice(len(rows), count, replace=False, p=chance/chance.sum())
    index = rows[picks]*width + columns[picks]
    level = np.clip(rng.pareto(2.2, count)*.25 + .25, 0, 1)
    pace, offset = rng.uniform(1, 3, count), rng.uniform(0, 2*math.pi, count)
    big = level > .7  # the brightest are two pixels square
    index = np.concatenate([index] + [index[big] + shift for shift in (1, width, width+1)])
    level, pace, offset = (np.concatenate([values] + [values[big]]*3) for values in (level, pace, offset))
    return Stars(index, level, pace, offset)


def bayer(size: int) -> np.ndarray:
    """Ordered-dither thresholds 0..size*size-1."""
    matrix = np.zeros((1, 1), np.uint16)
    while matrix.shape[0] < size:
        matrix = np.block([[4*matrix, 4*matrix+2], [4*matrix+3, 4*matrix+1]])
    return matrix


_TILES: dict[tuple, np.ndarray] = {}


def tiled(size: int, shape: tuple[int, int]) -> np.ndarray:
    key = (size, shape)
    if key not in _TILES:
        height, width = shape
        _TILES[key] = np.tile(bayer(size), (height//size+1, width//size+1))[:height, :width]
    return _TILES[key]


def pack565(rgb: np.ndarray) -> np.ndarray:
    packed = (rgb[..., 0] >> 3).astype(np.uint16) << 11
    packed |= (rgb[..., 1] >> 2).astype(np.uint16) << 5
    packed |= rgb[..., 2] >> 3
    return packed


def dissolve(current: np.ndarray, upcoming: np.ndarray, amount: float) -> np.ndarray:
    """Crossfade by ordered dithering: cheap on RGB565 and smooth at a distance."""
    return np.where(tiled(8, current.shape) < round(amount*64), upcoming, current)


# The moon and the weather --------------------------------------------------

MOON_RADIUS = 26  # pixels on a 1080-line screen: larger than life, to read across a room
MOON = np.array([236, 238, 242], np.float32)
HALO = np.array([150, 170, 215], np.float32)
# Seas of the near side as (x, y, size, depth) on the unit disc, north up.
MARIA = ((-.55, -.05, .34, .22), (-.25, -.42, .26, .25), (.22, -.38, .17, .22), (.35, -.05, .21, .22),
         (.68, -.3, .11, .25), (.55, .22, .14, .2), (-.2, .35, .17, .18), (-.45, .42, .1, .2))
_MOONS: dict[tuple, tuple] = {}


def moon_disc(phase: float, bright_side: float, radius: int = MOON_RADIUS):
    """The moon's 8-bit colours, how much of it covers the sky, and its halo, on a square patch.

    The lit part faces the sun (bright_side, degrees clockwise from up). The
    rest lets the night sky through, brightened a little by earthshine.
    Phase is 0 for full, 180 for new."""
    key = (round(phase), round(bright_side/2)*2, radius)
    if key not in _MOONS:
        if len(_MOONS) > 8:
            _MOONS.clear()
        half = 3*radius
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        distance = np.hypot(x, y)
        u, v = x/radius, y/radius
        albedo = np.ones_like(u)
        for sea_x, sea_y, size, depth in MARIA:
            albedo -= depth*np.exp(-((u - sea_x)**2 + (v - sea_y)**2)/size**2)
        rim = np.sqrt(np.clip(1 - u*u - v*v, 0, 1))
        turn = math.radians(bright_side)
        towards, across = u*math.sin(turn) - v*math.cos(turn), u*math.cos(turn) + v*math.sin(turn)
        terminator = -math.cos(math.radians(phase))*np.sqrt(np.clip(1 - across**2, 0, 1))
        lit = np.clip((towards - terminator)/.08 + .5, 0, 1)
        colour = MOON*(albedo*(.8 + .2*rim))[..., None]
        cover = np.clip(radius + .5 - distance, 0, 1)*(lit + .07*(1 - lit))
        halo = (1 + math.cos(math.radians(phase)))/2*.3*np.exp(-np.maximum(distance - radius, 0)/(.9*radius))
        _MOONS[key] = colour, cover, halo
    return _MOONS[key]


def moon_room(sky: np.ndarray, radius: int = MOON_RADIUS) -> np.ndarray:
    """Where the moon's centre can go: the disc and a margin around it lie in clear sky."""
    height, width = sky.shape
    side = 2*(radius + 4) + 1
    total = np.zeros((height + 1, width + 1), np.int32)
    total[1:, 1:] = sky.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    inside = total[side:, side:] - total[:-side, side:] - total[side:, :-side] + total[:-side, :-side]
    room = np.zeros_like(sky)
    room[side//2:side//2 + inside.shape[0], side//2:side//2 + inside.shape[1]] = inside == side*side
    return room


def moon_spot(sky: np.ndarray, moon: Moon, room: np.ndarray | None = None) -> tuple[int, int] | None:
    """Where the moon shows in a picture's clear sky, or None while it is down or no sky is near.

    Every picture is taken to face south: the moon rises on the left, sets on
    the right and stands as high above the horizon as it really is. If cloud,
    a peak or a tree stands there, it shows in the nearest clear sky instead."""
    if moon.altitude < 1 or not 75 <= moon.azimuth <= 285:
        return None
    room = moon_room(sky) if room is None else room
    height, width = sky.shape
    rows = np.flatnonzero(sky.any(axis=1))
    if not len(rows):
        return None
    horizon = rows[-1]
    x = round(width*(.5 + (moon.azimuth - 180)/210))
    y = round(horizon - (horizon - MOON_RADIUS)*min(1, moon.altitude/55))
    reach, stride = width//8, 3
    left = max(0, x - reach)
    ys, xs = np.nonzero(room[::stride, left:x + reach + 1:stride])
    if not len(ys):
        return None
    ys, xs = ys*stride, xs*stride + left
    nearest = np.argmin((xs - x)**2 + (ys - y)**2)
    return int(xs[nearest]), int(ys[nearest])


def draw_moon(image: np.ndarray, spot: tuple[int, int], moon: Moon, strength: float,
              frame: np.ndarray | None = None, first: np.ndarray | None = None) -> None:
    """Draw the moon into an RGB565 picture.

    Given the loop's current and first frames, it hides behind whatever
    passes in front of it, such as balloons and birds."""
    colour, cover, halo = moon_disc(moon.phase, moon.bright_side)
    half = cover.shape[0]//2
    x, y = spot
    top, left = max(0, y - half), max(0, x - half)
    bottom, right = min(image.shape[0], y + half + 1), min(image.shape[1], x + half + 1)
    patch = (slice(top - y + half, bottom - y + half), slice(left - x + half, right - x + half))
    area = image[top:bottom, left:right]
    behind = _RGB8[area]
    mixed = behind + (HALO - behind)*(halo[patch]*strength)[..., None]
    amount = (cover[patch]*strength)[..., None]
    mixed += (colour[patch] - mixed)*amount
    if frame is not None and first is not None:
        passing = ~same_colour565(frame[top:bottom, left:right], first[top:bottom, left:right])
        mixed = np.where(passing[..., None], behind, mixed)
    cells = np.array(DITHER, np.float32)[tiled(2, image.shape)[top:bottom, left:right]]
    area[:] = to565(mixed, cells)  # dithered like the night sky, so the halo does not band


def lightning(seconds: float) -> float:
    """How bright a lightning flash is at a moment: a double flicker every few seconds, 0 between."""
    bucket = int(seconds//9)
    rng = np.random.default_rng(bucket)
    if rng.random() > .6:
        return 0.
    since = seconds - bucket*9 - rng.uniform(0, 8)
    return 1. if 0 <= since < .15 else .5 if .3 <= since < .45 else 0.


class Precipitation:
    """Rain and snow over a picture: the same drops on every pass, so they fall smoothly.

    Drops fall straight, or slant with the wind, and are mixed into the picture
    so that it shows through them. Their colour is lit like the picture.
    Sizes are for a 1080-line screen seen from across a room."""
    COUNT = 2400

    def __init__(self, width: int, height: int):
        rng = np.random.default_rng(5)
        self.size = width, height
        self.x, self.y = rng.random(self.COUNT), rng.random(self.COUNT)
        self.pace = rng.uniform(.8, 1.25, self.COUNT)
        self.near = rng.random(self.COUNT) < .3
        self.sway = rng.uniform(0, 2*math.pi, self.COUNT)
        self.flake = rng.choice([2, 2, 3, 3, 4, 5], self.COUNT)

    def draw(self, image: np.ndarray, seconds: float, weather: Weather, light: Light) -> np.ndarray:
        """A copy of the picture with the rain and snow of a moment drawn in."""
        image = image.copy()
        flat = image.reshape(-1)
        if weather.rain:
            self._rain(flat, seconds, weather, light)
        if weather.snow:
            self._snow(flat, seconds, weather, light)
        return image

    @staticmethod
    def _colour(painted: np.ndarray, light: Light) -> np.ndarray:
        """Drops lit like the picture, but brighter after dark: they catch every light there is."""
        return np.clip(relight(painted, light)*(1 + .7*night_amount(light.elevation)), 0, 255)

    def _mix(self, flat: np.ndarray, rows: np.ndarray, columns: np.ndarray, table: np.ndarray) -> None:
        width, height = self.size
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        index = rows[inside]*width + columns[inside]
        flat[index] = table[flat[index]]

    def _rain(self, flat, seconds, weather, light) -> None:
        width, height = self.size
        count = int(self.COUNT*weather.rain)
        slant = .35*weather.wind  # pixels sideways for each pixel down
        colour = self._colour(RAIN, light)
        for near, length, speed, alpha in ((False, 28, 1400, .3), (True, 56, 2200, .45)):
            drops = np.flatnonzero(self.near[:count] == near)
            fall = height + length
            top = (self.y[drops]*fall + speed*self.pace[drops]*seconds) % fall - length
            rows = (top[:, None] + np.arange(length)).astype(int)
            columns = (self.x[drops, None]*width*1.4 - width*.2 + slant*rows).astype(int)
            for dx in (0, 1) if near else (0,):
                self._mix(flat, rows, columns + dx, blend565(colour, alpha))

    def _snow(self, flat, seconds, weather, light) -> None:
        width, height = self.size
        count = int(self.COUNT*.8*weather.snow)
        size, pace = self.flake[:count], self.pace[:count]
        fall = height + 12
        rows = (self.y[:count]*fall + (25 + 40*size)*pace*seconds) % fall - 6  # nearer flakes are bigger and faster
        sway = (4 + 4*size)*np.sin(.7*pace*seconds + self.sway[:count])
        columns = (self.x[:count]*width*1.4 - width*.2 + .5*weather.wind*rows + sway) % width
        colour = self._colour(SNOW, light)
        core, rim = blend565(colour, .9), blend565(colour, .4)
        for side in np.unique(size):
            flakes = size == side
            top, left = rows[flakes].astype(int), columns[flakes].astype(int)
            for dy in range(side):
                for dx in range(side):
                    self._mix(flat, top + dy, left + dx, core)
            for edge in range(side):  # a soft edge, so a flake looks round
                for dy, dx in ((-1, edge), (side, edge), (edge, -1), (edge, side)):
                    self._mix(flat, top + dy, left + dx, rim)


# Decoded loops -----------------------------------------------------------

@dataclass(eq=False)  # compared by identity: the run loop asks "is this the loop on screen?"
class Loop:
    frames: np.ndarray            # (count, height, width) RGB565
    ends: list[int]               # when each frame ends, in ms from the loop start
    glow: np.ndarray | None = None  # flat indices of pixels that glow at night
    moonlit: bool = True          # False for a scene painted at night, which stays as it is
    stars: Stars | None = None
    clear: np.ndarray | None = None  # (frame, star): nothing is passing in front of the star
    sky: np.ndarray | None = None    # clear sky in the first frame, where the moon can show
    weather: bool = True             # False for a scene that keeps its own weather

    def __post_init__(self):
        if self.glow is not None:
            rows, columns = np.divmod(self.glow, self.frames.shape[2])
            self.glow_cell = rows % 2*2 + columns % 2
        self.room = None  # moon_room(sky), worked out on the first night

    def frame_at(self, seconds: float) -> int:
        return bisect.bisect_right(self.ends, int(seconds*1000) % self.ends[-1])

    def frame(self, seconds: float) -> np.ndarray:
        return self.frames[self.frame_at(seconds)]

    def weathered(self, light: tuple) -> Light:
        """The light on this scene: without the weather if it keeps its own."""
        light = Light(*light)
        return light if self.weather else light._replace(cloud=0., fog=0.)

    def twinkles(self, light: tuple) -> bool:
        """Whether the picture changes between its frames: stars are out."""
        return self.stars is not None and star_strength(self.weathered(light)) > 0

    def lit(self, seconds: float, light: tuple, moon: Moon | None = None, flash: float = 0.) -> np.ndarray:
        """The frame showing at a moment, lit, with the moon and a lightning flash if given.

        In clear daylight it is the decoded frame itself: don't modify it."""
        light = self.weathered(light)
        index = self.frame_at(seconds)
        source = self.frames[index]
        if as_painted(light, self.moonlit) and not self.twinkles(light):
            image = source
        else:
            image = self._relit(seconds, index, light, moon)
        if flash and self.weather:
            image = dissolve(image, flash565()[source], .85*flash)
        return image

    def _relit(self, seconds: float, index: int, light: Light, moon: Moon | None) -> np.ndarray:
        source = self.frames[index]
        tables = lut565(light, moonlit=self.moonlit)
        image = np.empty_like(source)
        for row in (0, 1):
            for column in (0, 1):
                image[row::2, column::2] = tables[row*2 + column][source[row::2, column::2]]
        flat = image.reshape(-1)
        if self.glow is not None:
            lamps = lut565(light, True, self.moonlit)
            flat[self.glow] = lamps[self.glow_cell, source.reshape(-1)[self.glow]]
        strength = star_strength(light)
        if self.stars is not None and strength:
            colours = self.stars.light(seconds, strength)
            if self.clear is not None:
                colours *= self.clear[index][:, None]
            flat[self.stars.index] = brighter565(flat[self.stars.index], colours)
        # The moon rises a little before the stars come out, and cloud and fog hide it.
        shows = min(1, max(0, (-light.elevation - 2)/6))*(1 - light.cloud)*(1 - light.fog)
        if moon is not None and self.sky is not None and self.moonlit and shows > .02:
            if self.room is None:
                self.room = moon_room(self.sky)
            spot = moon_spot(self.sky, moon, self.room)
            if spot is not None:
                draw_moon(image, spot, moon, shows, source, self.frames[0])
        return image


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to cover the screen and crop the overflow; pixel art keeps hard edges."""
    if image.size == size:
        return image
    scale = max(size[0]/image.width, size[1]/image.height)
    method = Image.Resampling.NEAREST if scale >= 3 else Image.Resampling.LANCZOS
    image = image.resize((math.ceil(image.width*scale), math.ceil(image.height*scale)), method)
    left, top = (image.width-size[0])//2, (image.height-size[1])//2
    return image.crop((left, top, left+size[0], top+size[1]))


def read_frames(path: Path, size: tuple[int, int], limit: int = MAX_FRAMES):
    """(count, iterator of (RGB image, duration ms)) for a still or an animation."""
    media = Image.open(path)
    total = getattr(media, "n_frames", 1)
    stride = max(1, math.ceil(total/limit))

    def frames():
        with media:
            for index in range(0, total, stride):
                media.seek(index)
                frame = media.convert("RGB")  # WebP sets the duration on decoding.
                # Browsers treat GIF delays under 20 ms as 100 ms; so do we.
                duration = media.info.get("duration") or 100
                duration = (duration if duration >= 20 else 100)*stride if total > 1 else 1000
                yield fit(frame, size), duration
    return len(range(0, total, stride)), frames()


def read_glow(path: Path, size: tuple[int, int]) -> Image.Image | None:
    if not path.exists():
        return None
    with Image.open(path) as mask:
        return fit(mask.convert("L"), size)


def decode_loop(name: str, path: Path, size: tuple[int, int], limit: int = MAX_FRAMES) -> Loop:
    count, frames = read_frames(path, size, limit)
    packed = np.empty((count, size[1], size[0]), np.uint16)
    ends, total = [], 0
    for index, (image, duration) in enumerate(frames):
        pixels = np.asarray(image)
        if index == 0:
            sky = clear_sky(pixels)
            stars = find_stars(pixels, sky)
        packed[index] = pack565(pixels)
        total += duration
        ends.append(total)
    mask = read_glow(glow_path(name, path), size)
    glow = None if mask is None else np.flatnonzero(np.asarray(mask) > 127)
    clear = None
    if stars is not None and count > 1:
        # A star shows only while its patch of sky looks as it did in the first frame,
        # so balloons, birds and snow pass in front of it.
        behind = packed.reshape(count, -1)[:, stars.index]
        clear = same_colour565(behind, behind[0])
    return Loop(packed, ends, glow, name not in PAINTED_AT_NIGHT, stars, clear,
                None if stars is None else sky, name not in OWN_WEATHER)


class LoopCache:
    """Decoded loops for the scenes on screen now and next, filled in the background."""

    def __init__(self, library: dict[str, Path], size: tuple[int, int]):
        self.library, self.size = library, size
        self.loops: dict[str, Loop] = {}
        self.previews: dict[str, Loop] = {}
        self.loading: set[str] = set()
        self.lock = threading.Lock()

    def ready(self, name: str) -> Loop | None:
        """The decoded loop, or None after starting to decode it."""
        with self.lock:
            loop = self.loops.get(name)
            if loop is None and name not in self.loading:
                self.loading.add(name)
                threading.Thread(target=self._decode, args=(name,), daemon=True).start()
        return loop

    def get(self, name: str) -> Loop:
        """Something to show now: the loop, or its first frame while it decodes."""
        loop = self.ready(name)
        if loop is None:
            loop = self.previews.get(name)
            if loop is None:
                loop = self.previews[name] = decode_loop(name, self.library[name], self.size, 1)
        return loop

    def keep(self, names: set[str]) -> None:
        with self.lock:
            for cache in (self.loops, self.previews):
                for name in set(cache) - names:
                    del cache[name]

    def _decode(self, name: str) -> None:
        try:
            loop = decode_loop(name, self.library[name], self.size)
        except Exception as error:  # A broken drop-in file must not stop the frame.
            print(f"Cannot decode {name}: {error}", file=sys.stderr, flush=True)
            loop = Loop(np.zeros((1, self.size[1], self.size[0]), np.uint16), [1000])
        with self.lock:
            self.loops[name] = loop
            self.loading.discard(name)


# Display -----------------------------------------------------------------

def screen_view(buffer, width: int, height: int, stride: int, xoffset: int, yoffset: int):
    """The visible RGB565 area of a framebuffer mapping, honouring stride and pan offsets."""
    return np.ndarray((height, width), "<u2", buffer, yoffset*stride + xoffset*2, (stride, 2))


class Framebuffer:
    """RGB565 fbdev output, held by one process at a time; restores the screen on close."""

    def __init__(self, device: str):
        self.file = open(device, "r+b", buffering=0)
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise SystemExit(f"{device} is in use by another pixel-frame, probably the service. "
                             "Stop it first: pixel-frame stop")
        var = bytearray(160)
        fcntl.ioctl(self.file, FBIOGET_VSCREENINFO, var, True)
        self.width, self.height, _, _, xoffset, yoffset, bpp = struct.unpack_from("7I", var)
        if bpp != 16 or struct.unpack_from("9I", var, 32) != (11, 5, 0, 5, 6, 0, 0, 5, 0):
            raise SystemExit("The framebuffer must be RGB565 (16 bits per pixel); try --tty.")
        stride = int((Path("/sys/class/graphics")/Path(device).name/"stride").read_text())
        self.map = mmap.mmap(self.file.fileno(), stride*(self.height+yoffset))
        self.original = self.map[:]
        self.screen = screen_view(self.map, self.width, self.height, stride, xoffset, yoffset)
        # Earlier versions switched the display off at night, and a run killed then left it off.
        try:
            fcntl.ioctl(self.file, FBIOBLANK, FB_BLANK_UNBLANK)
        except OSError:
            pass

    def show(self, image: np.ndarray) -> None:
        np.copyto(self.screen, image)

    def close(self) -> None:
        self.screen = None
        self.map[:] = self.original
        self.map.close()
        self.file.close()


class Console:
    """Keep the kernel console from drawing over the picture.

    On a virtual terminal (the service runs on tty1) switch it to graphics mode
    and pause while another terminal is shown, so Ctrl+Alt+F2 still reaches a
    login prompt. The kernel ignores that key on a graphics console unless its
    program takes part in switching, so the frame does: it lets the screen go
    between two pictures. Started over SSH, only hide the visible console's cursor."""

    def __init__(self):
        self.vt = self.cursor = None
        self.leaving = False  # another console asked for the screen
        try:
            name = os.ttyname(0)
        except OSError:
            name = ""
        if name.startswith("/dev/tty") and name[8:].isdigit():
            try:
                fcntl.ioctl(0, KDSETMODE, KD_GRAPHICS)
                self.vt = name[5:]
            except OSError:
                pass
            else:
                signal.signal(signal.SIGUSR1, self._asked_to_leave)
                signal.signal(signal.SIGUSR2, self._back)
                try:
                    fcntl.ioctl(0, VT_SETMODE, struct.pack("bbhhh", VT_PROCESS, 0,
                                                           signal.SIGUSR1, signal.SIGUSR2, 0))
                except OSError as error:
                    print(f"Ctrl+Alt+F2 will not work: {error}", file=sys.stderr, flush=True)
                return
        try:
            self.cursor = os.open("/dev/" + self.active(), os.O_WRONLY | os.O_NOCTTY)
            os.write(self.cursor, b"\x1b[?25l")
        except OSError:
            self.cursor = None

    def _asked_to_leave(self, _signal, _frame) -> None:
        self.leaving = True  # let go in visible(), between two pictures, never halfway through one

    def _back(self, _signal, _frame) -> None:
        fcntl.ioctl(0, VT_RELDISP, VT_ACKACQ)

    @staticmethod
    def active() -> str:
        return Path("/sys/class/tty/tty0/active").read_text().strip()

    def _let_go(self) -> None:
        if self.leaving:
            self.leaving = False
            try:
                fcntl.ioctl(0, VT_RELDISP, 1)
            except OSError:
                pass  # the switch was called off

    def visible(self) -> bool:
        """Whether the frame may draw; first hands the screen over if another console asked."""
        self._let_go()
        return self.vt is None or self.active() == self.vt

    def restore(self) -> None:
        if self.vt is not None:
            self._let_go()
            fcntl.ioctl(0, VT_SETMODE, struct.pack("bbhhh", VT_AUTO, 0, 0, 0, 0))
            fcntl.ioctl(0, KDSETMODE, KD_TEXT)
        if self.cursor is not None:
            os.write(self.cursor, b"\x1b[?25h")
            os.close(self.cursor)


@dataclass
class Settings:
    seconds_per_scene: float = SECONDS_PER_SCENE
    crossfade: float = 3
    fps: float = 8
    day_seconds: float | None = None  # a fast demo cycle instead of the clock
    night: tuple[int, int] = (23*60, 7*60)  # minutes of the day that night starts and ends
    start_scene: str | None = None
    place: tuple[float, float] | None = None  # (latitude, longitude) for the moon and the weather
    weather: str | None = None  # a preset to show instead of the live report

    def light(self, now: float) -> tuple[float, bool]:
        """The sun's elevation and whether it is rising, in the steps the lighting moves in.

        Full day outside the night hours. Inside them the sun sets in the first
        NIGHTFALL minutes and rises in the last. Elevation is rounded to an
        eighth of a degree and held at the edges of twilight."""
        if self.day_seconds:
            turn = 2*math.pi*now/self.day_seconds
            elevation, rising = 50*math.sin(turn), math.cos(turn) > 0
        else:
            local = time.localtime(now)
            start, end = self.night
            length = (end - start) % 1440
            into = (local.tm_hour*60 + local.tm_min + local.tm_sec/60 - start) % 1440
            if into >= length:
                return TWILIGHT, False
            elevation = TWILIGHT*(1 - 2*min(into, length - into)/NIGHTFALL)
            rising = into > length/2
        return round(max(-TWILIGHT, min(TWILIGHT, elevation))*8)/8, rising and abs(elevation) < TWILIGHT


def schedule(names, settings: Settings, started: float, now: float):
    """(current scene, upcoming scene, seconds until the change)."""
    if settings.start_scene and now < started + settings.seconds_per_scene:
        current, remaining = settings.start_scene, started + settings.seconds_per_scene - now
    else:
        current, remaining = scene_at(names, now, settings.seconds_per_scene)
    upcoming, _ = scene_at(names, now + remaining + .001, settings.seconds_per_scene)
    return current, upcoming, remaining


def run(device: str, library: dict[str, Path], settings: Settings) -> None:
    running = True

    def stop(_signal, _frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    fb = Framebuffer(device)
    console = Console()
    try:
        cache = LoopCache(library, (fb.width, fb.height))
        sky = Sky(settings.place, PRESETS.get(settings.weather))
        falling = Precipitation(fb.width, fb.height)
        names, started, shown = list(library), time.time(), None
        drawn = None  # (loop, frame, light) on the screen; None when the next tick must draw
        while running:
            tick = time.monotonic()
            now = time.time()
            if not console.visible():
                drawn = None  # the other console draws over the picture
                time.sleep(.2)
                continue
            current, upcoming, remaining = schedule(names, settings, started, now)
            if current != shown:
                print(f"{time.strftime('%H:%M')} {current}", flush=True)
                shown = current
            cache.keep({current, upcoming})
            weather, moon = sky.weather(now), sky.moon(now)
            light = Light(*settings.light(now), moonshine(moon), step(weather.cloud), step(weather.fog))
            loop = cache.get(current)
            # Decode the next scene once the current one is ready, not alongside it.
            nearby = cache.ready(upcoming) if current in cache.loops else None
            fading = remaining < settings.crossfade and upcoming != current and nearby is not None
            flash = lightning(tick) if weather.storm else 0.
            pouring = bool(weather.rain or weather.snow) and loop.weather
            frame = (loop, loop.frame_at(tick), light)
            # A still by day, or a loop between its frames, needs no work until something changes.
            moving = fading or loop.twinkles(light) or pouring or bool(flash)
            if moving or frame != drawn:
                image = loop.lit(tick, light, moon, flash)
                if fading:
                    image = dissolve(image, nearby.lit(tick, light, moon, flash), 1 - remaining/settings.crossfade)
                if pouring:
                    image = falling.draw(image, tick, weather, light)
                fb.show(image)
                drawn = None if moving else frame
            time.sleep(max(0, 1/settings.fps - (time.monotonic()-tick)))
    finally:
        try:
            fb.close()
        finally:
            console.restore()


# Exports -----------------------------------------------------------------

def relit_frames(name: str, path: Path, light: tuple[float, bool], limit: int = MAX_FRAMES):
    """A scene's frames as the display lights them, at full 8-bit precision, with their durations."""
    glow = read_glow(glow_path(name, path), SCREEN)
    lamps = None if glow is None else np.asarray(glow) > 127
    strength = star_strength(light)
    _, frames = read_frames(path, SCREEN, limit)
    elapsed = 0
    for index, (image, duration) in enumerate(frames):
        pixels = np.asarray(image)
        if index == 0:
            stars = find_stars(pixels)
            sky = None if stars is None else pixels.reshape(-1, 3)[stars.index].astype(np.int16)
        moonlit = name not in PAINTED_AT_NIGHT
        lit = relight(pixels, light, moonlit=moonlit)
        if lamps is not None:
            lit[lamps] = relight(pixels[lamps], light, True, moonlit)
        if stars is not None and strength:
            clear = (np.abs(pixels.reshape(-1, 3)[stars.index] - sky) <= 16).all(axis=1)
            flat = lit.reshape(-1, 3)
            flat[stars.index] = np.maximum(flat[stars.index],
                                           stars.light(elapsed/1000, strength)*clear[:, None])
        yield Image.fromarray(np.clip(lit + .5, 0, 255).astype(np.uint8)), duration
        elapsed += duration


def export_still(name: str, path: Path, light: tuple[float, bool], output: Path) -> None:
    image, _ = next(relit_frames(name, path, light, 1))
    image.save(output)


def export_gif(name: str, path: Path, light: tuple[float, bool], output: Path, width: int = 640) -> None:
    """A looping GIF small enough to send in a chat.

    Every frame shares one palette and pixels that did not change are
    transparent, so the still parts of the painting are stored once."""
    size = (width, round(width*SCREEN[1]/SCREEN[0]))
    dither = (tiled(8, size[::-1]).astype(np.int16) - 32)//4  # -8..7, the same every frame
    smalls, durations = [], []
    for image, duration in relit_frames(name, path, light):
        small = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
        smalls.append(Image.fromarray(np.clip(small + dither[..., None], 0, 255).astype(np.uint8)))
        durations.append(duration)
    # One palette for the whole loop, from frames spread across it.
    picks = smalls[::max(1, len(smalls)//8)]
    mosaic = Image.new("RGB", (size[0], size[1]*len(picks)))
    for index, small in enumerate(picks):
        mosaic.paste(small, (0, index*size[1]))
    palette = mosaic.quantize(255, dither=Image.Dither.NONE)
    indexed = [np.asarray(small.quantize(palette=palette, dither=Image.Dither.NONE)) for small in smalls]
    colors = (palette.getpalette() + [0]*768)[:255*3]
    frames_out = []
    for index, current in enumerate(indexed):
        delta = current if index == 0 else np.where(current == indexed[index-1], 255, current)
        frame = Image.fromarray(delta.astype(np.uint8), "P")
        frame.putpalette(colors + [0, 0, 0])
        frames_out.append(frame)
    frames_out[0].save(output, save_all=True, append_images=frames_out[1:], duration=durations,
                       loop=0, transparency=255, disposal=1, optimize=False)


# Command line ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Painted, animated landscapes for a framebuffer picture frame")
    parser.add_argument("--device", default="/dev/fb0", help="framebuffer device (default: /dev/fb0)")
    parser.add_argument("--scenes-dir", type=Path, default=PICTURES,
                        help="folder of your own stills, GIFs or WebPs to add to the playlist")
    parser.add_argument("--start-scene", help="scene to show first; the clock-based playlist follows")
    parser.add_argument("--seconds-per-scene", type=float, default=SECONDS_PER_SCENE,
                        help="seconds per scene (default: 720)")
    parser.add_argument("--crossfade", type=float, default=3, help="seconds to fade between scenes (default: 3)")
    parser.add_argument("--fps", type=float, default=8, help="frames per second (default: 8, as the loops)")
    parser.add_argument("--day-seconds", type=float,
                        help="run a fast day/night cycle of this length instead of following the sun")
    parser.add_argument("--night", default="23:00-07:00",
                        help="night hours; the rest of the day is shown in daylight (default: 23:00-07:00)")
    parser.add_argument("--location", metavar="LAT,LON",
                        help="show the real moon and the live weather for this place, e.g. 52.23,21.01")
    parser.add_argument("--weather", choices=sorted(PRESETS),
                        help="show this weather instead of the live report, to try it out")
    parser.add_argument("--scene", help="scene to export (default: woodland)")
    parser.add_argument("--at", help="time of day to light an export, HH:MM (default: now)")
    parser.add_argument("--export", type=Path, metavar="FILE", help="write a still (.png, .jpg, .ppm)")
    parser.add_argument("--export-gif", type=Path, metavar="FILE", help="write a shareable looping GIF")
    parser.add_argument("--gif-width", type=int, default=640, help="GIF width in pixels (default: 640)")
    parser.add_argument("--tty", action="store_true", help="animated ASCII woodland for a text terminal")
    args = parser.parse_args()

    if args.tty:
        from tty_scene import run_tty
        try:
            run_tty(1, int(args.fps), args.day_seconds or 120)
        except KeyboardInterrupt:
            pass
        return

    if min(args.fps, args.seconds_per_scene) <= 0 or args.crossfade < 0 or (args.day_seconds or 1) <= 0:
        parser.error("durations and fps must be positive")
    try:
        night = parse_night(args.night)
        at = clock_minutes(args.at) if args.at else None
        place = parse_place(args.location) if args.location else None
    except ValueError as error:
        parser.error(str(error))
    library = scene_library(args.scenes_dir)
    for name in (args.scene, args.start_scene):
        if name and name not in library:
            parser.error(f"unknown scene {name!r}; choose from {', '.join(library)}")
    settings = Settings(args.seconds_per_scene, args.crossfade, args.fps, args.day_seconds,
                        night, args.start_scene, place, args.weather)

    if args.export or args.export_gif:
        name = args.scene or SCENES[0]
        when = dt.datetime.now().astimezone()
        if at is not None:
            when = when.replace(hour=at//60, minute=at % 60, second=0, microsecond=0)
        light = settings.light(when.timestamp())
        if args.export:
            export_still(name, library[name], light, args.export)
            print(f"Wrote {args.export}")
        if args.export_gif:
            export_gif(name, library[name], light, args.export_gif, args.gif_width)
            print(f"Wrote {args.export_gif} ({args.export_gif.stat().st_size/1e6:.1f} MB)")
        return

    run(args.device, library, settings)


if __name__ == "__main__":
    main()

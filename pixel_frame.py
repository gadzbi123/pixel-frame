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

import numpy as np
from PIL import Image

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
MEDIA = {".webp", ".gif", ".png", ".jpg", ".jpeg"}
SCREEN = (1920, 1080)
SECONDS_PER_SCENE = 720
MAX_FRAMES = 480  # Longer drop-in animations are sampled to bound memory.

FBIOGET_VSCREENINFO, FBIOBLANK, FB_BLANK_UNBLANK = 0x4600, 0x4611, 0
KDSETMODE, KD_TEXT, KD_GRAPHICS = 0x4B3A, 0, 1


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


def night_amount(elevation: float) -> float:
    """0 until the sun touches the horizon, 1 once it is 12 degrees under it.

    The eye judges brightness by ratios, so the mix moves fastest at first:
    each degree then darkens the picture by about the same share."""
    dusk = min(1, max(0, (1 - elevation)/(1 + TWILIGHT)))
    return (1 - .2**dusk)/.8


def star_strength(elevation: float) -> float:
    """Stars come out once the sun is 5 degrees down, the brightest first."""
    return min(1, max(0, (-elevation - 5)/(TWILIGHT - 5)))**2


def moonlight(rgb: np.ndarray) -> np.ndarray:
    """Daylight colours as a moonlit night: dark, blue, with a trace of their colour left."""
    x = rgb/255
    luma = x @ LUMA
    # Night vision hardly sees red: poppies and tulips are the first things to go dark.
    red = np.clip(x[..., 0] - np.maximum(x[..., 1], x[..., 2]) - .25, 0, 1)
    seen = np.clip(x @ RED_FILTER - .6*red, 0, 1)
    # Whites are held back at the top, or daisies and blossom look lit from inside.
    tone = .55*seen**1.4*(1 - .25*seen**4)
    colour = (x - luma[..., None])*(tone/np.maximum(luma, .02))[..., None]
    return (tone[..., None] + .15*colour)*MOONLIGHT*255 + SHADOW


def relight(rgb, elevation: float, rising: bool, glow: bool = False, moonlit: bool = True) -> np.ndarray:
    """Light 8-bit colours for the sun's position; floats on the same scale, not yet clipped.

    Glowing pixels (windows, lanterns, lava) keep their brightness after dark
    and warm up instead of turning to moonlight. A scene that is not moonlit
    keeps its painted colours all night."""
    rgb = np.asarray(rgb, np.float32)
    low = max(0, 1 - ((elevation-1)/11)**2)**3  # how close the sun is to the horizon; 0 outside twilight
    lit = rgb*(1 - .15*low)*(1 - low*(1 - (DAWN if rising else DUSK)))
    night = night_amount(elevation)
    if night and moonlit:
        lit += ((rgb*.946 + LAMPLIGHT*.14 if glow else moonlight(rgb)) - lit)*night
    return lit


_CODES = np.arange(65536, dtype=np.uint32)
_R5, _G6, _B5 = _CODES >> 11, _CODES >> 5 & 63, _CODES & 31
_LEVELS = np.array([31, 63, 31], np.float32)
_RGB8 = np.stack([_R5, _G6, _B5], axis=1)*(255/_LEVELS)  # every RGB565 colour, on the 8-bit scale
# 16-bit colour has few dark levels, and a night sky would show them as bands.
# Each cell of a 2x2 pattern rounds a little differently, which averages them out.
DITHER = (-.375, .125, .375, -.125)  # by cell: row*2 + column
_LUT565: dict[tuple, np.ndarray] = {}


def lut565(light: tuple[float, bool], glow: bool = False, moonlit: bool = True) -> np.ndarray:
    """Relight an RGB565 pixel with one lookup: four tables of 65536, one per dither cell."""
    key = (light, glow, moonlit)
    tables = _LUT565.get(key)
    if tables is None:
        if len(_LUT565) > 32:
            _LUT565.clear()
        levels = np.clip(relight(_RGB8, *light, glow, moonlit), 0, 255)*(_LEVELS/255)
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
        return (self.level*twinkle*strength)[:, None]*STARLIGHT


def find_stars(rgb: np.ndarray) -> Stars | None:
    """Scatter stars over the clear blue sky of a daylight picture, or None if it shows little.

    Sky is what a column reaches from near the top through blue and cloud. Stars
    go only on smooth blue and thin out towards the horizon, which keeps them
    off clouds and peaks and, where sea meets sky, off the water."""
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
    horizon = int(.55*height)
    clear[horizon:] = clear[:, :2] = clear[:, -2:] = False
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
    index = np.concatenate([index] + [index[big] + step for step in (1, width, width+1)])
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
    np.copyto(current, upcoming, where=tiled(8, current.shape) < round(amount*64))
    return current


# Decoded loops -----------------------------------------------------------

@dataclass
class Loop:
    frames: np.ndarray            # (count, height, width) RGB565
    ends: list[int]               # when each frame ends, in ms from the loop start
    glow: np.ndarray | None = None  # flat indices of pixels that glow at night
    moonlit: bool = True          # False for a scene painted at night, which stays as it is
    stars: Stars | None = None
    clear: np.ndarray | None = None  # (frame, star): nothing is passing in front of the star

    def __post_init__(self):
        if self.glow is not None:
            rows, columns = np.divmod(self.glow, self.frames.shape[2])
            self.glow_cell = rows % 2*2 + columns % 2

    def frame_at(self, seconds: float) -> int:
        return bisect.bisect_right(self.ends, int(seconds*1000) % self.ends[-1])

    def frame(self, seconds: float) -> np.ndarray:
        return self.frames[self.frame_at(seconds)]

    def lit(self, seconds: float, light: tuple[float, bool]) -> np.ndarray:
        index = self.frame_at(seconds)
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
        strength = star_strength(light[0])
        if self.stars is not None and strength:
            colours = self.stars.light(seconds, strength)
            if self.clear is not None:
                colours *= self.clear[index][:, None]
            flat[self.stars.index] = brighter565(flat[self.stars.index], colours)
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
            stars = find_stars(pixels)
        packed[index] = pack565(pixels)
        total += duration
        ends.append(total)
    mask = read_glow(glow_path(name, path), size)
    glow = None if mask is None else np.flatnonzero(np.asarray(mask) > 127)
    clear = None
    if stars is not None and count > 1:
        # A star shows only while its patch of sky looks as it did in the first frame,
        # so balloons, birds and snow pass in front of it.
        sky = packed.reshape(count, -1)[:, stars.index]
        clear = same_colour565(sky, sky[0])
    return Loop(packed, ends, glow, name not in PAINTED_AT_NIGHT, stars, clear)


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
    login prompt. Started over SSH, only hide the visible console's cursor."""

    def __init__(self):
        self.vt = self.cursor = None
        try:
            name = os.ttyname(0)
        except OSError:
            name = ""
        if name.startswith("/dev/tty") and name[8:].isdigit():
            try:
                fcntl.ioctl(0, KDSETMODE, KD_GRAPHICS)
                self.vt = name[5:]
                return
            except OSError:
                pass
        try:
            self.cursor = os.open("/dev/" + self.active(), os.O_WRONLY | os.O_NOCTTY)
            os.write(self.cursor, b"\x1b[?25l")
        except OSError:
            self.cursor = None

    @staticmethod
    def active() -> str:
        return Path("/sys/class/tty/tty0/active").read_text().strip()

    def visible(self) -> bool:
        return self.vt is None or self.active() == self.vt

    def restore(self) -> None:
        if self.vt is not None:
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
        names, started, shown = list(library), time.time(), None
        while running:
            tick = time.monotonic()
            now = time.time()
            if not console.visible():
                time.sleep(1)
                continue
            current, upcoming, remaining = schedule(names, settings, started, now)
            if current != shown:
                print(f"{time.strftime('%H:%M')} {current}", flush=True)
                shown = current
            cache.keep({current, upcoming})
            light = settings.light(now)
            image = cache.get(current).lit(tick, light)
            # Decode the next scene once the current one is ready, not alongside it.
            nearby = cache.ready(upcoming) if current in cache.loops else None
            if remaining < settings.crossfade and upcoming != current and nearby is not None:
                image = dissolve(image, nearby.lit(tick, light), 1 - remaining/settings.crossfade)
            fb.show(image)
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
    strength = star_strength(light[0])
    _, frames = read_frames(path, SCREEN, limit)
    elapsed = 0
    for index, (image, duration) in enumerate(frames):
        pixels = np.asarray(image)
        if index == 0:
            stars = find_stars(pixels)
            sky = None if stars is None else pixels.reshape(-1, 3)[stars.index].astype(np.int16)
        moonlit = name not in PAINTED_AT_NIGHT
        lit = relight(pixels, *light, moonlit=moonlit)
        if lamps is not None:
            lit[lamps] = relight(pixels[lamps], *light, True, moonlit)
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
    except ValueError as error:
        parser.error(str(error))
    library = scene_library(args.scenes_dir)
    for name in (args.scene, args.start_scene):
        if name and name not in library:
            parser.error(f"unknown scene {name!r}; choose from {', '.join(library)}")
    settings = Settings(args.seconds_per_scene, args.crossfade, args.fps, args.day_seconds,
                        night, args.start_scene)

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

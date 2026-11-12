#!/usr/bin/env python3
"""Painted, animated landscapes for a Linux framebuffer picture frame.

Show the playlist on the display (the systemd service runs this at boot):
    python3 pixel_frame.py

Export a still or a shareable GIF, lit for a time of day:
    python3 pixel_frame.py --scene pond --at 20:30 --export pond.png
    python3 pixel_frame.py --scene pond --at 20:30 --export-gif pond.gif
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
ASSET_DIR = ROOT / "assets"
SCENES_DIR = ROOT / "scenes"
SCENES = ("woodland", "meadow", "pond", "autumn", "winter",
          "christmas", "tornado", "tsunami", "balloons", "sakura")
# Months in which a scene joins the playlist. Unlisted scenes play all year.
SEASONS = {"sakura": {3, 4, 5}, "autumn": {9, 10, 11},
           "winter": {12, 1, 2}, "christmas": {12}}
MEDIA = {".webp", ".gif", ".png", ".jpg", ".jpeg"}
SCREEN = (1920, 1080)
SECONDS_PER_SCENE = 720
LIGHT_STEPS = 240
MAX_FRAMES = 480  # Longer drop-in animations are sampled to bound memory.

FBIOGET_VSCREENINFO, FBIOBLANK = 0x4600, 0x4611
FB_BLANK_UNBLANK, FB_BLANK_POWERDOWN = 0, 4
KDSETMODE, KD_TEXT, KD_GRAPHICS = 0x4B3A, 0, 1


# Scenes and the playlist -------------------------------------------------

def scene_library(scenes_dir: Path = SCENES_DIR) -> dict[str, Path]:
    """Built-in loops, then any still or animation dropped into scenes_dir."""
    library = {name: ASSET_DIR / f"{name}-animated.webp" for name in SCENES}
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


# Daylight ----------------------------------------------------------------

def sun_events(day: dt.date, latitude: float, longitude: float):
    """Sunrise, solar noon and sunset in UTC, after NOAA's approximation.

    In polar day or night the hour angle is clamped, so the day collapses into
    one long day or night with a brief dawn and dusk."""
    g = 2*math.pi*(day.timetuple().tm_yday - 1)/365
    equation = 229.18*(.000075 + .001868*math.cos(g) - .032077*math.sin(g)
                       - .014615*math.cos(2*g) - .040849*math.sin(2*g))
    declination = (.006918 - .399912*math.cos(g) + .070257*math.sin(g)
                   - .006758*math.cos(2*g) + .000907*math.sin(2*g)
                   - .002697*math.cos(3*g) + .00148*math.sin(3*g))
    lat = math.radians(latitude)
    cos_hour = (math.cos(math.radians(90.833))/(math.cos(lat)*math.cos(declination))
                - math.tan(lat)*math.tan(declination))
    half = dt.timedelta(minutes=4*math.degrees(math.acos(max(-1, min(1, cos_hour)))))
    noon = (dt.datetime.combine(day, dt.time(), dt.timezone.utc)
            + dt.timedelta(minutes=720 - 4*longitude - equation))
    return noon - half, noon, noon + half


def sun_phase(when: dt.datetime, latitude: float, longitude: float) -> float:
    """Place real time on the lighting cycle: sunrise 0, noon .25, sunset .5, midnight .75."""
    today = when.astimezone(dt.timezone.utc).date()
    days = [sun_events(today + dt.timedelta(days=offset), latitude, longitude)
            for offset in (-1, 0, 1, 2)]
    keys = []
    for index, (rise, noon, sunset) in enumerate(days):
        keys += [(rise, index), (noon, index+.25), (sunset, index+.5)]
        if index+1 < len(days):
            keys.append((sunset + (days[index+1][0]-sunset)/2, index+.75))
    for (start, first), (end, last) in zip(keys, keys[1:]):
        if start <= when < end:
            return (first + (last-first)*(when-start)/(end-start)) % 1
    raise ValueError(f"{when} is outside the computed days")


def zone_location() -> tuple[float, float] | None:
    """Coordinates of the system time zone's principal city, from tzdata."""
    try:
        zone = Path("/etc/timezone").read_text().strip()
    except OSError:
        zone = os.path.realpath("/etc/localtime").partition("zoneinfo/")[2]
    try:
        table = Path("/usr/share/zoneinfo/zone1970.tab").read_text()
    except OSError:
        return None
    for line in table.splitlines():
        fields = line.split("\t")
        if not line.startswith("#") and len(fields) > 2 and fields[2] == zone:
            return iso6709(fields[1])
    return None


def iso6709(text: str) -> tuple[float, float]:
    """Parse tzdata coordinates such as +5215+02100 or -332318+0703716."""
    split = max(text.rfind("+"), text.rfind("-"))

    def degrees(part: str, width: int) -> float:
        digits = part[1:]
        value = (int(digits[:width]) + int(digits[width:width+2])/60
                 + int(digits[width+2:] or 0)/3600)
        return -value if part[0] == "-" else value
    return degrees(text[:split], 2), degrees(text[split:], 3)


def quiet_state(minute: int, quiet: tuple[int, int] | None, dim_minutes: int = 60):
    """(blank, dim) for a minute of the day.

    The picture dims during the hour before quiet hours and the display is
    switched off during them."""
    if quiet is None:
        return False, 1.0
    start, end = quiet
    if (minute-start) % 1440 < (end-start) % 1440:
        return True, 1.0
    before = (start-minute) % 1440
    if before < dim_minutes:
        return False, 1 - .4*(1 - before/dim_minutes)
    return False, 1.0


def parse_quiet(text: str) -> tuple[int, int] | None:
    if text.lower() in ("off", "none", ""):
        return None
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"quiet hours look like 23:00-07:00, not {text}")
    return clock_minutes(parts[0]), clock_minutes(parts[1])


def clock_minutes(text: str) -> int:
    hours, minutes = text.strip().split(":")
    if not (0 <= int(hours) < 24 and 0 <= int(minutes) < 60):
        raise ValueError(f"not a time of day: {text}")
    return int(hours)*60 + int(minutes)


# Lighting ----------------------------------------------------------------

def daylight(phase: float) -> float:
    return (1 + math.cos((phase-.25)*2*math.pi)) / 2


def lighting_parameters(phase: float, glow: bool = False):
    """Brightness and tint at a point in the day.

    Glowing pixels (windows, lanterns) keep their brightness and warm up after
    dark instead of taking on the blue night tint."""
    day = daylight(phase)
    brightness = .58 + .42*day
    dawn_distance = min(phase, 1-phase)
    weights = [.08*math.exp(-(dawn_distance/.10)**2),
               .12*math.exp(-((phase-.52)/.10)**2),
               .24*(1-day)**2]
    colors = [(255, 181, 145), (244, 145, 92), (38, 61, 112)]
    if glow:
        night = 1 - day
        brightness += (1.1-brightness)*night
        weights[2] = .14*night
        colors[2] = (255, 196, 120)
    strength = sum(weights)
    tint = tuple(sum(w*color[c] for w, color in zip(weights, colors))/strength
                 for c in range(3))
    return brightness, tint, strength


def channel_lut(phase: float, dim: float = 1.0, glow: bool = False) -> np.ndarray:
    """A 3x256 table that lights each 8-bit channel."""
    brightness, tint, strength = lighting_parameters(phase, glow)
    values = np.clip(np.arange(256, dtype=np.float32)*brightness, 0, 255)
    lut = np.array([(values*(1-strength) + c*strength)*dim for c in tint])
    return np.clip(lut, 0, 255).astype(np.uint8)


_CODES = np.arange(65536, dtype=np.uint32)
_R5, _G6, _B5 = _CODES >> 11, _CODES >> 5 & 63, _CODES & 31
_RGB8 = (_R5 << 3 | _R5 >> 2, _G6 << 2 | _G6 >> 4, _B5 << 3 | _B5 >> 2)
_LUT565: dict[tuple, np.ndarray] = {}


def lut565(step: int, dim: float = 1.0, glow: bool = False) -> np.ndarray:
    """Light an RGB565 pixel with one lookup: 65536 entries, one per colour."""
    key = (step, dim, glow)
    lut = _LUT565.get(key)
    if lut is None:
        if len(_LUT565) > 32:
            _LUT565.clear()
        channels = channel_lut(step/LIGHT_STEPS, dim, glow).astype(np.uint32)
        r, g, b = (channels[c][_RGB8[c]] for c in range(3))
        # Round to the nearest RGB565 level as it expands back to 8 bits.
        lut = ((r*31+127)//255 << 11 | (g*63+127)//255 << 5 | (b*31+127)//255).astype(np.uint16)
        _LUT565[key] = lut
    return lut


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

    def frame(self, seconds: float) -> np.ndarray:
        return self.frames[bisect.bisect_right(self.ends, int(seconds*1000) % self.ends[-1])]

    def lit(self, seconds: float, step: int, dim: float) -> np.ndarray:
        source = self.frame(seconds)
        image = lut565(step, dim)[source]
        if self.glow is not None:
            image.reshape(-1)[self.glow] = lut565(step, dim, True)[source.reshape(-1)[self.glow]]
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
        packed[index] = pack565(np.asarray(image))
        total += duration
        ends.append(total)
    mask = read_glow(glow_path(name, path), size)
    glow = None if mask is None else np.flatnonzero(np.asarray(mask) > 127)
    return Loop(packed, ends, glow)


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
        self.blanked = False

    def show(self, image: np.ndarray) -> None:
        np.copyto(self.screen, image)

    def blank(self, off: bool) -> None:
        if off == self.blanked:
            return
        self.blanked = off
        try:
            fcntl.ioctl(self.file, FBIOBLANK, FB_BLANK_POWERDOWN if off else FB_BLANK_UNBLANK)
        except OSError:
            if off:
                self.screen[:] = 0

    def close(self) -> None:
        self.blank(False)
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
    day_seconds: float | None = None  # a fast demo cycle instead of the real sun
    location: tuple[float, float] = (0.0, 0.0)
    quiet: tuple[int, int] | None = None
    start_scene: str | None = None

    def phase(self, now: float) -> float:
        if self.day_seconds:
            return (now/self.day_seconds) % 1
        return sun_phase(dt.datetime.fromtimestamp(now, dt.timezone.utc), *self.location)


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
            local = time.localtime(now)
            off, dim = quiet_state(local.tm_hour*60 + local.tm_min, settings.quiet)
            fb.blank(off)
            if off or not console.visible():
                time.sleep(1)
                continue
            current, upcoming, remaining = schedule(names, settings, started, now)
            if current != shown:
                print(f"{time.strftime('%H:%M')} {current}", flush=True)
                shown = current
            cache.keep({current, upcoming})
            step = int(settings.phase(now)*LIGHT_STEPS) % LIGHT_STEPS
            dim = round(dim*50)/50
            image = cache.get(current).lit(tick, step, dim)
            # Decode the next scene once the current one is ready, not alongside it.
            nearby = cache.ready(upcoming) if current in cache.loops else None
            if remaining < settings.crossfade and upcoming != current and nearby is not None:
                image = dissolve(image, nearby.lit(tick, step, dim), 1 - remaining/settings.crossfade)
            fb.show(image)
            time.sleep(max(0, 1/settings.fps - (time.monotonic()-tick)))
    finally:
        try:
            fb.close()
        finally:
            console.restore()


# Exports -----------------------------------------------------------------

def lit_image(image: Image.Image, phase: float, glow: Image.Image | None) -> Image.Image:
    """Light an RGB image at full 8-bit precision for exports."""
    lit = image.point(channel_lut(phase).reshape(-1).tolist())
    if glow is not None:
        lit = Image.composite(image.point(channel_lut(phase, glow=True).reshape(-1).tolist()), lit, glow)
    return lit


def export_still(name: str, path: Path, phase: float, output: Path) -> None:
    _, frames = read_frames(path, SCREEN, 1)
    image, _ = next(frames)
    lit_image(image, phase, read_glow(glow_path(name, path), SCREEN)).save(output)


def export_gif(name: str, path: Path, phase: float, output: Path, width: int = 640) -> None:
    """A looping GIF small enough to send in a chat.

    Every frame shares one palette and pixels that did not change are
    transparent, so the still parts of the painting are stored once."""
    size = (width, round(width*SCREEN[1]/SCREEN[0]))
    glow = read_glow(glow_path(name, path), SCREEN)
    _, frames = read_frames(path, SCREEN)
    dither = (tiled(8, size[::-1]).astype(np.int16) - 32)//4  # -8..7, the same every frame
    smalls, durations = [], []
    for image, duration in frames:
        small = np.asarray(lit_image(image, phase, glow).resize(size, Image.Resampling.LANCZOS))
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
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR,
                        help="folder of your own stills, GIFs or WebPs to add to the playlist")
    parser.add_argument("--start-scene", help="scene to show first; the clock-based playlist follows")
    parser.add_argument("--seconds-per-scene", type=float, default=SECONDS_PER_SCENE,
                        help="seconds per scene (default: 720)")
    parser.add_argument("--crossfade", type=float, default=3, help="seconds to fade between scenes (default: 3)")
    parser.add_argument("--fps", type=float, default=8, help="frames per second (default: 8, as the loops)")
    parser.add_argument("--day-seconds", type=float,
                        help="run a fast day/night cycle of this length instead of following the sun")
    parser.add_argument("--latitude", type=float, help="for sunrise and sunset (default: the time zone's city)")
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--quiet-hours", default="23:00-07:00",
                        help="switch the display off between these times, or 'off' (default: 23:00-07:00)")
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
    if (args.latitude is None) != (args.longitude is None):
        parser.error("give both --latitude and --longitude")
    try:
        quiet = parse_quiet(args.quiet_hours)
        at = clock_minutes(args.at) if args.at else None
    except ValueError as error:
        parser.error(str(error))
    library = scene_library(args.scenes_dir)
    for name in (args.scene, args.start_scene):
        if name and name not in library:
            parser.error(f"unknown scene {name!r}; choose from {', '.join(library)}")
    if args.latitude is not None:
        location = (args.latitude, args.longitude)
    else:
        # Without tzdata, an equator-like day centred on local noon.
        offset = dt.datetime.now().astimezone().utcoffset().total_seconds()/3600
        location = zone_location() or (0.0, 15*offset)
    settings = Settings(args.seconds_per_scene, args.crossfade, args.fps, args.day_seconds,
                        location, quiet, args.start_scene)

    if args.export or args.export_gif:
        name = args.scene or SCENES[0]
        when = dt.datetime.now().astimezone()
        if at is not None:
            when = when.replace(hour=at//60, minute=at % 60, second=0, microsecond=0)
        phase = settings.phase(when.timestamp())
        if args.export:
            export_still(name, library[name], phase, args.export)
            print(f"Wrote {args.export}")
        if args.export_gif:
            export_gif(name, library[name], phase, args.export_gif, args.gif_width)
            print(f"Wrote {args.export_gif} ({args.export_gif.stat().st_size/1e6:.1f} MB)")
        return

    run(args.device, library, settings)


if __name__ == "__main__":
    main()

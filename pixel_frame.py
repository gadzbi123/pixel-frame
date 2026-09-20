#!/usr/bin/env python3
"""A dependency-free generative pixel-art frame for Raspberry Pi.

Render a still image:
    python3 pixel_frame.py --export scene.ppm

Run it on the Pi framebuffer (Ctrl-C restores the previous screen):
    sudo python3 pixel_frame.py --framebuffer
"""

from __future__ import annotations

import argparse
import gc
import math
import mmap
import os
import random
import signal
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # The original low-resolution renderer still has a fallback.
    np = None

W, H = 200, 120
SCENES = ("woodland", "meadow", "pond", "autumn", "winter",
          "christmas", "tornado", "tsunami", "balloons", "sakura")
SECONDS_PER_SCENE = 720
ASSET_DIR = Path(__file__).with_name("assets")
ILLUSTRATED_ASSETS = {name: ASSET_DIR / f"{name}-illustrated.png" for name in SCENES}
# The GIF is kept as the requested shareable asset.  The equivalent WebP is
# used on the Pi because it is much smaller and faster to decode.
ANIMATED_ASSETS = {name: ASSET_DIR / f"{name}-animated.webp" for name in SCENES}
ANIMATED_FRAME_MS = {name: 125 for name in SCENES}
_ILLUSTRATED_CACHE = None
_ANIMATED_CACHE = {}
_ANIMATED_LOADING = set()
_ANIMATED_LOCK = threading.Lock()
_ACTIVE_ANIMATED_THEME = None
_LIGHTING_LUTS = {}


def mix(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * amount) for x, y in zip(a, b))


class Canvas:
    def __init__(self, width: int = W, height: int = H):
        self.width, self.height = width, height
        self.data = bytearray(width * height * 3)

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.data[i] = color[0]
            self.data[i + 1] = color[1]
            self.data[i + 2] = color[2]

    def rect(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x), min(self.width, x + width)
        y0, y1 = max(0, y), min(self.height, y + height)
        row = bytes(color) * max(0, x1 - x0)
        for yy in range(y0, y1):
            i = (yy * self.width + x0) * 3
            self.data[i : i + len(row)] = row

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                return
            twice = 2 * err
            if twice >= dy:
                err += dy
                x0 += sx
            if twice <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.pixel(x, y, color)


def illustrated_variants(theme):
    """Keep a lighting atlas only for the current scene to bound memory use."""
    global _ILLUSTRATED_CACHE
    if _ILLUSTRATED_CACHE is not None and _ILLUSTRATED_CACHE[0] == theme:
        return _ILLUSTRATED_CACHE[1]
    _ILLUSTRATED_CACHE = None
    gc.collect()
    try:
        from PIL import Image, ImageEnhance
    except ImportError as exc:
        raise RuntimeError("Detailed landscapes require the python3-pil package") from exc
    base = Image.open(ILLUSTRATED_ASSETS[theme]).convert("RGB")
    if base.size != (1920, 1080):
        raise RuntimeError(f"Expected a 1920x1080 {theme} illustration, got {base.size}")
    frames = []
    for step in range(12):
        phase = step / 12
        daylight = (1 + math.cos((phase-.25)*2*math.pi)) / 2
        brightness = .58 + .42*daylight
        image = ImageEnhance.Brightness(base).enhance(brightness)
        if phase < .16 or phase > .91:
            tint, strength = (255, 181, 145), .08
        elif .43 < phase < .63:
            tint, strength = (244, 145, 92), .12
        elif .60 <= phase <= .91:
            tint, strength = (38, 61, 112), .24
        else:
            tint, strength = (255, 255, 255), 0
        if strength:
            image = Image.blend(image, Image.new("RGB", image.size, tint), strength)
        frames.append(image.tobytes())
    _ILLUSTRATED_CACHE = (theme, tuple(frames))
    return _ILLUSTRATED_CACHE[1]


def load_animated_asset(theme):
    """Decode one animation once; scenes are preloaded before their playlist slot."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Animated landscapes require the python3-pil package") from exc
    with _ANIMATED_LOCK:
        if theme in _ANIMATED_CACHE:
            return _ANIMATED_CACHE[theme]
        if theme in _ANIMATED_LOADING:
            return None
        _ANIMATED_LOADING.add(theme)
    try:
        path = ANIMATED_ASSETS[theme]
        animation = Image.open(path)
        duration = ANIMATED_FRAME_MS.get(theme) or animation.info.get("duration") or 125
        decoded = []
        for index in range(animation.n_frames):
            animation.seek(index)
            frame = animation.convert("RGB")
            if frame.size != (1920, 1080):
                resampling = getattr(Image, "Resampling", Image)
                frame = frame.resize((1920, 1080), resampling.LANCZOS)
            decoded.append(frame.tobytes())
        animation.close()
        entry = [tuple(decoded), duration, {}, set()]
        with _ANIMATED_LOCK:
            keep = {theme}
            if _ACTIVE_ANIMATED_THEME is not None:
                keep.add(_ACTIVE_ANIMATED_THEME)
            for cached_theme in tuple(_ANIMATED_CACHE):
                if cached_theme not in keep:
                    del _ANIMATED_CACHE[cached_theme]
            _ANIMATED_CACHE[theme] = entry
        return entry
    finally:
        with _ANIMATED_LOCK:
            _ANIMATED_LOADING.discard(theme)


def animation_lighting_lut(light_step):
    lut = _LIGHTING_LUTS.get(light_step)
    if lut is not None:
        return lut
    lighting_phase = light_step/12
    daylight = (1 + math.cos((lighting_phase-.25)*2*math.pi)) / 2
    brightness = .58 + .42*daylight
    if lighting_phase < .16 or lighting_phase > .91:
        tint, strength = (255, 181, 145), .08
    elif .43 < lighting_phase < .63:
        tint, strength = (244, 145, 92), .12
    elif .60 <= lighting_phase <= .91:
        tint, strength = (38, 61, 112), .24
    else:
        tint, strength = (255, 255, 255), 0
    values = np.arange(256, dtype=np.float32)
    bright = np.clip(values*brightness, 0, 255)
    lut = np.empty((3, 256), dtype=np.uint8)
    for channel in range(3):
        lut[channel] = bright*(1-strength)+tint[channel]*strength
    _LIGHTING_LUTS[light_step] = lut
    return lut


def compute_lit_loop(theme, light_step):
    entry = _ANIMATED_CACHE[theme]
    try:
        if np is None:
            from PIL import Image
            phase = light_step/12
            lit = tuple(
                apply_animation_lighting(Image.frombytes("RGB", (1920,1080), raw), phase)
                for raw in entry[0]
            )
        else:
            lut = animation_lighting_lut(light_step)
            lit_frames = []
            for raw in entry[0]:
                source = np.frombuffer(raw, dtype=np.uint8).reshape(-1,3)
                result = np.empty_like(source)
                result[:,0] = lut[0,source[:,0]]
                result[:,1] = lut[1,source[:,1]]
                result[:,2] = lut[2,source[:,2]]
                lit_frames.append(result.tobytes())
            lit = tuple(lit_frames)
        with _ANIMATED_LOCK:
            entry[2][light_step] = lit
    finally:
        with _ANIMATED_LOCK:
            entry[3].discard(light_step)


def apply_animation_lighting(image, phase):
    from PIL import Image, ImageEnhance
    daylight = (1 + math.cos((phase-.25)*2*math.pi)) / 2
    brightness = .58 + .42*daylight
    image = ImageEnhance.Brightness(image).enhance(brightness)
    if phase < .16 or phase > .91:
        tint, strength = (255,181,145), .08
    elif .43 < phase < .63:
        tint, strength = (244,145,92), .12
    elif .60 <= phase <= .91:
        tint, strength = (38,61,112), .24
    else:
        tint, strength = (255,255,255), 0
    if strength:
        image = Image.blend(image, Image.new("RGB", image.size, tint), strength)
    return image.tobytes()


def request_lit_loop(theme, light_step):
    entry = _ANIMATED_CACHE[theme]
    with _ANIMATED_LOCK:
        if light_step in entry[2] or light_step in entry[3]:
            return
        entry[3].add(light_step)
    threading.Thread(target=compute_lit_loop, args=(theme,light_step), daemon=True).start()


def animated_illustration(theme, t, phase):
    """Return one pre-lit frame from a decoded high-resolution animation."""
    global _ACTIVE_ANIMATED_THEME
    entry = _ANIMATED_CACHE.get(theme) or load_animated_asset(theme)
    # A background preload of this same theme can only occur around a manual
    # scene jump. Waiting here avoids decoding the asset twice.
    while entry is None:
        time.sleep(.01)
        entry = _ANIMATED_CACHE.get(theme) or load_animated_asset(theme)
    if _ACTIVE_ANIMATED_THEME != theme:
        previous = _ANIMATED_CACHE.get(_ACTIVE_ANIMATED_THEME)
        if previous is not None:
            previous[2].clear()
        _ACTIVE_ANIMATED_THEME = theme
    frames, duration, lit_loops, _ = entry
    light_step = int(phase*12) % 12
    request_lit_loop(theme, light_step)
    while light_step not in lit_loops:
        time.sleep(.01)
    next_step = (light_step+1) % 12
    request_lit_loop(theme, next_step)
    for cached_step in tuple(lit_loops):
        if cached_step not in (light_step, next_step):
            del lit_loops[cached_step]
    frame_index = int(t*1000/duration) % len(frames)
    return lit_loops[light_step][frame_index]


def preload_animation(theme):
    if theme in ANIMATED_ASSETS and ANIMATED_ASSETS[theme].exists():
        def preload():
            entry = load_animated_asset(theme)
            if entry is not None:
                request_lit_loop(theme, 0)
        threading.Thread(target=preload, daemon=True).start()


def draw_glints(c, t, y0, y1, center, near_half, far_half, color, count=42):
    for n in range(count):
        y = y0+(n*67+round(t*13))%max(1,y1-y0)
        progress = (y-y0)/max(1,y1-y0)
        half = round(far_half+(near_half-far_half)*progress)
        x = center-half+(n*157+round(t*21))%max(1,half*2)
        c.line(x,y,min(center+half,x+8+n%24),y,color)


def draw_illustrated_motion(c, theme, t, phase):
    night = .60 < phase < .91
    water_light = (202,235,232) if not night else (104,137,174)
    if theme == 'woodland':
        # The river bends toward the lower-right rather than covering the bank.
        for n in range(42):
            y=600+(n*67+round(t*13))%440
            progress=(y-600)/440
            left=round(820+650*progress); right=round(1470+450*progress)
            x=left+(n*157+round(t*21))%max(1,right-left)
            c.line(x,y,min(right,x+8+n%24),y,water_light)
    elif theme == 'meadow':
        draw_glints(c,t,540,850,1120,600,120,water_light,34)
        for n in range(12):
            x=(n*179+round(t*(18+n%3)))%1920
            y=420+(n*83+round(24*math.sin(t+n)))%360
            col=((242,156,73),(227,121,166),(151,186,225))[n%3]
            c.circle(x-4,y,4,col); c.circle(x+4,y,4,col); c.pixel(x,y,(48,43,42))
    elif theme == 'pond':
        draw_glints(c,t,390,970,960,850,280,water_light,65)
        for n in range(9):
            x=250+(n*211+round(t*35))%1450; y=520+(n*97)%390
            radius=4+(round(t*5)+n)%13
            c.line(x-radius,y,x+radius,y,(185,224,218))
    elif theme == 'autumn':
        for n in range(58):
            x=round((n*223+t*(22+n%5)+30*math.sin(t+n))%2000)-40
            y=round((n*131+t*(31+n%4))%1160)-40
            col=((206,92,48),(235,147,54),(169,64,43))[n%3]
            c.rect(x,y,5+n%4,3,col)
    elif theme in ('winter','christmas'):
        snow=(235,243,247)
        for n in range(130):
            x=round((n*277+t*(10+n%5)+18*math.sin(t+n))%1960)-20
            y=round((n*149+t*(25+n%7))%1120)-20
            c.circle(x,y,1+n%3,snow)
    elif theme == 'tornado':
        center_x=1080+round(80*math.sin(t*.25))
        for n in range(45):
            angle=t*3+n*.83; radius=120+(n%8)*38
            x=round(center_x+radius*math.cos(angle)); y=570+round(radius*.55*math.sin(angle))
            c.line(x,y,x+12*round(math.cos(angle)),y+7*round(math.sin(angle)),
                   ((119,87,61),(180,151,104),(76,70,67))[n%3])
        if 19.5 < t%48 < 19.8:
            c.line(1550,80,1490,210,(248,236,190)); c.line(1490,210,1530,205,(248,236,190))
    elif theme == 'tsunami':
        draw_glints(c,t,690,1030,980,920,520,water_light,60)
        for n in range(75):
            x=(n*251+round(t*43))%1920; y=650+(n*79+round(t*19))%410
            radius=2+(n+round(t*3))%7
            c.circle(x,y,radius,(226,246,240))
            c.circle(x,y,max(0,radius-2),(84,170,188))
    elif theme == 'balloons':
        for n in range(9):
            x=(n*307+round(t*11))%1940-10; y=210+(n*59)%260
            c.line(x-8,y,x,y-4,(62,69,68)); c.line(x,y-4,x+8,y,(62,69,68))
    elif theme == 'sakura':
        draw_glints(c,t,610,1050,1010,760,180,water_light,55)
        petal=(250,188,205) if not night else (148,128,168)
        for n in range(70):
            x=round((n*257+t*(13+n%5)+24*math.sin(t*.6+n))%1980)-30
            y=round((n*137+t*(24+n%4))%1140)-30
            c.circle(x,y,2+n%3,petal); c.pixel(x+5,y+2,petal)
    if night and theme in ('woodland','pond','sakura'):
        for n in range(24):
            if math.sin(t*2+n)>.15:
                c.circle(150+(n*347)%1620,610+(n*113)%350,2,(236,226,128))


def render_illustrated(theme, t, cycle):
    phase = (t/cycle)%1
    if theme in ANIMATED_ASSETS and ANIMATED_ASSETS[theme].exists():
        c = Canvas(1920,1080)
        c.data[:] = animated_illustration(theme, t, phase)
        return c
    frames = illustrated_variants(theme)
    c = Canvas(1920,1080)
    c.data[:] = frames[int(phase*len(frames))%len(frames)]
    draw_illustrated_motion(c,theme,t,phase)
    return c


PALETTES = (
    {"name": "Dawn", "sky_top": (39, 53, 108), "sky_bottom": (244, 142, 113), "sun": (255, 224, 142), "far": (80, 74, 126), "near": (45, 50, 87), "land": (28, 58, 68), "water": (46, 101, 134), "cloud": (247, 174, 151), "light": (255, 220, 123)},
    {"name": "Golden hour", "sky_top": (38, 91, 151), "sky_bottom": (248, 179, 91), "sun": (255, 234, 157), "far": (79, 107, 131), "near": (45, 76, 99), "land": (27, 77, 73), "water": (36, 117, 146), "cloud": (255, 205, 133), "light": (255, 235, 150)},
    {"name": "Midnight", "sky_top": (9, 17, 48), "sky_bottom": (38, 48, 90), "sun": (224, 231, 220), "far": (39, 52, 92), "near": (21, 35, 63), "land": (12, 44, 48), "water": (20, 66, 103), "cloud": (51, 60, 98), "light": (243, 211, 120)},
)


@dataclass
class Scene:
    seed: int
    palette: dict
    sun_x: int
    sun_y: int
    sun_r: int
    water_y: int
    mountains: list[tuple[int, int, int]]
    trees: list[tuple[int, int, int]]
    clouds: list[tuple[int, int, int]]
    stars: list[tuple[int, int, int]]
    cabin_x: int


def make_scene(seed: int | None = None) -> Scene:
    rng = random.Random(seed if seed is not None else time.time_ns())
    palette = rng.choice(PALETTES)
    water_y = rng.randint(61, 69)
    mountains = [(rng.randint(-20, 15) + i * rng.randint(21, 34), rng.randint(34, 56), rng.randint(15, 29)) for i in range(7)]
    trees = [(rng.randrange(0, W), rng.randrange(water_y - 4, H - 4), rng.randint(7, 17)) for _ in range(rng.randint(13, 21))]
    clouds = [(rng.randrange(-20, W), rng.randint(9, 34), rng.randint(7, 17)) for _ in range(rng.randint(2, 4))]
    stars = [(rng.randrange(W), rng.randrange(3, 48), rng.randint(0, 2)) for _ in range(70)]
    return Scene(rng.randrange(2**31), palette, rng.randint(24, 132), rng.randint(19, 34), rng.randint(5, 9), water_y, mountains, trees, clouds, stars, rng.randint(30, 126))


def draw_mountain(c: Canvas, x: int, peak_y: int, half_width: int, base_y: int, color: tuple[int, int, int]) -> None:
    for xx in range(x - half_width, x + half_width + 1):
        slope = abs(xx - x) / half_width
        top = round(peak_y + (base_y - peak_y) * slope)
        # A vertical rectangle delegates the repeated write to bytearray rather
        # than calling Canvas.pixel once per landscape pixel.
        c.rect(xx, top, 1, base_y - top + 1, color)


def draw_tree(c: Canvas, x: int, bottom: int, size: int, trunk: tuple[int, int, int], needles: tuple[int, int, int]) -> None:
    c.rect(x, bottom - size // 3, 2, size // 3, trunk)
    for level in range(3):
        y = bottom - size + level * (size // 4)
        half = 2 + level * (size // 7)
        for row in range(size // 3 + 1):
            w = max(1, half - row // 2)
            c.rect(x - w, y + row, w * 2 + 1, 1, needles)


# Hand-authored, code-native sprites stay crisp at four physical pixels per cell.
FOX = [
    "  OO    OO          ",
    " OWWO  OWWO         ",
    " OOOOOOOOOO         ",
    "OOOKOOOOKOOO        ",
    " OOWWWWWWOO         ",
    "  WWWKKWWWOOOOOO    ",
    "   OOOOOOOOOOOOOO  W",
    "   OOOOOOOOOOOOOO WW",
    "    OOOOOOOOOOOOOOWW",
    "    OO OO    OO OO  ",
    "    KK KK    KK KK  ",
]
RABBIT = [
    " WW  WW ", " WP  PW ", " WW  WW ", " WWWWWW ",
    "WKWWWWKW", " WWWPWW ", " WWWWWW ", "WWWWWWWW", " WW  WW ",
]
SLEEPING_FOX = [
    "          OO OO          ",
    "         OOOOOOO         ",
    "        OOKCCCCK         ",
    "       OOOCCCCOO         ",
    "   OOOOOOOOOOOOOOOCCC    ",
    " OOOOOOOOOOOOOOOOOOCCCC  ",
    "OOOOOOOOOOOOOOOOOOOOOCCC ",
    "OOOCCOOOOOOOOOOOOOOOOCCC ",
    " OOCCCCOOOOOOOOOOOOOOCC  ",
    "  CCCCCOOOOOOOOOOOOOO    ",
    "    OOOOOOOOOOOOOOO      ",
    "       OOOOOOOOO         ",
]
DEER = [
    "               A A ",
    "             A A A ",
    "              A A  ",
    "             BBBB  ",
    "            BBKBBB ",
    "            BBB BB ",
    "             BBB   ",
    "   BBBBBBBBBBBBB   ",
    " BBBBBBBBBBBBBBB   ",
    " BBBBBBBBBBBBBBB   ",
    "  BBBBBBBBBBBB     ",
    "   BB  BB   BB     ",
    "   BB  BB   BB     ",
    "   KK  KK   KK     ",
]
COW = [
    " HH                ",
    "HBBH               ",
    "BKBBB              ",
    " BBBBWWBBBBBBBB    ",
    "  BBBWWBBBBBBBBB   ",
    "  BBBBBBBBBBBBBB   ",
    "    BB  BB   BB    ",
    "    BB  BB   BB    ",
    "    KK  KK   KK    ",
]
FROG = [
    "        GG ",
    "    GGGGKG ",
    " GGGGGGGG  ",
    "GGGGGGGG   ",
    " GG  GG    ",
    "LLLLLLLLLL ",
]


def sprite(c, rows, x, y, colors, flip=False):
    width = max(map(len, rows))
    for yy, row in enumerate(rows):
        row = row.ljust(width)
        for xx, key in enumerate(row[::-1] if flip else row):
            if key in colors:
                c.pixel(x+xx, y+yy, colors[key])


def celestial_position(phase):
    progress = (phase * 2) % 1
    return round(25 + 150 * progress), round(91 - 78 * math.sin(progress * math.pi))


def pond_bounds(y):
    if not 88 <= y <= 118:
        return 100, 100
    half = round(79 * math.sqrt(max(0, 1 - ((y - 103) / 16) ** 2)))
    return 100 - half, 100 + half


def pond_ripple(c, x, y, width, color):
    left, right = pond_bounds(y)
    start, end = max(left, x), min(right, x + width)
    if end > start:
        c.rect(start, y, end - start, 1, color)


def duck_position(t, n):
    # Separate lanes leave a full pixel gap between bodies and wakes.
    angle = t * .065 + n * 2 * math.pi / 3
    return round(94 - 39 * math.cos(angle)), 91 + n * 8, math.sin(angle) >= 0


def sakura_river_bounds(y):
    """Perspective river: narrow at the horizon and broad in the foreground."""
    if not 81 <= y < H:
        return 103, 103
    progress = (y - 81) / (H - 81)
    center = 103 + round(4 * math.sin(progress * 2.5))
    half = round(3 + 31 * progress)
    return center - half, center + half


def render(scene: Scene, tick: int = 0, label: bool = True, cycle: float = 120, theme: str = "woodland") -> Canvas:
    t = tick / 12
    if theme in ILLUSTRATED_ASSETS and ILLUSTRATED_ASSETS[theme].exists():
        return render_illustrated(theme,t,cycle)
    phase = (t / cycle) % 1
    # Dawn -> blue daylight -> amber sunset -> moonlit blue, with smooth blends.
    day = {"sky_top": (62, 143, 189), "sky_bottom": (186, 223, 205),
           "far": (101, 151, 157), "near": (64, 116, 123),
           "land": (38, 98, 77), "water": (48, 133, 155),
           "cloud": (240, 236, 220), "light": (255, 230, 155)}
    keys = [PALETTES[0], day, PALETTES[1], PALETTES[2], PALETTES[0]]
    segment = phase*4
    amount = segment % 1
    amount = amount*amount*(3-2*amount)
    p = {key: mix(keys[int(segment)][key], keys[int(segment)+1][key], amount) for key in day}
    c = Canvas()
    night = max(0, 1-abs(phase-.75)*8)
    for y in range(H):
        c.rect(0, y, W, 1, mix(p['sky_top'], p['sky_bottom'], min(1, y/77)))
    if night > .1:
        for x, y, shimmer in scene.stars:
            c.pixel(x, y, mix(p['sky_top'], (225, 236, 249), night*(.6+.4*math.sin(t*2+x)**2)))
    celestial_x, celestial_y = celestial_position(phase)
    # Leave the actual sky untouched inside the crescent; keep sun RGB fixed.
    for dy in range(-7, 8):
        for dx in range(-7, 8):
            if dx*dx + dy*dy <= 49 and not (phase >= .5 and (dx-3)**2 + (dy+2)**2 <= 36):
                c.pixel(celestial_x+dx, celestial_y+dy,
                        (244, 239, 196) if phase >= .5 else (255, 225, 138))
    for x, y, size in scene.clouds:
        xx = int((x+t*.8) % (W+50))-25
        c.rect(xx, y, size*2, 3, p['cloud'])
        c.rect(xx+4, y-3, size, 6, p['cloud'])
        c.rect(xx+size, y-1, size, 4, p['cloud'])
    for x, peak, width in scene.mountains:
        draw_mountain(c, x, peak+8, width+20, 80, p['far'])
        draw_mountain(c, x+8, peak+20, width+12, 84, p['near'])
    # Close gaps between random mountain silhouettes at the horizon.
    c.rect(0, 77, W, H-77, p['near'])
    if theme in SCENES[5:]:
        draw_special_scene(c, theme, t, p, night)
        return c
    c.rect(0, 78, W, 42, p['water'])
    # Broken reflections move gently across the open river.
    for y in range(81, H, 3):
        for x in range(-20, W, 23):
            xx = x+round(5*math.sin(t*.8+y))
            c.rect(xx, y, 7+(y%5), 1, mix(p['water'], p['light'], .24))
    for y in range(80, 116, 4):
        c.rect(celestial_x-(y-76)//4+round(3*math.sin(t+y)), y, (y-76)//2, 1, mix(p['water'], p['light'], .48))
    # Clear stage: bank below, forest at the edges, animals against open grass.
    ground_colors = {"woodland": (91, 140, 76), "meadow": (123, 161, 84),
                     "pond": (91, 137, 95), "autumn": (163, 115, 60),
                     "winter": (215, 231, 230)}
    grass = mix(ground_colors[theme], (55, 76, 91) if theme == 'winter' else (32, 65, 69), night)
    if theme == 'pond':
        # An enclosed pool, rather than a river cutting through its far bank.
        c.rect(0, 78, W, H-78, grass)
    for x in range(W):
        bank = round(89+3*math.sin(x*.035))
        c.line(x, bank, x, H-1, grass)
        c.pixel(x, bank, mix(grass, p['light'], .4))
    if theme in ('woodland', 'winter'):
        for x, bottom, size in [(12, 98, 39), (31, 94, 30), (175, 96, 35), (193, 101, 44)]:
            c.rect(x-1, bottom-size//2, 3, size//2, (83, 69, 58))
            for level in range(3):
                top = bottom-size+level*size//5
                sway = round(math.sin(t*1.6+x+level*.4)*(2-level*.5))
                for row in range(size//2):
                    half = 1+row//2
                    col = mix(p['land'], grass, .24 if level%2 else .05)
                    c.rect(x+sway-half, top+row, half*2+1, 1, col)
                    c.pixel(x+sway-half, top+row, mix(col, p['light'], .17))
    if theme in ('woodland', 'winter'):
        # Cabin and chimney smoke.
        c.rect(124, 76, 25, 17, (116, 77, 60))
        for y in range(76, 93, 4):
            c.rect(124, y, 25, 1, (85, 59, 52))
        for row in range(12):
            c.rect(121+row, 76-row, 31-row*2, 1, (67, 62, 66))
        c.rect(142, 63, 4, 9, (97, 74, 67))
        c.rect(127, 80, 6, 6, p['light'])
        c.rect(138, 82, 6, 11, (58, 50, 49))
        c.pixel(139, 87, p['light'])
        for i in range(4):
            rise = (t*3+i*5)%22
            c.rect(143+round(math.sin(t+i)*2), 61-int(rise), 3+i//2, 2, mix(p['sky_bottom'], p['cloud'], .6))
    for n in range(4):
        x = int((t*9+n*9) % (W+40))-20
        y = 25+n*3+round(2*math.sin(t*.8))
        wing = -2 if int(t*5+n)%2 else 2
        c.line(x-3, y+wing, x, y, p['near'])
        c.line(x, y, x+3, y+wing, p['near'])
    rng = random.Random(scene.seed)
    for _ in range(65):
        x, y = rng.randrange(W), rng.randrange(98, H)
        sway = round(math.sin(t*2+x))
        c.line(x, y, x+sway, y-2, mix(grass, p['land'], .5))
        if x%5==0:
            c.pixel(x+sway, y-3, (230, 189, 114))
    draw_habitat(c, theme, t, p, grass, night)
    action = t%40
    fx = round(-22+action*10) if action<8 else (58 if action<26 else round(58+(action-26)*12))
    rows = FOX.copy()
    if 12<action<23:  # A quiet sniff and blink before continuing downstream.
        if int(t)%4==0:
            rows[3] = rows[3].replace('K', 'O')
        fy = 94+int(math.sin(t*2)>0)
    else:
        fy = 94+int(math.sin(t*8)>0)
        if int(t*6)%2:
            rows[-1] = '     KK KK  KK KK   '
    if night > .45 and theme in ('woodland', 'autumn', 'winter'):
        rows = SLEEPING_FOX
        fx, fy = 53, 105
    if theme in ('woodland', 'autumn', 'winter'):
        sprite(c, rows, fx, fy,
               {'O': (217, 116, 55), 'W': (255, 224, 173),
                'C': (255, 224, 173), 'K': (39, 36, 42)}, flip=True)
    hop = round(max(0, math.sin(t*3))*4) if int(t)%13<6 else 0
    if theme != 'pond' and night < .45:
        sprite(c, RABBIT, 109+round(5*math.sin(t*.25)), 109-hop,
               {'W': (229, 225, 207), 'P': (208, 149, 145), 'K': (43, 43, 57)})
    if night>.2:
        for n in range(14):
            if math.sin(t*2+n)>.2:
                c.pixel((n*17+round(math.sin(t+n)*4))%W, 84+round(10*math.sin(n+t*.5)), (242, 237, 139))
    draw_season(c, theme, t, night)
    draw_events(c, theme, t, night)
    draw_weather(c, theme, t, scene.seed)
    return c


def draw_habitat(c, theme, t, p, grass, night):
    """Each habitat has its own silhouette and characteristic moving subjects."""
    if theme == 'meadow':
        # Windmill on a flower-covered hill; the sails rotate in the breeze.
        c.rect(145, 66, 13, 28, (211, 186, 136))
        draw_mountain(c, 151, 57, 10, 68, (108, 73, 66))
        c.rect(149, 85, 5, 9, (74, 65, 59))
        for n in range(4):
            angle = t*.6+n*math.pi/2
            x, y = round(151+19*math.cos(angle)), round(69+19*math.sin(angle))
            c.line(151, 69, x, y, (244, 227, 186))
            c.line(152, 69, x+1, y, (244, 227, 186))
            c.line(x, y, round(151+11*math.cos(angle)-4*math.sin(angle)),
                   round(69+11*math.sin(angle)+4*math.cos(angle)), (244, 227, 186))
        c.circle(151, 69, 2, (98, 72, 60))
        for n in range(45):
            x, y = (n*37)%W, 96+(n*13)%24
            sway = round(math.sin(t*1.8+n))
            c.line(x, y, x+sway, y-4, (52, 110, 70))
            c.circle(x+sway, y-5, 1, [(240, 157, 170), (249, 219, 133), (190, 172, 220)][n%3])
        for n in range(6):
            x = round(20+n*26+8*math.sin(t*.7+n))
            y = round(72+10*math.sin(t+n*2))
            wing = 1+int((math.sin(t*12+n)+1)*1.5)
            color = [(253, 194, 101), (241, 151, 185), (169, 210, 242)][n%3]
            c.rect(x-wing, y-1, wing*2+1, 3, color)
            c.line(x, y-2, x, y+2, (64, 51, 65))
    elif theme == 'pond':
        # A broad oval pool replaces the front lawn; ducks have room to swim.
        for y in range(88, 119):
            left, right = pond_bounds(y)
            c.rect(left, y, right-left, 1, p['water'])
            for x in range(left+3, right-3, 17):
                pond_ripple(c, x+int(t*2+y)%5, y, 4, mix(p['water'], p['light'], .18))
        for n in range(9):
            x, y = 40+n*14, 104+n%3*4
            c.rect(x, y, 7, 2, (54, 117, 77))
            if n%3==0:
                c.circle(x+3, y-1, 2, (237, 167, 187))
        for x in (12, 20, 180, 188):
            bend = round(2*math.sin(t*1.5+x))
            c.line(x, 113, x+bend, 87, (74, 116, 65))
            c.rect(x+bend-1, 84, 3, 7, (111, 70, 49))
        duck = ['         GG   ', '        GKGYYY', '        GWG Y ',
                '   BBBBBBB    ', ' BBBBBBBBBBB  ', '  BBBBBBBBB   ']
        for n in range(3):
            x, y, right = duck_position(t, n)
            diving = 16 < (t+n*9)%31 < 19
            pose = ['            ', '            ', '    WW      ', '   WWWW     ', '    YY      ', '            '] if diving else duck
            sprite(c, pose, x, y, {'G': (45, 100, 74), 'K': (22, 32, 38),
                                 'Y': (248, 187, 77), 'W': (231, 222, 189),
                                 'B': (139, 100, 63)}, flip=not right)
            pond_ripple(c, x-5 if right else x+13, y+6, 5, mix(p['water'], p['light'], .55))
        # Frog waits on the near bank and makes short hops.
        hop = round(5*max(0, math.sin(t*4))) if int(t)%9<2 else 0
        sprite(c, FROG, 56, 113-hop,
               {'G': (90, 143, 73), 'K': (25, 37, 34), 'L': (65, 122, 78)})
        hunt = t % 17
        if 10 < hunt < 14:
            fly_x, fly_y = 72+round(2*math.sin(t*5)), 110
            c.pixel(fly_x, fly_y, (32, 38, 44))
            if hunt > 13.5:
                c.line(64, 117-hop, fly_x, fly_y, (233, 151, 154))
    elif theme == 'autumn':
        for x, y, radius in [(18, 70, 17), (173, 64, 21), (193, 78, 15)]:
            c.rect(x-2, y, 4, 100-y, (91, 65, 50))
            sway = round(2*math.sin(t*1.2+x))
            for dx, dy, r, color in [(-8, 0, radius-3, (166, 70, 50)),
                                     (7, -5, radius-2, (216, 124, 58)), (0, -12, radius-4, (233, 165, 70))]:
                c.circle(x+dx+sway, y+dy, r, mix(color, p['near'], night*.45))
        for n in range(30):
            x = round((n*31+t*4+4*math.sin(t+n))%(W+8))-4
            y = round((n*17+t*(3+n%3))%125)
            c.rect(x, y, 3 if int(t*4+n)%2 else 1, 2, [(225, 149, 64), (190, 83, 54), (237, 185, 93)][n%3])
        for x in (42, 87, 155):
            c.rect(x, 111, 2, 5, (238, 220, 179))
            c.rect(x-2, 109, 6, 3, (184, 70, 50))
            c.pixel(x, 109, (255, 231, 192))
    elif theme == 'winter':
        # Snow caps on trees and cabin, with slow wind-driven snowflakes.
        for x, y in [(12, 62), (31, 67), (175, 64), (193, 60)]:
            for row in range(5):
                c.rect(x-row, y+row, row*2+1, 1, (222, 237, 238))
        for row in range(3):
            c.line(123, 76+row, 136, 65+row, (231, 239, 235))
            c.line(136, 65+row, 149, 76+row, (231, 239, 235))
        c.circle(40, 102, 6, (238, 242, 232))
        c.circle(40, 93, 4, (238, 242, 232))
        c.rect(35, 89, 10, 2, (53, 60, 74))
        c.rect(37, 85, 6, 5, (53, 60, 74))
        c.pixel(39, 92, (39, 45, 55))
        c.rect(41, 94, 4, 1, (234, 136, 60))
        c.rect(37, 97, 7, 2, (179, 76, 73))
        for n in range(70):
            x = round((n*47+t*(2+n%3)+3*math.sin(t+n))%W)
            y = round((n*29+t*(4+n%4))%H)
            c.pixel(x, y, (226, 237, 244))
            if n%7==0:
                c.pixel(x+1, y, (226, 237, 244))


def draw_special_scene(c, theme, t, p, night):
    """Distinct stages keep ocean waves, weather and animals in their habitats."""
    snow = (227, 238, 240)
    if theme == 'christmas':
        c.rect(0, 78, W, 42, mix(snow, (87, 111, 146), night*.6))
        # Nativity stable: the figures and manger are framed as one readable scene.
        c.rect(9, 73, 53, 35, (83, 61, 48))
        for row in range(4):
            c.line(4, 72-row, 35, 54-row, (122, 84, 53))
            c.line(35, 54-row, 67, 72-row, (122, 84, 53))
        c.rect(11, 74, 3, 35, (151, 103, 61))
        c.rect(58, 74, 3, 35, (151, 103, 61))
        c.rect(13, 75, 45, 28, (73, 55, 48))
        # Bethlehem star gently pulses above the stable.
        star = (255, 225, 117)
        ray = 4 + int((math.sin(t*3)+1)*2)
        c.line(35-ray, 45, 35+ray, 45, star)
        c.line(35, 45-ray, 35, 45+ray, star)
        c.line(32, 42, 38, 48, star)
        c.line(38, 42, 32, 48, star)
        # Joseph, Mary and the infant in a straw-filled manger.
        c.circle(22, 82, 3, (210, 163, 119))
        c.rect(18, 85, 8, 16, (126, 91, 62))
        c.rect(17, 98, 10, 3, (68, 58, 53))
        c.line(28, 82, 28, 103, (169, 125, 72))
        c.circle(48, 86, 3, (210, 163, 119))
        for row in range(14):
            c.rect(48-row//2, 88+row, row+1, 1, (72, 117, 161))
        c.rect(31, 96, 14, 3, (181, 126, 66))
        c.line(31, 99, 29, 104, (133, 89, 52))
        c.line(44, 99, 46, 104, (133, 89, 52))
        c.circle(37, 94, 2, (231, 190, 143))
        c.rect(34, 96, 8, 2, (245, 218, 139))

        # Village, smoking chimney and decorated tree change continuously.
        c.rect(158, 74, 34, 29, (128, 70, 59))
        draw_mountain(c, 175, 55, 22, 76, snow)
        c.rect(163, 83, 7, 8, (255, 216, 112))
        c.rect(180, 86, 7, 17, (68, 52, 56))
        c.rect(184, 58, 5, 12, (96, 72, 64))
        for n in range(4):
            rise = (t*4+n*6)%25
            c.circle(186+round(2*math.sin(t+n)), 55-round(rise), 2,
                     mix(p['cloud'], p['sky_top'], .3))
        c.rect(111, 87, 5, 22, (108, 76, 50))
        for top, half, base in ((38,12,64),(51,18,83),(67,25,103)):
            draw_mountain(c,113,top,half,base,(29,105,70))
        sprite(c, ['    S    ', '    S    ', 'SSSSSSSSS', ' SSSSSSS ',
                   '  SSSSS  ', ' SSS SSS ', ' SS   SS '],109,31,{'S':star})
        for n in range(24):
            y = 49+n*2
            x = 113+round((y-35)*.32*math.sin(n*2.4))
            colors = ((255,102,100),(255,226,107),(116,201,255))
            c.pixel(x,y,colors[(n+int(t*2))%3])
        for x,col in ((91,(211,67,77)),(122,(70,139,196)),(138,(190,102,186))):
            c.rect(x,103,11,9,col)
            c.rect(x+5,103,2,9,(255,223,128))
            c.rect(x,106,11,2,(255,223,128))
        # A snowman waves; Santa crosses only during the darkest portion.
        # Give the snowman contrast against the pale snow, with connected balls.
        c.circle(75, 104, 8, (167,193,208))
        c.circle(75, 103, 7, (250,250,243))
        c.circle(75, 94, 6, (167,193,208))
        c.circle(75, 93, 5, (250,250,243))
        c.rect(70,88,11,2,(48,54,68))
        c.rect(72,84,7,5,(48,54,68))
        c.rect(71,98,9,2,(190,65,64))
        c.rect(78,99,2,5,(190,65,64))
        c.pixel(75,103,(48,54,68))
        c.pixel(75,107,(48,54,68))
        c.pixel(73, 92, (35, 40, 45)); c.pixel(77, 92, (35, 40, 45))
        c.rect(77, 95, 4, 1, (226, 115, 65))
        c.line(69, 100, 64, 95+round(2*math.sin(t*3)), (104, 75, 52))
        if night > .35:
            sleigh_x = round((t*10)%(W+70))-45
            c.line(sleigh_x, 25, sleigh_x+20, 25, (197, 60, 55))
            c.rect(sleigh_x+5, 20, 12, 5, (197, 60, 55))
            sprite(c, DEER, sleigh_x+23, 16,
                   {'B': (150, 104, 64), 'A': (102, 78, 57), 'K': (31, 32, 34)})
        for n in range(45):
            c.pixel(round((n*43+t*3)%W),round((n*29+t*6)%H),snow)
    elif theme == 'tornado':
        c.rect(0,0,W,78,(64,79,91))
        c.rect(0,78,W,42,(98,112,73))
        for n in range(7):
            c.circle(round((n*36-t*5)%(W+40))-20,17+n%2*5,23,(45,57,69))
        # Farm buildings at both edges show the storm's scale and damage.
        c.rect(3, 86, 41, 25, (151, 79, 61))
        c.rect(8, 93, 10, 18, (74, 55, 49))
        c.rect(29, 92, 8, 8, (202, 187, 130))
        left_roof_x = 6+round(9*math.sin(t*1.7))
        left_roof_y = 77-round(10*(math.sin(t*.8)**2))
        draw_mountain(c, left_roof_x+20, left_roof_y-10, 24, left_roof_y+4,
                      (83, 58, 52))
        c.rect(166, 88, 34, 23, (132, 110, 76))
        c.rect(174, 96, 8, 15, (61, 53, 48))
        for row in range(10):
            c.rect(163+row, 88-row, 43-row*2, 1, (76, 60, 55))
        center = 100+round(24*math.sin(t*.13))
        for y in range(27,107):
            half = round(4+(107-y)*.32)
            x = center+round(5*math.sin(t*2-y*.09))
            c.rect(x-half,y,half*2,1,(99,111,117))
            stripe = round(half*math.sin(t*8-y*.3))
            c.rect(x+stripe-2,y,4,1,(157,163,157))
        for n in range(30):
            angle = t*4+n*2.4
            x = center+round((15+n%5*5)*math.cos(angle))
            y = 103-n%7*8+round(3*math.sin(angle))
            debris = ((190,167,115), (91,67,57), (171,79,61))[n%3]
            c.rect(x,y,2+n%4,1+n%2,debris)
        # One cow circles the funnel while another flees across the foreground.
        orbit = t*2.3
        cow_x = center+round(43*math.cos(orbit))-6
        cow_y = 61+round(20*math.sin(orbit))
        sprite(c, COW, cow_x, cow_y,
               {'B': (115, 72, 48), 'W': (235, 226, 199), 'K': (31, 31, 33),
                'H': (208, 181, 126)},
               flip=math.cos(orbit)<0)
        runner_x = round((t*15)%(W+40))-20
        runner = COW.copy()
        if int(t*6)%2:
            runner[-2] = "     BB BB  BB      "; runner[-1] = "     KK KK  KK      "
        sprite(c, runner, runner_x, 108,
               {'B': (111, 70, 47), 'W': (237, 228, 204), 'K': (29, 30, 31),
                'H': (208, 181, 126)}, flip=True)
        # A skidding car and airborne roof panels follow different paths.
        car_phase = t%18
        if car_phase < 6:
            car_x = round((W+24)*(1-car_phase/6)+(center-9)*car_phase/6)
            car_y = 105
        else:
            lift = (car_phase-6)/12
            radius = 34*math.sin(math.pi*min(1,lift*2))
            car_x = round(center-9+radius*math.sin(lift*math.pi*8))
            car_y = round(105-145*lift)
        c.rect(car_x, car_y, 18, 6, (183, 60, 55))
        c.rect(car_x+4, car_y-4, 10, 4, (126, 166, 176))
        c.circle(car_x+4, car_y+6, 2, (31, 34, 38))
        c.circle(car_x+15, car_y+6, 2, (31, 34, 38))
        roof_angle = t*2
        roof_x = center+round(58*math.cos(roof_angle))
        roof_y = 49+round(20*math.sin(roof_angle))
        c.line(roof_x-8,roof_y,roof_x+8,roof_y+round(4*math.sin(t*5)),(78,54,48))
        for x in range(0,W,9):
            c.line(x,119,x+5,115,(65,87,62))
    elif theme == 'tsunami':
        c.rect(0,72,W,48,(29,112,152))
        # A looping offshore wave rises, travels, and subsides smoothly.
        phase = (t%32)/32
        # Build, curl, collapse, then leave a field of fading bubbles.
        collapse = max(0,min(1,(phase-.50)/.22))
        amplitude = 46*math.sin(math.pi*min(phase,.5))**2*(1-collapse)
        center = -45+290*phase
        for x in range(W):
            crest = round(84-amplitude*math.exp(-((x-center)/24)**2))
            c.rect(x,crest,1,H-crest,(38,148,177))
            c.rect(x,crest,1,2,(215,244,236))
            c.pixel(x,crest+5,(115,205,214))
            if abs(x-center)<20 and x%4==0:
                c.rect(x, crest-3-round(3*math.sin(t*4+x)), 2, 3, (232,248,239))
        for n in range(18):
            x = round((n*17+t*12)%W)
            c.rect(x,105+n%3*4,6,1,(102,197,209))
        if .36 < phase < .72:
            curl = min(1,(phase-.36)/.14)
            tip_x = center+18*curl
            tip_y = 84-amplitude+23*collapse
            for step in range(32):
                angle = -math.pi/2+step/31*math.pi*1.3
                x = round(tip_x+12*curl*math.cos(angle))
                y = round(tip_y+10*curl*(1+math.sin(angle)))
                c.circle(x,y,2,(223,246,237))
        if .53 < phase < .99:
            age = (phase-.53)/.46
            for n in range(round(60*(1-age))):
                bx = round(center+(n*37%119)-59+age*28)
                by = 87+(n*17%26)
                radius = 1+(n+int(t*3))%3
                if 2 <= bx < W-2 and by+radius < 115:
                    color = mix((235,250,243),(83,175,192),age)
                    c.line(bx-radius,by,bx,by-radius,color)
                    c.line(bx,by-radius,bx+radius,by,color)
                    c.line(bx+radius,by,bx,by+radius,color)
                    c.line(bx,by+radius,bx-radius,by,color)
        # Lighthouse beam sweeps the storm while a rescue boat climbs the swell.
        c.rect(8, 62, 12, 48, (230, 220, 190))
        c.rect(8, 74, 12, 7, (190, 67, 57))
        c.rect(6, 57, 16, 7, (62, 57, 55))
        c.rect(10, 59, 8, 4, (255, 228, 139))
        beam_y = 43+round(18*math.sin(t*.8))
        c.line(18, 60, 86, beam_y, mix((255,232,159), p['sky_bottom'], .25))
        boat_x = round((t*13)%(W+65))-25
        boat_surface = round(84-amplitude*math.exp(-((boat_x+12-center)/24)**2))
        c.line(boat_x, boat_surface-4, boat_x+24, boat_surface-4, (238, 229, 193))
        c.rect(boat_x+4, boat_surface-10, 14, 6, (224, 91, 57))
        c.rect(boat_x+9, boat_surface-14, 7, 4, (205, 232, 224))
        c.line(boat_x+12,boat_surface-14,boat_x+12,boat_surface-20,(65,61,60))
        c.pixel(boat_x+13,boat_surface-20,(255,211,95))
        # Coastguard helicopter and bobbing buoys keep the scene active.
        heli_x = round((t*9)%(W+60))-30
        heli_y = 27+round(4*math.sin(t*1.4))
        c.rect(heli_x,heli_y,20,6,(232,159,55))
        c.rect(heli_x+4,heli_y-5,9,5,(114,167,178))
        c.rect(heli_x+18,heli_y+1,11,2,(232,159,55))
        c.rect(heli_x+8,heli_y-6,2,7,(47,51,55))
        c.line(heli_x+29,heli_y-2,heli_x+29,heli_y+5,(47,51,55))
        c.line(heli_x+4,heli_y+5,heli_x+4,heli_y+9,(47,51,55))
        c.line(heli_x+15,heli_y+5,heli_x+15,heli_y+9,(47,51,55))
        c.line(heli_x+1,heli_y+9,heli_x+19,heli_y+9,(47,51,55))
        c.line(heli_x-7-int(5*math.sin(t*12)),heli_y-6,
               heli_x+25+int(5*math.sin(t*12)),heli_y-6,(47,51,55))
        for n in range(3):
            buoy_x = 65+n*48+round(4*math.sin(t+n))
            buoy_y = 103+n%2*5+round(2*math.sin(t*2+n))
            c.rect(buoy_x,buoy_y-5,2,6,(241,191,67))
            c.circle(buoy_x+1,buoy_y,3,(210,72,57))
        for n in range(3):
            bird_x = round((t*7+n*49)%(W+20))-10
            bird_y = 38+n*7+round(3*math.sin(t+n))
            c.line(bird_x-3,bird_y,bird_x,bird_y-2,(35,43,48))
            c.line(bird_x,bird_y-2,bird_x+3,bird_y,(35,43,48))
        # A dry foreground lookout stays above the ocean.
        c.rect(0,115,W,5,(191,175,125))
    elif theme == 'balloons':
        c.rect(0,78,W,42,(116,169,89))
        for n in range(4):
            x = round((n*57+t*2)%(W+44))-22
            y = 25+n%2*16+round(3*math.sin(t*.6+n))
            col = ((238,123,112),(246,193,99),(150,134,220),(100,201,194))[n]
            c.circle(x,y,12,col)
            c.rect(x-3,y-11,6,23,mix(col,(255,242,194),.5))
            c.line(x-6,y+10,x-3,y+19,(97,85,70))
            c.line(x+6,y+10,x+3,y+19,(97,85,70))
            c.rect(x-4,y+18,9,5,(159,107,65))
        for n in range(32):
            x,y=n*37%W,94+n*11%25
            c.line(x,y,x,y-4,(61,122,72))
            c.line(x,y-1,x+2,y-2,(61,122,72))
            if n%3 == 0:  # White daisies with gold centres.
                for dx,dy in ((-2,0),(2,0),(0,-2),(0,2)):
                    c.pixel(x+dx,y-5+dy,(244,239,226))
                c.pixel(x,y-5,(229,179,78))
            elif n%3 == 1:  # Pink tulip cups.
                c.rect(x-1,y-6,3,3,(226,126,168))
                c.pixel(x-2,y-7,(241,159,189))
                c.pixel(x+2,y-7,(241,159,189))
            else:  # Purple flower spikes.
                for dy in range(3):
                    c.rect(x-1,y-5-dy*2,3,1,(164+dy*12,137,208))
    elif theme == 'sakura':
        # A river runs toward the viewer; the bridge is fixed across both banks.
        c.rect(0,78,W,42,(134,165,113))
        for y in range(81, H):
            left, right = sakura_river_bounds(y)
            c.rect(left, y, right-left, 1, (86,156,175))
            for n in range(3):
                x = left + ((n*19+round(t*3)+y) % max(1, right-left))
                c.rect(x, y, min(5, right-x), 1, (172,218,218))
        for x,y in ((25,66),(171,60)):
            c.rect(x-2,y,5,38,(108,72,72))
            for dx,dy in ((-13,0),(12,-2),(0,-13)):
                c.circle(x+dx,y+dy,17,(231,154,180))
                c.circle(x+dx-4,y+dy-4,10,(249,191,207))
        bridge_left, bridge_right = sakura_river_bounds(102)
        bridge_left -= 8; bridge_right += 8
        for x in range(bridge_left, bridge_right):
            progress = (x-bridge_left)/(bridge_right-bridge_left-1)
            deck_y = round(101-7*math.sin(progress*math.pi))
            c.rect(x,deck_y,1,5,(158,94,81))
            c.pixel(x,deck_y,(211,139,105))
            if (x-bridge_left)%7==0:
                c.rect(x,deck_y-7,2,8,(139,82,72))
        # Iris and fern clusters mark both riverbanks beside the bridge.
        for base_x in (bridge_left-12, bridge_left-5, bridge_right+5, bridge_right+13):
            for stem in range(3):
                sway = round(math.sin(t*1.2+base_x+stem))
                c.line(base_x+stem*3,113,base_x+stem*3+sway,105-stem%2*3,(51,111,70))
                c.circle(base_x+stem*3+sway,104-stem%2*3,1,
                         (137,116,201) if stem%2 else (240,198,220))
        # A white crane fishes on the left bank; koi move below the bridge.
        crane_x = 55+round(3*math.sin(t*.35))
        c.line(crane_x,99,crane_x-1,116,(85,79,68))
        c.line(crane_x+3,98,crane_x+4,116,(85,79,68))
        c.circle(crane_x,96,5,(238,235,219))
        c.line(crane_x+3,97,crane_x+7,87,(238,235,219))
        c.circle(crane_x+7,86,3,(238,235,219))
        c.line(crane_x+9,86,crane_x+15,88,(215,136,78))
        c.pixel(crane_x+8,85,(31,35,37))
        for n in range(3):
            fish_y = 108+n*4
            left, right = sakura_river_bounds(fish_y)
            fish_x = left+5+round((t*(3+n)+n*17) % max(1,right-left-15))
            sprite(c, ['T   FFF   ', 'TT FFFFFF ', ' TFFFFFFFK',
                       'TT FFFFFF ', 'T   FFF   '],fish_x,fish_y-2,
                   {'F':((230,116,82),(242,188,91),(222,221,207))[n],
                    'T':(223,153,106),'K':(35,48,49)})
        for n in range(25):
            c.rect(round((n*31+t*4+3*math.sin(t+n))%W),round((n*17+t*5)%H),2,1,(255,204,218))


def weather_at(t, seed=0):
    slot = int(t//48)
    kind = ('clear', 'rain', 'fog', 'clear', 'storm', 'clearing')[(slot+seed%6)%6]
    local = t%48
    strength = min(1, local/8, (48-local)/8)
    return kind, strength


def draw_weather(c, theme, t, seed):
    kind, strength = weather_at(t, seed)
    if kind == 'clear' or strength <= 0:
        return
    # Sparse drifting fog bands keep the pixel grid crisp and rendering cheap.
    if kind in ('fog', 'clearing'):
        for band in range(4):
            y = 47+band*13
            x = round((t*2+band*53)%(W+80))-80
            for yy in range(y, y+4):
                for xx in range(max(0,x), min(W,x+80)):
                    i = (yy*W+xx)*3
                    c.pixel(xx, yy, mix(tuple(c.data[i:i+3]), (182, 199, 202), strength*.32))
    if kind not in ('rain', 'storm'):
        return
    snow = theme == 'winter'
    for n in range(round((75 if kind == 'storm' else 40)*strength)):
        x = round((n*43+t*(5 if snow else 19))%W)
        y = round((n*29+t*(9 if snow else 67))%H)
        if snow:
            c.pixel(x,y,(232,240,244))
        else:
            c.line(x,y,x-1,y+2,(139,181,200))
    if not snow:
        for n in range(8):
            y = 94+n*3%23 if theme == 'pond' else 82+n%3*2
            x = 40+n*17
            radius = int((t*4+n)%4)
            if theme == 'pond':
                pond_ripple(c,x-radius,y,radius*2+1,(130,182,196))
            else:
                c.rect(x-radius,y,radius*2+1,1,(130,182,196))
    # A brief distant bolt, without a full-screen flash.
    if kind == 'storm' and 20 < t%48 < 20.35:
        c.line(153,8,147,20,(255,242,183))
        c.line(147,20,153,19,(255,242,183))
        c.line(153,19,145,33,(255,242,183))


def draw_season(c, theme, t, night):
    if theme == 'meadow':  # Spring petals.
        for n in range(16):
            x = round((n*29+t*3+3*math.sin(t+n))%W)
            y = round((n*17+t*4)%H)
            c.rect(x,y,2,1,mix((250,183,202),(117,112,145),night))
    elif theme == 'woodland':  # Summer berries on the foreground edges.
        for x in (5, 18, 179, 192):
            c.line(x,119,x,109,(54,92,59))
            c.line(x,114,x-4,111,(54,92,59))
            c.line(x,116,x+4,113,(54,92,59))
            c.rect(x-5,109,4,3,(48,111,69))
            c.rect(x+2,111,5,3,(48,111,69))
            c.rect(x-3,115,4,3,(48,111,69))
            c.pixel(x-4,110,(208,98,114))
            c.pixel(x+5,113,(208,98,114))
    elif theme == 'autumn':
        for n in range(min(65,int(t/3))):
            c.rect((n*37)%W,117-n%3,3,1,(185+n%3*17,104+n%4*12,54))
    elif theme == 'winter':
        depth = min(4,1+int(t/90))
        c.rect(0,H-depth,W,depth,(233,241,239))
        c.rect(126,75-depth,21,depth,(233,241,239))


def draw_events(c, theme, t, night):
    # Rare visits use animation time, independent of the real-world clock.
    event = t%180
    if night > .3 and 85 < event < 87:
        x = round(35+(event-85)*55)
        y = round(8+(event-85)*10)
        c.line(x-13,y-3,x,y,(194,210,224))
        c.pixel(x,y,(255,247,211))
    if theme != 'pond' and 130 < event < 158:
        x = round(-24+(event-130)*9)
        sprite(c, DEER, x, 101,
               {'B':(173,130,87),'A':(104,77,55),'K':(31,35,39)})
    if theme != 'pond' and 35 < t%240 < 67:
        x = round(-18+(t%240-35)*7)
        c.rect(x,82,14,2,(143,86,57))
        c.rect(x+2,84,10,1,(143,86,57))
        c.line(x+7,81,x+7,70,(228,214,172))
        for row in range(8):
            c.rect(x+8,72+row,1+row//2,1,(241,225,181))
    if night > .4 and 90 < event < 105:
        for n in range(12):
            x = round(95+17*math.sin(t*.8+n))
            y = round(75+9*math.cos(t+n*2))
            c.pixel(x,y,(241,235,137))


def scene_at(seconds: float, seconds_per_scene: float = SECONDS_PER_SCENE):
    """Ten equal slots, repeating after two hours with the default duration."""
    slot = int(seconds // seconds_per_scene) % len(SCENES)
    return SCENES[slot], seconds % seconds_per_scene


def render_playlist(scene, seconds, seconds_per_scene=SECONDS_PER_SCENE, cycle=120):
    theme, local_time = scene_at(seconds, seconds_per_scene)
    return render(scene, local_time*12, cycle=cycle, theme=theme)


def write_ppm(canvas: Canvas, path: Path) -> None:
    path.write_bytes(f"P6\n{canvas.width} {canvas.height}\n255\n".encode() + canvas.data)


class Framebuffer:
    """RGB565 fbdev output. It preserves and restores the previous framebuffer."""
    def __init__(self, device: str):
        self.device = device
        import fcntl
        # fb_var_screeninfo: visible dimensions, offsets and pixel channel layout.
        with open(device, "rb", buffering=0) as info:
            var = bytearray(160)
            fcntl.ioctl(info.fileno(), 0x4600, var, True)
        self.width, self.height, _, _, self.xoffset, self.yoffset, bpp = struct.unpack_from("7I", var)
        channels = struct.unpack_from("9I", var, 32)
        if bpp != 16 or channels != (11, 5, 0, 5, 6, 0, 0, 5, 0):
            raise RuntimeError("Framebuffer mode must be RGB565 (16 bits per pixel); use --tty for other modes.")
        sysfs = Path('/sys/class/graphics') / Path(device).name
        self.line_length = int((sysfs / 'stride').read_text().strip())
        self.size = self.line_length * (self.height + self.yoffset)
        self.file = open(device, "r+b", buffering=0)
        self.map = mmap.mmap(self.file.fileno(), self.size)
        self.original = self.map[:]
        # The artwork has a small, repeating palette.  Cache RGB565 conversion
        # and scaled pixels instead of packing every source pixel every frame.
        self._pixels: dict[tuple[int, int, int], bytes] = {}
        self._expanded: dict[tuple[tuple[int, int, int], int], bytes] = {}

    def show(self, canvas: Canvas) -> None:
        # Permit lightweight framebuffer doubles in renderer checks.
        if not hasattr(self, "_pixels"):
            self._pixels = {}
            self._expanded = {}
        scale = min(self.width // canvas.width, self.height // canvas.height)
        if scale < 1:
            raise RuntimeError(f"Framebuffer {self.width}x{self.height} is smaller than {canvas.width}x{canvas.height}.")
        image_w, image_h = canvas.width * scale, canvas.height * scale
        x_pad, y_pad = (self.width - image_w) // 2, (self.height - image_h) // 2
        if np is not None:
            source = np.frombuffer(canvas.data, dtype=np.uint8).reshape(
                canvas.height, canvas.width, 3).astype(np.uint16)
            packed = ((source[:, :, 0] >> 3) << 11) | ((source[:, :, 1] >> 2) << 5) | (source[:, :, 2] >> 3)
            if scale > 1:
                packed = packed.repeat(scale, axis=0).repeat(scale, axis=1)
            raw = np.zeros(self.size // 2, dtype='<u2')
            screen = raw.reshape(self.size // self.line_length, self.line_length // 2)
            top, left = y_pad+self.yoffset, x_pad+self.xoffset
            screen[top:top+image_h, left:left+image_w] = packed
            self.map[:] = raw.tobytes()
            return
        raw = bytearray(self.size)
        for y in range(canvas.height):
            row = bytearray()
            for x in range(canvas.width):
                i = (y * canvas.width + x) * 3
                color = tuple(canvas.data[i : i + 3])
                key = (color, scale)
                expanded_pixel = self._expanded.get(key)
                if expanded_pixel is None:
                    pixel = self._pixels.get(color)
                    if pixel is None:
                        r, g, b = color
                        pixel = struct.pack("<H", ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3))
                        self._pixels[color] = pixel
                    expanded_pixel = pixel * scale
                    self._expanded[key] = expanded_pixel
                row.extend(expanded_pixel)
            expanded = bytes(row)
            for yy in range(y_pad + y * scale, y_pad + (y + 1) * scale):
                start = (yy+self.yoffset) * self.line_length + (x_pad+self.xoffset) * 2
                raw[start : start + len(expanded)] = expanded
        self.map[:] = raw

    def close(self) -> None:
        self.map[:] = self.original
        self.map.close()
        self.file.close()


class ConsoleCursor:
    """Hide the Linux console cursor while fbdev owns the display."""
    def __init__(self):
        self.fd = None

    def hide(self):
        console = Path('/sys/class/tty/tty0/active').read_text().strip()
        if not console.startswith('tty') or not console[3:].isdigit():
            raise RuntimeError('Cannot identify the active Linux console')
        self.fd = os.open('/dev/' + console, os.O_WRONLY | os.O_NOCTTY)
        try:
            # DECSC saves the cursor position; CUP homes it; DECTCEM hides it.
            os.write(self.fd, b'\x1b7\x1b[H\x1b[?25l')
        except BaseException:
            self.restore()
            raise

    def restore(self):
        if self.fd is not None:
            try:
                os.write(self.fd, b'\x1b8\x1b[?25h')
            finally:
                os.close(self.fd)
                self.fd = None


def run_framebuffer(device: str, seconds_per_scene: int, fps: int, seed=None,
                    day_seconds=120, start_scene: str = "woodland") -> None:
    fb = None
    cursor = ConsoleCursor()
    running = True

    def stop(_sig: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        cursor.hide()
        fb = Framebuffer(device)
        scene = make_scene(seed)
        scene_started = time.monotonic()
        start_offset = SCENES.index(start_scene) * seconds_per_scene
        prepared_theme = None
        while running:
            started = time.monotonic()
            playlist_time = start_offset + started-scene_started
            theme, _ = scene_at(playlist_time, seconds_per_scene)
            fb.show(render_playlist(scene, playlist_time, seconds_per_scene, day_seconds))
            if theme != prepared_theme:
                next_theme = SCENES[(SCENES.index(theme)+1) % len(SCENES)]
                preload_animation(next_theme)
                prepared_theme = theme
            time.sleep(max(0, 1 / fps-(time.monotonic()-started)))
    finally:
        try:
            if fb is not None:
                fb.close()
        finally:
            cursor.restore()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dependency-free generative pixel-art frame")
    parser.add_argument("--export", metavar="FILE", help="write a one-frame PPM preview")
    parser.add_argument("--seed", type=int, help="repeat a particular scene")
    parser.add_argument("--framebuffer", action="store_true", help="animate on /dev/fb0 (default); requires framebuffer access")
    parser.add_argument("--device", default="/dev/fb0", help="framebuffer device (default: /dev/fb0)")
    parser.add_argument("--seconds-per-scene", type=int, default=SECONDS_PER_SCENE, help="seconds per habitat (default: 720; full loop: two hours)")
    parser.add_argument("--day-seconds", type=int, default=120, help="duration of a day/night cycle")
    parser.add_argument("--scene", choices=SCENES, help="habitat for a still export; otherwise --time selects from the playlist")
    parser.add_argument("--start-scene", choices=SCENES, default="woodland",
                        help="first habitat to show in the looping framebuffer playlist")
    parser.add_argument("--fps", type=int, default=6,
                        help="frames per second (default: 6; 12 is available for smoother motion)")
    parser.add_argument("--tty", action="store_true", help="optional animated ASCII woodland")
    parser.add_argument("--time", type=float, default=0, help="preview time in seconds")
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds_per_scene <= 0 or args.day_seconds <= 0:
        parser.error("fps and cycle duration must be positive")
    if args.scene and not args.export:
        parser.error("--scene requires --export")
    if args.tty and args.framebuffer:
        parser.error("choose either --tty or --framebuffer")


    scene = make_scene(args.seed)
    if args.export:
        output = Path(args.export).expanduser()
        canvas = (render(scene, args.time*12, cycle=args.day_seconds, theme=args.scene)
                  if args.scene else render_playlist(scene, args.time, args.seconds_per_scene, args.day_seconds))
        write_ppm(canvas, output)
        print(f"Wrote {output} ({W}x{H} pixels, nearest-neighbour scalable).")
    if args.framebuffer or (not args.tty and not args.export):
        run_framebuffer(args.device, args.seconds_per_scene, args.fps, args.seed,
                        args.day_seconds, args.start_scene)
    elif args.tty:
        from tty_scene import run_tty
        try:
            run_tty(args.seed if args.seed is not None else 1, args.fps, args.day_seconds)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

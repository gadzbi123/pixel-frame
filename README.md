# Woodland frame

## Detailed illustrated landscapes

All ten scenes now use hand-painted 1920×1080 illustrations rather than the
original 200×120 pixel renderer. They retain the accelerated day/night lighting
cycle and add scene-specific motion: flowing water, drifting butterflies and
balloons, swimming ducks, rotating blades, foliage, leaves, snow, and petals.

Native illustrations are in `assets/<scene>-illustrated.png`; their generated
sources are preserved beside them as `*-illustrated-source.png`. Pillow decodes
the loops; NumPy combines lighting and RGB565 conversion in a single pass for
the 8 FPS player.

All ten scenes use seamless 1920×1080 animated WebP loops at 8 FPS. Water,
weather, smoke, foliage, animals, and scene-specific landmarks move within the
painted artwork rather than relying only on foreground particles. The original
four-keyframe sheets remain in `assets/`, and woodland also includes a shareable
GIF. The next scene is decoded in the background. Lighting is applied only to
the displayed frame, with gradual brightness changes and no whole-loop lighting jobs.

The loops have 48 frames over six seconds. `scene_motion.py` selects water,
foliage, smoke, and weather regions for periodic motion with a fixed camera.
Ducks, balloons, butterflies, airborne objects, the boat, and helicopter move
as separate foreground layers. Windmill blades and the helicopter rotor rotate
independently. The meadow uses `meadow-motion-background.png`, a repaired
painting without blades, so rotating the windmill does not move the tower.
Foxes and rabbits have subtle local movement rather than a walking gait. These
are layered animations of painted artwork, not full character animation.

Open `previews/animations.html` to browse the rebuilt loops. Run
`python3 audit_animations.py` to check frame counts, timing, visible motion, and
the last-to-first transition. The older woodland GIF remains a historical
export; the display uses the rebuilt WebPs.

Rebuild a loop with build-only OpenCV dependencies (the player needs no OpenCV):

```bash
python3 -m venv --system-site-packages .venv-animation
.venv-animation/bin/pip install opencv-python-headless
.venv-animation/bin/python build_animated_asset.py woodland
```

Console auto-login on tty1 is configured independently of the player. Run the
player in the foreground when you want the display; Ctrl-C restores the screen
and cursor and returns to your shell.

An illustrated landscape collection for a Linux framebuffer. No desktop or touchscreen is needed. A two-minute dawn/day/dusk/night cycle changes the lighting.

Run directly on the display from your TTY:

```bash
sudo python3 pixel_frame.py --framebuffer
```

No desktop is needed. **Ctrl-C** quits and restores the previous screen. `--framebuffer` is also accepted explicitly.

Every illustration matches the detected 1920×1080 framebuffer. The renderer requires an RGB565 (16-bit) Linux framebuffer and reads its visible resolution, offsets, and stride. Ctrl-C in foreground mode restores the saved framebuffer and cursor.

The default illustrated playlist repeats every **two hours**, with **12 minutes per scene**:

| Minutes | Scene | Distinct animation |
| --- | --- | --- |
| 0–12 | Woodland | Flowing water, foliage, subtle fox movement, cabin smoke |
| 12–24 | Meadow | Butterflies, flowers, rotating windmill |
| 24–36 | Pond | Ducks glide, water ripples, reeds sway |
| 36–48 | Autumn | Orange tree canopies, drifting leaves, fox |
| 48–60 | Winter | Falling snow, snow-covered cabin, snowman, fox |
| 60–72 | Christmas | Falling snow, sparkling tree lights |
| 72–84 | Tornado | Swirling funnel, airborne car and cow |
| 84–96 | Tsunami | Rolling wave, bobbing boat, helicopter with spinning rotor |
| 96–108 | Balloons | Balloons drift above swaying flowers |
| 108–120 | Sakura | Cherry blossoms, falling petals, stream and arched bridge |

Each habitat retains the two-minute day/night cycle. After sakura, the playlist starts again automatically.

Timing options:

```bash
sudo python3 pixel_frame.py --seed 42 --fps 4 --seconds-per-scene 720 --day-seconds 120
```

`--seconds-per-scene` sets each habitat’s duration; `--day-seconds` independently sets the day/night cycle. For a quick tour use `--seconds-per-scene 10` (all ten in 100 seconds). Animation speed is independent of frame rate. `--device /dev/fb1` selects another framebuffer.

Choose the first scene with `--start-scene`:

```bash
sudo python3 pixel_frame.py --start-scene pond
```

Valid names are `woodland`, `meadow`, `pond`, `autumn`, `winter`, `christmas`, `tornado`, `tsunami`, `balloons`, and `sakura`. The next scenes continue in the usual order and loop after two hours.

The default is 8 FPS to match the animation assets. Use `--fps 4` for the lightest load or `--fps 12` if you prefer smoother movement and have CPU headroom.

Export a pixel preview at any point in the cycle:

```bash
python3 pixel_frame.py --seed 42 --time 30 --export preview.ppm
python3 pixel_frame.py --seed 42 --time 90 --export night.ppm
python3 pixel_frame.py --seed 42 --scene pond --time 30 --export pond.ppm
```

Run rendering checks:

```bash
python3 -m unittest -v
```

The ten-scene playlist is for framebuffer output. An optional ASCII version is available with `python3 pixel_frame.py --tty`. In that mode, Q / Esc quits, Space pauses, and N advances a quarter-day. It uses standard console colors and needs at least 40×18 characters.

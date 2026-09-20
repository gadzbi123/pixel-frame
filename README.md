# Woodland frame

## Detailed illustrated landscapes

All ten scenes now use hand-painted 1920×1080 illustrations rather than the
original 200×120 pixel renderer. They retain the accelerated day/night lighting
cycle and add scene-specific motion: water highlights, butterflies, ripples,
leaves, snow, swirling debris, sea foam, birds, petals, and fireflies.

Native illustrations are in `assets/<scene>-illustrated.png`; their generated
sources are preserved beside them as `*-illustrated-source.png`. Pillow prepares
the lighting variants, and NumPy keeps native-resolution framebuffer conversion
fast enough for the 8 FPS service.

All ten scenes use seamless 1920×1080 animated WebP loops at 8 FPS. Water,
weather, smoke, foliage, animals, and scene-specific landmarks move within the
painted artwork rather than relying only on foreground particles. The original
four-keyframe sheets remain in `assets/`, and woodland also includes a shareable
GIF. Day/night lighting phases and the next scene are prepared in the background
so lighting and playlist changes do not pause the display.

The display now runs automatically through the user service `pixel-frame`.
The service uses the Pi's existing passwordless sudo access to control the
active Linux console. On launch it saves the cursor position, moves to the
top-left, and hides the blinking cursor. Stopping restores the saved position
and shows the cursor again, including normal service stops and Ctrl-C.
Saving `pixel_frame.py` triggers a syntax check and automatic restart on the
same framebuffer. A restart begins the playlist again at woodland. The service
starts after reboot without logging in. To stop it, run
`pixel-frame stop`; to start it, run `pixel-frame start`. Use `pixel-frame run`
to run in the current TTY so Ctrl-C directly stops it and restores the cursor.

Weather cycles through clear skies, rain, fog, distant lightning, and clearing
mist; winter precipitation is snow. Ducks dabble, frogs catch flies, foxes sleep,
and rabbits hide at night. Occasional shooting stars, boats, deer, and firefly
gatherings appear. Spring petals, summer berries, fallen autumn leaves, and
gradually accumulating winter snow follow the existing habitat playlist.
The accelerated two-minute day/night cycle is unchanged.

The newer scenes include a nativity and changing village activity at Christmas;
moving cattle, cars, roofs and debris in the tornado; and a lighthouse, rescue
boat, helicopter, buoys, birds and animated foam in the tsunami. The sakura
bridge is fixed across a perspective river, with irises, ferns, koi and a crane.
The balloon scene stays deliberately calm and uncluttered.

An illustrated landscape collection for a Linux framebuffer. No desktop or touchscreen is needed. A two-minute dawn/day/dusk/night cycle brings out stars and fireflies.

Run directly on the display from your TTY:

```bash
pixel-frame run
```

No desktop is needed. **Ctrl-C** quits and restores the previous screen. `--framebuffer` is also accepted explicitly.

Every illustration matches the detected 1920×1080 framebuffer. The renderer requires an RGB565 (16-bit) Linux framebuffer and reads its visible resolution, offsets, and stride. Ctrl-C in foreground mode restores the saved framebuffer and cursor.

The default illustrated playlist repeats every **two hours**, with **12 minutes per scene**:

| Minutes | Scene | Distinct animation |
| --- | --- | --- |
| 0–12 | Woodland | Fox visits, rabbit hops, cabin smoke |
| 12–24 | Meadow | Butterflies, flowers, rotating windmill |
| 24–36 | Pond | Ducks swim and turn, frog hops, reeds sway |
| 36–48 | Autumn | Orange tree canopies, drifting leaves, fox |
| 48–60 | Winter | Falling snow, snow-covered cabin, snowman, fox |
| 60–72 | Christmas | Snowy village, decorated tree, twinkling lights, presents |
| 72–84 | Tornado | Rotating funnel, drifting storm clouds, swirling debris |
| 84–96 | Tsunami | Offshore wave that rises, travels, and subsides |
| 96–108 | Balloons | Colourful hot-air balloons above a flower meadow |
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

The default is 6 FPS to keep CPU use modest on a Pi-class system. Use `--fps 4` for the lightest load or `--fps 12` if you prefer smoother movement and have CPU headroom.

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

# Woodland frame

The display now runs automatically through the user service `pixel-frame`.
The service uses the Pi's existing passwordless sudo access to control the
active Linux console. On launch it saves the cursor position, moves to the
top-left, and hides the blinking cursor. Stopping restores the saved position
and shows the cursor again, including normal service stops and Ctrl-C.
Saving `pixel_frame.py` triggers a syntax check and automatic restart on the
same framebuffer. A restart begins the playlist again at the pond. The service
starts after reboot without logging in. To stop it, run
`systemctl --user stop pixel-frame`; to start it, run
`systemctl --user start pixel-frame`.

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

An animated woodland for a Linux text console. No desktop, touchscreen, Python packages, or network connection needed. Foxes visit and pause, rabbits hop and blink, birds flap across the sky, trees and grass sway, clouds drift, smoke rises, and water ripples. A two-minute dawn/day/dusk/night cycle brings out stars and fireflies.

Run pixel art directly on the display from your TTY (the default):

```bash
cd /home/gadzbi/pixel-frame
sudo python3 pixel_frame.py
```

No desktop is needed. **Ctrl-C** quits and restores the previous screen. `--framebuffer` is also accepted explicitly.

The 200×120 scene scales to exactly 800×480 with crisp 4× pixels; other framebuffer sizes are centered using integer scaling. Requires an RGB565 (16-bit) Linux framebuffer. Reads visible resolution, offsets, and stride from the selected device; unsupported color formats produce an error. Ctrl-C restores the saved framebuffer. The program does not change display resolution or console blanking settings. The framebuffer on the development machine currently reports 1920×1080; physical 800×480 hardware still needs an on-device check.

The default pixel-art playlist repeats every **two hours**, with **12 minutes per scene**:

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

The ten-scene playlist is for pixel output. An optional ASCII version is available with `python3 pixel_frame.py --tty`. In that mode, Q / Esc quits, Space pauses, and N advances a quarter-day. It uses standard console colors and needs at least 40×18 characters.

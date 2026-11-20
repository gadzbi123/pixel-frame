# Pixel Frame

Painted, animated landscapes for a Raspberry Pi picture frame. Ten 1920×1080
scenes play as seamless 24-second loops on the Linux framebuffer. There is no
desktop, touchscreen or network service involved.

![Woodland in late-afternoon light](pictures/previews/woodland.gif)

## Start it at boot

Run this once:

```bash
sudo ./install-service.sh
```

This installs `pixel-frame.service`, which shows the frame on the display's
first console (tty1) at every boot and restarts it if it crashes. It runs as
your account, not as root.

The service owns only tty1. SSH sessions are not involved: they get a normal
shell, with no login hook and no flag to remember. On a keyboard attached to
the display (or through a KVM), press Ctrl+Alt+F2 for a login prompt and
Ctrl+Alt+F1 to return to the frame. While another console is shown, the frame
pauses instead of drawing over it.

Control it without sudo, from anywhere:

```bash
pixel-frame status     # running? which scene?
pixel-frame logs       # recent scene changes and errors
pixel-frame stop       # tty1 gets its login prompt back
pixel-frame start
pixel-frame restart    # after editing the code
pixel-frame run --start-scene pond   # foreground run, Ctrl-C to quit
```

Only one copy can draw at a time. A second copy, for example `pixel-frame run`
while the service is running, exits with a message instead of fighting over
the screen.

To change options for the service, put them in `/etc/default/pixel-frame` and
restart:

```bash
PIXEL_FRAME_ARGS="--night 22:00-06:30 --seconds-per-scene 600"
```

## What you see

**Day and night.** Night is from 23:00 to 07:00; the rest of the day a scene is
shown as painted. At 23:00 the picture turns golden and night falls over 15
minutes, and from 06:45 it brightens back to day by 07:00. The display stays on
all night. Change the hours with `--night 22:30-06:00`. `--day-seconds 120`
runs a fast two-minute day instead, for demos.

**Night.** After dark the painting is relit as moonlight. The sky goes deep
blue and stars come out and twinkle, clouds and snow turn silver, and colours
fade. A sun in the painting fades to a soft glow. The Christmas scene is
painted at night already and keeps its own light.

**Lights at night.** Cabin windows, lanterns, the village, the tree lights and
lava keep glowing warm after dark.

**The real moon and weather.** Tell the frame where it is, and the night and
the weather follow the sky outside:

```bash
PIXEL_FRAME_ARGS="--location 52.23,21.01"   # latitude,longitude, in /etc/default/pixel-frame
```

- The moon rises in tonight's phase, from a thin crescent with faint
  earthshine to full, lit on the side facing the sun. It stands as high as the
  real moon and moves across the sky through the night: every painting is taken
  to face south, so it rises on the left and sets on the right. If a cloud,
  peak or tree in the painting stands in the way, it shows in the nearest clear
  sky. Scenes with little clear sky show no moon.
- A moonless night is darker, with brighter stars. A full moon lights the
  night more and washes out the faintest stars.
- Rain and snow fall over the picture, slanting with the wind. Grey cloud
  flattens the daylight and hides the moon and stars, fog veils the picture
  (lamps glow through it), and a thunderstorm brings lightning. Weather comes
  on and clears over a minute or two.
- Winter, Christmas, Tornado, Tsunami and Reef keep the weather they are
  painted with.

The weather comes from [Open-Meteo](https://open-meteo.com), which is free and
needs no account. The frame asks for the current weather every 15 minutes and
sends nothing but the location, rounded to about a kilometre. Without a
network it keeps the last report for three hours, then shows clear sky.
`pixel-frame logs` shows each change, such as `21:30 weather: rain 65%, cloud 100%`.
To see an effect now, run it in the foreground with
`--weather rain` (or `clear`, `cloudy`, `fog`, `drizzle`, `snow`, `storm`).

**Seasons.** Each scene shows for 12 minutes, then crossfades to the next.
Seasonal scenes join the playlist in their months; the rest play all year.

| Scene | Months | Motion |
| --- | --- | --- |
| Woodland | all year | River glints, swaying plants, the fox and rabbit |
| Meadow | all year | Butterflies, flowers, the windmill turning |
| Pond | all year | Ducks gliding, surface glints, reeds |
| Autumn | Sep–Nov | Drifting maple leaves, the fox |
| Winter | Dec–Feb | Falling snow, the lit cabin, the snowman and fox |
| Christmas | Dec | Snow, twinkling tree lights, the nativity |
| Tornado | all year | The funnel, the airborne car and cow |
| Tsunami | all year | Foam, the bobbing boat, the helicopter |
| Balloons | all year | Balloons drifting over flowers |
| Sakura | Mar–May | Falling petals, the stream under the bridge |

Scene slots follow the clock, so after a restart the frame shows the scene that
was due rather than starting over. `--start-scene NAME` shows one scene first,
whatever the season, and then rejoins the playlist. `--seconds-per-scene 20`
gives a quick tour.

## Your own scenes

Put stills or animations in `pictures/`: PNG, JPG, animated GIF or WebP. Each
file joins the playlist all year under its file name, with the same day and
night lighting. Pictures are scaled to fill the screen. Small pixel art keeps
its hard edges.

Stars come out by themselves over clear blue sky. To make windows or lamps
glow at night, add a mask with the same name and a `-glow.png` suffix, white
where the light is: `pictures/harbour.png` and `pictures/harbour-glow.png`. Trace
each light closely: the mask is on or off per pixel, and at night a lit wall
beside a window shows.

## Share a scene

GIFs play everywhere, so the frame exports them, lit for any time of day:

```bash
python3 pixel_frame.py --scene christmas --at 21:00 --export-gif christmas.gif
python3 pixel_frame.py --scene pond --at 07:30 --export pond.png
```

GIFs are 640 pixels wide by default (`--gif-width`). Every frame shares one
palette and only stores the pixels that changed, which keeps a 24-second loop
small enough to send in a chat. `pictures/previews/woodland.gif` above is made
this way.

## How the loops are made

Each scene starts as one painted keyframe. `upscale_keyframes.py` enlarges the
836×470 painting to 1920×1080 with Real-ESRGAN, so the loops are sharp rather
than stretched. `build_animated_asset.py` then animates the keyframe:
`scene_motion.py` marks water, foliage, smoke and weather regions for periodic
motion with a fixed camera, and lifts ducks, balloons, butterflies, the boat
and the helicopter onto their own moving layers. The windmill sails and the
rotor turn independently. Motion keeps a six-second beat; a gust that rises and
falls once per loop, and particles that take a new path on every fall, keep
the 24 seconds from repeating.

The build needs OpenCV and PyTorch, the player needs neither:

```bash
python3 -m venv --system-site-packages .venv-animation
.venv-animation/bin/pip install opencv-python-headless
.venv-animation/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
mkdir -p ~/.cache/pixel-frame/models && cd ~/.cache/pixel-frame/models
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd -
.venv-animation/bin/python upscale_keyframes.py woodland   # 15–30 minutes on a Pi 4
.venv-animation/bin/python build_animated_asset.py woodland
python3 audit_animations.py
```

`audit_animations.py` checks frame counts, timing, visible motion and the
last-to-first transition, and refreshes the posters for
`pictures/previews/animations.html`, a browser gallery of the loops.

## Under the hood

The player decodes the current and the next scene in the background and keeps
their frames in the display's RGB565 format. Lighting a frame is then one
table lookup per pixel, about 35 ms on a Pi 4 at 8 FPS. The tables come in
fours, one per cell of a 2×2 dither, because 16-bit colour has too few dark
levels for a night sky and would band. It needs an RGB565 (16-bit) Linux
framebuffer and reads its resolution, offsets and stride.

```bash
python3 -m unittest -v                                        # player
.venv-animation/bin/python -m unittest test_animation_encoder # build
```

`python3 pixel_frame.py --tty` shows an ASCII woodland in a text terminal
instead: Q or Esc quits, Space pauses and N skips ahead a quarter of a day.

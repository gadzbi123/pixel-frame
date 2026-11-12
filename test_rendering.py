import datetime as dt
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import pixel_frame as pf
from tty_scene import render_text

UTC = dt.timezone.utc
WARSAW = (52.2297, 21.0122)


def unpack(code):
    r, g, b = code >> 11 & 31, code >> 5 & 63, code & 31
    return r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2


class LightingTests(unittest.TestCase):
    def test_lighting_changes_gradually_including_day_wrap(self):
        luts = [pf.channel_lut(n/240).astype(int) for n in range(240)]
        for first, second in zip(luts, luts[1:]+luts[:1]):
            self.assertLessEqual(np.abs(first-second).max(), 4)

    def test_single_lookup_matches_per_channel_lighting(self):
        for step in (0, 60, 130, 180):
            channels, lut = pf.channel_lut(step/240), pf.lut565(step)
            for code in (0xFFFF, 0x8410, 0x1D5D, 0xF800, 0x07E0, 0x39E7):
                lit = int(lut[code])
                fields = (lit >> 11, lit >> 5 & 63, lit & 31)
                for channel, value, levels in zip(range(3), unpack(code), (31, 63, 31)):
                    # Within one RGB565 step of lighting the 8-bit colour.
                    self.assertLessEqual(abs(fields[channel] - int(channels[channel][value])*levels/255), 1)

    def test_night_is_darker_but_glowing_windows_are_not(self):
        noon, midnight = pf.LIGHT_STEPS//4, pf.LIGHT_STEPS*3//4
        warm = 0xFE60  # a lit-window yellow
        self.assertLess(sum(unpack(int(pf.lut565(midnight)[warm]))),
                        sum(unpack(int(pf.lut565(noon)[warm]))))
        self.assertGreaterEqual(sum(unpack(int(pf.lut565(midnight, glow=True)[warm]))),
                                sum(unpack(int(pf.lut565(noon)[warm]))))
        self.assertEqual(pf.lut565(noon, glow=True)[warm], pf.lut565(noon)[warm])

    def test_dimming_darkens(self):
        self.assertLess(sum(unpack(int(pf.lut565(60, .6)[0xFFFF]))),
                        sum(unpack(int(pf.lut565(60)[0xFFFF]))))

    def test_glowing_pixels_keep_their_brightness_at_night(self):
        frames = np.full((1, 2, 2), 0xFE60, np.uint16)
        loop = pf.Loop(frames, [1000], np.array([0]))
        image = loop.lit(0, pf.LIGHT_STEPS*3//4, 1.0).reshape(-1)
        self.assertGreater(sum(unpack(int(image[0]))), sum(unpack(int(image[1]))))


class SunTests(unittest.TestCase):
    def assertClose(self, actual, hour, minute, tolerance=4):
        expected = actual.replace(hour=hour, minute=minute, second=0, microsecond=0)
        self.assertLessEqual(abs((actual-expected).total_seconds()), tolerance*60, actual)

    def test_warsaw_solstices_match_the_almanac(self):
        rise, _, sunset = pf.sun_events(dt.date(2026, 6, 21), *WARSAW)
        self.assertClose(rise, 2, 14)    # 04:14 CEST
        self.assertClose(sunset, 19, 1)  # 21:01 CEST
        rise, _, sunset = pf.sun_events(dt.date(2026, 12, 21), *WARSAW)
        self.assertClose(rise, 6, 43)    # 07:43 CET
        self.assertClose(sunset, 14, 25)  # 15:25 CET

    def test_phase_landmarks_follow_the_sun(self):
        day = dt.date(2026, 11, 11)
        rise, noon, sunset = pf.sun_events(day, *WARSAW)
        next_rise = pf.sun_events(day + dt.timedelta(days=1), *WARSAW)[0]
        for when, phase in ((rise, 0), (noon, .25), (sunset, .5),
                            (sunset + (next_rise-sunset)/2, .75)):
            value = pf.sun_phase(when + dt.timedelta(seconds=1), *WARSAW)
            self.assertLess(min(abs(value-phase), 1-abs(value-phase)), .001)

    def test_phase_moves_smoothly_through_a_year_of_days(self):
        start = dt.datetime(2026, 1, 1, tzinfo=UTC)
        previous = pf.sun_phase(start, *WARSAW)
        for minutes in range(10, 366*24*60, 10*24*60 + 7):
            for step in range(0, 24*60, 10):
                value = pf.sun_phase(start + dt.timedelta(minutes=minutes+step), *WARSAW)
                if step:
                    self.assertLess((value-previous) % 1, .02)
                previous = value

    def test_polar_night_and_day_do_not_fail(self):
        for day in (dt.datetime(2026, 12, 21, 12, tzinfo=UTC), dt.datetime(2026, 6, 21, 0, tzinfo=UTC)):
            self.assertTrue(0 <= pf.sun_phase(day, 78.2, 15.6) < 1)

    def test_tzdata_coordinates(self):
        self.assertEqual(pf.iso6709("+5215+02100"), (52.25, 21.0))
        latitude, longitude = pf.iso6709("-332318-0703716")
        self.assertAlmostEqual(latitude, -(33 + 23/60 + 18/3600))
        self.assertAlmostEqual(longitude, -(70 + 37/60 + 16/3600))


class PlaylistTests(unittest.TestCase):
    def test_seasonal_scenes_appear_in_their_months(self):
        december, july, april = (pf.playlist(pf.SCENES, m) for m in (12, 7, 4))
        self.assertIn("christmas", december)
        self.assertIn("winter", december)
        self.assertNotIn("sakura", december)
        self.assertFalse({"christmas", "winter", "autumn", "sakura"} & set(july))
        self.assertIn("sakura", april)
        self.assertIn("woodland", july)

    def test_slots_follow_the_clock(self):
        start = time.mktime((2026, 7, 15, 12, 0, 0, 0, 0, -1))
        shown = pf.playlist(pf.SCENES, 7)
        seen = [pf.scene_at(pf.SCENES, start + n*720, 720) for n in range(len(shown))]
        self.assertEqual(sorted(name for name, _ in seen), sorted(shown))
        self.assertTrue(all(remaining == 720 for _, remaining in seen))
        self.assertEqual(pf.scene_at(pf.SCENES, start+100, 720), (seen[0][0], 620))

    def test_upcoming_scene_is_the_one_shown_next(self):
        settings = pf.Settings(seconds_per_scene=720)
        for moment in ((2026, 11, 30, 23, 55), (2026, 12, 31, 23, 50), (2026, 7, 15, 9, 7)):
            now = time.mktime(moment + (0, 0, 0, -1))
            for offset in range(0, 3600, 97):
                _, upcoming, remaining = pf.schedule(pf.SCENES, settings, now, now+offset)
                after = pf.schedule(pf.SCENES, settings, now, now+offset+remaining+.01)[0]
                self.assertEqual(after, upcoming)

    def test_start_scene_shows_first_then_the_clock(self):
        settings = pf.Settings(seconds_per_scene=720, start_scene="christmas")
        now = time.mktime((2026, 7, 15, 12, 5, 0, 0, 0, -1))
        current, upcoming, remaining = pf.schedule(pf.SCENES, settings, now, now)
        self.assertEqual((current, remaining), ("christmas", 720))
        self.assertEqual(pf.schedule(pf.SCENES, settings, now, now+721)[0], upcoming)


class QuietHoursTests(unittest.TestCase):
    def test_dims_before_and_switches_off_during_quiet_hours(self):
        quiet = pf.parse_quiet("23:00-07:00")
        state = lambda clock: pf.quiet_state(pf.clock_minutes(clock), quiet)
        self.assertEqual(state("23:30"), (True, 1.0))
        self.assertEqual(state("06:59"), (True, 1.0))
        self.assertEqual(state("07:00"), (False, 1.0))
        self.assertEqual(state("21:59"), (False, 1.0))
        self.assertAlmostEqual(state("22:30")[1], .8)
        self.assertGreater(state("22:10")[1], state("22:50")[1])

    def test_off_and_invalid_values(self):
        self.assertIsNone(pf.parse_quiet("off"))
        self.assertEqual(pf.quiet_state(0, None), (False, 1.0))
        for text in ("23:00", "25:00-07:00", "noon-07:00"):
            with self.assertRaises(ValueError):
                pf.parse_quiet(text)


class FrameTests(unittest.TestCase):
    def test_screen_view_honours_stride_and_offsets(self):
        buffer = bytearray(8*3)
        view = pf.screen_view(buffer, 2, 2, 8, 1, 1)
        view[:] = [[1, 2], [3, 4]]
        rows = np.frombuffer(bytes(buffer), "<u2").reshape(3, 4)
        np.testing.assert_array_equal(rows, [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]])

    def test_dissolve_moves_from_current_to_upcoming(self):
        counts = []
        for amount in (0, .25, .5, .75, 1):
            image = pf.dissolve(np.zeros((64, 64), np.uint16), np.ones((64, 64), np.uint16), amount)
            counts.append(int(image.sum()))
        self.assertEqual(counts, [0, 1024, 2048, 3072, 4096])

    def test_fit_covers_the_screen(self):
        for size in ((100, 100), (4000, 1000), (40, 30)):
            self.assertEqual(pf.fit(Image.new("RGB", size), (192, 108)).size, (192, 108))


def tiny_animation(path, colors=((200, 40, 40), (40, 200, 40), (40, 40, 200)), durations=(50, 100, 150)):
    frames = [Image.new("RGB", (32, 18), color) for color in colors]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=list(durations), loop=0)


class DropInTests(unittest.TestCase):
    def test_gifs_and_stills_in_the_scenes_folder_join_the_playlist(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            tiny_animation(folder/"tiny.gif")
            Image.new("RGB", (64, 36), (90, 90, 90)).save(folder/"still.png")
            mask = Image.new("L", (64, 36))
            mask.paste(255, (0, 0, 8, 4))
            mask.save(folder/"still-glow.png")
            (folder/"broken.gif").write_bytes(b"not a gif")
            library = pf.scene_library(folder)
            self.assertEqual(list(library)[:10], list(pf.SCENES))
            self.assertEqual(list(library)[10:], ["still", "tiny"])

            loop = pf.decode_loop("tiny", library["tiny"], (64, 36))
            self.assertEqual(loop.frames.shape, (3, 36, 64))
            self.assertEqual(loop.ends, [50, 150, 300])
            tiny_animation(folder/"tiny.webp")
            self.assertEqual(pf.decode_loop("tiny", folder/"tiny.webp", (64, 36)).ends, [50, 150, 300])
            self.assertEqual(int(loop.frame(.12)[0, 0]), int(loop.frames[1][0, 0]))
            self.assertEqual(int(loop.frame(.31)[0, 0]), int(loop.frames[0][0, 0]))

            still = pf.decode_loop("still", library["still"], (64, 36))
            self.assertEqual(len(still.glow), 8*4)

    def test_gif_export_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            tiny_animation(folder/"tiny.gif", durations=(125, 125, 125))
            output = folder/"share.gif"
            pf.export_gif("tiny", folder/"tiny.gif", .25, output, width=64)
            source = [np.asarray(pf.lit_image(Image.new("RGB", (64, 36), color), .25, None)).astype(int)
                      for color in ((200, 40, 40), (40, 200, 40), (40, 40, 200))]
            with Image.open(output) as gif:
                self.assertEqual((gif.n_frames, gif.info["loop"]), (3, 0))
                for index, expected in enumerate(source):
                    gif.seek(index)
                    frame = np.asarray(gif.convert("RGB")).astype(int)
                    self.assertLess(np.abs(frame-expected).mean(), 10, index)


class TerminalTests(unittest.TestCase):
    def test_terminal_sizes_and_phases(self):
        for width, height in [(100, 30), (80, 24), (50, 30), (40, 18), (20, 8)]:
            for t in (0, 15, 30, 60, 90):
                c = render_text(width, height, t)
                self.assertEqual(len(c.cells), height)
                self.assertTrue(all(len(row) == width for row in c.cells))
                self.assertTrue(all(ord(ch) < 128 for row in c.cells for ch, _ in row))


if __name__ == "__main__":
    unittest.main()

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import pixel_frame as pf
from tty_scene import render_text



def unpack(code):
    r, g, b = code >> 11 & 31, code >> 5 & 63, code & 31
    return r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2


DAY, NIGHT = (12.0, False), (-12.0, False)


def brightness(code):
    return sum(unpack(int(code)))


class LightingTests(unittest.TestCase):
    def test_light_changes_gradually_from_day_to_night(self):
        grid = np.stack(np.meshgrid(*[np.arange(0, 256, 51)]*3), -1).reshape(-1, 3)
        for rising in (False, True):
            steps = [pf.relight(grid, eighths/8, rising) for eighths in range(96, -97, -1)]
            for first, second in zip(steps, steps[1:]):
                self.assertLessEqual(np.abs(first-second).max(), 6)

    def test_every_dither_table_is_within_a_level_of_the_lighting(self):
        for light in (DAY, (0.0, False), (-4.0, True), NIGHT):
            tables = pf.lut565(light)
            for code in (0xFFFF, 0x8410, 0x1D5D, 0xF800, 0x07E0, 0x39E7):
                exact = np.clip(pf.relight(np.array(unpack(code)), *light), 0, 255)*np.array([31, 63, 31])/255
                for table in tables:
                    lit = int(table[code])
                    for field, level in zip((lit >> 11, lit >> 5 & 63, lit & 31), exact):
                        self.assertLessEqual(abs(field - level), 1)

    def test_dither_cells_average_out_the_coarse_night_levels(self):
        tables = pf.lut565(NIGHT)
        for code in (0x1D5D, 0x8410, 0x5AFF):
            exact = np.clip(pf.relight(np.array(unpack(code)), *NIGHT), 0, 255)[2]*31/255
            self.assertLess(abs(np.mean([int(table[code]) & 31 for table in tables]) - exact), .26)

    def test_by_day_the_painting_is_shown_as_painted(self):
        for code in (0xFFFF, 0x8410, 0x1D5D, 0xF800, 0x07E0, 0x39E7):
            self.assertEqual({int(table[code]) for table in pf.lut565(DAY)}, {code})

    def test_night_is_dark_and_blue_with_the_sky_darker_than_the_clouds(self):
        sky, cloud, grass = (pf.relight(np.array(colour), *NIGHT)
                             for colour in ((70, 140, 230), (240, 240, 245), (120, 180, 60)))
        self.assertLess(sky.sum(), 130)
        self.assertLess(sky.sum(), cloud.sum()/2)
        self.assertLess(cloud.sum(), (240 + 240 + 245)*.6)
        for colour in (sky, cloud, grass):
            self.assertGreater(colour[2], colour[1])
            self.assertGreater(colour[1], colour[0])

    def test_a_scene_painted_at_night_stays_as_painted(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("christmas", "meadow"):
                Image.new("RGB", (64, 36), (230, 150, 60)).save(Path(folder)/f"{name}.png")
                Image.new("L", (64, 36), 255).crop((0, 0, 64, 36)).save(Path(folder)/f"{name}-glow.png")
            christmas, meadow = (pf.decode_loop(name, Path(folder)/f"{name}.png", (64, 36)) for name in ("christmas", "meadow"))
        np.testing.assert_array_equal(christmas.lit(0, NIGHT), christmas.lit(0, DAY))
        self.assertFalse((meadow.lit(0, NIGHT) == meadow.lit(0, DAY)).any())

    def test_low_sun_warms_the_picture(self):
        grey = np.array((180, 180, 180))
        for rising in (True, False):
            red, green, blue = pf.relight(grey, 1.0, rising)
            self.assertGreater(red, green + 15)
            self.assertGreater(green, blue + 15)
        self.assertGreater(pf.relight(grey, 1.0, True)[2], pf.relight(grey, 1.0, False)[2])  # dawn is the cooler

    def test_dusk_darkens_step_by_step_without_a_jump(self):
        for colour in ((120, 160, 90), (70, 140, 230)):   # grass, sky
            levels = [pf.relight(np.array(colour), float(elevation), False).sum() for elevation in range(0, -13, -2)]
            for before, after in zip(levels, levels[1:]):
                self.assertLess(after, before*.99)
                self.assertGreater(after, before*.8)

    def test_night_is_darker_but_glowing_windows_are_not(self):
        warm = 0xFE60  # a lit-window yellow
        self.assertLess(brightness(pf.lut565(NIGHT)[0][warm]), brightness(pf.lut565(DAY)[0][warm]))
        self.assertGreaterEqual(brightness(pf.lut565(NIGHT, glow=True)[0][warm]),
                                brightness(pf.lut565(DAY)[0][warm]))
        self.assertEqual(pf.lut565(DAY, glow=True)[0][warm], pf.lut565(DAY)[0][warm])

    def test_glowing_pixels_keep_their_brightness_at_night(self):
        frames = np.full((1, 2, 2), 0xFE60, np.uint16)
        loop = pf.Loop(frames, [1000], np.array([0]))
        image = loop.lit(0, NIGHT).reshape(-1)
        self.assertGreater(brightness(image[0]), brightness(image[1]))


def landscape(size=(320, 180), balloon=False):
    """Blue sky with a cloud over green land and a lake the colour of the sky."""
    image = Image.new("RGB", size, (60, 130, 230))
    width, height = size
    image.paste((245, 245, 250), (width//2, height//8, width*3//4, height//4))  # cloud
    image.paste((70, 140, 60), (0, height*2//5, width, height))                 # land
    image.paste((60, 130, 230), (width//4, height*3//5, width*3//4, height*4//5))  # lake
    if balloon:
        image.paste((220, 40, 40), (0, 0, width//4, height//4))
    return image


class StarTests(unittest.TestCase):
    def test_stars_come_out_only_in_clear_sky(self):
        picture = landscape()
        stars = pf.find_stars(np.asarray(picture))
        rows, columns = np.divmod(stars.index, 320)
        self.assertGreater(len(rows), 20)
        self.assertLess(rows.max(), 180*2//5)                                  # not on land or lake
        self.assertFalse(((rows >= 180//8) & (rows < 180//4) & (columns >= 160) & (columns < 240)).any())
        self.assertIsNone(pf.find_stars(np.asarray(Image.new("RGB", (320, 180), (70, 140, 60)))))

    def test_stars_shine_at_night_and_hide_behind_what_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"sky.gif"
            landscape().save(path, save_all=True, append_images=[landscape(balloon=True)], duration=[100, 100], loop=0)
            loop = pf.decode_loop("sky", path, (320, 180))
        rows, columns = np.divmod(loop.stars.index, 320)
        behind = (rows < 180//4) & (columns < 320//4)
        self.assertTrue(behind.any() and (~behind).any())
        self.assertTrue(loop.clear[0].all())
        np.testing.assert_array_equal(loop.clear[1], ~behind)

        night = loop.lit(0, NIGHT).reshape(-1).astype(int)
        starless = pf.Loop(loop.frames, loop.ends)
        plain = starless.lit(0, NIGHT).reshape(-1).astype(int)
        changed = np.flatnonzero(night != plain)
        self.assertGreater(len(changed), 10)
        self.assertTrue(set(changed) <= set(loop.stars.index))
        self.assertTrue(all(brightness(night[i]) > brightness(plain[i]) for i in changed))
        hidden = loop.lit(.1, NIGHT).reshape(-1)[loop.stars.index[behind]]
        np.testing.assert_array_equal(hidden, starless.lit(.1, NIGHT).reshape(-1)[loop.stars.index[behind]])

    def test_no_stars_by_day(self):
        frames = pf.pack565(np.asarray(landscape()))[None]
        stars = pf.find_stars(np.asarray(landscape()))
        np.testing.assert_array_equal(pf.Loop(frames, [1000], stars=stars).lit(0, DAY),
                                      pf.Loop(frames, [1000]).lit(0, DAY))


def local(hour, minute, second=0):
    return time.mktime((2026, 11, 14, hour, minute, second, 0, 0, -1))


class NightHoursTests(unittest.TestCase):
    def test_day_outside_the_night_hours_and_night_inside_them(self):
        light = pf.Settings().light
        for clock in ((7, 0), (7, 1), (12, 0), (16, 30), (20, 0), (22, 59)):
            self.assertEqual(light(local(*clock)), DAY, clock)
        for clock in ((23, 15), (0, 0), (3, 0), (6, 45)):
            self.assertEqual(light(local(*clock)), NIGHT, clock)

    def test_the_sun_sets_at_eleven_and_rises_by_seven(self):
        light = pf.Settings().light
        self.assertEqual(light(local(23, 0)), DAY)
        dusk = light(local(23, 10))
        self.assertFalse(dusk[1])
        self.assertTrue(0 < pf.night_amount(dusk[0]) < 1)
        dawn = light(local(6, 50))
        self.assertTrue(dawn[1])
        self.assertTrue(0 < pf.night_amount(dawn[0]) < 1)
        self.assertEqual(pf.star_strength(light(local(23, 1))[0]), 0)

    def test_light_moves_in_small_steps_through_dusk(self):
        light = pf.Settings().light
        elevations = [light(local(22, 55) + 2*n)[0] for n in range(30*25)]
        self.assertEqual((max(elevations), min(elevations)), (12, -12))
        self.assertLessEqual(max(abs(a-b) for a, b in zip(elevations, elevations[1:])), .125)

    def test_other_night_hours(self):
        light = pf.Settings(night=pf.parse_night("01:30-05:00")).light
        self.assertEqual(light(local(23, 30)), DAY)
        self.assertEqual(light(local(3, 0)), NIGHT)
        self.assertEqual(light(local(5, 0)), DAY)

    def test_invalid_night_hours(self):
        self.assertEqual(pf.parse_night("23:00-07:00"), (23*60, 7*60))
        for text in ("23:00", "25:00-07:00", "noon-07:00", "off"):
            with self.assertRaises(ValueError):
                pf.parse_night(text)

    def test_a_demo_day_runs_from_full_day_to_full_night(self):
        settings = pf.Settings(day_seconds=120)
        lights = {settings.light(second) for second in range(120)}
        self.assertTrue({DAY, NIGHT} <= lights)
        self.assertTrue(any(rising for _, rising in lights))


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
            pf.export_gif("tiny", folder/"tiny.gif", DAY, output, width=64)
            source = [np.full((36, 64, 3), color) for color in ((200, 40, 40), (40, 200, 40), (40, 40, 200))]
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

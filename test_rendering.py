import unittest
from unittest.mock import patch
import pixel_frame
from pixel_frame import Canvas, Framebuffer, make_scene, render, render_playlist, scene_at, SCENES, FOX, sprite
from tty_scene import render_text, phase_at


class RenderingTests(unittest.TestCase):
    def test_celestial_resets_are_hidden_below_horizon(self):
        for phase in (0, .0001, .4999, .5, .5001, .9999):
            _, y = pixel_frame.celestial_position(phase)
            self.assertGreaterEqual(y-7, 78)
        self.assertEqual(pixel_frame.celestial_position(.25)[1], 13)

    def test_pond_ripples_never_write_land(self):
        c = Canvas()
        for y in range(120):
            pixel_frame.pond_ripple(c, -10, y, 220, (255, 255, 255))
            left, right = pixel_frame.pond_bounds(y)
            for x in range(200):
                self.assertEqual(bool(c.data[(y*200+x)*3]), left <= x < right)

    def test_ducks_and_wakes_stay_in_pool_without_collisions(self):
        for step in range(2400):
            ducks = [pixel_frame.duck_position(step*.1, n) for n in range(3)]
            for x, y, right in ducks:
                for row in range(y, y+7):
                    left, edge = pixel_frame.pond_bounds(row)
                    self.assertGreaterEqual(x-5, left)
                    self.assertLess(x+18, edge)
            for a, b in zip(ducks, ducks[1:]):
                self.assertGreater(b[1], a[1]+6)

    def test_sakura_bridge_crosses_the_perspective_river(self):
        narrow = pixel_frame.sakura_river_bounds(82)
        wide = pixel_frame.sakura_river_bounds(119)
        bridge = pixel_frame.sakura_river_bounds(102)
        self.assertLess(narrow[1]-narrow[0], bridge[1]-bridge[0])
        self.assertLess(bridge[1]-bridge[0], wide[1]-wide[0])

    def test_animal_silhouettes_are_readable_sizes(self):
        self.assertGreaterEqual(len(pixel_frame.SLEEPING_FOX), 8)
        self.assertGreaterEqual(max(map(len, pixel_frame.SLEEPING_FOX)), 20)
        self.assertGreaterEqual(len(pixel_frame.DEER), 12)
        self.assertIn('A', ''.join(pixel_frame.DEER))
        self.assertGreaterEqual(len(pixel_frame.COW), 7)

    def test_sleeping_fox_has_no_transparent_colour_keys(self):
        original = pixel_frame.sprite
        with patch('pixel_frame.sprite', wraps=original) as draw:
            pixel_frame.render(make_scene(42), 90*12, theme='woodland')
        fox = draw.call_args_list[0]
        keys = set(''.join(fox.args[1])) - {' '}
        self.assertLessEqual(keys, set(fox.args[4]))

    def test_framebuffer_repeats_pixels_with_stride_padding(self):
        fb = Framebuffer.__new__(Framebuffer)
        fb.width, fb.height, fb.line_length, fb.size = 4, 2, 12, 24
        fb.xoffset = fb.yoffset = 0
        fb.map = bytearray(24)
        c = Canvas(2, 1)
        c.pixel(0, 0, (255, 0, 0))
        c.pixel(1, 0, (0, 0, 255))
        fb.show(c)
        self.assertEqual(fb.map[:8], b'\x00\xf8'*2+b'\x1f\x00'*2)
        self.assertEqual(fb.map[:12], fb.map[12:])
        self.assertEqual(len(fb.map), 24)

    def test_cycle_and_motion(self):
        scene = make_scene(42)
        frames = [bytes(render(scene, t*12).data) for t in (0, 1, 30, 60, 90)]
        self.assertEqual(len(set(frames)), 5)
        self.assertEqual(frames[0], bytes(render(make_scene(42), 0).data))
        for frame in frames:
            self.assertEqual(len(frame), 200*120*3)
        self.assertEqual(phase_at(120, 120), (0, 'DAWN'))

    def test_hour_playlist_boundaries_and_repeat(self):
        for i, theme in enumerate(SCENES):
            self.assertEqual(scene_at(i*720), (theme, 0))
            self.assertEqual(scene_at(i*720+719.9)[0], theme)
        self.assertEqual(len(SCENES), 10)
        self.assertEqual(scene_at(3600), ('christmas', 0))
        self.assertEqual(scene_at(7200), ('woodland', 0))
        self.assertEqual(scene_at(10, 10), ('meadow', 0))
        scene = make_scene(42)
        self.assertEqual(render_playlist(scene, 0).data, render_playlist(scene, 7200).data)

    def test_start_scene_offsets_the_playlist(self):
        scene = make_scene(42)
        for index, theme in enumerate(SCENES):
            offset = index * 720
            self.assertEqual(scene_at(offset)[0], theme)
            self.assertEqual(
                render_playlist(scene, offset).data,
                render(scene, 0, theme=theme).data,
            )

    def test_every_habitat_animates_by_day_and_night(self):
        scene = make_scene(42)
        frames = []
        for theme in SCENES:
            for seconds in (15, 90):
                first = render(scene, seconds*12, theme=theme)
                second = render(scene, (seconds+.5)*12, theme=theme)
                self.assertNotEqual(first.data, second.data, theme)
                self.assertEqual(len(first.data), 200*120*3)
                frames.append(bytes(first.data))
        self.assertEqual(len(set(frames)), 20)

    def test_fox_faces_right_while_walking_right(self):
        positions = []
        original = pixel_frame.sprite
        for seconds in (2, 4, 28, 30):
            with patch('pixel_frame.sprite', wraps=original) as draw:
                render(make_scene(42), seconds*12)
                fox = draw.call_args_list[0]
                self.assertTrue(fox.kwargs['flip'])
                positions.append(fox.args[2])
        self.assertLess(positions[0], positions[1])
        self.assertLess(positions[2], positions[3])
        # Mirroring must use the whole bounding box even for ragged rows.
        colors = {'O': (200, 100, 50), 'K': (30, 30, 30), 'W': (240, 230, 210)}
        width = max(map(len, FOX))
        left, right = Canvas(width, len(FOX)), Canvas(width, len(FOX))
        sprite(left, FOX, 0, 0, colors)
        sprite(right, FOX, 0, 0, colors, flip=True)
        for y in range(len(FOX)):
            for x in range(width):
                a, b = (y*width+x)*3, (y*width+width-1-x)*3
                self.assertEqual(left.data[a:a+3], right.data[b:b+3])

    def test_terminal_sizes_and_phases(self):
        for width, height in [(100,30), (80,24), (50,30), (40,18), (20,8)]:
            for t in (0, 15, 30, 60, 90):
                c = render_text(width, height, t)
                self.assertEqual(len(c.cells), height)
                self.assertTrue(all(len(row)==width for row in c.cells))
                self.assertTrue(all(ord(ch)<128 for row in c.cells for ch, _ in row))


if __name__ == '__main__':
    unittest.main()

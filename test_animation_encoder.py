"""Build-time tests; use .venv-animation/bin/python -m unittest here."""
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from build_animated_asset import stabilize, coherent_frames
    from scene_motion import scene_frames,subjects,SUBJECTS
    from PIL import Image
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV is a build-only dependency")
class AnimationEncoderTests(unittest.TestCase):
    def test_all_selected_subjects_are_extracted(self):
        assets=Path(__file__).with_name('assets')
        for name,specs in SUBJECTS.items():
            with self.subTest(scene=name):
                with Image.open(assets/f'{name}-animation-sheet.png') as sheet:
                    w,h=sheet.size
                    base=np.array(sheet.crop((0,0,w//2,h//2)).resize((960,540)))
                if name=='meadow':
                    with Image.open(assets/'meadow-motion-background.png') as clean:
                        base=np.array(clean.convert('RGB').resize((960,540)))
                _,layers=subjects(base,name)
                self.assertEqual(len(layers),len(specs))

    def test_each_scene_moves_and_wraps_without_a_large_jump(self):
        assets = Path(__file__).with_name('assets')
        names = ('woodland','meadow','pond','autumn','winter','christmas',
                 'tornado','tsunami','balloons','sakura')
        cv2.setNumThreads(2)
        for name in names:
            with self.subTest(scene=name):
                with Image.open(assets/f'{name}-animation-sheet.png') as sheet:
                    w,h=sheet.size
                    base=np.array(sheet.crop((0,0,w//2,h//2)).resize((320,180)))
                frames=[np.asarray(f).astype(np.float32) for f in scene_frames(base,name,24)]
                changes=[np.abs(a-b).mean() for a,b in zip(frames,frames[1:])]
                self.assertGreater(max(np.abs(f-frames[0]).mean() for f in frames),.5)
                seam=np.abs(frames[-1]-frames[0]).mean()
                self.assertLess(seam,max(changes)*2+.1)

    def test_stationary_bridge_stays_fixed_under_water_motion(self):
        assets = Path(__file__).with_name('assets')
        with Image.open(assets/'sakura-animation-sheet.png') as sheet:
            w,h=sheet.size
            base=np.array(sheet.crop((0,0,w//2,h//2)).resize((320,180)))
        frames=[np.asarray(f) for f in scene_frames(base,'sakura',24,particles=False)]
        # These pixels belong to the painted bridge above the water, where
        # neither its structure nor its position should be warped.
        first=frames[0][116:121,186:233].astype(float)
        for frame in frames[1:]:
            self.assertLess(np.abs(first-frame[116:121,186:233]).mean(),1)

    def test_camera_translation_is_removed(self):
        original = np.random.default_rng(42).integers(
            0, 256, (180, 320, 3), dtype=np.uint8)
        shifted = cv2.warpAffine(original, np.float32([[1,0,8],[0,1,-4]]),
                                 (320,180), borderMode=cv2.BORDER_REFLECT_101)
        result = stabilize([original, shifted])[1]
        error = np.abs(original[15:-15,15:-15].astype(float)
                       - result[15:-15,15:-15]).mean()
        self.assertLess(error, 5)

    def test_stationary_landmarks_do_not_flicker(self):
        original = np.random.default_rng(7).integers(
            0, 256, (96, 160, 3), dtype=np.uint8)
        frames = list(coherent_frames([original]*4, 12))
        self.assertEqual(len(frames), 48)
        for frame in frames:
            np.testing.assert_array_equal(np.asarray(frame), original)


if __name__ == "__main__":
    unittest.main()

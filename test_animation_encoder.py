"""Build-time tests; use .venv-animation/bin/python -m unittest here."""
import unittest

try:
    import cv2
    import numpy as np
    from build_animated_asset import stabilize, coherent_frames
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV is a build-only dependency")
class AnimationEncoderTests(unittest.TestCase):
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

"""Build-time tests; use .venv-animation/bin/python -m unittest here."""
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import cv2
    import numpy as np
    from scene_motion import (scene_frames, subjects, SUBJECTS, water_shimmer, WATER, FOLIAGE,
                              GLOW, glow_mask)
    from PIL import Image
except ImportError:
    cv2 = None

ASSETS = Path(__file__).with_name('assets')
NAMES = ('woodland', 'meadow', 'pond', 'autumn', 'winter', 'christmas',
         'tornado', 'tsunami', 'balloons', 'sakura')


def keyframe(name, size):
    with Image.open(ASSETS/f'{name}-keyframe.png') as image:
        return np.array(image.convert('RGB').resize(size, Image.Resampling.LANCZOS))


@unittest.skipIf(cv2 is None, "OpenCV is a build-only dependency")
class AnimationEncoderTests(unittest.TestCase):
    def test_overlapping_foliage_motion_does_not_warp_water(self):
        base=np.zeros((90,160,3),np.uint8)
        base[:]=(38,81,97)
        base[:,::8]=(158,203,216)
        pool=[(150,90),(810,90),(810,450),(150,450)]
        wind=[(0,0),(959,0),(959,539),(0,539)]
        with patch('scene_motion.subjects',return_value=(base,[])), \
             patch.dict(WATER,{'pond':pool},clear=True), \
             patch.dict(FOLIAGE,{'pond':[wind]},clear=True):
            for frame in scene_frames(base,'pond',24,beats=2,particles=False):
                interior=np.array(frame)[22:68,32:128]
                original=base[22:68,32:128]
                self.assertLessEqual(np.abs(interior.astype(int)-original).max(),4)
                np.testing.assert_array_equal(interior[:,:,0]>100,original[:,:,0]>100)

    def test_water_highlights_cannot_displace_or_strongly_recolor_the_surface(self):
        # Alternating reflected landmarks make any geometric warp obvious.
        base=np.zeros((90,160,3),np.uint8)
        base[:]=(38,81,97)
        base[:,::8]=(158,203,216)
        mask=np.zeros((90,160),np.float32)
        mask[15:75,20:140]=1
        frames=[water_shimmer(base,mask,u/12,4) for u in range(12)]
        for frame in frames:
            delta=frame.astype(int)-base.astype(int)
            self.assertGreaterEqual(delta.min(),0)
            self.assertLessEqual(delta.max(),4)
            np.testing.assert_array_equal(frame[mask==0],base[mask==0])
            np.testing.assert_array_equal(frame[:,:,0]>100,base[:,:,0]>100)
        self.assertGreater(len({frame.tobytes() for frame in frames}),1)
        np.testing.assert_array_equal(water_shimmer(base,mask,0,4),water_shimmer(base,mask,1,4))

    def test_all_selected_subjects_are_extracted(self):
        for name,specs in SUBJECTS.items():
            with self.subTest(scene=name):
                _,layers=subjects(keyframe(name,(960,540)),name)
                self.assertEqual(len(layers),len(specs))

    def test_each_scene_moves_and_wraps_without_a_large_jump(self):
        cv2.setNumThreads(2)
        for name in NAMES:
            with self.subTest(scene=name):
                frames=[np.asarray(f).astype(np.float32)
                        for f in scene_frames(keyframe(name,(320,180)),name,48,beats=2)]
                changes=[np.abs(a-b).mean() for a,b in zip(frames,frames[1:])]
                self.assertGreater(max(np.abs(f-frames[0]).mean() for f in frames),.5)
                seam=np.abs(frames[-1]-frames[0]).mean()
                self.assertLess(seam,max(changes)*2+.1)

    def test_a_long_loop_does_not_repeat_its_six_second_beat(self):
        for name in ('woodland','winter','balloons'):
            with self.subTest(scene=name):
                frames=[np.asarray(f).astype(np.float32)
                        for f in scene_frames(keyframe(name,(320,180)),name,96,beats=2)]
                repeat=np.mean([np.abs(frames[i]-frames[i+48]).mean() for i in range(0,48,6)])
                step=np.mean([np.abs(a-b).mean() for a,b in zip(frames,frames[1:])])
                self.assertGreater(repeat,step)

    def test_stationary_bridge_stays_fixed_under_water_motion(self):
        frames=[np.asarray(f) for f in scene_frames(keyframe('sakura',(320,180)),'sakura',24,particles=False)]
        # These pixels belong to the painted bridge above the water, where
        # neither its structure nor its position should be warped.
        first=frames[0][116:121,186:233].astype(float)
        for frame in frames[1:]:
            self.assertLess(np.abs(first-frame[116:121,186:233]).mean(),1)

    def test_glow_masks_mark_only_the_marked_light_sources(self):
        for name,regions in GLOW.items():
            with self.subTest(scene=name):
                base=keyframe(name,(960,540))
                mask=glow_mask(base,name)>0
                marked=np.zeros(mask.shape,np.uint8)
                for points,_ in regions:
                    cv2.fillPoly(marked,[np.int32(points)],1)
                allowed=cv2.dilate(marked,np.ones((15,15),np.uint8)).astype(bool)
                self.assertGreater(mask.sum(),40)
                self.assertFalse((mask & ~allowed).any())
                # Only the lamp-lit pixels glow, never a whole marked region.
                self.assertLess(mask.sum(),marked.sum()/2)


if __name__ == "__main__":
    unittest.main()

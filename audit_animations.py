"""Inspect every saved loop and write a portable visual review page."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from pixel_frame import SCENES


def audit():
    root=Path(__file__).resolve().parent
    preview=root/'pictures'/'previews'
    preview.mkdir(exist_ok=True)
    results=[]
    storyboard=Image.new('RGB',(960,len(SCENES)*202))
    for row,name in enumerate(SCENES):
        with Image.open(root/'pictures'/f'{name}.webp') as animation:
            assert animation.size==(1920,1080),name
            assert animation.n_frames%48==0,(name,animation.n_frames)
            picks=(0,animation.n_frames//6,animation.n_frames//2)
            assert animation.info.get('loop')==0,name
            frames=[]
            durations=[]
            for i in range(animation.n_frames):
                animation.seek(i)
                frame=animation.convert('RGB')
                durations.append(animation.info['duration'])
                frames.append(np.asarray(frame.resize((480,270))).astype(np.float32))
                if i==0:
                    frame.resize((960,540)).save(preview/f'{name}-poster.jpg',quality=90)
                if i in picks:
                    column=picks.index(i)
                    storyboard.paste(frame.resize((320,180)),(column*320,row*202))
            assert set(durations)=={125},(name,durations)
            changes=[float(np.abs(a-b).mean()) for a,b in zip(frames,frames[1:])]
            seam=float(np.abs(frames[-1]-frames[0]).mean())
            movement=max(float(np.abs(f-frames[0]).mean()) for f in frames)
            assert min(changes)>.05,(name,'frozen frame')
            assert movement>.5,(name,'no visible motion')
            assert seam<=max(changes)*2+.1,(name,'loop jump',seam,max(changes))
            result={'scene':name,'frames':len(frames),'duration_ms':sum(durations),
                    'mean_frame_change':round(float(np.mean(changes)),3),
                    'loop_seam_change':round(seam,3),'maximum_motion':round(movement,3)}
            results.append(result)
            print(json.dumps(result),flush=True)
            ImageDraw.Draw(storyboard).text((8,row*202+182),name,fill='white')
    (preview/'animation-audit.json').write_text(json.dumps(results,indent=2)+'\n')
    storyboard.save(preview/'new-motion-storyboard.jpg',quality=92)
    return results


if __name__=='__main__':
    audit()

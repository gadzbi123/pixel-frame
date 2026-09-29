"""Build-time motion for the painted scenes; no OpenCV in the player."""
from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image


def region(shape, points, softness=5):
    """Feather a hand-selected region without moving the surrounding camera."""
    height, width = shape[:2]
    points = np.asarray(points, dtype=np.float32) * (width / 960, height / 540)
    mask = np.zeros((height, width), np.float32)
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return cv2.GaussianBlur(mask, (0, 0), softness * width / 960)


WATER = {
    'woodland': [(500,330),(690,335),(960,405),(960,540),(830,510),(705,420),(585,375)],
    'meadow': [(465,269),(590,276),(715,310),(960,344),(960,395),(724,387),(470,327)],
    'pond': [(83,211),(615,178),(910,199),(945,353),(740,450),(255,464),(85,373)],
    'autumn': [(685,343),(878,320),(915,375),(760,420),(663,401)],
    'winter': [(493,382),(660,350),(825,367),(815,411),(960,486),(960,540),(751,499),(598,431)],
    'tsunami': [(0,384),(449,298),(960,347),(960,540),(0,540)],
    'balloons': [(527,392),(600,375),(708,394),(719,444),(622,434)],
    'sakura': [(590,379),(711,370),(806,414),(927,540),(400,540),(478,450)],
}
FOLIAGE = {
    'woodland': [[(0,0),(375,0),(322,120),(75,190),(0,261)],[(0,395),(479,426),(577,540),(0,540)]],
    'meadow': [[(0,0),(405,0),(277,136),(0,234)],[(0,400),(358,385),(676,453),(960,407),(960,540),(0,540)]],
    'pond': [[(0,0),(390,0),(290,87),(0,170)],[(0,333),(161,334),(225,540),(0,540)],[(806,197),(960,168),(960,360),(847,328)]],
    'autumn': [[(0,0),(610,0),(548,108),(150,153),(0,298)]],
    'winter': [[(0,0),(270,0),(159,167),(0,288)]],
    'balloons': [[(0,0),(175,0),(146,215),(0,268)],[(0,433),(420,483),(600,540),(0,540)]],
    'sakura': [[(0,0),(439,0),(370,103),(95,256),(0,302)],[(584,0),(960,0),(960,294),(818,179)]],
}
LOCAL = {
    'woodland': [([(327,265),(405,268),(420,329),(367,357)],3,1),
                 ([(854,163),(886,160),(918,238),(881,252)],4,2)],
    'autumn': [([(369,283),(436,277),(458,348),(395,365)],3,1)],
    'winter': [([(328,323),(384,332),(377,407),(323,403)],2,1),
               ([(856,184),(900,173),(932,298),(882,335)],5,2)],
    'meadow': [([(161,274),(255,274),(255,369),(145,376)],1,1)],
    'pond': [([(670,424),(749,425),(753,476),(679,480)],1,2)],
}
# Rectangles isolate subjects from their stationary backgrounds. Coordinates
# refer to the 960x540 working canvas, regardless of the encoder's resolution.
SUBJECTS = {
    'balloons': [((227,3,155,222), 19, 12),((652,10,130,185), -14, 18),
                 ((415,141,51,77), 12, 9),((815,102,65,86), -9, 13)],
    'pond': [((199,235,134,69), 18, 2),((675,234,140,94), -22, 2),
             ((403,179,85,60), 13, 1)],
    'meadow': [((160,105,81,75), 14, 10),((641,248,88,91), -13, 7)],
    'tsunami': [((243,232,237,195), 3, 7),((691,20,265,136),9,5)],
    'tornado': [((645,46,116,94),27,18),((450,110,83,99),-21,24)],
}


@dataclass
class Subject:
    color: np.ndarray
    alpha: np.ndarray
    origin: tuple
    travel: tuple


def subjects(base, theme):
    """Lift foreground silhouettes, then fill the vacated stationary background."""
    height, width = base.shape[:2]
    scale = width / 960
    background = base.copy()
    result = []
    erase = np.zeros((height, width), np.uint8)
    for index,(rect, dx, dy) in enumerate(SUBJECTS.get(theme, ())):
        x, y, w, h = [round(value * scale) for value in rect]
        crop = base[y:y+h, x:x+w].copy()
        labels = np.zeros((h, w), np.uint8)
        cv2.setRNGSeed(0)
        cv2.grabCut(crop, labels, (1, 1, w-2, h-2),
                    np.zeros((1,65)), np.zeros((1,65)), 4, cv2.GC_INIT_WITH_RECT)
        mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
        if theme == 'pond' and index == 2:
            # Keep the white body attached to the dark head; reflections can
            # otherwise make the classifier discard the pale feathers.
            outline=((410,205),(451,203),(455,198),(458,186),(467,183),
                     (474,187),(474,192),(483,196),(474,200),(473,211),
                     (466,219),(429,222),(411,215))
            polygon=np.int32(np.array(outline)*scale-(x,y))
            cv2.fillPoly(mask,[polygon],1)
        if theme == 'tsunami' and index == 0:
            # The thin pale mast is easily classified as background by GrabCut.
            for start,end in (((355,236),(348,300)),((331,269),(379,270))):
                a=tuple(round(v*scale-origin) for v,origin in zip(start,(x,y)))
                b=tuple(round(v*scale-origin) for v,origin in zip(end,(x,y)))
                cv2.line(mask,a,b,1,max(1,round(2*scale)))
        # A failed cutout must not lift an entire rectangular patch of sky.
        if not .02 < mask.mean() < .85:
            continue
        erase[y:y+h, x:x+w] |= mask * 255
        if theme == 'tsunami' and index == 1:
            # The pale fuselage is too similar to the sky for classification.
            # Use its silhouette and preserve the slender landing skids.
            outline=((763,107),(770,96),(786,88),(803,84),(813,80),(831,76),
                     (842,82),(853,97),(867,108),(897,115),(913,114),(919,90),
                     (923,86),(924,96),(919,117),(930,129),(924,134),
                     (912,136),(902,128),(883,128),(867,132),(855,137),
                     (834,141),(808,137),(791,130),(774,121),(763,115))
            mask[:]=0
            cv2.fillPoly(mask,[np.int32(np.array(outline)*scale-(x,y))],1)
            for a,b in (((781,130),(785,139)),((785,139),(855,146)),
                        ((792,125),(793,138))):
                start=tuple(np.int32(np.array(a)*scale-(x,y)))
                end=tuple(np.int32(np.array(b)*scale-(x,y)))
                cv2.line(mask,start,end,1,max(1,round(2*scale)))
            erase[y:y+h,x:x+w] |= mask*255
            # Remove the old frozen rotor before painting a rotating one.
            for a,b in (((703,28),(826,66)),((826,66),(944,26)),
                        ((697,73),(826,66)),((826,66),(959,70))):
                cv2.line(erase,tuple(np.int32(np.array(a)*scale)),
                         tuple(np.int32(np.array(b)*scale)),255,max(1,round(14*scale)))
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (0,0), .65)
        result.append(Subject(crop, alpha, (x,y), (dx*scale,dy*scale)))
    if erase.any():
        erase = cv2.dilate(erase, np.ones((3,3),np.uint8))
        background = cv2.inpaint(background, erase, 5, cv2.INPAINT_TELEA)
    return background, result


def composite(image, subject, u, index):
    dx, dy = subject.travel
    angle = 2*math.pi*u + index*.9
    x, y = subject.origin
    h, w = subject.alpha.shape
    matrix = np.float32([[1,0,x+dx*math.sin(angle)],
                        [0,1,y+dy*math.cos(angle)]])
    alpha = cv2.warpAffine(subject.alpha, matrix, (image.shape[1],image.shape[0]))[...,None]
    # Premultiplication avoids dark fringes at a moving silhouette's edges.
    color = cv2.warpAffine(subject.color.astype(np.float32)*subject.alpha[...,None],
                           matrix, (image.shape[1],image.shape[0]))
    image[:] = np.clip(image*(1-alpha)+color,0,255).astype(np.uint8)


def rotor(base, clean=False):
    mask = np.zeros(base.shape[:2],np.uint8)
    scale = base.shape[1]/960
    polygons = [[(733,43),(750,29),(798,85),(786,101)],
                [(810,83),(870,22),(885,39),(824,99)],
                [(823,121),(841,110),(870,179),(852,192)],
                [(781,117),(794,134),(737,187),(721,171)]]
    for points in polygons:
        cv2.fillPoly(mask,[np.int32(np.array(points)*scale)],255)
    for end in ((789,92),(818,91),(834,126),(785,126)):
        cv2.line(mask,tuple(np.int32(np.array((807,111))*scale)),
                 tuple(np.int32(np.array(end)*scale)),255,max(1,round(4*scale)))
    # Remove the complete painted lattice, including its antialiased edges.
    # Sampling the old blades also samples the tower visible through the grid;
    # that would rotate chunks of the building along with the sails.
    background = base.copy() if clean else cv2.inpaint(base, cv2.dilate(mask,np.ones((13,13),np.uint8)),9,cv2.INPAINT_TELEA)
    color = np.zeros_like(base)
    alpha = np.zeros(base.shape[:2],np.uint8)
    center=np.array((807,111),np.float32)
    for degrees in (-133,-43,47,137):
        angle=math.radians(degrees)
        forward=np.array((math.cos(angle),math.sin(angle)))
        across=np.array((-forward[1],forward[0]))
        def point(radius,offset):
            return tuple(np.round((center+radius*forward+offset*across)*scale).astype(int))
        def line(a,b):
            cv2.line(alpha,a,b,255,max(1,round(2*scale)),cv2.LINE_AA)
            cv2.line(color,a,b,(80,65,49),max(1,round(2*scale)),cv2.LINE_AA)
        for side in (-9,0,9):
            line(point(27,side),point(112,side))
        for radius in range(27,113,9):
            line(point(radius,-9),point(radius,9))
        line(point(0,0),point(27,0))
    return background,color,alpha.astype(np.float32)/255


def scene_frames(base, theme, count=48, *, clean_rotor=False, particles=True):
    """Six-second periodic motion, with stable landmarks and independently moving subjects."""
    height,width = base.shape[:2]
    scale = width/960
    y,x = np.mgrid[:height,:width].astype(np.float32)
    water = region(base.shape,WATER[theme]) if theme in WATER else np.zeros_like(x)
    if theme == 'sakura':
        water *= 1-region(base.shape,[(471,321),(771,313),(800,388),(492,425)],2)
    foliage = sum((region(base.shape,p) for p in FOLIAGE.get(theme,())),np.zeros_like(x))
    background,layers = subjects(base,theme)
    local = [(region(base.shape,p,8),dx,dy) for p,dx,dy in LOCAL.get(theme,())]
    rotor_alpha = None
    if theme == 'meadow':
        background,rotor_color,rotor_alpha = rotor(background,clean=clean_rotor)
    funnel = region(base.shape,[(321,0),(677,0),(598,208),(561,381),(494,398),(430,212)],12) if theme=='tornado' else None
    wave = region(base.shape,[(0,0),(425,0),(640,254),(566,363),(175,395),(0,281)],12) if theme=='tsunami' else None
    for frame in range(count):
        u = frame/count
        angle = 2*math.pi*u
        fx = water*scale*(3.5*np.sin(y/(12*scale)-angle*2)+1.5*np.sin(x/(25*scale)+angle))
        fy = water*scale*1.7*np.sin(x/(31*scale)-angle*2)
        fx += foliage*scale*3.5*np.sin(angle+x/(160*scale))
        fy += foliage*scale*.8*np.cos(angle+y/(90*scale))
        for index,(mask,dx,dy) in enumerate(local):
            fx += mask*scale*dx*math.sin(angle+index)
            fy += mask*scale*dy*math.cos(angle*2+index)
        if funnel is not None:
            fx += funnel*scale*14*np.sin(angle*2+y/(22*scale))
            fy += funnel*scale*5*np.cos(angle*2+x/(45*scale))
        if wave is not None:
            fx += wave*scale*11*np.sin(angle+y/(70*scale))
            fy += wave*scale*15*np.sin(angle+x/(95*scale))
        image = cv2.remap(background,x+fx,y+fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        if rotor_alpha is not None:
            rotation = cv2.getRotationMatrix2D((807*scale,111*scale),u*360,1)
            alpha=cv2.warpAffine(rotor_alpha,rotation,(width,height))[...,None]
            color=cv2.warpAffine(rotor_color.astype(np.float32)*rotor_alpha[...,None],rotation,(width,height))
            image=np.clip(image*(1-alpha)+color,0,255).astype(np.uint8)
        for index,subject in enumerate(layers):
            composite(image,subject,u,index)
        if theme=='tsunami' and len(layers)==2:
            center=(826+9*math.sin(angle+.9),69+5*math.cos(angle+.9))
            cv2.line(image,tuple(round(v*scale) for v in center),
                     (round((center[0]+3)*scale),round((center[1]+15)*scale)),
                     (62,60,56),max(1,round(3*scale)),cv2.LINE_AA)
            for n in range(4):
                spin=angle*5+n*math.pi/2
                start=tuple(round(v*scale) for v in center)
                end=(round((center[0]+122*math.cos(spin))*scale),
                     round((center[1]+24*math.sin(spin))*scale))
                cv2.line(image,start,end,(71,68,62),max(1,round(2*scale)),cv2.LINE_AA)
        if particles and theme in ('autumn','winter','christmas','sakura'):
            for n in range(85 if theme in ('winter','christmas') else 42):
                px=((n*137.3+19*math.sin(angle+n))%980-10)*scale
                py=((n*79.7+u*560)%560-10)*scale
                color=(238,244,250) if theme in ('winter','christmas') else ((222,126,42) if theme=='autumn' else (249,180,205))
                axes=(max(1,round((1+n%2)*scale)),max(1,round((1 if theme in ('winter','christmas') else 2)*scale)))
                cv2.ellipse(image,(round(px),round(py)),axes,n*37+u*360,0,360,color,-1,cv2.LINE_AA)
        if theme=='christmas':
            # Existing painted tree lights sparkle without changing the village.
            for n in range(24):
                py=96+n*11; half=(py-75)*.25
                px=928+half*math.sin(n*2.4)
                strength=.5+.5*math.sin(angle*2+n)
                cv2.circle(image,(round(px*scale),round(py*scale)),max(1,round(2*scale)),
                           (round(170+85*strength),round(125+115*strength),round(45+100*strength)),-1,cv2.LINE_AA)
        yield Image.fromarray(image)

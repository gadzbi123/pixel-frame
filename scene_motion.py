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
    'woodland': [[(0,0),(375,0),(322,120),(75,190),(0,261)],[(0,330),(73,332),(82,437),(169,452),(149,540),(0,540)]],
    'meadow': [[(0,0),(405,0),(277,136),(0,234)],[(0,400),(358,385),(676,453),(960,407),(960,540),(0,540)]],
    'pond': [[(0,0),(390,0),(290,87),(0,170)],[(0,333),(161,334),(225,540),(0,540)],[(806,197),(960,168),(960,360),(847,328)]],
    'autumn': [[(0,0),(610,0),(548,108),(150,153),(0,298)]],
    'winter': [[(0,0),(270,0),(159,167),(0,288)]],
    'balloons': [[(0,0),(175,0),(146,215),(0,268)],[(0,433),(420,483),(600,540),(0,540)]],
    'sakura': [[(0,0),(439,0),(370,103),(95,256),(0,302)],[(584,0),(960,0),(960,294),(818,179)]],
}
LOCAL = {
    'woodland': [([(207,301),(249,287),(306,297),(348,311),(340,351),(259,346),(211,328)],2,1),
                 ([(348,291),(360,281),(370,304),(391,288),(389,318),(401,336),(382,345),(354,333)],2.5,1),
                 ([(129,342),(173,324),(211,306),(224,319),(181,345),(142,358)],3,2),
                 ([(84,384),(118,380),(134,364),(148,378),(163,397),(143,413),(101,420)],1.5,1)],
    'autumn': [([(184,207),(251,185),(308,209),(350,235),(330,292),(254,302),(189,266)],2,1),
               ([(360,202),(379,183),(389,216),(421,195),(408,232),(433,256),(443,269),(409,279),(369,263)],2.5,1),
               ([(49,270),(100,245),(158,215),(180,229),(127,272),(73,296)],3,2)],
    'winter': [([(328,323),(384,332),(377,407),(323,403)],2,1),
               ([(856,184),(900,173),(932,298),(882,335)],5,2)],
    'meadow': [([(161,274),(255,274),(255,369),(145,376)],1,1)],
    'pond': [([(670,424),(749,425),(753,476),(679,480)],1,2)],
}
# Rectangles isolate subjects from their stationary backgrounds. Coordinates
# refer to the 960x540 working canvas, regardless of the encoder's resolution.
SUBJECTS = {
    'balloons': [((220,8,180,212), 19, 12),((652,10,130,185), -14, 18),
                 ((415,141,51,77), 12, 9),((815,102,65,86), -9, 13)],
    'pond': [((199,235,134,69), 18, 2),((675,234,140,94), -22, 2),
             ((403,179,85,60), 13, 1)],
    'meadow': [((160,105,81,75), 14, 10),((634,242,98,101), -13, 7)],
    'tsunami': [((243,232,237,195), 3, 7),((691,20,265,136),9,5)],
    'tornado': [((645,46,116,94),27,18),((450,110,83,99),-21,24)],
}
# Light sources that keep glowing at night: (polygon, kind). Inside each region
# 'warm' keeps lamp-lit yellow and orange pixels; 'bright' keeps near-white light.
# 'window' is for a region traced around one window: it also keeps the deep
# orange of curtains, so the pane glows whole. Trace windows closely: sunlit log
# ends and leaves beside them are just as warm, and against a moonlit wall they
# would glow like fairy lights.
def box(left,top,right,bottom):
    return [(left,top),(right,top),(right,bottom),(left,bottom)]

GLOW = {
    'woodland': [(box(843,263.5,851.5,281),'window'),(box(853,261.5,866,283.5),'window'),
                 (box(874.5,263,881.5,284.5),'window'),(box(923.5,262,940.5,281.5),'window')],
    'winter': [(box(770,234.5,778.5,250.5),'window'),(box(769.5,275.5,787,299.5),'window'),
               (box(836,276,850.5,298),'window'),(box(899.5,278,915.5,296.5),'window'),
               (box(758,388,802,448),'warm')],  # the window's reflection in the stream
    'christmas': [([(55,165),(425,195),(425,425),(55,425)],'warm'),
                  ([(110,400),(160,400),(160,485),(110,485)],'warm'),
                  ([(470,225),(740,225),(740,345),(470,345)],'warm'),
                  ([(500,330),(545,330),(545,390),(500,390)],'warm'),
                  ([(605,425),(640,425),(640,465),(605,465)],'warm'),
                  ([(825,30),(960,30),(960,425),(825,425)],'warm'),
                  ([(320,15),(380,15),(380,75),(320,75)],'bright')],
}
# Subjects that drift once per loop instead of moving at the six-second beat.
SLOW_SUBJECTS = {'pond', 'balloons'}


def christmas_lights(scale):
    """Centres of the tree lights that twinkle in the christmas scene."""
    for n in range(24):
        py=96+n*11; half=(py-75)*.25
        yield n,(round((928+half*math.sin(n*2.4))*scale),round(py*scale))


def glow_mask(base, theme):
    """0/255 mask of the pixels that stay lit at night, for the player."""
    height,width=base.shape[:2]
    scale=width/960
    rgb=base.astype(np.int32)
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    warm=((r>170)&(g>110)&(b<g*.8)&(r>=g)).astype(np.uint8)
    # The hottest part of a lamp is near white; keep it when it borders warm light,
    # or it shows as a cold hole in the middle of a lit window.
    near=cv2.dilate(warm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,round(9*scale)),)*2))
    lit=warm|(near&(rgb.min(axis=2)>150)&(r>=b))
    lit=cv2.morphologyEx(lit,cv2.MORPH_CLOSE,np.ones((max(3,round(3*scale)),)*2,np.uint8))
    deep=(r>150)&(r-b>70)&(r>=g)
    kinds={'warm':lit.astype(bool),'bright':rgb.min(axis=2)>215,'window':lit.astype(bool)|deep}
    mask=np.zeros((height,width),np.uint8)
    for points,kind in GLOW.get(theme,()):
        inside=np.zeros_like(mask)
        cv2.fillPoly(inside,[np.int32(np.array(points)*scale)],1)
        mask|=inside&kinds[kind]
    if theme=='christmas':
        for _,center in christmas_lights(scale):
            cv2.circle(mask,center,max(1,round(2*scale)),1,-1)
    return cv2.dilate(mask,np.ones((3,3),np.uint8))*255


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
        if theme == 'balloons' and index == 0:
            # Include the full rim, basket and rigging; the former crop clipped
            # the right edge and left a fixed sliver behind in the sky.
            outline=((307,18),(331,20),(355,29),(373,46),(382,67),
                     (384,89),(377,115),(364,138),(340,164),(328,179),
                     (324,190),(324,207),(297,210),(296,191),(287,175),
                     (275,156),(257,135),(243,112),(237,88),(237,63),
                     (246,42),(263,28),(285,20))
            mask[:]=0
            cv2.fillPoly(mask,[np.int32(np.array(outline)*scale-(x,y))],1)
        if theme == 'meadow' and index == 1:
            # A solid wing silhouette preserves pale paint and dark veins.
            # Colour classification had made holes throughout the white wing.
            outline=((633,286),(647,284),(662,289),(675,291),(681,296),
                     (684,281),(690,267),(701,254),(711,246),(720,247),
                     (723,258),(722,279),(719,290),(724,297),(725,310),
                     (722,319),(712,325),(703,322),(696,319),(699,331),
                     (692,334),(686,329),(678,327),(666,319),(663,310),
                     (650,302),(640,297))
            mask[:]=0
            cv2.fillPoly(mask,[np.int32(np.array(outline)*scale-(x,y))],1)
            for a,b in (((676,287),(682,299)),((692,328),(696,342))):
                cv2.line(mask,tuple(np.int32(np.array(a)*scale-(x,y))),
                         tuple(np.int32(np.array(b)*scale-(x,y))),1,max(1,round(2*scale)))
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
        erase = cv2.dilate(erase, np.ones((5,5),np.uint8))
        background = cv2.inpaint(background, erase, 5, cv2.INPAINT_TELEA)
    return background, result


def drift(u, index, cycles):
    """Periodic offsets in -1..1; unequal x and y rates trace a loop that rarely repeats."""
    return (math.sin(2*math.pi*u*cycles + index*.9),
            math.cos(2*math.pi*u*(cycles + (cycles > 1)) + index*.9))


def composite(image, subject, u, index, cycles=1):
    dx, dy = subject.travel
    x, y = subject.origin
    ox, oy = drift(u, index, cycles)
    matrix = np.float32([[1,0,x+dx*ox],
                        [0,1,y+dy*oy]])
    alpha = cv2.warpAffine(subject.alpha, matrix, (image.shape[1],image.shape[0]))[...,None]
    # Premultiplication avoids dark fringes at a moving silhouette's edges.
    color = cv2.warpAffine(subject.color.astype(np.float32)*subject.alpha[...,None],
                           matrix, (image.shape[1],image.shape[0]))
    image[:] = np.clip(image*(1-alpha)+color,0,255).astype(np.uint8)


def rotor(base):
    """Paint windmill sails to rotate over the bladeless meadow keyframe."""
    scale = base.shape[1]/960
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
    return color,alpha.astype(np.float32)/255


def water_shimmer(image, mask, u, beats=1):
    """Animate faint surface highlights without displacing the painted water."""
    u %= 1
    height,width=image.shape[:2]
    scale=width/960
    y,x=np.mgrid[:height,:width].astype(np.float32)
    rgb=image.astype(np.float32)
    # Restrict highlights to cool water and foam inside the selected surface.
    cool=(rgb[:,:,1]>rgb[:,:,0]+4)&(rgb[:,:,2]>rgb[:,:,0]-12)
    foam=np.min(rgb,axis=2)>170
    glint=(.5+.5*np.sin(y/(2.2*scale)+.25*np.sin(x/(43*scale))-4*math.pi*u*beats))**8
    envelope=.5+.5*np.sin(x/(87*scale)-2*math.pi*u)
    amount=mask*(cool|foam)*glint*envelope*4
    return np.clip(rgb+amount[...,None]*np.array((.6,.9,1.)),0,255).astype(np.uint8)


def falling_particles(image, theme, u, beats=1):
    """Paint drifting snow, lobed leaves or petals with a soft depth fade.

    Each particle falls `beats` times per loop and starts every fall at a new
    column, so a long loop does not repeat its six-second pattern."""
    height,width=image.shape[:2]
    scale=width/960
    angle=2*math.pi*u*beats
    overlay=image.copy()
    snow=theme in ('winter','christmas')
    count=70 if snow else 28
    for n in range(count):
        travel=n*79.7+u*580*beats
        lap=int(travel//580)%beats
        px=((n*137.3+lap*331.7+12*math.sin(angle+n))%980-10)*scale
        # Integral travel across one padded height makes the loop periodic.
        py=(travel%580-20)*scale
        fade=min(1,max(0,py/(20*scale)),max(0,(height-py)/(20*scale)))
        center=(round(px),round(py))
        if snow:
            radius=(.65+(n%4)*.3)*scale
            color=(247,248,250)
            cv2.circle(overlay,center,max(1,round(radius)),color,-1,cv2.LINE_AA)
            if n%7==0:
                for a in (0,math.pi/3,2*math.pi/3):
                    delta=np.array((math.cos(a),math.sin(a)))*2.2*scale
                    cv2.line(overlay,tuple(np.round(np.array(center)-delta).astype(int)),
                             tuple(np.round(np.array(center)+delta).astype(int)),color,
                             max(1,round(scale)),cv2.LINE_AA)
        elif theme=='autumn':
            # Maple-like lobes, a pointed tip and a thin central vein.
            points=np.array([(0,-5),(1,-2),(3,-3),(2,0),(5,0),(2,2),
                             (2,4),(0,3),(-2,4),(-2,2),(-5,0),(-2,0),(-3,-3),(-1,-2)],np.float32)
            spin=angle+n*2.4
            rotation=np.array([[math.cos(spin),-math.sin(spin)],[math.sin(spin),math.cos(spin)]])
            points=points@rotation.T*scale*(.7+n%3*.2)+center
            color=((180,76,28),(199,108,35),(163,58,28),(207,139,47))[n%4]
            cv2.fillPoly(overlay,[np.round(points).astype(np.int32)],color,cv2.LINE_AA)
            tip=np.array((0,4))*scale@rotation.T+center
            cv2.line(overlay,center,tuple(np.round(tip).astype(int)),(112,69,29),max(1,round(scale)),cv2.LINE_AA)
        else:
            cv2.ellipse(overlay,center,(max(1,round(1.4*scale)),max(1,round(2.5*scale))),
                        n*37+u*360*beats,0,360,(249,190,210),-1,cv2.LINE_AA)
        # Blend only this particle's small patch; preserve the rest of the image.
        reach=max(1,round(8*scale))
        left,right=max(0,center[0]-reach),min(width,center[0]+reach+1)
        top,bottom=max(0,center[1]-reach),min(height,center[1]+reach+1)
        if left<right and top<bottom:
            a=(.65 if snow else .8)*fade
            image[top:bottom,left:right]=cv2.addWeighted(image[top:bottom,left:right],1-a,
                                                        overlay[top:bottom,left:right],a,0)
            overlay[top:bottom,left:right]=image[top:bottom,left:right]
    return image


def scene_frames(base, theme, count=48, *, beats=1, particles=True):
    """Periodic motion with stable landmarks and independently moving subjects.

    Motion keeps a six-second beat; a loop of `count` frames holds `beats` of
    them, varied by a gust that rises and falls once per loop."""
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
        rotor_color,rotor_alpha = rotor(background)
    funnel = None
    if theme == 'tornado':
        # Follow the visible funnel's sloping axis. End above the ground cows.
        axis=np.interp(y/scale,[0,120,220,325],[535,585,566,548])
        radius=np.interp(y/scale,[0,120,220,325],[210,110,58,16])
        radial=(x/scale-axis)/radius
        funnel=np.clip(1-radial**2,0,1)**2
        funnel *= np.clip((325-y/scale)/25,0,1)
    if theme == 'woodland':
        # The boulders and fallen log are stationary; only selected plants sway.
        foliage *= 1-region(base.shape,[(164,438),(273,421),(396,463),(438,540),(164,540)],2)
        foliage *= 1-region(base.shape,[(425,474),(685,500),(727,540),(425,540)],2)
    if theme=='tsunami':
        crest=region(base.shape,[(0,0),(425,0),(640,254),(566,363),(175,395),(0,281)],12)
        water=np.maximum(water,crest)
    tempo = 1 if theme in SLOW_SUBJECTS else beats
    for frame in range(count):
        u = frame/count
        angle = 2*math.pi*u*beats
        gust = .75+.25*math.sin(2*math.pi*u+1.3) if beats > 1 else 1
        fx = foliage*scale*3.5*gust*np.sin(angle+x/(160*scale))
        fy = foliage*scale*.8*gust*np.cos(angle+y/(90*scale))
        for index,(mask,dx,dy) in enumerate(local):
            fx += mask*scale*dx*gust*math.sin(angle+index)
            fy += mask*scale*dy*gust*math.cos(angle*2+index)
        if funnel is not None:
            fx += funnel*scale*5*np.sin(angle*2+radial*math.pi)
            # Texture moves across the cylindrical funnel, without translating
            # its axis or wobbling the cows and ground beneath it.
        # Foliage masks can overlap the shore. Keep those other displacement
        # fields from bending the water or its reflected landmarks as well.
        protection=np.clip(water*4,0,1)
        fx *= 1-protection
        fy *= 1-protection
        image = cv2.remap(background,x+fx,y+fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        if theme in WATER:
            image=water_shimmer(image,water,u,beats)
        if rotor_alpha is not None:
            rotation = cv2.getRotationMatrix2D((807*scale,111*scale),u*360*beats,1)
            alpha=cv2.warpAffine(rotor_alpha,rotation,(width,height))[...,None]
            color=cv2.warpAffine(rotor_color.astype(np.float32)*rotor_alpha[...,None],rotation,(width,height))
            image=np.clip(image*(1-alpha)+color,0,255).astype(np.uint8)
        for index,subject in enumerate(layers):
            composite(image,subject,u,index,tempo)
        if theme=='tsunami' and len(layers)==2:
            ox,oy=drift(u,1,tempo)
            center=(826+9*ox,69+5*oy)
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
            image = falling_particles(image, theme, u, beats)
        if theme=='christmas':
            # Existing painted tree lights sparkle without changing the village.
            for n,center in christmas_lights(scale):
                # Each light twinkles at its own whole-number rate per loop.
                strength=.5+.5*math.sin(angle*2+2*math.pi*u*(n%3)+n)
                cv2.circle(image,center,max(1,round(2*scale)),
                           (round(170+85*strength),round(125+115*strength),round(45+100*strength)),-1,cv2.LINE_AA)
        yield Image.fromarray(image)

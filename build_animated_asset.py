#!/usr/bin/env python3
"""Build an eight-FPS 1080p animated WebP from a four-panel keyframe sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import cv2
import numpy as np


def stabilize(frames):
    """Register stationary landmarks, rejecting independently moving objects."""
    detector = cv2.ORB_create(nfeatures=6000)
    keypoints, descriptors = detector.detectAndCompute(frames[0], None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    aligned = [frames[0]]
    for frame in frames[1:]:
        points, desc = detector.detectAndCompute(frame, None)
        if desc is None or descriptors is None:
            raise ValueError("Insufficient landmarks to stabilize keyframe")
        matches = [pair[0] for pair in matcher.knnMatch(desc, descriptors, k=2)
                   if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        if len(matches) < 20:
            raise ValueError("Keyframes differ too much for reliable registration")
        source = np.float32([points[m.queryIdx].pt for m in matches])
        target = np.float32([keypoints[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=2)
        if matrix is None or int(inliers.sum()) < 15:
            raise ValueError("Unreliable keyframe alignment")
        print(f"alignment: {int(inliers.sum())} landmarks; offset {matrix[:,2].round(1)}", flush=True)
        aligned.append(cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                                     borderMode=cv2.BORDER_REFLECT_101))
    return aligned


def coherent_frames(keys, steps):
    """Use generated poses only as motion guides for one consistent painted image."""
    first = keys[0]
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    a_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    y, x = np.mgrid[:first.shape[0], :first.shape[1]].astype(np.float32)
    fields = [np.zeros((*a_gray.shape,2), dtype=np.float32)]
    reliability = np.ones(a_gray.shape, dtype=np.float32)
    for key in keys[1:]:
        gray = cv2.cvtColor(key, cv2.COLOR_RGB2GRAY)
        forward = flow.calc(a_gray, gray, None)
        backward = flow.calc(gray, a_gray, None)
        inverse = cv2.remap(backward, x+forward[:,:,0], y+forward[:,:,1],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        error = np.linalg.norm(forward+inverse, axis=2)
        magnitude = np.linalg.norm(forward, axis=2)
        # Large/inconsistent motion is not supported by these sparse keyframes.
        confidence = np.clip(1-error/3, 0, 1) * np.clip((24-magnitude)/12, 0, 1)
        reliability = np.minimum(reliability, confidence)
        fields.append(cv2.GaussianBlur(backward, (0,0), 1.5))
    reliability = cv2.GaussianBlur(reliability, (0,0), 3)[...,None]
    fields = [field*reliability for field in fields]
    for i in range(len(fields)):
        p0,p1,p2,p3 = [fields[j % len(fields)] for j in (i-1,i,i+1,i+2)]
        for index in range(steps):
            u = index/steps
            # Periodic Catmull-Rom: continuous position and velocity at every
            # keyframe and at the wrap, without pausing at each pose.
            field = .5*((2*p1)+(-p0+p2)*u+
                         (2*p0-5*p1+4*p2-p3)*u*u+
                         (-p0+3*p1-3*p2+p3)*u*u*u)
            image = cv2.remap(first, x+field[:,:,0], y+field[:,:,1],
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            yield Image.fromarray(image)


def build(theme: str, steps: int = 12) -> Path:
    if steps < 2:
        raise ValueError("steps must be at least two")
    cv2.setNumThreads(2)
    assets = Path(__file__).with_name("assets")
    source = assets / f"{theme}-animation-sheet.png"
    output = assets / f"{theme}-animated.webp"
    sheet = Image.open(source).convert("RGB")
    width, height = sheet.size
    half_width, half_height = width // 2, height // 2
    boxes = (
        (0, 0, half_width, half_height),
        (half_width, 0, width, half_height),
        (0, half_height, half_width, height),
        (half_width, half_height, width, height),
    )
    resampling = getattr(Image, "Resampling", Image)
    keyframes = [
        np.array(sheet.crop(box).resize((960, 540), resampling.LANCZOS))
        for box in boxes
    ]
    keyframes = stabilize(keyframes)
    frames = [frame.resize((1920,1080), resampling.LANCZOS)
              for frame in coherent_frames(keyframes, steps)]
    temporary = output.with_suffix(".building.webp")
    frames[0].save(
        temporary,
        save_all=True,
        append_images=frames[1:],
        duration=125,
        loop=0,
        quality=84,
        method=3,
    )
    temporary.replace(output)
    print(f"{theme}: {len(frames)} frames -> {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("themes", nargs="+")
    parser.add_argument("--steps", type=int, default=12)
    arguments = parser.parse_args()
    for theme in arguments.themes:
        build(theme, arguments.steps)

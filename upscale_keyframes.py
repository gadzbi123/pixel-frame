#!/usr/bin/env python3
"""Make each scene's 1920x1080 keyframe from its 836x470 painting with Real-ESRGAN.

The motion masks in scene_motion.py are traced on this composition, so the
keyframe must be the same painting, only sharper. Build-only; needs torch in
.venv-animation and the weights from the Real-ESRGAN releases in
~/.cache/pixel-frame/models:

    .venv-animation/bin/python upscale_keyframes.py woodland pond
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ASSETS = Path(__file__).with_name("assets")
MODELS = Path.home()/".cache"/"pixel-frame"/"models"


class Compact(nn.Module):
    """SRVGGNetCompact, the network of realesr-general-x4v3."""

    def __init__(self, features=64, convolutions=32, scale=4):
        super().__init__()
        self.scale = scale
        layers = [nn.Conv2d(3, features, 3, 1, 1), nn.PReLU(features)]
        for _ in range(convolutions):
            layers += [nn.Conv2d(features, features, 3, 1, 1), nn.PReLU(features)]
        layers.append(nn.Conv2d(features, 3*scale*scale, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


class DenseBlock(nn.Module):
    def __init__(self, features=64, growth=32):
        super().__init__()
        for index in range(1, 6):
            width = features + (index-1)*growth
            setattr(self, f"conv{index}", nn.Conv2d(width, growth if index < 5 else features, 3, 1, 1))
        self.lrelu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        outputs = [x]
        for index in range(1, 5):
            outputs.append(self.lrelu(getattr(self, f"conv{index}")(torch.cat(outputs, 1))))
        return self.conv5(torch.cat(outputs, 1))*.2 + x


class RRDB(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = DenseBlock(features), DenseBlock(features), DenseBlock(features)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x)))*.2 + x


class RRDBNet(nn.Module):
    """The network of RealESRGAN_x4plus."""

    def __init__(self, features=64, blocks=23):
        super().__init__()
        self.conv_first = nn.Conv2d(3, features, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(features) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


NETWORKS = {
    "compact": (Compact, "realesr-general-x4v3.pth"),
    "x4plus": (RRDBNet, "RealESRGAN_x4plus.pth"),
}


def load(name: str) -> nn.Module:
    network, weights = NETWORKS[name]
    model = network()
    state = torch.load(MODELS/weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params_ema", state.get("params", state)))
    # Channels-last convolutions are about 20% faster on the Pi 4, with the same output.
    return model.eval().to(memory_format=torch.channels_last)


@torch.inference_mode()
def upscale(model: nn.Module, image: Image.Image, tile: int = 192, pad: int = 12) -> Image.Image:
    """4x upscale in overlapping tiles, to bound memory on the Pi."""
    source = torch.from_numpy(np.asarray(image.convert("RGB"), np.float32)/255).permute(2, 0, 1)[None]
    _, _, height, width = source.shape
    output = torch.zeros((1, 3, height*4, width*4))
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            y0, x0 = max(0, top-pad), max(0, left-pad)
            y1, x1 = min(height, top+tile+pad), min(width, left+tile+pad)
            result = model(source[:, :, y0:y1, x0:x1].contiguous(memory_format=torch.channels_last))
            inner_y, inner_x = (top-y0)*4, (left-x0)*4
            rows, cols = (min(height, top+tile)-top)*4, (min(width, left+tile)-left)*4
            output[:, :, top*4:top*4+rows, left*4:left*4+cols] = \
                result[:, :, inner_y:inner_y+rows, inner_x:inner_x+cols]
    array = (output[0].clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
    return Image.fromarray(array)


def source_painting(theme: str) -> Image.Image:
    """The first keyframe of the scene's sheet; meadow uses its bladeless repaint."""
    if theme == "meadow":
        # Repainted from the 836x470 keyframe at a larger size, without more detail.
        return (Image.open(ASSETS/"meadow-motion-background.png").convert("RGB")
                .resize((836, 470), Image.Resampling.LANCZOS))
    sheet = Image.open(ASSETS/f"{theme}-animation-sheet.png").convert("RGB")
    return sheet.crop((0, 0, sheet.width//2, sheet.height//2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("themes", nargs="+")
    parser.add_argument("--model", choices=NETWORKS, default="x4plus")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    model = load(arguments.model)
    for theme in arguments.themes:
        large = upscale(model, source_painting(theme))
        keyframe = large.resize((1920, 1080), Image.Resampling.LANCZOS)
        temporary = ASSETS/f"{theme}-keyframe.partial.png"
        keyframe.save(temporary, optimize=True)
        temporary.replace(ASSETS/f"{theme}-keyframe.png")  # Builds never see half a file.
        print(f"{theme}: {keyframe.size} -> {ASSETS/f'{theme}-keyframe.png'}", flush=True)

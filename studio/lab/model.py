"""The network a camera is given (BUILD_PLAN_NOISE F.3). Needs PyTorch: imported only by training.

A compact U-Net on the four colour planes after the variance-stabilising transform, where noise has strength one
at every ISO. The planes are split inside the network into a smooth part (brightness: context) and the rest
(detail + noise, which is what has to be told apart), and the network outputs **the noise it found**, which is
subtracted. So an untrained network changes nothing, and what it removes can be inspected and tested (the detail
guard, engine/raw/guard.py). There is no adversarial training and nothing generative in it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {            # width, blocks per level (down), blocks in the middle
    "fast": (24, (1, 2), 2),
    "balanced": (40, (2, 2), 4),
    "best": (64, (2, 4), 6),
}


class Block(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.g = nn.Parameter(torch.zeros(1, c, 1, 1))        # starts as "do nothing": training is stable

    def forward(self, x):
        return x + self.g * self.b(F.gelu(self.a(x)))


class CameraNet(nn.Module):
    def __init__(self, size: str = "balanced"):
        super().__init__()
        w, down, mid = SIZES[size]
        self.size = size
        self.head = nn.Conv2d(8, w, 3, padding=1)
        self.enc1 = nn.Sequential(*[Block(w) for _ in range(down[0])])
        self.down1 = nn.Conv2d(w, 2 * w, 2, stride=2)
        self.enc2 = nn.Sequential(*[Block(2 * w) for _ in range(down[1])])
        self.down2 = nn.Conv2d(2 * w, 4 * w, 2, stride=2)
        self.mid = nn.Sequential(*[Block(4 * w) for _ in range(mid)])
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = nn.Sequential(*[Block(2 * w) for _ in range(down[1])])
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = nn.Sequential(*[Block(w) for _ in range(down[0])])
        self.tail = nn.Conv2d(w, 4, 3, padding=1)
        nn.init.zeros_(self.tail.weight), nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        smooth = F.avg_pool2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        f = self.head(torch.cat([x - smooth, smooth * 0.02], dim=1))
        e1 = self.enc1(f)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(self.up2(m) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        return x - self.tail(d1)                              # the photo minus the noise the network found


def export_onnx(net: "CameraNet", path: str) -> None:
    net = net.eval().cpu()
    dummy = torch.zeros(1, 4, 64, 64)
    kw = dict(input_names=["planes"], output_names=["clean"], opset_version=17,
              dynamic_axes={"planes": {0: "n", 2: "h", 3: "w"}, "clean": {0: "n", 2: "h", 3: "w"}})
    try:
        torch.onnx.export(net, dummy, path, dynamo=False, **kw)
    except TypeError:                                         # an older PyTorch has no "dynamo" argument
        torch.onnx.export(net, dummy, path, **kw)

"""Colour spaces, white points, chromatic adaptation and transfer functions.

Working space (decision 0.4): linear ProPhoto RGB (ROMM primaries), D50, 32-bit float, scene-referred.
Default export: Adobe RGB (1998). Everything is derived from primaries + white point, so the numbers
are exact and the ICC profiles we write (icc.py) agree with the pixel maths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# CIE xy white points
D65_XY = (0.3127, 0.3290)
D50_XY = (0.3457, 0.3585)
# ICC PCS illuminant (exactly as the ICC spec writes it)
ICC_D50_XYZ = np.array([0.9642, 1.0, 0.8249])

BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])


def xy_to_XYZ(xy, Y=1.0) -> np.ndarray:
    x, y = xy
    return np.array([x * Y / y, Y, (1 - x - y) * Y / y])


def XYZ_to_xy(XYZ) -> tuple[float, float]:
    X, Y, Z = XYZ
    s = X + Y + Z
    return (X / s, Y / s)


def bradford(src_XYZ, dst_XYZ) -> np.ndarray:
    """3x3 Bradford chromatic adaptation matrix from source white to destination white (XYZ)."""
    s = BRADFORD @ np.asarray(src_XYZ, float)
    d = BRADFORD @ np.asarray(dst_XYZ, float)
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def rgb_to_xyz_matrix(primaries, white_xy) -> np.ndarray:
    """Linear RGB -> XYZ (under the space's own white) from xy primaries."""
    P = np.array([xy_to_XYZ(p) for p in primaries]).T          # columns = primaries at Y=1
    W = xy_to_XYZ(white_xy)
    S = np.linalg.solve(P, W)
    return P * S[None, :]


# ------------------------------------------------------------------------------------------ transfer curves
def srgb_encode(x):
    x = np.asarray(x)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def srgb_decode(v):
    v = np.asarray(v)
    return np.where(v <= 0.04045, v / 12.92, np.power((np.maximum(v, 0) + 0.055) / 1.055, 2.4))


def gamma_encode(g):
    def f(x):
        return np.power(np.maximum(np.asarray(x), 0), 1.0 / g)
    return f


def gamma_decode(g):
    def f(v):
        return np.power(np.maximum(np.asarray(v), 0), g)
    return f


ADOBE_GAMMA = 563.0 / 256.0          # 2.19921875, exactly as the Adobe RGB (1998) spec and ICC u8Fixed8
PROPHOTO_GAMMA = 1.8


@dataclass(frozen=True)
class RGBSpace:
    key: str
    name: str
    primaries: tuple
    white_xy: tuple
    trc: str                 # 'srgb' or 'gamma'
    gamma: float = 1.0

    @property
    def to_xyz(self) -> np.ndarray:
        return rgb_to_xyz_matrix(self.primaries, self.white_xy)

    @property
    def to_xyz_d50(self) -> np.ndarray:
        """RGB -> XYZ adapted to D50 (what an ICC profile stores as rXYZ/gXYZ/bXYZ)."""
        M = self.to_xyz
        A = bradford(xy_to_XYZ(self.white_xy), xy_to_XYZ(D50_XY))
        return A @ M

    def encode(self, x):
        return srgb_encode(x) if self.trc == "srgb" else gamma_encode(self.gamma)(x)

    def decode(self, v):
        return srgb_decode(v) if self.trc == "srgb" else gamma_decode(self.gamma)(v)


SRGB = RGBSpace("srgb", "sRGB", ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), D65_XY, "srgb")
ADOBE_RGB = RGBSpace("adobergb", "Adobe RGB (1998) compatible", ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)),
                     D65_XY, "gamma", ADOBE_GAMMA)
PROPHOTO = RGBSpace("prophoto", "ProPhoto RGB (ROMM) compatible",
                    ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), D50_XY, "gamma", PROPHOTO_GAMMA)
DISPLAY_P3 = RGBSpace("displayp3", "Display P3", ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), D65_XY, "srgb")
REC2020 = RGBSpace("rec2020", "Rec. 2020 (linear render space)", ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
                   D65_XY, "gamma", 2.4)

SPACES = {s.key: s for s in (SRGB, ADOBE_RGB, PROPHOTO, DISPLAY_P3, REC2020)}


def conversion_matrix(src: RGBSpace, dst: RGBSpace) -> np.ndarray:
    """Linear src RGB -> linear dst RGB, adapting white points with Bradford via D50 (ICC-consistent)."""
    return np.linalg.inv(dst.to_xyz_d50) @ src.to_xyz_d50


def apply_matrix(img: np.ndarray, M: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """img HxWx3 float32 times 3x3 matrix (applied to each pixel's RGB column)."""
    M = np.asarray(M, np.float32)
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    res = flat @ M.T
    res = res.reshape(h, w, 3)
    if out is not None:
        out[...] = res
        return out
    return res.astype(np.float32, copy=False)


# ------------------------------------------------------------------------------------------ Lab (for tests)
def xyz_to_lab(XYZ, white=None):
    white = ICC_D50_XYZ if white is None else np.asarray(white)
    t = np.asarray(XYZ) / white
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

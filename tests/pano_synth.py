"""A synthetic world for the panorama tests: an equirectangular texture and frames rendered from it by a rotating
camera with known rotations and focal length (the truth the stitcher must recover)."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def world(h=1024, w=2048, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.zeros((h, w), np.float32)
    for s, a in ((1.0, 0.35), (2.5, 0.5), (6, 0.6), (18, 0.8), (60, 1.0)):
        t += a * gaussian_filter(rng.normal(0, 1, (h, w)), s, mode="wrap") * s ** 0.5
    t = (t - t.mean()) / t.std()
    img = 0.25 * np.exp(0.6 * t)
    # a few hard-edged shapes (buildings, poles) so there are corners at every place
    for _ in range(160):
        y, x = rng.integers(0, h), rng.integers(0, w)
        hh, ww = rng.integers(4, 40), rng.integers(4, 40)
        img[y:y + hh, x:x + ww] *= rng.uniform(0.4, 2.0)
    return np.clip(img, 0.005, 0.95).astype(np.float32)


def rot(yaw=0.0, pitch=0.0, roll=0.0) -> np.ndarray:
    """World -> camera rotation for a camera turned by yaw (right), pitch (up) and roll, in degrees."""
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    cam_to_world = Ry @ Rx @ Rz
    return cam_to_world.T


def render(tex: np.ndarray, R: np.ndarray, f: float, w: int, h: int, gain: float = 1.0,
           vignette: float = 0.0) -> np.ndarray:
    H, W = tex.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ray = np.stack([xx - (w - 1) / 2.0, yy - (h - 1) / 2.0, np.full_like(xx, f)], -1)
    d = ray @ R                                        # R^T ray
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    th = np.arctan2(d[..., 0], d[..., 2])
    ph = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = (th / (2 * math.pi) + 0.5) * W - 0.5
    v = (ph / math.pi + 0.5) * H - 0.5
    out = map_coordinates(tex, [v, u], order=3, mode="wrap").astype(np.float32)
    if vignette:
        r = np.hypot(xx - w / 2, yy - h / 2) / math.hypot(w / 2, h / 2)
        out *= (1 - vignette * r ** 2).astype(np.float32)
    return np.clip(out * gain, 0, 1).astype(np.float32)

"""Transforming a layer or a selection (BUILD_PLAN_PAINT P4): free (move, scale, rotate, skew, flip),
perspective (four corners), warp (a grid of points) and liquify (push pixels with a brush).

Every transform is worked out from the original pixels kept when it started, and the result is resampled once
(plan 0.9): moving a layer ten times is as sharp as moving it once. Resampling is bilinear on premultiplied paint, so
edges stay clean.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

KINDS = {
    "free": ("Free", "moving, scaling, rotating, skewing and flipping", "bending: use Warp"),
    "perspective": ("Perspective", "fitting a picture onto a surface seen at an angle: drag the four corners",
                    "curved surfaces"),
    "warp": ("Warp", "bending a shape smoothly: a 4 x 4 grid of points", "small precise changes"),
    "liquify": ("Liquify", "pushing paint around with a brush, as in wet clay", "large straight moves"),
    "cage": ("Cage", "bending a shape by moving the points of a cage drawn around it (arms, tails, bottles)",
             "anything outside the cage stays put"),
    "mesh": ("Mesh", "fine bending with a dense grid of points (cloth, labels on curved things)", "many points to "
             "move for a simple change"),
}


def mean_value_coords(px, py, cage):
    """Mean-value coordinates (Floater 2003; Hormann & Floater 2006) of points (px, py) in a closed polygon:
    (N, k) weights summing to 1 that move smoothly with the cage."""
    v = np.asarray(cage, np.float64)
    k = len(v)
    dx = v[None, :, 0] - px[:, None]
    dy = v[None, :, 1] - py[:, None]
    r = np.maximum(np.hypot(dx, dy), 1e-9)
    nx, ny = np.roll(dx, -1, 1), np.roll(dy, -1, 1)
    cross = dx * ny - dy * nx
    dot = dx * nx + dy * ny
    ang = np.arctan2(cross, dot)                                 # angle at the point between v_i and v_i+1
    t = np.tan(ang / 2)
    w = (np.roll(t, 1, 1) + t) / r
    s = w.sum(1, keepdims=True)
    return w / np.where(np.abs(s) < 1e-12, 1e-12, s)


def cage_warp(img, src_cage, dst_cage, step: int = 6):
    """Deform what is inside ``src_cage`` so that the cage becomes ``dst_cage``. Each output pixel inside the new
    cage looks up its source through mean-value coordinates of the new cage applied to the old one (worked out on a
    coarse grid, then smoothed up to full size)."""
    from .select import polygon
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h + step:step, 0:w + step:step].astype(np.float64)
    lam = mean_value_coords(gx.ravel(), gy.ravel(), dst_cage)
    src = np.asarray(src_cage, np.float64)
    sx = (lam @ src[:, 0]).reshape(gx.shape)
    sy = (lam @ src[:, 1]).reshape(gx.shape)
    X = ndimage.zoom(sx, step, order=1)[:h, :w]
    Y = ndimage.zoom(sy, step, order=1)[:h, :w]
    inside_src = polygon(w, h, list(map(tuple, src_cage)))
    out = sample(img * inside_src[..., None], X, Y)
    return out * polygon(w, h, list(map(tuple, dst_cage)))[..., None]


def affine(tx=0.0, ty=0.0, sx=1.0, sy=1.0, angle=0.0, shx=0.0, cx=0.0, cy=0.0):
    """3x3 matrix: about (cx, cy) skew, scale, rotate (degrees), then move."""
    a = math.radians(angle)
    T = lambda x, y: np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    S = np.diag([sx, sy, 1.0])
    K = np.array([[1, shx, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    return T(cx + tx, cy + ty) @ R @ S @ K @ T(-cx, -cy)


def homography(src, dst):
    """3x3 H with H @ src_i ~ dst_i for four point pairs."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(A, np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def warp_with_matrix(img, M, out_shape):
    """Each output pixel looks up the source through the inverse of M (bilinear)."""
    Minv = np.linalg.inv(M)
    h, w = out_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X = Minv[0, 0] * xx + Minv[0, 1] * yy + Minv[0, 2]
    Y = Minv[1, 0] * xx + Minv[1, 1] * yy + Minv[1, 2]
    Z = Minv[2, 0] * xx + Minv[2, 1] * yy + Minv[2, 2]
    return sample(img, X / Z, Y / Z)


def sample(img, X, Y):
    out = np.empty(X.shape + (img.shape[2],), np.float32)
    for c in range(img.shape[2]):
        out[..., c] = ndimage.map_coordinates(img[..., c], [Y, X], order=1, mode="constant", cval=0.0)
    return out


def grid_warp(img, src_pts, dst_pts):
    """Warp by a grid of points (n x n): the displacement between grid points is interpolated smoothly, and each
    output pixel looks up where it came from."""
    h, w = img.shape[:2]
    n = src_pts.shape[0]
    disp = (src_pts - dst_pts)                                # backward displacement at the grid points (n, n, 2)
    zy = (h / (n - 1), w / (n - 1))
    dx = ndimage.zoom(disp[..., 0], zy, order=3)[:h, :w]
    dy = ndimage.zoom(disp[..., 1], zy, order=3)[:h, :w]
    dx = np.pad(dx, ((0, h - dx.shape[0]), (0, w - dx.shape[1])), mode="edge")
    dy = np.pad(dy, ((0, h - dy.shape[0]), (0, w - dy.shape[1])), mode="edge")
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return sample(img, xx + dx, yy + dy)


class Liquify:
    """A displacement field over the layer, pushed by a brush; the picture is resampled from the original."""

    def __init__(self, original):
        self.orig = original
        h, w = original.shape[:2]
        self.dx = np.zeros((h, w), np.float32)
        self.dy = np.zeros((h, w), np.float32)

    def push(self, x, y, mx, my, radius, strength=0.6):
        """Drag the paint under a round brush at (x, y) by (mx, my)."""
        h, w = self.dx.shape
        r = int(math.ceil(radius))
        x0, y0, x1, y1 = max(0, int(x) - r), max(0, int(y) - r), min(w, int(x) + r + 1), min(h, int(y) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        f = np.clip(1 - ((xx - x) ** 2 + (yy - y) ** 2) / (radius * radius), 0, 1) ** 2 * strength
        self.dx[y0:y1, x0:x1] -= f * mx                       # backward map: look up where the paint came from
        self.dy[y0:y1, x0:x1] -= f * my
        return x0, y0, x1, y1

    def result(self, rect=None):
        h, w = self.dx.shape
        x0, y0, x1, y1 = rect or (0, 0, w, h)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        return sample(self.orig, xx + self.dx[y0:y1, x0:x1], yy + self.dy[y0:y1, x0:x1])

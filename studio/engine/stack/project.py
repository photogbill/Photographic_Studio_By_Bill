"""Panorama projections (BUILD_PLAN_PANO_IR.md 2.5): forward (direction -> canvas) and inverse (canvas -> direction)
maps, and where a canvas pixel falls in a frame.

World coordinates: x right, y down, z forward (the panorama's middle). A canvas is in pixels at ``scale`` canvas pixels
per radian (``scale = f``: full resolution at the middle). Every projection with its best-for / weak-at text:

* rectilinear: straight lines stay straight; up to about 120 degrees wide, beyond that the edges stretch.
* cylindrical: wide horizontal sweeps; vertical lines stay vertical, the horizon bends if it is not level.
* spherical (equirectangular): anything up to 360 x 180; the format 360 viewers read (GPano fields written).
* panini (vedutismo, d = 1): wide architecture and interiors: verticals and lines through the middle stay straight
  with far less stretching than rectilinear.
* mercator: tall panoramas; keeps shapes locally but stretches the top and bottom.
"""
from __future__ import annotations

import math

import numpy as np

METHOD_INFO = {
    "auto": {"name": "Auto", "does": "Rectilinear when the panorama is narrower than 100 degrees each way, "
             "cylindrical for a wide sweep, spherical for anything taller.",
             "best": "Most panoramas.", "weak": "May not be the look you want for architecture: try Panini."},
    "rectilinear": {"name": "Rectilinear", "does": "Like one very wide lens: straight lines stay straight.",
                    "best": "Up to about 120 degrees; buildings, horizons.",
                    "weak": "Past 120 degrees the edges stretch badly."},
    "cylindrical": {"name": "Cylindrical", "does": "Wrapped around a cylinder: verticals stay vertical.",
                    "best": "Wide horizontal sweeps (landscapes, skylines).",
                    "weak": "Horizontal lines away from the horizon bend; poor for tall panoramas."},
    "spherical": {"name": "Spherical (360)", "does": "Longitude across, latitude down (equirectangular).",
                  "best": "Full 360 x 180 panoramas for 360 viewers; anything very tall.",
                  "weak": "Straight lines curve; the top and bottom stretch."},
    "panini": {"name": "Panini", "does": "The painters' wide view (vedutismo): verticals and lines through the "
               "middle stay straight.", "best": "Wide interiors and architecture.",
               "weak": "Horizontal lines away from the middle curve."},
    "mercator": {"name": "Mercator", "does": "Cylindrical with the vertical stretched to keep shapes locally.",
                 "best": "Tall panoramas that should keep round things round.",
                 "weak": "The top and bottom stretch; not for near-vertical views."},
}
PROJECTIONS = ("rectilinear", "cylindrical", "spherical", "panini", "mercator")


def to_canvas(d: np.ndarray, projection: str, scale: float) -> tuple:
    """World directions (N x 3, any length) -> canvas (u, v) in pixels around the middle, and a valid mask."""
    d = np.asarray(d, float)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    n = np.sqrt(x * x + y * y + z * z) + 1e-300
    x, y, z = x / n, y / n, z / n
    if projection == "rectilinear":
        ok = z > 1e-6
        zz = np.where(ok, z, 1e-6)
        return scale * x / zz, scale * y / zz, ok
    th = np.arctan2(x, z)
    if projection == "cylindrical":
        rho = np.hypot(x, z)
        ok = rho > 1e-9
        return scale * th, scale * y / np.where(ok, rho, 1e-9), ok
    phi = np.arcsin(np.clip(y, -1, 1))
    if projection == "spherical":
        return scale * th, scale * phi, np.ones_like(th, bool)
    if projection == "mercator":
        ok = np.abs(phi) < math.radians(85)
        return scale * th, scale * np.log(np.tan(np.pi / 4 + np.clip(phi, -1.48, 1.48) / 2)), ok
    if projection == "panini":
        ok = np.abs(th) < math.radians(170)
        S = 2.0 / (1.0 + np.cos(th))
        return scale * S * np.sin(th), scale * S * np.tan(np.clip(phi, -1.5, 1.5)), ok
    raise ValueError(f"unknown projection {projection!r}")


def from_canvas(u: np.ndarray, v: np.ndarray, projection: str, scale: float) -> np.ndarray:
    """Canvas (u, v) pixels around the middle -> world directions (... x 3, unit length)."""
    a = np.asarray(u, float) / scale
    b = np.asarray(v, float) / scale
    if projection == "rectilinear":
        d = np.stack([a, b, np.ones_like(a)], -1)
    elif projection == "cylindrical":
        d = np.stack([np.sin(a), b, np.cos(a)], -1)
    elif projection == "spherical":
        cb = np.cos(b)
        d = np.stack([np.sin(a) * cb, np.sin(b), np.cos(a) * cb], -1)
    elif projection == "mercator":
        phi = np.arctan(np.sinh(b))
        cb = np.cos(phi)
        d = np.stack([np.sin(a) * cb, np.sin(phi), np.cos(a) * cb], -1)
    elif projection == "panini":
        th = 2.0 * np.arctan(a / 2.0)
        S = 2.0 / (1.0 + np.cos(th))
        tphi = b / S
        d = np.stack([np.sin(th), tphi, np.cos(th)], -1)
    else:
        raise ValueError(f"unknown projection {projection!r}")
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def to_camera(d: np.ndarray, R: np.ndarray, f: float, cx: float, cy: float) -> tuple:
    """World directions -> frame pixel (x, y) and a mask of directions in front of the camera."""
    r = np.asarray(d, float) @ np.asarray(R).T
    z = r[..., 2]
    ok = z > 1e-6
    zz = np.where(ok, z, 1e-6)
    return f * r[..., 0] / zz + cx, f * r[..., 1] / zz + cy, ok


def from_camera(x: np.ndarray, y: np.ndarray, R: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    ray = np.stack([np.asarray(x, float) - cx, np.asarray(y, float) - cy, np.full(np.shape(x), float(f))], -1)
    d = ray @ np.asarray(R)                            # R^T ray
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def auto_projection(extent_h_deg: float, extent_v_deg: float) -> str:
    if extent_h_deg < 100 and extent_v_deg < 100:
        return "rectilinear"
    if extent_v_deg < 100:
        return "cylindrical"
    return "spherical"


def frame_outline(R, f, w, h, n: int = 24) -> np.ndarray:
    """Directions along a frame's border (for its outline on the layout and the canvas size)."""
    t = np.linspace(0, 1, n)
    xs = np.r_[t * (w - 1), np.full(n, w - 1.0), (1 - t) * (w - 1), np.zeros(n)]
    ys = np.r_[np.zeros(n), t * (h - 1), np.full(n, h - 1.0), (1 - t) * (h - 1)]
    return from_camera(xs, ys, R, f, (w - 1) / 2.0, (h - 1) / 2.0)


def canvas_box(Rs: list, f: float, w: int, h: int, projection: str, scale: float, pad: int = 2) -> tuple:
    """(u0, v0, width, height) of the canvas that holds every placed frame, in canvas pixels."""
    us, vs = [], []
    for R in Rs:
        if R is None:
            continue
        d = frame_outline(R, f, w, h)
        u, v, ok = to_canvas(d, projection, scale)
        us.append(u[ok])
        vs.append(v[ok])
    u = np.concatenate(us)
    v = np.concatenate(vs)
    u0, u1 = math.floor(u.min()) - pad, math.ceil(u.max()) + pad
    v0, v1 = math.floor(v.min()) - pad, math.ceil(v.max()) + pad
    if projection in ("spherical", "cylindrical", "mercator", "panini"):
        lim = math.pi * scale
        u0, u1 = max(u0, -lim), min(u1, lim)
    return int(u0), int(v0), int(u1 - u0), int(v1 - v0)


def extents_deg(Rs: list, f: float, w: int, h: int) -> tuple:
    """Horizontal and vertical extent of the frames in degrees (spherical coordinates)."""
    th, ph = [], []
    for R in Rs:
        if R is None:
            continue
        d = frame_outline(R, f, w, h)
        th.append(np.arctan2(d[:, 0], d[:, 2]))
        ph.append(np.arcsin(np.clip(d[:, 1], -1, 1)))
    th = np.unwrap(np.sort(np.concatenate(th)))
    ph = np.concatenate(ph)
    return math.degrees(float(th.max() - th.min())), math.degrees(float(ph.max() - ph.min()))

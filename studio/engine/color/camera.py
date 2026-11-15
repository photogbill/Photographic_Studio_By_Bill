"""Stage 11: white balance and camera RGB -> working space (linear ProPhoto, D50).

Phase 1 uses the single D65 camera matrix LibRaw provides (Adobe DNG ColorMatrix2 style, XYZ -> camera).
The dcraw/LibRaw construction is used: rows of (camera <- linear sRGB) are normalised so that a
white-balanced neutral (1,1,1) in camera space maps to the white point, then inverted.
Dual-illuminant (DCP-style) profiles and ColorChecker profiles come with calibration (Phase 2/7).

Temperature/tint <-> multipliers uses the same matrix: a white of chromaticity xy is converted to camera
RGB and the multipliers are its reciprocal. This is approximate away from D65 (a single matrix), which
is the standard trade-off until dual-illuminant profiles exist.
"""
from __future__ import annotations

import math

import numpy as np

from .spaces import PROPHOTO, SRGB, bradford, xy_to_XYZ, D50_XY, D65_XY


def camera_to_working(cam_xyz: np.ndarray, working=PROPHOTO) -> np.ndarray:
    """3x3 matrix: white-balanced camera RGB -> linear working RGB (default ProPhoto D50)."""
    cam_xyz = np.asarray(cam_xyz, float)[:3, :3]
    if not np.isfinite(cam_xyz).all() or abs(np.linalg.det(cam_xyz)) < 1e-9:
        raise ValueError("camera matrix missing or singular")
    xyz_srgb = SRGB.to_xyz                         # linear sRGB (D65) -> XYZ D65
    cam_srgb = cam_xyz @ xyz_srgb                  # linear sRGB -> camera
    cam_srgb = cam_srgb / cam_srgb.sum(axis=1, keepdims=True)
    srgb_cam = np.linalg.inv(cam_srgb)             # WB'd camera -> linear sRGB
    # linear sRGB (D65) -> XYZ D65 -> Bradford D50 -> working
    to_d50 = bradford(xy_to_XYZ(D65_XY), xy_to_XYZ(D50_XY)) @ xyz_srgb
    return np.linalg.inv(working.to_xyz_d50) @ to_d50 @ srgb_cam


def normalise_wb(mult) -> np.ndarray:
    """4 LibRaw-style multipliers (R, G, B, G2) -> 3 multipliers (R, G, B) with G = 1."""
    m = [float(x) for x in mult]
    g = m[1] if m[1] > 0 else 1.0
    r = m[0] / g if m[0] > 0 else 1.0
    b = m[2] / g if m[2] > 0 else 1.0
    return np.array([r, 1.0, b])


# ------------------------------------------------------------------------------------------ temperature / tint
def cct_to_xy(T: float) -> tuple[float, float]:
    """Correlated colour temperature (K) -> CIE xy on the Planckian locus (Kim et al. 2002 cubic spline),
    valid 1667 K - 25000 K."""
    T = float(min(max(T, 1667.0), 25000.0))
    if T <= 4000:
        x = -0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.240390
    if T <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return x, y


def _xy_to_uv(x, y):
    d = -2 * x + 12 * y + 3
    return 4 * x / d, 6 * y / d          # CIE 1960 uv


def _uv_to_xy(u, v):
    d = 2 * u - 8 * v + 4
    return 3 * u / d, 2 * v / d


def temp_tint_to_xy(T: float, tint: float = 0.0) -> tuple[float, float]:
    """Temperature + tint -> xy. Tint moves perpendicular to the locus in CIE 1960 uv;
    +tint = towards magenta, -tint = towards green, 1 unit = 0.0001 uv (Adobe-like scale /30)."""
    x0, y0 = cct_to_xy(T)
    if tint == 0:
        return x0, y0
    x1, y1 = cct_to_xy(T * 1.01)
    u0, v0 = _xy_to_uv(x0, y0)
    u1, v1 = _xy_to_uv(x1, y1)
    du, dv = u1 - u0, v1 - v0
    n = math.hypot(du, dv) or 1.0
    # perpendicular, oriented so positive tint goes below the locus (magenta)
    pu, pv = dv / n, -du / n
    if pv > 0:
        pu, pv = -pu, -pv
    k = tint * 0.0003
    return _uv_to_xy(u0 + pu * k, v0 + pv * k)


def xy_to_cct(x: float, y: float) -> float:
    """McCamy 1992 approximation (good to a few K between 2856 K and 6504 K; fine for display)."""
    n = (x - 0.3320) / (0.1858 - y)
    return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33


def multipliers_for_white(cam_xyz: np.ndarray, xy) -> np.ndarray:
    """Camera multipliers (R, G, B; G=1) that make a light source of chromaticity xy neutral."""
    cam = np.asarray(cam_xyz, float)[:3, :3] @ xy_to_XYZ(xy)
    cam = np.maximum(cam, 1e-6)
    m = 1.0 / cam
    return m / m[1]


def white_for_multipliers(cam_xyz: np.ndarray, mult) -> tuple[float, float]:
    """Inverse of multipliers_for_white: the xy the given multipliers neutralise."""
    m = np.asarray(mult, float)[:3]
    cam = 1.0 / np.maximum(m, 1e-9)
    XYZ = np.linalg.solve(np.asarray(cam_xyz, float)[:3, :3], cam)
    s = XYZ.sum()
    return XYZ[0] / s, XYZ[1] / s


def temp_tint_from_multipliers(cam_xyz, mult) -> tuple[float, float]:
    """Best-fit temperature and tint for the given multipliers (for showing as-shot WB as sliders)."""
    x, y = white_for_multipliers(cam_xyz, mult)
    T = min(max(xy_to_cct(x, y), 2000.0), 25000.0)
    # refine T and solve tint by small search (cheap, robust)
    best = (1e9, T, 0.0)
    for _ in range(3):
        _, T0, t0 = best
        for T1 in np.geomspace(T0 * 0.9, T0 * 1.1, 21):
            for tn in np.linspace(t0 - 60, t0 + 60, 25):
                xx, yy = temp_tint_to_xy(T1, tn)
                err = (xx - x) ** 2 + (yy - y) ** 2
                if err < best[0]:
                    best = (err, T1, tn)
    return float(best[1]), float(best[2])

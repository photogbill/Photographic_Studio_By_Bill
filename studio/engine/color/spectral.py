"""Camera colour from Solomatov and Akkaynak's data (arXiv:2304.11549, 2023, CC BY 4.0): two-light matrices (the
Studio's default colour from camera_matrix version 2) and spectral profiles (a choice in the Camera profile list).

Two-light matrices (``dual_profile``): for 1,013 cameras their data holds the camera's colour matrix for daylight
(D65: the same one the Studio's raw reader has, and it is used only when the two agree) and for tungsten (CIE A).
The photo's white balance chooses the mix between them, as the DNG specification describes, so colour in warm
light no longer drifts warm and saturated the way a daylight-only matrix does. Daylight photos are unchanged.

Spectral camera profiles: colour for any light, worked out from the camera's own colour sensitivities.

Solomatov and Akkaynak ("Spectral Sensitivity Estimation Without a Camera", arXiv:2304.11549, 2023, CC BY 4.0)
predicted the red, green and blue sensitivity curves of 1,012 cameras without a camera or a colour chart
(profiles/spectral/camera_sensitivities.json, with their attribution). With a camera's curves the Studio can
calculate, for the light a photo was taken in, how the camera sees every colour and how the eye sees it, and fit
the 3x3 matrix between the two: a profile made for that exact light, from candle flame (about 1,850 K) to shade,
where a single daylight matrix (the camera matrix) drifts warm and saturated in warm light and a two-light profile
stops at tungsten (2,856 K).

How a profile is made (all from published standards, no Adobe data):
* The light: a black body (Planck's law) below 4,000 K, CIE daylight (its S0, S1, S2 basis) from 4,000 K up.
* Its temperature: the one whose white the camera sees with the photo's own white-balance multipliers
  (searched on the camera's curves; tint is left to the white balance itself).
* The colours: a fixed set of smooth reflectance spectra spread across real surface colours (made here, seeded),
  plus white.
* The fit: least squares from white-balanced camera RGB to XYZ, with white held exactly, then Bradford to D50
  (the DNG ForwardMatrix convention), then to the working space.

The data files are loaded once; matrices are cached per camera and per 25 K.
"""
from __future__ import annotations

import json
import math
import os
import threading

import numpy as np

from ... import log as slog
from ... import paths
from .spaces import D50_XY, PROPHOTO, bradford, xy_to_XYZ

log = slog.get("engine.spectral")

_LOCK = threading.Lock()
_DATA: dict = {}
_CACHE: dict = {}
CITATION = ("Solomatov & Akkaynak, \"Spectral Sensitivity Estimation Without a Camera\", arXiv:2304.11549 (2023), "
            "CC BY 4.0")
NAME = "Spectral (Solomatov & Akkaynak 2023)"
KEY = "spectral:"                               # the edit's "profile" value for this method


def data_dir() -> str:
    return os.path.join(str(paths.PROFILES), "spectral")


def _load():
    with _LOCK:
        if not _DATA:
            from .dcp import _norm
            with open(os.path.join(data_dir(), "camera_sensitivities.json"), encoding="utf-8") as fh:
                d = json.load(fh)
            with open(os.path.join(data_dir(), "cie_400_700_10nm.json"), encoding="utf-8") as fh:
                c = json.load(fh)
            if d["wavelengths"] != c["wavelengths"]:
                raise ValueError("spectral data: wavelength grids differ")
            _DATA["wl"] = np.asarray(c["wavelengths"], np.float64)
            _DATA["cmf"] = np.asarray(c["cmf_xyz"], np.float64)
            _DATA["S"] = [np.asarray(c[k], np.float64) for k in ("daylight_S0", "daylight_S1", "daylight_S2")]
            _DATA["cams"] = {_norm(k): (k, np.asarray(v, np.float64)) for k, v in d["cameras"].items()}
            _DATA["refl"] = _reflectances(_DATA["wl"])
            log.info("spectral: sensitivities of %d cameras loaded", len(_DATA["cams"]))
        return _DATA


def available() -> bool:
    return os.path.isfile(os.path.join(data_dir(), "camera_sensitivities.json"))


def camera_name(make: str, model: str) -> str | None:
    """The data's name for this camera, or None when it is not among the 1,012."""
    if not available():
        return None
    from .dcp import camera_keys
    cams = _load()["cams"]
    for k in camera_keys(make, model):
        if k in cams:
            return cams[k][0]
    return None


def _ssf(make, model):
    from .dcp import camera_keys
    cams = _load()["cams"]
    for k in camera_keys(make, model):
        if k in cams:
            return cams[k][1]
    return None


# ------------------------------------------------------------------------------------------ light and colours
def illuminant(cct: float) -> np.ndarray:
    """Relative spectral power of the light at ``cct`` kelvin on the data's grid: Planck below 4,000 K, CIE daylight
    from 4,000 K (the same split the DNG specification uses for its temperature scale)."""
    d = _load()
    wl = d["wl"] * 1e-9
    T = float(min(max(cct, 1000.0), 25000.0))
    if T < 4000.0:
        c2 = 1.4388e-2
        spd = 1.0 / (wl ** 5 * (np.exp(c2 / (wl * T)) - 1.0))
    else:
        x = (-4.6070e9 / T ** 3 + 2.9678e6 / T ** 2 + 0.09911e3 / T + 0.244063) if T <= 7000 else \
            (-2.0064e9 / T ** 3 + 1.9018e6 / T ** 2 + 0.24748e3 / T + 0.237040)
        y = -3.0 * x * x + 2.870 * x - 0.275
        m = 0.0241 + 0.2562 * x - 0.7341 * y
        m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m
        m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m
        S0, S1, S2 = d["S"]
        spd = S0 + m1 * S1 + m2 * S2
    return spd / spd.max()


def _reflectances(wl: np.ndarray, n: int = 600, seed: int = 20231123) -> np.ndarray:
    """Smooth surface reflectances (0.02 to 0.95) of every hue and strength, plus greys: sums of one to three soft
    bumps and steps, the shapes real dyes, skin, foliage and paints have. Seeded, so every PC fits the same matrix."""
    rng = np.random.default_rng(seed)
    x = (wl - wl[0]) / (wl[-1] - wl[0])
    out = []
    for k in range(n):
        r = np.full_like(x, rng.uniform(0.02, 0.25))
        for _ in range(rng.integers(1, 4)):
            c, w, a = rng.uniform(-0.1, 1.1), rng.uniform(0.06, 0.4), rng.uniform(0.1, 0.8)
            if rng.random() < 0.35:                               # a step: yellows, oranges, reds
                r = r + a / (1.0 + np.exp(-(x - c) / (0.3 * w)))
            else:
                r = r + a * np.exp(-0.5 * ((x - c) / w) ** 2)
        out.append(np.clip(r, 0.02, 0.95))
    for g in (0.03, 0.1, 0.2, 0.4, 0.6, 0.9):
        out.append(np.full_like(x, g))
    return np.asarray(out)


# ------------------------------------------------------------------------------------------ the profile
def camera_white(ssf: np.ndarray, cct: float) -> np.ndarray:
    """What the camera records (R, G, B) for a white lit by ``cct``."""
    return ssf.T @ illuminant(cct)


def cct_for_multipliers(ssf: np.ndarray, mult) -> float:
    """The light temperature whose white the photo's white-balance multipliers (R, G, B; G = 1) make neutral."""
    m = np.asarray(mult, np.float64)[:3]
    want = np.array([1.0 / max(m[0], 1e-9), 1.0, 1.0 / max(m[2], 1e-9)])     # the camera's white, G = 1
    Ts = np.exp(np.linspace(math.log(1500.0), math.log(15000.0), 241))
    best, err = 5000.0, None
    for T in Ts:
        w = camera_white(ssf, T)
        w = w / w[1]
        e = (math.log(w[0] / want[0])) ** 2 + (math.log(w[2] / want[2])) ** 2
        if err is None or e < err:
            best, err = float(T), e
    lo, hi = best / 1.03, best * 1.03                              # refine
    for _ in range(30):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        ea = _wb_err(ssf, a, want)
        eb = _wb_err(ssf, b, want)
        lo, hi = (lo, b) if ea < eb else (a, hi)
    return float(0.5 * (lo + hi))


def _wb_err(ssf, T, want):
    w = camera_white(ssf, T)
    w = w / w[1]
    return (math.log(w[0] / want[0])) ** 2 + (math.log(w[2] / want[2])) ** 2


def forward_matrix(ssf: np.ndarray, cct: float) -> np.ndarray:
    """White-balanced camera RGB -> XYZ D50 for a photo lit by ``cct`` (DNG ForwardMatrix convention): fitted on the
    reflectance set with white held exactly."""
    d = _load()
    E = illuminant(cct)
    R = d["refl"] * E[None, :]
    cam = R @ ssf                                                 # (n, 3) camera RGB
    xyz = R @ d["cmf"]                                            # (n, 3) XYZ
    cw = ssf.T @ E
    xw = d["cmf"].T @ E
    C = cam / cw[None, :]                                         # white-balanced: white -> (1, 1, 1)
    X = xyz / xw[1]                                               # white luminance 1
    xw = xw / xw[1]
    # weight darker colours up a little, so the fit is not ruled by the brightest patches
    wgt = 1.0 / np.sqrt(np.maximum(X[:, 1], 0.02))
    Cw, Xw = C * wgt[:, None], X * wgt[:, None]
    G = np.linalg.inv(Cw.T @ Cw)
    M0 = (G @ Cw.T @ Xw).T
    one = np.ones(3)
    g1 = G @ one
    M = M0 + np.outer(xw - M0 @ one, g1) / float(one @ g1)        # M @ (1, 1, 1) = white exactly
    wxy = (xw[0] / xw.sum(), xw[1] / xw.sum())
    return bradford(xy_to_XYZ(wxy), xy_to_XYZ(D50_XY)) @ M


def camera_to_working(make: str, model: str, mult, working=PROPHOTO, cam_xyz=None) -> tuple[np.ndarray, float]:
    """(3x3 white-balanced camera -> working RGB, the light's temperature) for this camera and white balance.
    The temperature comes from the camera matrix when given (``cam_xyz``): the predicted curves are scaled per
    channel, so they fix the shape of each colour's response but not the balance between red, green and blue that
    the multipliers measure; the fit itself does not depend on that balance (white is divided out).
    ValueError when the camera is not in the data."""
    ssf = _ssf(make, model)
    if ssf is None:
        raise ValueError(f"no spectral data for {make} {model}")
    if cam_xyz is not None:
        from .camera import temp_tint_from_multipliers
        T = float(temp_tint_from_multipliers(cam_xyz, mult)[0])
    else:
        T = cct_for_multipliers(ssf, mult)
    key = (make, model, int(round(T / 25.0)))
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is None:
        F = forward_matrix(ssf, key[2] * 25.0)
        hit = np.linalg.inv(working.to_xyz_d50) @ F
        with _LOCK:
            if len(_CACHE) > 256:
                _CACHE.clear()
            _CACHE[key] = hit
        log.info("spectral profile: %s %s at %.0f K", make, model, T)
    return hit, T


# ------------------------------------------------------------------------------------------ two-light matrices
_DUAL: dict = {}


def dual_available() -> bool:
    return os.path.isfile(os.path.join(data_dir(), "dual_matrices.json"))


def _duals() -> dict:
    with _LOCK:
        if not _DUAL and dual_available():
            from .dcp import _norm
            with open(os.path.join(data_dir(), "dual_matrices.json"), encoding="utf-8") as fh:
                d = json.load(fh)
            for k, v in d["cameras"].items():
                _DUAL[_norm(k)] = (k, np.asarray(v["A"], float), np.asarray(v["D65"], float))
            log.info("two-light matrices of %d cameras loaded", len(_DUAL))
        return _DUAL


def dual_profile(make: str, model: str, cam_xyz=None, tol: float = 0.02):
    """A two-light (tungsten + daylight) profile for this camera as a dcp.Profile, or None when the camera is not
    in the data, or when its daylight matrix disagrees with ``cam_xyz`` (the raw reader's, by more than ``tol`` in
    any entry): then the raw reader's own matrix stays, so daylight colour never changes underneath anyone."""
    from .dcp import Profile, camera_keys
    duals = _duals()
    for k in camera_keys(make, model):
        if k in duals:
            name, A, D = duals[k]
            if cam_xyz is not None and float(np.abs(np.asarray(cam_xyz, float)[:3, :3] - D).max()) > tol:
                log.info("two-light matrices for %s differ from the raw reader's daylight matrix; not used", name)
                return None
            return Profile(name="Studio two-light", camera=name, copyright=CITATION, illuminants=[17, 21],
                           color_matrices=[A, D])
    return None

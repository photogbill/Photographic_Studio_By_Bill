"""What a camera's calibration takes out of a photo before Clean sees it (BUILD_PLAN_NOISE block C).

Not everything in a raw that is not picture is noise. Some of it is the same in every photo this camera takes, and
the dark frames measured it. That part is not cleaned, it is **subtracted**: nothing is guessed and no detail is
at risk. In order:

  1. **Hot and warm photosites**, from the camera's defect map (the ones that stood out in nearly every dark frame
     at this ISO; for exposures of a second or more also the ones of the nearest long, warm dark frames). A mapped
     photosite is healed only when it shows in THIS photo: when it stands above its same-colour neighbours by more
     than the photo's own noise there. In a bright area a warm photosite is hidden under photon noise and is left
     alone, because it still carries the picture. Truly hot ones are always healed. Only mapped positions are ever
     touched, so a star, a catchlight or a candle spark cannot be mistaken for a defect.
  2. **The fixed banding**: the row and column pattern that is the same in every frame (deep calibration),
     subtracted exactly. The per-photo row measurement from the masked border (raw/banding.py) keeps the part it
     is sure of; the calibration supplies the rest of the fixed part.
  3. **The black level**: a photo takes its black from the masked border beside the picture; the dark frames show
     by how much that misses the picture area's true black, per colour position.

Everything here needs a profile for this camera (Cameras page); the fixed banding needs Deep calibration too. With
neither, nothing is done and nothing is claimed. Each step reports what it did (cyan: measured on this body).
"""
from __future__ import annotations

import os
import threading

import numpy as np

from ... import log as slog

log = slog.get("engine.calibrated")

SHOW_SIGMAS = 3.0           # a mapped warm photosite is healed when it stands this far above its neighbours
LONG_S = 1.0
_LOCK = threading.Lock()
_CACHE: dict = {}


class CameraFix:
    """One camera's subtractable faults, loaded once."""

    def __init__(self, make: str, model: str):
        from ..calib.deep import find_bank
        from ..calib.profile import find_profile, local_data_dir
        self.profile = find_profile(make, model)
        self.bank = find_bank(make, model)
        self.defects, self.width = {}, 0
        path = os.path.join(local_data_dir(make, model), "defects.npz")
        if os.path.isfile(path):
            try:
                with np.load(path, allow_pickle=False) as z:
                    self.width = int(z["width"])
                    for k in z.files:
                        if k != "width" and not k.endswith("|dn"):
                            self.defects[k] = (z[k].astype(np.int64), z[k + "|dn"].astype(np.float32))
            except Exception:                                 # noqa: BLE001  (a damaged map is no map)
                slog.log_exception(f"defect map unreadable: {path}", log)
                self.defects = {}

    @property
    def any(self) -> bool:
        return self.profile is not None

    def entry(self, iso):
        return self.profile.entry_for(iso) if self.profile is not None else None

    def defect_list(self, iso: float, exposure_s, temp_c) -> tuple | None:
        """(flat indices, DN above black) for a photo at this ISO, shutter time and camera temperature."""
        e = self.entry(iso)
        if e is None:
            return None
        got = [self.defects[k] for k in (f"{e.iso:g}|short",) if k in self.defects]
        if exposure_s and exposure_s >= LONG_S:
            best, score = None, None
            for k in self.defects:
                parts = k.split("|")
                if parts[0] != f"{e.iso:g}" or parts[1] == "short":
                    continue
                sec = float(parts[1])
                tmp = float(parts[2]) if len(parts) > 2 and parts[2] else None
                if not 0.4 <= sec / exposure_s <= 2.5:
                    continue
                d = abs(np.log2(sec / exposure_s)) + (abs(tmp - temp_c) / 8.0 if tmp is not None and temp_c is not None
                                                     else 0.5)
                if score is None or d < score:
                    best, score = k, d
            if best is not None:
                got.append(self.defects[best])
        if not got:
            return None
        idx = np.concatenate([g[0] for g in got])
        dn = np.concatenate([g[1] for g in got])
        order = np.argsort(-dn, kind="stable")                 # the same photosite twice: keep its larger level
        idx, dn = idx[order], dn[order]
        _, first = np.unique(idx, return_index=True)
        return idx[first], dn[first]


def for_camera(make: str, model: str) -> CameraFix:
    from ..calib.profile import local_data_dir, local_profile_path
    d = local_data_dir(make, model)
    stamp = tuple(os.path.getmtime(p) if os.path.isfile(p) else 0.0 for p in (
        local_profile_path(make, model), os.path.join(d, "defects.npz"), os.path.join(d, "fixed_pattern.npz")))
    key = (make, model)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is None or hit[0] != stamp:
            _CACHE[key] = (stamp, CameraFix(make, model))
        return _CACHE[key][1]


def _neighbour_estimate(mos: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """What each photosite should be, from its same-colour neighbours two away, along the direction in which the
    picture changes least (so an edge passing through is followed, not blurred across)."""
    pairs = (((-2, 0), (2, 0)), ((0, -2), (0, 2)), ((-2, -2), (2, 2)), ((-2, 2), (2, -2)))
    best, grad = None, None
    for (dy1, dx1), (dy2, dx2) in pairs:
        a, b = mos[y + dy1, x + dx1], mos[y + dy2, x + dx2]
        g = np.abs(a - b)
        m = 0.5 * (a + b)
        if best is None:
            best, grad = m, g
        else:
            take = g < grad
            best, grad = np.where(take, m, best), np.where(take, g, grad)
    return best


def heal(mos: np.ndarray, idx: np.ndarray, dn: np.ndarray, width: int, x0: int, y0: int, span: float,
         sigma_of, gain: float | None = None) -> dict:
    """Heal, in place, the mapped photosites that show in this (region of a) normalised mosaic.
    ``sigma_of(level, i, j)``: the photo's noise at a brightness, per colour position. ``gain``: electrons per raw
    level at the photo's ISO, which says which mapped photosites are hot (profile.hot_limit_dn). Returns counts."""
    from ..calib.profile import hot_limit_dn
    h, w = mos.shape
    yy, xx = idx // width - y0, idx % width - x0
    inside = (yy >= 2) & (yy < h - 2) & (xx >= 2) & (xx < w - 2)
    yy, xx, dn = yy[inside], xx[inside], dn[inside]
    if yy.size == 0:
        return dict(hot=0, warm=0, mapped=0)
    # two mapped photosites two apart would heal from each other: estimate all first, then write
    est = _neighbour_estimate(mos, yy, xx)
    val = mos[yy, xx]
    sig = np.empty_like(est)
    for i in (0, 1):
        for j in (0, 1):
            m = ((yy + y0) % 2 == i) & ((xx + x0) % 2 == j)
            if m.any():
                sig[m] = sigma_of(np.maximum(est[m], 0.0), i, j)
    hot = dn >= hot_limit_dn(span, gain)
    shows = (val - est) > SHOW_SIGMAS * sig
    fix = hot | shows
    mos[yy[fix], xx[fix]] = est[fix]
    return dict(hot=int(np.count_nonzero(hot)), warm=int(np.count_nonzero(fix & ~hot)), mapped=int(yy.size))


def apply(mos: np.ndarray, image, x0: int, y0: int, row_weight: float | None, use_optical_black: bool = True) -> dict:
    """Take out of a normalised mosaic region (first photosite = visible-area (x0, y0)) what this camera's
    calibration measured, in place. ``row_weight``: how much of each row's offset the per-photo banding step
    removes (None: that step is off or has nothing), so the fixed rows are not taken out twice.
    Returns a report {hot, warm, mapped, rows_dn, cols_dn, black, used: [...]} (empty when nothing applies)."""
    m = image.meta
    fix = for_camera(m.make or "", m.model or "")
    if not fix.any:
        return {}
    e = fix.entry(m.iso)
    if e is None:
        return {}
    lv = image.levels(use_optical_black)
    span = float(np.mean(lv.range))
    rep: dict = dict(used=[])
    # 3. black first: the neighbours a defect is judged against should stand on the right level
    if e.black_offset_dn and lv.source == "optical-black":
        off = np.asarray(e.black_offset_dn, np.float32)
        for i in (0, 1):
            for j in (0, 1):
                mos[(i - y0) % 2::2, (j - x0) % 2::2] -= off[i, j] / np.float32(lv.range[i, j])
        rep["black"] = [round(float(v), 3) for v in off.ravel()]
        rep["used"].append("black")
    # 2. the banding that is the same in every photo
    lines = fix.bank.fixed_lines(e.iso) if fix.bank is not None else None
    if lines is not None and lines[0].size == image.raw.height and lines[1].size == image.raw.width:
        rows, cols = lines
        h, w = mos.shape
        keep = 1.0 if row_weight is None else float(np.clip(1.0 - row_weight, 0.0, 1.0))
        r = rows[y0:y0 + h] * np.float32(keep / span)
        c = cols[x0:x0 + w] / np.float32(span)
        mos -= r[:, None]
        mos -= c[None, :]
        rep["rows_dn"], rep["cols_dn"] = round(float(np.std(rows) * keep), 3), round(float(np.std(cols)), 3)
        rep["used"].append("fixed pattern")
    # 1. defects that show
    got = fix.defect_list(m.iso, m.exposure_s, m.camera_temperature_c) if fix.width == image.raw.width else None
    if got is not None:
        model = image.noise_model(use_optical_black)

        def sigma_of(level, i, j):
            return np.sqrt(np.maximum(float(model.a[i, j]) * level + float(model.b[i, j]), 1e-20))
        rep.update(heal(mos, got[0], got[1], fix.width, x0, y0, span, sigma_of, fix.profile.gain_at(m.iso)))
        rep["used"].append("hot pixels")
    return rep if rep["used"] else {}

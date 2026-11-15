"""Calibration builder (section 7, Phase 2): checklist frames in, camera profile out. Headless; the wizard
(Phase 7) is a friendly face on top of this.

The user never sorts or renames anything: every frame is classified from its own pixels and metadata.
  * dark frame (Part A, body cap on): signal within a few read-noise sigmas of black
  * flat frame (Part B, evenly lit): bright and smooth across the centre of the frame
  * anything else is ignored and reported, with the reason

Per ISO, from dark frames: black level and temporal read noise per CFA position, row and column banding
(white-noise part removed), hot pixels, and dark current against exposure time and camera temperature.
From pairs of flats at the same ISO and shutter speed: gain (e-/DN) by photon transfer,
gain = mean / (var(F1 - F2) / 2 - read_noise^2), on the centre of the frame where vignetting is small.

Quick profile = dark frames only (section 21 change 2): read noise, black, banding, hot pixels. The signal part of
the noise model is then estimated per image. Full profile = darks + flat pairs.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from ..io.rawfile import RawImage
from ..raw.linearize import clipped_stats
from .profile import CameraProfile, IsoEntry, local_data_dir

log = slog.get("engine.calib")


@dataclass
class FrameInfo:
    path: str
    kind: str                     # dark | flat | other
    iso: float | None
    exposure_s: float | None
    temp_c: float | None
    mean_dn: float = 0.0
    reason: str = ""


@dataclass
class BuildReport:
    frames: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def counts(self) -> dict:
        c = defaultdict(int)
        for f in self.frames:
            c[f.kind] += 1
        return dict(c)


def _centre(a: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = a.shape
    ch, cw = int(h * frac) // 2 * 2, int(w * frac) // 2 * 2
    y0, x0 = ((h - ch) // 2) // 2 * 2, ((w - cw) // 2) // 2 * 2
    return a[y0:y0 + ch, x0:x0 + cw]


def _ob_black_and_noise(img: RawImage) -> tuple[float, float]:
    """Mean black and read noise from the left masked columns (for classification only)."""
    if img.left >= 16:
        ob = img.mosaic[img.top:img.top + img.height, 4:img.left - 4].astype(np.float32)
        return clipped_stats(ob[::4])
    return float(np.mean(img.black_libraw)), float("nan")


def classify(img: RawImage) -> FrameInfo:
    m = img.meta
    info = FrameInfo(path=img.path, kind="other", iso=m.iso, exposure_s=m.exposure_s, temp_c=m.camera_temperature_c)
    black, rn = _ob_black_and_noise(img)
    c = _centre(img.visible.astype(np.float32))
    mean = float(np.median(c[::3, ::3])) - black
    info.mean_dn = mean
    rng = img.clip_level() - black
    if not math.isfinite(rn) or rn <= 0:
        rn = max(1.0, float(clipped_stats(c[::5, ::5])[1]))
    if abs(mean) < max(3.0 * rn / 10.0, 0.002 * rng) + 2.0:
        info.kind, info.reason = "dark", "signal at black level"
        return info
    if mean < 0.02 * rng:
        info.reason = "too dark for a flat, too bright for a dark frame (light leak?)"
        return info
    if mean > 0.92 * rng:
        info.reason = "over-exposed (clipped)"
        return info
    # smoothness: block means of one green plane vary little across the centre
    g = c[0::2, 1::2] if int(img.cfa_rgb[0, 1]) == 1 else c[0::2, 0::2]
    bh, bw = g.shape[0] // 16, g.shape[1] // 16
    blocks = g[:bh * 16, :bw * 16].reshape(16, bh, 16, bw).mean(axis=(1, 3)) - black
    cv = float(blocks.std() / max(blocks.mean(), 1e-6))
    if cv < 0.08:
        info.kind, info.reason = "flat", f"even light (variation {100 * cv:.1f}%)"
    else:
        info.reason = f"not even enough for a flat (variation {100 * cv:.0f}%)"
    return info


def _per_position(a: np.ndarray, fn) -> np.ndarray:
    return np.array([[fn(a[i::2, j::2]) for j in (0, 1)] for i in (0, 1)], float)


def _dark_stats(frames: list[RawImage]) -> dict:
    vis = [_centre(f.visible.astype(np.float32), 0.9) for f in frames]
    black = np.mean([_per_position(v, lambda p: clipped_stats(p[::2, ::2])[0]) for v in vis], axis=0)
    if len(vis) >= 2:                                  # temporal noise from a frame difference (fixed pattern cancels)
        d = vis[0] - vis[1]
        d = d - np.median(d, axis=1, keepdims=True)    # row banding is reported separately, not as pixel noise
        rn = _per_position(d, lambda p: clipped_stats(p[::2, ::2])[1]) / math.sqrt(2.0)
    else:
        rn = _per_position(vis[0], lambda p: clipped_stats(p[::2, ::2])[1])
    # banding: variance of row (column) means beyond what white noise predicts
    v0 = vis[0] - np.kron(np.ones((vis[0].shape[0] // 2, vis[0].shape[1] // 2)), black)[:vis[0].shape[0],
                                                                                         :vis[0].shape[1]]
    s_all = float(np.sqrt(np.mean(rn ** 2)))
    keep = np.abs(v0) < 6 * s_all
    rows = np.where(keep, v0, 0).sum(1) / np.maximum(keep.sum(1), 1)
    cols = np.where(keep, v0, 0).sum(0) / np.maximum(keep.sum(0), 1)
    row_b = math.sqrt(max(float(rows.var()) - s_all ** 2 / v0.shape[1], 0.0))
    col_b = math.sqrt(max(float(cols.var()) - s_all ** 2 / v0.shape[0], 0.0))
    # hot pixels: far above black in every dark frame at this ISO (persistent, not cosmic rays)
    hot = None
    for f in frames:
        v = f.visible.astype(np.float32)
        thr = np.kron(np.ones((v.shape[0] // 2 + 1, v.shape[1] // 2 + 1)), black + 10 * rn)[:v.shape[0], :v.shape[1]]
        h = v > thr
        hot = h if hot is None else (hot & h)
    return dict(black=black, rn=rn, row=row_b, col=col_b, hot=hot)


def _gain_from_pairs(pairs: list, black: np.ndarray, rn: np.ndarray) -> tuple[float | None, int, float | None]:
    gains, wells = [], []
    for f1, f2 in pairs:
        a = _centre(f1.visible.astype(np.float32), 0.4)
        b = _centre(f2.visible.astype(np.float32), 0.4)
        for i in (0, 1):
            for j in (0, 1):
                pa, pb = a[i::2, j::2], b[i::2, j::2]
                mu = 0.5 * float(pa.mean() + pb.mean()) - black[i, j]
                var = float(np.var(pa - pb)) / 2.0 - rn[i, j] ** 2
                if mu > 0 and var > 0:
                    gains.append(mu / var)
        wells.append(f1.clip_level())
    if not gains:
        return None, 0, None
    g = float(np.median(gains))
    fw = g * (float(np.median(wells)) - float(np.mean(black)))
    return g, len(pairs), fw


def build_profile(raws: list[RawImage], save_local_data: bool = True) -> tuple[CameraProfile, BuildReport]:
    report = BuildReport()
    infos = [classify(r) for r in raws]
    report.frames = infos
    if not raws:
        raise ValueError("no frames given")
    models = {(r.meta.make, r.meta.model) for r in raws}
    if len(models) > 1:
        report.warnings.append(f"Frames from more than one camera model: {sorted(models)}. Using the most common.")
    make, model = max(models, key=lambda mm: sum((r.meta.make, r.meta.model) == mm for r in raws))
    by_iso_dark, by_iso_flat = defaultdict(list), defaultdict(list)
    for r, inf in zip(raws, infos):
        if (r.meta.make, r.meta.model) != (make, model) or not inf.iso:
            continue
        if inf.kind == "dark":
            by_iso_dark[inf.iso].append((r, inf))
        elif inf.kind == "flat":
            by_iso_flat[inf.iso].append((r, inf))
    entries = []
    hot_maps = {}
    for iso in sorted(set(by_iso_dark) | set(by_iso_flat)):
        darks = by_iso_dark.get(iso, [])
        e = IsoEntry(iso=float(iso), frames=len(darks) + len(by_iso_flat.get(iso, [])))
        if darks:
            # read noise / black / banding from the shortest exposures (no dark current)
            tmin = min((inf.exposure_s or 0) for _, inf in darks)
            short = [r for r, inf in darks if (inf.exposure_s or 0) <= tmin * 1.5 + 1e-9] or [darks[0][0]]
            st = _dark_stats(short[:4])
            e.black_dn = st["black"].round(3).tolist()
            e.read_noise_dn = st["rn"].round(3).tolist()
            e.row_banding_dn = round(st["row"], 3)
            e.col_banding_dn = round(st["col"], 3)
            if st["hot"] is not None:
                e.hot_pixels = int(st["hot"].sum())
                hot_maps[iso] = np.argwhere(st["hot"])
            base = float(np.mean(st["black"]))
            for r, inf in darks:                      # dark current: excess over the short-exposure black
                v = _centre(r.visible.astype(np.float32), 0.5)
                e.dark_current.append(dict(exposure_s=inf.exposure_s, temp_c=inf.temp_c,
                                           excess_dn=round(float(clipped_stats(v[::3, ::3])[0]) - base, 3)))
        else:
            report.warnings.append(f"ISO {iso:g}: flats but no dark frames; read noise from the masked border only.")
            r0 = by_iso_flat[iso][0][0]
            black, rn = _ob_black_and_noise(r0)
            e.black_dn = np.full((2, 2), round(black, 3)).tolist()
            e.read_noise_dn = np.full((2, 2), round(rn, 3)).tolist()
        flats = by_iso_flat.get(iso, [])
        groups = defaultdict(list)
        for r, inf in flats:
            groups[round(math.log2(inf.exposure_s or 1), 1)].append(r)
        pairs = [(g[k], g[k + 1]) for g in groups.values() for k in range(0, len(g) - 1, 2)]
        if pairs:
            g, n, fw = _gain_from_pairs(pairs, np.asarray(e.black_dn), np.asarray(e.read_noise_dn))
            e.gain_e_per_dn = None if g is None else round(g, 5)
            e.gain_pairs = n
            e.full_well_e = None if fw is None else round(fw, 0)
        elif flats:
            report.warnings.append(f"ISO {iso:g}: a flat without a partner (same ISO and shutter speed); "
                                   f"shoot flats in pairs to measure gain.")
        entries.append(e)
    method = "full" if any(e.gain_e_per_dn for e in entries) else "quick"
    n_dark = sum(len(v) for v in by_iso_dark.values())
    n_flat = sum(len(v) for v in by_iso_flat.values())
    quality = dict(isos=len(entries), isos_with_gain=sum(1 for e in entries if e.gain_e_per_dn),
                   dark_frames=n_dark, flat_frames=n_flat, ignored=sum(1 for i in infos if i.kind == "other"),
                   score=round(min(100.0, 10 * len(entries) + 5 * sum(1 for e in entries if e.gain_e_per_dn)
                                   + (10 if any(len(e.dark_current) > 1 for e in entries) else 0)), 0))
    prof = CameraProfile(make=make, model=model, method=method, isos=entries, quality=quality)
    if save_local_data and hot_maps:
        d = local_data_dir(make, model)
        with open(os.path.join(d, "hot_pixels.json"), "w", encoding="utf-8") as fh:   # private, never shared
            json.dump({str(k): v.tolist() for k, v in hot_maps.items()}, fh)
    log.info("calibration: %s %s, %s profile, %d ISOs, %d darks, %d flats", make, model, method, len(entries),
             n_dark, n_flat)
    return prof, report

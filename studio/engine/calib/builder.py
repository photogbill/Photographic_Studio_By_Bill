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

Since 2026-11-18 the builder works from per-frame measurements (frames.py), not from opened frames, so a pile of
any size can be built, frames of several cameras can be mixed (one profile each), and frames at different camera
temperatures are kept apart: read noise per temperature, and how fast the noise of long exposures grows with heat.
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
    left_out: list = field(default_factory=list)      # frames of cameras that got no profile (no dark frames)

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


FLAT_CENTRE = 0.04          # the centre of a flat varies less than this (where gain is measured)
FLAT_STEP = 0.065           # neighbouring areas of a flat differ by less than this: lens falloff is gentle
FLAT_FALLOFF = 0.35         # and the whole half-frame by less than this


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
    # A flat is smooth, not uniform. Bill's first flats (2026-11-18, a fast lens wide open) fell off 8 to 11% across
    # the half-frame from the lens alone and 78 of 89 were left out by the old "varies under 8%" test. What gain
    # is measured from is the small centre crop, so: the very centre must be even, the wider area may fall off
    # as a lens does (gently, block to block), and it must not have the texture or edges of a real photo.
    g = c[0::2, 1::2] if int(img.cfa_rgb[0, 1]) == 1 else c[0::2, 0::2]
    bh, bw = g.shape[0] // 16, g.shape[1] // 16
    blocks = g[:bh * 16, :bw * 16].reshape(16, bh, 16, bw).mean(axis=(1, 3)) - black
    level = max(float(blocks.mean()), 1e-6)
    cv = float(blocks.std() / level)
    inner = blocks[6:10, 6:10]
    cv_in = float(inner.std() / max(float(inner.mean()), 1e-6))
    steps = np.concatenate([np.abs(np.diff(blocks, axis=0)).ravel(), np.abs(np.diff(blocks, axis=1)).ravel()])
    rough = float(np.percentile(steps, 90) / level)
    if cv_in < FLAT_CENTRE and rough < FLAT_STEP and cv < FLAT_FALLOFF:
        info.kind, info.reason = "flat", f"even light (centre varies {100 * cv_in:.1f}%, falloff {100 * cv:.0f}%)"
    else:
        info.reason = f"not even enough for a flat (variation {100 * cv:.0f}%)"
    return info


# ================================================================================================ building
TEMP_STEP = 4.0             # camera temperatures are grouped in steps of this many degrees C
SHORT_S = 0.05              # darks this short (or the shortest given) carry no dark current: read noise comes from them
LONG_S = 1.0                # from here on a dark frame says something about heat


def temp_bin(t) -> float | None:
    return None if t is None else float(round(float(t) / TEMP_STEP) * TEMP_STEP)


def _exp_key(t) -> float:
    return round(math.log2(t), 1) if t and t > 0 else -99.0


def _rms(a) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def _order(frames: list) -> list:
    return sorted(frames, key=lambda f: (f.datetime or "", f.name))


def _pair_noise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Temporal noise (2x2, DN) from two frames' centre crops: the fixed pattern cancels in the difference."""
    d = a.astype(np.float32) - b.astype(np.float32)
    d = d - np.median(d, axis=1, keepdims=True)        # row banding is reported separately, not as pixel noise
    return np.array([[clipped_stats(d[i::2, j::2])[1] for j in (0, 1)] for i in (0, 1)], float) / math.sqrt(2.0)


PAIR_MAX_DT = 4.0           # two frames are a pair when the camera was within this many degrees C for both


def _pairs_noise(frames: list) -> np.ndarray | None:
    """Temporal noise from neighbours in time. A pair is two frames at nearly the same camera temperature,
    whichever temperature step each falls in: Bill's first pile had its two frames per ISO at 26 and 27 C, one
    each side of a step, and no pair was made of them."""
    fr = [f for f in _order(frames) if f.crop is not None]
    est = []
    for k in range(len(fr) - 1):
        a, b = fr[k], fr[k + 1]
        near = a.temp_c is None or b.temp_c is None or abs(a.temp_c - b.temp_c) <= PAIR_MAX_DT
        if near and a.crop.shape == b.crop.shape:
            est.append(_pair_noise(a.crop, b.crop))
    return np.median(est, axis=0) if est else None


HOT_FRACTION = 0.01         # a HOT pixel stands at least this much of the sensor's range above black with no
#                             light: a bright dot in any photo. Below that it is WARM: it stands out of the
#                             noise at this ISO (and is healed), but only shows when shadows are lifted.


def _persistent_hot(frames: list) -> np.ndarray | None:
    """Photosites that stand out in (nearly) every frame: defects, not cosmic rays or noise."""
    lists = [f.hot for f in frames if f.hot is not None]
    if not lists:
        return None
    if len(lists) == 1:
        return np.asarray(lists[0], np.int64)
    idx, n = np.unique(np.concatenate([np.asarray(h, np.int64) for h in lists]), return_counts=True)
    return idx[n >= max(2, math.ceil(0.7 * len(lists)))]


def _hot_and_warm(frames: list, keep: np.ndarray, black: float) -> tuple[int, int]:
    """Of the persistent photosites ``keep``: how many are hot (HOT_FRACTION of the range above black, median
    over the frames) and how many only warm. Frames measured before the excess was kept count all as warm."""
    if keep is None or keep.size == 0:
        return 0, 0
    fr = [f for f in frames if f.hot is not None and f.hot_dn is not None]
    if not fr:
        return 0, int(keep.size)
    rows = []
    for f in fr:
        h = np.asarray(f.hot, np.int64)
        pos = np.clip(np.searchsorted(h, keep), 0, max(h.size - 1, 0))
        found = h.size > 0 and h[pos] == keep
        vals = np.where(found, np.asarray(f.hot_dn, np.float64)[pos] if h.size else 0.0, np.nan)
        rows.append(vals)
    med = np.nanmedian(np.stack(rows), axis=0)
    limit = HOT_FRACTION * max(float(np.median([f.clip_dn for f in fr])) - black, 1.0)
    hot = int(np.sum(med >= limit))
    return hot, int(keep.size) - hot


def _gain_from_flats(flats: list, black: np.ndarray, rn: np.ndarray, warnings: list, iso: float):
    groups = defaultdict(list)
    for f in flats:
        if f.crop is not None:
            groups[_exp_key(f.exposure_s)].append(f)
    gains, wells, pairs, odd, uneven = [], [], 0, 0, 0
    for g in groups.values():
        g = _order(g)
        odd += len(g) % 2
        for k in range(0, len(g) - 1, 2):
            a, b = g[k].crop.astype(np.float32), g[k + 1].crop.astype(np.float32)
            if a.shape != b.shape:
                continue
            ma, mb = float(a.mean()), float(b.mean())
            if abs(ma - mb) > 0.03 * max(ma, mb):         # the light changed between the two: not a pair
                uneven += 1
                continue
            pairs += 1
            for i in (0, 1):
                for j in (0, 1):
                    pa, pb = a[i::2, j::2], b[i::2, j::2]
                    mu = 0.5 * float(pa.mean() + pb.mean()) - black[i, j]
                    var = float(np.var(pa - pb)) / 2.0 - rn[i, j] ** 2
                    if mu > 0 and var > 0:
                        gains.append(mu / var)
            wells.append(g[k].clip_dn)
    if uneven:
        warnings.append(f"ISO {iso:g}: {uneven} pair(s) of flats left out: the light changed between the two frames.")
    if not gains:
        if flats:
            warnings.append(f"ISO {iso:g}: a flat without a partner (same ISO and shutter speed); "
                            f"shoot flats in pairs to measure gain.")
        return None, 0, None
    g = float(np.median(gains))
    return g, pairs, g * (float(np.median(wells)) - float(np.mean(black)))


def _thermal(rows: list, rn: float) -> dict | None:
    """How fast the noise of a long exposure grows: extra variance per second, and the temperature step that
    doubles it when the frames span enough degrees to tell. (Canon subtracts the dark level itself, so the mean
    of a long dark frame says little; its noise says it all.)"""
    pts = []
    for r in rows:
        t, s = r.get("exposure_s"), r.get("noise_dn")
        if t and t >= LONG_S and s:
            extra = s * s - rn * rn
            if extra > 0.02 * rn * rn:
                pts.append((r.get("temp_c"), extra / t))
    if not pts:
        return None
    rates = np.array([p[1] for p in pts], float)
    temps = [p[0] for p in pts if p[0] is not None]
    out = dict(var_dn2_per_s=round(float(np.median(rates)), 5),
               at_temp_c=round(float(np.median(temps)), 1) if temps else None, doubling_c=None, points=len(pts))
    known = [(p[0], p[1]) for p in pts if p[0] is not None]
    if len(known) >= 3 and max(k[0] for k in known) - min(k[0] for k in known) >= 6.0:
        T = np.array([k[0] for k in known], float)
        y = np.log2(np.array([k[1] for k in known], float))
        slope, icpt = np.polyfit(T, y, 1)
        if slope > 0 and 2.0 <= 1.0 / slope <= 30.0:
            t0 = float(np.median(T))
            out.update(var_dn2_per_s=round(float(2.0 ** (icpt + slope * t0)), 5), at_temp_c=round(t0, 1),
                       doubling_c=round(float(1.0 / slope), 2))
    return out


def _advice(entries: list, cov: dict, n_flat: int) -> list:
    """What would make this profile better, in the photographer's words. Most important first."""
    out = []
    single = [e.iso for e in entries if 0 < cov["short"].get(e.iso, 0) < 2]
    none = [e.iso for e in entries if cov["short"].get(e.iso, 0) == 0 and e.read_noise_dn is not None]
    if single:
        out.append("Only one short dark frame at ISO " + ", ".join(f"{i:g}" for i in single[:12]) +
                   (" ..." if len(single) > 12 else "") + ": with two, noise can be told from the sensor's fixed "
                   "pattern. Shoot each ISO at least twice.")
    if none:
        out.append("No short dark frames at ISO " + ", ".join(f"{i:g}" for i in none[:12]) + " (only long "
                   "exposures): read noise there includes heat. Add a few at the fastest shutter speed.")
    temps = cov["temps"]
    if not temps:
        out.append("These files do not record the camera's temperature, so heat cannot be told apart.")
    elif max(temps) - min(temps) < 8:
        out.append(f"All frames are between {min(temps):g} and {max(temps):g} °C. A second pass with the camera "
                   f"warm from use (or cold, from outdoors) shows how the noise changes with heat.")
    if not any(e.thermal for e in entries):
        out.append("No long dark frames (1 second or more), so the noise of long exposures is not measured. "
                   "A few at 1 s, 8 s and 30 s, cap on, is enough.")
    if not n_flat:
        out.append("No flat frames: gain is estimated from each photo (a Quick profile). Pairs of evenly lit "
                   "frames per ISO make it exact.")
    return out


def _build_camera(make: str, model: str, frames: list, save_local_data: bool) -> tuple[CameraProfile, BuildReport]:
    report = BuildReport(frames=list(frames))
    by_dark, by_flat = defaultdict(list), defaultdict(list)
    for f in frames:
        if f.kind in ("dark", "flat") and not f.iso:
            report.warnings.append(f"{f.name}: no ISO in the file, left out.")
            continue
        if f.kind == "dark":
            by_dark[float(f.iso)].append(f)
        elif f.kind == "flat":
            by_flat[float(f.iso)].append(f)
    for f in frames:
        for n in f.notes:
            report.warnings.append(f"{f.name}: {n}")
    entries, hot_short, hot_long = [], {}, {}
    cov = dict(short={}, temps=[], cells={})
    all_temps = set()
    for iso in sorted(set(by_dark) | set(by_flat)):
        darks, flats = by_dark.get(iso, []), by_flat.get(iso, [])
        e = IsoEntry(iso=float(iso), frames=len(darks) + len(flats))
        rn = None
        if darks:
            tmin = min((f.exposure_s or 0) for f in darks)
            short = [f for f in darks if (f.exposure_s or 0) <= max(tmin * 1.5, SHORT_S) + 1e-9]
            if tmin >= LONG_S:
                report.warnings.append(f"ISO {iso:g}: only long dark frames ({tmin:g} s and up); read noise from "
                                       f"them includes heat.")
            cov["short"][iso] = 0 if tmin >= LONG_S else len(short)
            by_e = defaultdict(list)
            for f in short:
                by_e[_exp_key(f.exposure_s)].append(f)
            est = [x for x in (_pairs_noise(g) for g in by_e.values()) if x is not None]
            rn = np.median(est, axis=0) if est else np.median([f.sigma for f in short], axis=0)
            e.read_noise_paired = bool(est)
            black = np.median([f.black for f in short], axis=0)
            e.black_dn = np.round(black, 3).tolist()
            e.read_noise_dn = np.round(rn, 3).tolist()
            e.row_banding_dn = round(float(np.median([f.row_band for f in short])), 3)
            e.col_banding_dn = round(float(np.median([f.col_band for f in short])), 3)
            hp = _persistent_hot(short)
            if hp is not None:
                e.hot_pixels, e.warm_pixels = _hot_and_warm(short, hp, float(np.mean(black)))
                hot_short[iso] = (hp, short[0].width)
            # per camera temperature (short frames): read noise, banding, black
            tb_groups = defaultdict(list)
            for f in short:
                if f.temp_c is not None:
                    tb_groups[temp_bin(f.temp_c)].append(f)
            for tb in sorted(tb_groups):
                g = tb_groups[tb]
                sub = defaultdict(list)
                for f in g:
                    sub[_exp_key(f.exposure_s)].append(f)
                pe = [x for x in (_pairs_noise(s) for s in sub.values()) if x is not None]
                rt = np.median(pe, axis=0) if pe else np.median([f.sigma for f in g], axis=0)
                e.by_temp.append(dict(temp_c=tb, frames=len(g), paired=bool(pe),
                                      read_noise_dn=np.round(rt, 3).tolist(),
                                      row_banding_dn=round(float(np.median([f.row_band for f in g])), 3),
                                      black_dn=round(float(np.mean([f.black for f in g])), 3)))
            # dark frames by exposure time and temperature: level, noise and hot pixels
            base = float(np.mean(black))
            cells = defaultdict(list)
            for f in darks:
                cells[(_exp_key(f.exposure_s), temp_bin(f.temp_c))].append(f)
                tb = temp_bin(f.temp_c)
                if tb is not None:
                    all_temps.add(tb)
                    cov["cells"].setdefault(f"{iso:g}", {})
                    cov["cells"][f"{iso:g}"][f"{tb:g}"] = cov["cells"][f"{iso:g}"].get(f"{tb:g}", 0) + 1
            for (ek, tb) in sorted(cells, key=lambda k: (k[0], -999 if k[1] is None else k[1])):
                g = cells[(ek, tb)]
                exp = float(np.median([f.exposure_s or 0 for f in g]))
                pn = _pairs_noise(g)
                row = dict(exposure_s=exp, temp_c=tb, frames=len(g),
                           excess_dn=round(float(np.median([f.centre_dn for f in g])) - base, 3),
                           noise_dn=round(float(np.median([_rms(f.sigma) for f in g])), 3),
                           temporal_dn=None if pn is None else round(_rms(pn), 3))
                if exp >= LONG_S:
                    hp = _persistent_hot(g)
                    if hp is not None:
                        row["hot_pixels"], row["warm_pixels"] = _hot_and_warm(g, hp, base)
                        hot_long[f"{iso:g}|{exp:g}|{'' if tb is None else f'{tb:g}'}"] = hp
                e.dark_current.append(row)
            e.thermal = _thermal(e.dark_current, _rms(rn))
        else:
            report.warnings.append(f"ISO {iso:g}: flat frames but no dark frames, so nothing is measured at this "
                                   f"ISO yet. Add dark frames (cap on) at ISO {iso:g}.")
        if flats and rn is not None:
            g, n, fw = _gain_from_flats(flats, np.asarray(e.black_dn), np.asarray(rn), report.warnings, iso)
            e.gain_e_per_dn = None if g is None else round(g, 5)
            e.gain_pairs = n
            e.full_well_e = None if fw is None else round(fw, 0)
        if e.read_noise_dn is not None:
            entries.append(e)
    cov["temps"] = sorted(all_temps)
    cov["short"] = {f"{k:g}": v for k, v in cov["short"].items()}
    method = "full" if any(e.gain_e_per_dn for e in entries) else "quick"
    n_dark = sum(len(v) for v in by_dark.values())
    n_flat = sum(len(v) for v in by_flat.values())
    n = max(len(entries), 1)
    span = (max(all_temps) - min(all_temps)) if all_temps else 0
    score = (min(40.0, 4.0 * len(entries))
             + 20.0 * sum(1 for e in entries if cov["short"].get(f"{e.iso:g}", 0) >= 2) / n
             + (15.0 if span >= 8 else 8.0 if span >= 4 else 0.0)
             + (10.0 if any(e.thermal for e in entries) else 0.0)
             + 15.0 * sum(1 for e in entries if e.gain_e_per_dn) / n)
    cov_short = {float(k): v for k, v in cov["short"].items()}
    quality = dict(isos=len(entries), isos_with_gain=sum(1 for e in entries if e.gain_e_per_dn),
                   dark_frames=n_dark, flat_frames=n_flat,
                   ignored=sum(1 for f in frames if f.kind not in ("dark", "flat")),
                   temp_min_c=min(all_temps) if all_temps else None, temp_max_c=max(all_temps) if all_temps else None,
                   score=round(min(100.0, score), 0), coverage=cov,
                   advice=_advice(entries, dict(short=cov_short, temps=cov["temps"]), n_flat))
    prof = CameraProfile(make=make, model=model, method=method, isos=entries, quality=quality)
    if save_local_data and (hot_short or hot_long):
        d = local_data_dir(make, model)
        with open(os.path.join(d, "hot_pixels.json"), "w", encoding="utf-8") as fh:   # private, never shared
            json.dump({str(k): np.stack([hp // w, hp % w], axis=1).tolist() for k, (hp, w) in hot_short.items()},
                      fh)
        if hot_long:                                     # by ISO | seconds | temperature: flat indices
            with open(os.path.join(d, "hot_pixels_long.npz"), "wb") as fh:
                np.savez_compressed(fh, **{k: v.astype(np.uint32) for k, v in hot_long.items()})
    log.info("calibration: %s %s, %s profile, %d ISOs, %d darks, %d flats, temperatures %s", make, model, method,
             len(entries), n_dark, n_flat, cov["temps"])
    return prof, report


def build_profiles(stats: list, save_local_data: bool = True) -> list:
    """Per-frame measurements (frames.py) of any number of cameras -> [(profile, report)], most frames first.
    A camera with no usable dark frame gets no profile; its frames are in ``left_out`` of the first report."""
    cams = defaultdict(list)
    for s in stats:
        cams[(s.make, s.model)].append(s)
    out, orphans = [], []
    for (make, model), frames in sorted(cams.items(), key=lambda kv: -len(kv[1])):
        if not any(f.kind == "dark" and f.iso for f in frames):
            orphans += frames
            continue
        out.append(_build_camera(make, model, frames, save_local_data))
    if out and orphans:
        out[0][1].left_out = orphans
    return out


def build_profile(raws: list[RawImage], save_local_data: bool = True) -> tuple[CameraProfile, BuildReport]:
    """Opened frames of one camera -> its profile (the first builder's call; small sets and tests). A big pile
    goes through frames.scan and build_profiles instead, one frame in memory at a time."""
    from .frames import measure
    if not raws:
        raise ValueError("no frames given")
    stats = [measure(r) for r in raws]
    built = build_profiles(stats, save_local_data)
    if not built:
        raise ValueError("no dark frames among these: a profile needs frames shot with the lens cap on")
    prof, report = built[0]
    if len({s.camera for s in stats}) > 1:
        report.warnings.append(f"Frames from more than one camera model: {sorted({s.camera for s in stats})}. "
                               f"Using the most common.")
    report.frames = stats
    return prof, report


@dataclass
class DropResult:
    scanned: list                 # FrameStats of the files given now (dark, flat, other, unreadable)
    built: list                   # [(profile, report, saved path)] for every camera in this drop
    cancelled: bool = False

    def counts(self) -> dict:
        c = defaultdict(int)
        for f in self.scanned:
            c[f.kind] += 1
        return dict(c)

    @property
    def left_out(self) -> list:
        return [f for f in self.scanned if f.kind not in ("dark", "flat")]


def calibrate_drop(files: list, progress=None, cancelled=None, workers: int | None = None,
                   save: bool = True) -> DropResult:
    """The whole job for a pile of files: measure each (frames.scan), then rebuild the profile of every camera
    in the pile from ALL its frames measured on this PC so far, so a second drop adds to the first."""
    from . import frames as F
    scanned = F.scan(files, progress=progress, cancelled=cancelled, workers=workers)
    stopped = bool(cancelled is not None and cancelled())
    cams = {s.camera for s in scanned if s.kind in ("dark", "flat")}
    built = []
    if cams:
        stored = [s for s in F.load_store() if s.camera in cams]
        for prof, rep in build_profiles(stored):
            built.append((prof, rep, prof.save() if save else None))
    return DropResult(scanned=scanned, built=built, cancelled=stopped)

"""Measured highlights: DESAT for cameras.

NASA's Solar Dynamics Observatory team recovers saturated flare cores from the diffraction fringes the telescope
spreads around them (DESAT; Schwartz, Torre and Piana, ApJL 2014): a known share of the core's light lands, in a
known pattern, on pixels that did not saturate. A camera lens does the same with its glow (veiling glare, the glare
spread function: Talvala, Adams, Horowitz and Levoy, SIGGRAPH 2007; McCann and Rizzi 2007), and astronomers measure
saturated stars the same way, by fitting the unsaturated wings of the star's spread pattern.

So the glow around a blown light tells how bright the light really was, channel by channel:

* tier "lens" (cyan): the lens's glow profile was measured once, from a bracket of a small light in the dark
  (measure_profile). The light's brightness in each channel is the largest the glow around it allows: the scene can
  add light around the lamp but never take it away, so the predicted glow may not exceed what was recorded anywhere
  around it (a lower-envelope fit; a low quantile, weighted to where the glow is strongest, keeps it robust).
* tier "photo" (amber): no profile, but the light blew out in some channels and not in another (often blue). The
  unclipped channel gives the light's brightness directly and the colour of the glow gives the other channels (glare
  in glass is close to colour-neutral, so the glow has the light's own colour).
* otherwise the light keeps the colour rebuild (raw/highlights.py) at the clip level.

The recovered excess is laid inside the clipped area as a dome that meets the clip level at the edge. The glow is
smooth, so it says how much light there was and its colour, not the fine detail inside the light.

Works on the half-size white-balanced planes of raw/highlights.py and returns maps in the same form, so the same
apply() lays them into any region.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
from scipy.signal import fftconvolve

from ... import log as slog
from .highlights import CLIP  # noqa: F401  (the clip level, for callers)

log = slog.get("engine.glow")

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")     # NumPy 2 renamed it

GRID = 640                       # long side of the coarse grid the glow is predicted on
MAX_LIGHTS = 40
MAX_LIGHT_FRAC = 0.003           # a blown area bigger than 0.3% of the frame is a surface (sky, a backlit leaf), not a
                                 # light: where its extra light sat is unknowable, so it keeps the colour rebuild
QUANTILE = 0.10                  # lower envelope


@dataclass
class Profile:
    """A lens's glow: the light that lands at distance r (half-size pixels) per unit of a light's brightness."""
    name: str = ""
    lens: str = ""
    camera: str = ""
    fnumber: float | None = None
    plane_width: int = 0                       # half-size width of the camera it was measured on
    r: list = field(default_factory=list)      # radii, half-size pixels (ascending)
    k: list = field(default_factory=list)      # 3 lists (R, G, B), same length as r
    wing: list = field(default_factory=list)   # share of each channel's light in the glow beyond r[0]
    frames: int = 0
    stops: float = 0.0                         # dynamic range the measurement covered
    made: str = ""

    def kernel(self, rr: np.ndarray, c: int) -> np.ndarray:
        """k at radii rr (log-log interpolation; power law past the measured range)."""
        r = np.asarray(self.r, float)
        k = np.maximum(np.asarray(self.k[c], float), 1e-30)
        lr, lk = np.log(r), np.log(k)
        x = np.log(np.clip(rr, r[0] / 8.0, None))                # inside the measured range: up to 8x closer
        out = np.interp(x, lr, lk)
        n = max(3, len(r) // 3)
        slope = min(np.polyfit(lr[-n:], lk[-n:], 1)[0], -1.0)    # outer slope (steeper than -1)
        far = x > lr[-1]
        out[far] = lk[-1] + slope * (x[far] - lr[-1])
        s_in = float(np.clip(np.polyfit(lr[:n], lk[:n], 1)[0], -3.0, -0.5))   # inner slope, bounded
        near = x < lr[0]
        out[near] = lk[0] + s_in * (x[near] - lr[0])
        return np.exp(out)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def profile_dir() -> str:
    from ... import paths
    d = paths.get("SETTINGS") / "glow"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", s).strip("_") or "lens"


def save_profile(p: Profile) -> str:
    p.name = p.name or _slug(f"{p.lens or 'lens'}" + (f"_f{p.fnumber:g}" if p.fnumber else ""))
    path = os.path.join(profile_dir(), p.name + ".json")
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(p.to_dict(), fh, indent=1)
    os.replace(tmp, path)
    return path


def list_profiles() -> list:
    d = profile_dir()
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(".json"):
            p = load_profile(f[:-5])
            if p is not None:
                out.append(p)
    return out


def load_profile(name: str) -> Profile | None:
    if not name:
        return None
    path = os.path.join(profile_dir(), name + ".json")
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        p = Profile(**{k: d[k] for k in Profile.__dataclass_fields__ if k in d})
        if len(p.r) < 4 or len(p.k) != 3:
            raise ValueError("incomplete profile")
        return p
    except (OSError, ValueError, TypeError) as err:
        log.warning("glow profile %r unusable (%s)", name, err)
        return None


def pick_profile(lens: str, fnumber: float | None) -> str:
    """The best stored profile for a lens: same lens, nearest aperture (any aperture for lenses that record none)."""
    best, bd = "", None
    for p in list_profiles():
        if not lens or p.lens.lower() != lens.lower():
            continue
        d = abs(math.log2(p.fnumber / fnumber)) if (p.fnumber and fnumber) else 0.5
        if bd is None or d < bd:
            best, bd = p.name, d
    return best


# ------------------------------------------------------------------------------------------ helpers
def _block(a: np.ndarray, d: int, how="mean") -> np.ndarray:
    if d == 1:
        return a.astype(np.float32) if how != "any" else a
    h, w = a.shape[:2]
    H, W = -(-h // d), -(-w // d)
    pad = [(0, H * d - h), (0, W * d - w)] + [(0, 0)] * (a.ndim - 2)
    if how == "any":
        b = np.pad(a, pad, constant_values=False).reshape(H, d, W, d, *a.shape[2:])
        return b.any(axis=(1, 3))
    b = np.pad(a.astype(np.float32), pad).reshape(H, d, W, d, *a.shape[2:])
    s = b.sum(axis=(1, 3))
    if how == "sum":
        return s
    n = np.pad(np.ones((h, w), np.float32), pad[:2]).reshape(H, d, W, d).sum(axis=(1, 3))
    return s / np.maximum(n, 1.0).reshape(H, W, *([1] * (a.ndim - 2)))


def _wq(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][min(int(np.searchsorted(cw, q * cw[-1])), v.size - 1)])


def envelope_fit(g: np.ndarray, v: np.ndarray, q: float = QUANTILE):
    """(F, b): the lower envelope of the recorded values v against the predicted glow g, v >= b + F g (the scene
    behind the glow only ever adds light). F is a low quantile of (v - b) / g weighted to where the glow is strongest;
    b, the darkest scene level, is read where the glow is weakest. Two rounds."""
    far = g <= np.quantile(g, 0.25)
    b = 0.0
    F = 0.0
    for _ in range(3):
        F = max(_wq((v - b) / g, g * g, q), 0.0)
        b = max(float(np.quantile(v[far] - F * g[far], q)), 0.0)
    return F, b


def noise_sigma(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-channel noise of the half-size planes (median absolute difference of neighbours, unclipped only)."""
    out = np.zeros(3, np.float32)
    for c in range(3):
        d = np.diff(P[..., c], axis=1)
        ok = ~(K[:, 1:, c] | K[:, :-1, c])
        v = np.abs(d[ok])
        out[c] = 1.4826 * float(np.median(v)) / math.sqrt(2.0) if v.size else 1e-3
    return np.maximum(out, 1e-6)


def _dome(mask: np.ndarray) -> np.ndarray:
    """Highest in the middle of the area, zero at its edge (a parabola for a disc), summing to 1."""
    d = distance_transform_edt(mask)
    dm = d.max()
    if dm <= 0:
        return mask.astype(np.float32) / max(mask.sum(), 1)
    u = d * (2.0 * dm - d)
    return (u / u.sum()).astype(np.float32)


def kernel_image(prof: Profile, c: int, d: int, shape, scale: float = 1.0) -> np.ndarray:
    """The glow kernel on the coarse grid (each cell d half-size pixels): response per half-size pixel to a unit
    light in the centre cell. ``scale`` converts this camera's half-size pixels to the profile camera's."""
    H, W = shape
    yy = np.arange(-H + 1, H)[:, None]
    xx = np.arange(-W + 1, W)[None, :]
    rr = np.hypot(yy, xx) * d * scale
    return prof.kernel(np.maximum(rr, 0.75 * d * scale), c).astype(np.float32)


# ------------------------------------------------------------------------------------------ the estimate
def estimate(P: np.ndarray, K: np.ndarray, mult, prof: Profile | None, base: np.ndarray | None = None):
    """(est, report): the planes with measured lights laid into their clipped channels, and one entry per light
    {y, x, area, tier, stops, colour}. ``base`` is where to start (the colour rebuild), P otherwise."""
    h2, w2 = P.shape[:2]
    est = (base if base is not None else P).astype(np.float32).copy()
    A = K.any(axis=2)
    report = []
    if not A.any():
        return est, report
    sig = noise_sigma(P, K)
    lab, n = label(binary_dilation(A, iterations=2), structure=np.ones((3, 3)))
    lab = np.where(A, lab, 0)
    idx = np.arange(1, n + 1)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    order = [int(i) for i in idx[np.argsort(-areas)] if areas[i - 1] >= 4][:MAX_LIGHTS]
    if not order:
        return est, report
    d = max(1, int(math.ceil(max(h2, w2) / float(GRID))))
    Pc = _block(P, d)                                      # coarse planes (residual: lights' glow is taken out)
    Kc = _block(K.any(axis=2), d, "any")
    labc_all = _block(lab > 0, d, "any")
    scale = (prof.plane_width / float(w2)) if (prof is not None and prof.plane_width) else 1.0
    kern = [kernel_image(prof, c, d, Pc.shape[:2], scale) for c in range(3)] if prof is not None else None
    objs = find_objects(lab)
    for j in order:
        sl = objs[j - 1]
        sub = lab[sl] == j
        ys, xs = np.nonzero(sub)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        area = int(ys.size)
        rj = math.sqrt(area / math.pi)
        rmax = max(1.0, float(np.hypot(ys - ys.mean(), xs - xs.mean()).max()))
        entry = {"y": int(ys.mean()) * 2, "x": int(xs.mean()) * 2, "area": area * 4, "tier": "rebuilt", "stops": 0.0,
                 "colour": [], "channels": [], "why": ""}
        if area > max(MAX_LIGHT_FRAC * h2 * w2, 3000) or area / (math.pi * rmax * rmax) < 0.2:
            entry["why"] = "a blown surface, not a light"
            report.append(entry)
            continue
        reach = int(3 * rj + 12)
        y0, y1 = max(0, ys.min() - reach), min(h2, ys.max() + reach + 1)
        x0, x1 = max(0, xs.min() - reach), min(w2, xs.max() + reach + 1)
        M = np.zeros((y1 - y0, x1 - x0), bool)
        M[ys - y0, xs - x0] = True
        Ms = M
        a1 = 2.0 + 0.3 * rj
        dist = distance_transform_edt(~Ms)
        L = dist <= a1                                     # the light, with its own soft edge
        Ps, Ks = P[y0:y1, x0:x1], K[y0:y1, x0:x1]
        rec = np.array([float(Ps[..., c][L].sum()) for c in range(3)])
        clipped = [c for c in range(3) if (Ks[..., c] & Ms).any()]
        F = rec.copy()
        tier, ok = "", {}
        if prof is not None:
            Mc = np.zeros(Pc.shape[:2], np.float32)
            np.add.at(Mc, (ys // d, xs // d), 1.0 / area)
            dc = distance_transform_edt(Mc <= 0) * d
            b1, b2 = a1 + d, a1 + d + max(6.0 * rj, 10.0 * d)
            ring = (dc >= b1) & (dc <= b2) & ~Kc & ~(labc_all & (dc > 1.5 * d))
            if ring.sum() >= 30:
                sc = sig / float(d)                        # noise of a coarse cell
                for c in clipped:
                    g = fftconvolve(Mc, kern[c], mode="same")
                    gr = np.maximum(g[ring], 1e-30)
                    Fc, _ = envelope_fit(gr, Pc[..., c][ring])
                    snr = Fc * math.sqrt(float((gr * gr).sum())) / float(sc[c])
                    seen = Fc * float(gr.max()) / float(sc[c])
                    log.debug("glow light %d ch %d: F %.4g rec %.4g snr %.1f seen %.1f ring %d", j, c, Fc, rec[c], snr,
                              seen, ring.sum())
                    if snr >= 8.0 and seen >= 2.0 and Fc > rec[c] * 1.02:
                        F[c], ok[c] = Fc, snr
                if ok:
                    tier = "lens"
                for c in range(3):                         # take this light's glow out before the next light
                    Pc[..., c] -= F[c] * fftconvolve(Mc, kern[c], mode="same")
        unclipped = [c for c in range(3) if c not in clipped]
        if not tier and unclipped and clipped:
            u = max(unclipped, key=lambda c: rec[c])
            a2 = a1 + max(6.0, 2.0 * rj)
            ring = (dist > a1) & (dist <= a2) & ~Ks.any(axis=2)
            outer = (dist > a2 + rj) & (dist <= a2 + 3 * rj + 10) & ~Ks.any(axis=2)
            if ring.sum() >= 20 and outer.sum() >= 20:
                bg = [float(np.median(Ps[..., c][outer])) for c in range(3)]
                su = float((Ps[..., u][ring] - bg[u]).sum())
                mid = 0.5 * (a1 + a2)                      # a glow fades with distance; a lit surface need not
                ri, ro = ring & (dist <= mid), ring & (dist > mid)
                ni, no = max(int(ri.sum()), 1), max(int(ro.sum()), 1)
                tot = Ps.sum(axis=2) - sum(bg)             # all channels: the clipped ones glow the most
                vin = float(tot[ri].mean()) if ri.any() else 0.0
                vout = float(tot[ro].mean()) if ro.any() else 0.0
                st = float(np.sqrt((sig ** 2).sum()))
                glowing = vin > 1.3 * max(vout, 0.0) and vin - vout > 3.0 * st * math.sqrt(1.0 / ni + 1.0 / no)
                # a light stands far above what is around it; a bright patch on a lit surface does not, and the
                # colour around it is the surface's, not the light's
                level = float(Ps.sum(axis=2)[L].mean())
                glowing = glowing and level > 8.0 * max(sum(bg), 3.0 * st)
                if glowing and su > 5.0 * float(sig[u]) * math.sqrt(ring.sum()) and rec[u] > 0:
                    for c in clipped:
                        sc = float((Ps[..., c][ring] - bg[c]).sum())
                        Fc = rec[u] * sc / su
                        if Fc > rec[c] * 1.02:
                            F[c], ok[c] = Fc, su / (float(sig[u]) * math.sqrt(ring.sum()))
                    if ok:
                        tier = "photo"
        if tier:
            unc = [c for c in range(3) if not (Ks[..., c] & Ms).any()]
            for c, _ in ok.items():
                Mcl = (Ks[..., c] & Ms)
                exc = F[c] - rec[c]
                if exc <= 0 or not Mcl.any():
                    continue
                if unc:                                    # a channel that did not clip shows the light's shape
                    shape = np.maximum(Ps[..., max(unc, key=lambda q: rec[q])], 0.0) * Mcl
                    w = shape / shape.sum() if shape.sum() > 0 else _dome(Mcl)
                else:                                      # nothing to show it: a dome, highest in the middle
                    w = _dome(Mcl)
                sub = est[y0:y1, x0:x1, c]
                sub[Mcl] = np.maximum(sub[Mcl], Ps[..., c][Mcl] + exc * w[Mcl])
        tot = float(sum(rec[c] for c in clipped)) or 1.0
        entry.update(tier=tier or "rebuilt", channels=sorted(ok),
                     stops=round(math.log2(max(float(sum(F[c] for c in clipped)) / tot, 1.0)), 2),
                     colour=[round(float(v), 4) for v in (F / max(F.max(), 1e-9))],
                     why="" if tier else "no glow strong enough to measure")
        report.append(entry)
    nm = sum(1 for r in report if r["tier"] != "rebuilt")
    log.info("glow: %d blown light(s), %d measured (%s)", len(report), nm,
             ", ".join(f"{r['tier']} +{r['stops']}" for r in report[:8]))
    return est, report


def maps_for(raw, levels, mult, prof: Profile | None):
    """Highlight maps (as raw/highlights.maps_for) with measured lights, plus the report."""
    from ..raw import linearize as lin
    from .highlights import _planes
    from .highlights import estimate as rebuild
    with slog.stage("glow.estimate", log, profile=prof.name if prof else ""):
        mos = lin.normalise(raw, levels)
        P, K = _planes(mos, raw.cfa_rgb)
        del mos
        P *= np.asarray(mult, np.float32).reshape(1, 1, 3)
        rb = rebuild(P, K)
        base = P.copy()
        if rb is not None:
            ry, rx, re_ = rb
            base[ry:ry + re_.shape[0], rx:rx + re_.shape[1]] = re_
        est, report = estimate(P, K, mult, prof, base)
    if not K.any():
        return None, report
    return (0, 0, est), report


# ------------------------------------------------------------------------------------------ calibration
def _tail_background(r: np.ndarray, m: np.ndarray) -> float:
    """The scene's own level under a measured glow: fit m = b + A r^-a to the outer half of the radii."""
    from scipy.optimize import least_squares
    n = len(r)
    if n < 6:
        return 0.0
    rr, mm = r[n // 2:], m[n // 2:]
    lo = float(np.min(mm))

    def res(p):
        b, la, a = p
        return (b + np.exp(la) * rr ** (-a) - mm) / np.maximum(np.abs(mm), 1e-12)
    p0 = [0.5 * lo, math.log(max(float(mm[0]) * rr[0] ** 2.0, 1e-30)), 2.0]
    try:
        f = least_squares(res, p0, bounds=([-abs(lo) - 1e-9, -200.0, 0.5], [lo, 200.0, 6.0]))
        return float(f.x[0])
    except (ValueError, RuntimeError):
        return 0.0


def measure_profile(paths: list, lens: str = "", progress=None) -> dict:
    """A lens's glow profile from a bracket of one small light in the dark: the same framing at several shutter
    speeds, from one where the light is not blown at all to several stops brighter. Returns a report; the profile
    is saved in settings/glow (yours, on this PC)."""
    from ..io.rawfile import open_raw
    from ..raw import linearize as lin
    from .highlights import _planes
    frames, skipped = [], []
    cam = None
    for k, p in enumerate(paths):
        if progress:
            progress(k, len(paths), os.path.basename(p))
        try:
            raw = open_raw(p)
        except Exception as err:                                  # noqa: BLE001  (unreadable files are skipped)
            skipped.append((p, f"unreadable ({err})"))
            continue
        t = raw.meta.exposure_s
        if not t:
            skipped.append((p, "no shutter speed recorded"))
            continue
        c = (raw.meta.make, raw.meta.model, raw.width, raw.height)
        if cam is None:
            cam = c
        elif c != cam:
            skipped.append((p, "another camera or size"))
            continue
        lv = lin.levels(raw)
        P, K = _planes(lin.normalise(raw, lv), raw.cfa_rgb)
        frames.append((float(t), P, K, raw))
    if len(frames) < 2:
        return {"ok": False, "reason": "need at least two frames of the bracket", "skipped": skipped}
    frames.sort(key=lambda f: f[0])
    t0, P0, K0, raw0 = frames[0]
    pk = np.unravel_index(np.argmax(P0.sum(axis=2)), P0.shape[:2])
    if K0[pk].any():
        return {"ok": False, "reason": "the light is blown even in the shortest frame; add a faster one",
                "skipped": skipped}
    h2, w2 = P0.shape[:2]
    # HDR merge in units of the shortest frame: each pixel from the longest frame where it is not near clipping
    hdr = np.zeros_like(P0)
    wsum = np.zeros_like(P0)
    for t, P, K, _ in frames:
        s = t0 / t
        good = (P < 0.9) & ~K
        w = good * t
        hdr += np.where(good, P * s, 0.0) * w
        wsum += w
    hdr = np.where(wsum > 0, hdr / np.maximum(wsum, 1e-12), P0).astype(np.float32)
    yy, xx = np.mgrid[0:h2, 0:w2]
    tot = hdr.sum(axis=2)
    core = tot > 0.02 * tot[pk]
    lab, _ = label(core)
    core = lab == lab[pk]
    wy = tot * core
    cy, cx = float((yy * wy).sum() / wy.sum()), float((xx * wy).sum() / wy.sum())
    rr = np.hypot(yy - cy, xx - cx)
    rc = max(2.0, float(rr[core].max()))
    rmax = float(min(max(cy, h2 - cy, cx, w2 - cx), math.hypot(h2, w2) * 0.6))
    edges = np.geomspace(2.0 * rc, rmax, 41)
    rm, med, cnt = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        ring = (rr >= a) & (rr < b)
        if ring.sum() < 8:
            continue
        rm.append(float(math.sqrt(a * b)))
        med.append([float(np.median(hdr[..., c][ring])) for c in range(3)])
        cnt.append(int(ring.sum()))
    rm, med, cnt = np.asarray(rm), np.asarray(med), np.asarray(cnt)
    noise = noise_sigma(P0, K0) * (t0 / frames[-1][0])           # noise of the longest frame, in merged units
    flux = np.array([float(hdr[..., c][rr <= 1.5 * rc].sum()) for c in range(3)])
    rs, ks, bgs = [], [[], [], []], []
    keep = np.ones(len(rm), bool)
    for c in range(3):
        bg = _tail_background(rm, med[:, c])                     # a dark room is rarely perfectly dark
        bgs.append(bg)
        sig = med[:, c] - bg
        keep &= sig > 3.0 * float(noise[c]) / np.sqrt(np.maximum(cnt, 1)) + 1e-12
    for i in np.nonzero(keep)[0]:
        rs.append(float(rm[i]))
        for c in range(3):
            ks[c].append((med[i, c] - bgs[c]) / flux[c])
    if len(rs) < 6:
        return {"ok": False, "reason": "the glow could not be measured (too faint, or the frames do not line up)",
                "skipped": skipped}
    stops = float(np.log2(hdr[pk].max() / max(float(noise.max()), 1e-9)))
    wing = []
    for c in range(3):
        r_ = np.asarray(rs)
        k_ = np.asarray(ks[c])
        wing.append(float(_trapz(k_ * 2 * np.pi * r_, r_)))
    md = raw0.meta
    prof = Profile(lens=lens or md.lens or "", camera=f"{md.make} {md.model}".strip(), fnumber=md.fnumber,
                   plane_width=w2, r=[round(v, 3) for v in rs], k=[[float(f"{v:.6g}") for v in kc] for kc in ks],
                   wing=[round(v, 5) for v in wing], frames=len(frames), stops=round(stops, 1),
                   made=_dt.date.today().isoformat())
    path = save_profile(prof)
    log.info("glow profile %s: %d radii, glow share %s, %d frames, %.1f stops", prof.name, len(rs),
             [round(w, 4) for w in wing], len(frames), stops)
    return {"ok": True, "name": prof.name, "path": path, "frames": len(frames), "stops": prof.stops,
            "wing": prof.wing, "skipped": skipped}

"""Clean for scans and JPEGs: noise measured in the picture itself, removed there.

On a raw, Clean knows each photosite's noise from the sensor's own model (Poisson-Gaussian, calib/noise.py) and
works before demosaicing, sharpening and compression have touched it: the best place there is. A scan or a JPEG
has none of that. The camera or scanner has already demosaiced, sharpened, tone-mapped and often compressed it,
so its noise is no longer independent per pixel, its level depends on brightness through a curve nobody wrote
down, and some of it is gone or smeared into the picture. So Clean on a scan does the honest second-best:

  1. Measure. The picture is taken into an orthonormal brightness / colour basis (Y, and two colour-difference
     planes) of its stored values, and each is split into starlet bands (the same astronomy wavelets as the raw
     denoiser). In flat places (the smoothest quarter at each brightness), the spread of each band IS the noise:
     that gives the noise level per band, per plane, as a curve over brightness (the noise level function of
     Liu, Szeliski, Kang, Zitnick, Freeman, IEEE PAMI 2008, measured on bands instead of pixels). Measuring per
     band takes care of the correlation demosaicing and JPEG put into the noise: no white-noise assumption.
  2. Remove. Every band is shrunk by the same two-pass empirical Wiener rule as the raw starlet denoiser, with
     the measured noise of that band at that brightness; brightness and colour have their own strengths (colour
     blotches can go much harder than grain), keep-grain puts a fine part back, and the safeguard keeps any pixel
     that moved by more than 4.5 times its measured noise (that was a spark or a star, not noise).

It does well on grain and colour mottling. It cannot undo what the camera has already done: smeared noise
blobs from in-camera noise reduction, sharpening halos around noise, compression. That is why it is shown with a
note that Clean does its best work on raw files.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import uniform_filter

from ... import log as slog
from ..raw.denoise import P2_MAX_SCALE, P2_NV, P2_WIN_ADD, starlet, starlet_noise_sigma

log = slog.get("engine.scanclean")

LUMA_SCALES, CHROMA_SCALES = 4, 6
BINS = 8
TILE, MAX_TILES = 256, 24
SAFEGUARD_K = 4.5
_S3, _S2, _S6 = math.sqrt(3.0), math.sqrt(2.0), math.sqrt(6.0)


def to_opp(rgb: np.ndarray) -> list:
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return [(R + G + B) / _S3, (R - B) / _S2, (R + B - 2.0 * G) / _S6]


def from_opp(Y, C1, C2) -> np.ndarray:
    R = Y / _S3 + C1 / _S2 + C2 / _S6
    G = Y / _S3 - 2.0 * C2 / _S6
    B = Y / _S3 - C1 / _S2 + C2 / _S6
    return np.stack([R, G, B], -1)


def _planes(enc: np.ndarray) -> list:
    return [enc.astype(np.float32)] if enc.ndim == 2 else [p.astype(np.float32) for p in to_opp(enc)]


def _level(enc: np.ndarray) -> np.ndarray:
    """Brightness 0..1 of the stored values (what the noise depends on)."""
    return (enc if enc.ndim == 2 else enc.mean(axis=2)).astype(np.float32)


def _tiles(h: int, w: int) -> list:
    if h <= TILE or w <= TILE:
        return [(0, 0, h, w)]
    ny, nx = max(1, h // TILE), max(1, w // TILE)
    ys = np.linspace(0, h - TILE, min(ny, 6)).astype(int)
    xs = np.linspace(0, w - TILE, min(nx, 6)).astype(int)
    out = [(int(y), int(x), TILE, TILE) for y in ys for x in xs]
    return out[:: max(1, len(out) // MAX_TILES)][:MAX_TILES]


PCT = 10.0                                   # the quietest tenth of the blocks at each brightness
CORR = 2.5                                   # most a coarse band's noise may exceed white noise's share


def _block(j: int) -> int:
    return 8 * 2 ** min(j, 2)


def _block_stats(v: np.ndarray, L: np.ndarray, n: int, clip: np.ndarray | None = None):
    """Std of ``v`` and mean of ``L`` in n x n blocks; blocks with clipped pixels (crushed black, blown white:
    no noise left there to see) are left out."""
    h, w = (v.shape[0] // n) * n, (v.shape[1] // n) * n
    vb = v[:h, :w].reshape(h // n, n, w // n, n)
    lb = L[:h, :w].reshape(h // n, n, w // n, n)
    sd, mn = vb.std(axis=(1, 3)).ravel(), lb.mean(axis=(1, 3)).ravel()
    if clip is not None:
        cb = clip[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3)).ravel()
        keep = cb < 0.02
        sd, mn = sd[keep], mn[keep]
    return sd, mn


_BIAS: dict = {}


def _bias(j: int, sc: int) -> float:
    """For pure white noise, the quietest-tenth block std of band j over its true std (measured once)."""
    key = (j, sc)
    if key not in _BIAS:
        rng = np.random.default_rng(11)
        n = rng.standard_normal((512, 512)).astype(np.float32)
        bands = starlet(n, sc)
        v = bands[j][16:-16, 16:-16]
        stds, _ = _block_stats(v, np.zeros_like(v), _block(j))
        _BIAS[key] = float(np.percentile(stds, PCT) / v.std())
    return _BIAS[key]


def measure(enc: np.ndarray) -> dict:
    """The picture's noise: {"edges": brightness bin edges, "sigma": [plane][band] -> per-bin noise std,
    "pixel": [plane] -> per-bin noise std per pixel (for the safeguard), "levels": the 8-bit noise of
    brightness in the mid-tones (for the report)}. ``enc``: stored values 0..1, (h, w) or (h, w, 3).
    Per band and brightness bin, the std inside each block is measured and the quietest tenth of the blocks is
    taken (a lower envelope: texture can only add to a block's spread), corrected by the same statistic for pure
    noise (Shin, Park, Lee 2005; Liu et al. 2008)."""
    t0 = time.perf_counter()
    h, w = enc.shape[:2]
    nplanes = 1 if enc.ndim == 2 else 3
    scales = [LUMA_SCALES] + [CHROMA_SCALES] * (nplanes - 1)
    Ls = _level(enc[:: max(1, h // 400), :: max(1, w // 400)]).ravel()      # bins follow the picture's own
    edges = np.unique(np.concatenate([[0.0], np.percentile(Ls, np.linspace(0, 100, BINS + 1))[1:-1], [1.0]]))
    if len(edges) < 3:                                                     # brightness spread (a dark photo
        edges = np.linspace(0.0, 1.0, BINS + 1)                            # gets its bins in the dark)
    nb = len(edges) - 1
    samples = [[[[] for _ in range(nb)] for _ in range(scales[p])] for p in range(nplanes)]
    m = 16
    for (y, x, th, tw) in _tiles(h, w):
        tile = enc[y:y + th, x:x + tw]
        L = _level(tile)[m:-m, m:-m]
        t3 = tile if tile.ndim == 3 else tile[..., None]
        clip = ((t3.min(axis=2) <= 1.5 / 255.0) | (t3.max(axis=2) >= 253.5 / 255.0))[m:-m, m:-m]
        for p, pl in enumerate(_planes(tile)):
            bands = starlet(pl, scales[p])
            for b in range(scales[p]):
                v = bands[b][m:-m, m:-m]
                stds, means = _block_stats(v, L, _block(b), clip)
                k = np.clip(np.searchsorted(edges, means, side="right") - 1, 0, nb - 1)
                for kk in range(nb):
                    samples[p][b][kk].append(stds[k == kk])
    sigma, pixel = [], []
    for p in range(nplanes):
        per_band = []
        for b in range(scales[p]):
            s = np.full(nb, np.nan)
            for kk in range(nb):
                vv = np.concatenate(samples[p][b][kk]) if samples[p][b][kk] else np.zeros(0)
                if vv.size >= 20:
                    s[kk] = float(np.percentile(vv, PCT)) / _bias(b, scales[p])
            ok = np.isfinite(s)
            if ok.any():
                s = np.interp(np.arange(nb), np.nonzero(ok)[0], s[ok])
            elif b == 0:
                s = np.zeros(nb)                                   # nothing measurable: no noise assumed
            per_band.append(s)                                     # (a coarse band left unmeasured gets the cap)
        # Coarse bands of a photo are mostly picture even in its quietest blocks, so their measurement is capped
        # by what the finest band implies: white noise scaled by the starlet's own band ratios, times CORR for the
        # correlation demosaicing and JPEG add (measured on JPEGs of known noise: 1.5-2.5x at band 1).
        # Colour planes of a JPEG are stored at half size and low-passed, so their noise is spread evenly over
        # the bands (measured: bands 0-2 alike) instead of falling off: there the cap is the finest band's level.
        # Told apart by the finest band: half-size colour has far less fine noise than brightness.
        wn = starlet_noise_sigma()
        y0 = float(np.median(sigma[0][0])) if sigma else 0.0
        subsampled = p > 0 and y0 > 0 and float(np.median(per_band[0])) < 0.35 * y0
        for b in range(1, len(per_band)):
            ratio = max(1.0, (wn[b] / wn[0]) * CORR) if subsampled else (wn[b] / wn[0]) * CORR
            cap = per_band[0] * ratio
            per_band[b] = np.where(np.isfinite(per_band[b]), np.minimum(per_band[b], cap), cap)
        sigma.append(per_band)
        pixel.append(per_band[0] / wn[0])                         # the finest band as white-noise equivalent
    mid = slice(nb // 4, max(nb // 4 + 1, 3 * nb // 4))
    levels = float(np.median(pixel[0][mid]) / (_S3 if nplanes == 3 else 1.0) * 255.0)
    res = {"edges": edges, "sigma": sigma, "pixel": pixel, "levels": levels, "ms": (time.perf_counter() - t0) * 1e3,
           "scales": scales}
    log.info("scan noise measured: about %.2f 8-bit levels in the mid-tones (%d planes, %.0f ms)", levels, nplanes,
             res["ms"])
    return res


def _wiener(wb: np.ndarray, nv: np.ndarray, j: int) -> np.ndarray:
    """The raw starlet denoiser's two-pass empirical Wiener shrinkage, with a noise variance per pixel."""
    win = 3 + 2 * min(j, 3)
    e = uniform_filter(wb * wb, size=win, mode="mirror")
    pilot = wb * np.clip(1.0 - nv / np.maximum(e, 1e-12), 0.0, 1.0)
    if j >= P2_MAX_SCALE:
        return pilot
    ep = uniform_filter(pilot * pilot, size=win + P2_WIN_ADD, mode="mirror")
    return wb * (ep / (ep + P2_NV * nv + 1e-20))


PAD = 160                    # context round a strip: the 6-scale starlet reaches 126 px, the Wiener windows ~10


def clean(enc: np.ndarray, noise: dict, luma: float = 1.0, chroma: float = 1.5, grain: float = 0.0,
          safeguard: bool = True, stats: dict | None = None) -> np.ndarray:
    """The picture with its measured noise removed (stored values 0..1, same shape). Done in horizontal strips
    with 160 px of context each, side by side on every core (the result is the same as in one piece: no filter
    reaches further than the context)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    h = enc.shape[0]
    n = max(1, min(os.cpu_count() or 1, h // 384))
    cuts = np.linspace(0, h, n + 1).astype(int)
    jobs = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

    def run(job):
        a, b = job
        y0, y1 = max(0, a - PAD), min(h, b + PAD)
        st: dict = {}
        part = _clean_block(enc[y0:y1], noise, luma, chroma, grain, safeguard, st, (a - y0, b - y0))
        return part[a - y0:b - y0], st.get("protected", 0)

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        done = list(ex.map(run, jobs))
    if stats is not None:
        stats["protected"], stats["sites"] = sum(d[1] for d in done), int(enc.shape[0] * enc.shape[1])
    return np.concatenate([d[0] for d in done], axis=0)


def _clean_block(enc: np.ndarray, noise: dict, luma: float, chroma: float, grain: float, safeguard: bool,
                 stats: dict, keep=None) -> np.ndarray:
    L = _level(enc)
    edges = noise["edges"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    pos = np.interp(L, centres, np.arange(len(centres))).astype(np.float32)   # once: where each pixel sits
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, len(centres) - 1)          # between the bins
    i1 = np.minimum(i0 + 1, len(centres) - 1)
    fr = pos - i0

    def curve_map(c):
        c = np.asarray(c, np.float32)
        return c[i0] + (c[i1] - c[i0]) * fr

    out, kept = [], 0
    for p, pl in enumerate(_planes(enc)):
        strength = luma if p == 0 else chroma
        sc = noise["scales"][p]
        if strength <= 0:
            out.append(pl)
            continue
        bands = starlet(pl, sc)
        res = bands[-1].copy()
        for j in range(sc):
            sj = curve_map(noise["sigma"][p][j])
            res += _wiener(bands[j], strength * sj * sj, j)
            bands[j] = None
        if p == 0 and grain > 0:
            res = res + float(grain) * (pl - res)
        if safeguard:
            bad = np.abs(res - pl) > np.maximum(SAFEGUARD_K * curve_map(noise["pixel"][p]), 1e-6)
            if keep is not None:
                kept += int(bad[keep[0]:keep[1]].sum())
            res = np.where(bad, pl, res)
        out.append(res.astype(np.float32))
    stats["protected"] = kept
    if enc.ndim == 2:
        return np.clip(out[0], 0.0, 1.0)
    return np.clip(from_opp(*out), 0.0, 1.0).astype(np.float32)

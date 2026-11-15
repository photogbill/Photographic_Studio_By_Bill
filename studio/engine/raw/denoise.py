"""Stage 7: raw denoise (section 8). One raw in, one clean result out: no stacking, ever.

Design (8.1):
 1. Know the noise: the Poisson-Gaussian model per CFA position (calib/noise.py).
 2. Work on the mosaic, before demosaic, on the four colour planes.
 3. Variance-stabilise each plane (generalised Anscombe) so the noise is unit-variance everywhere.
 4. Rotate the four planes into an orthonormal luminance / chroma basis (unit noise stays unit noise):
        Y  = (R + G1 + G2 + B) / 2        luminance
        C1 = (R - B) / sqrt 2             red-blue chroma
        C2 = (G1 + G2 - R - B) / 2        green-magenta chroma
        C3 = (G1 - G2) / sqrt 2           green split: mostly noise, some diagonal detail
    so chroma can be cleaned much harder than luminance (colour blotches are what ruin high-ISO files).
 5. Denoise each component with a candidate method, rotate back, exact unbiased inverse VST.

Candidates (8.1, the ladder; each is measured on the benchmark harness, tools/bench.py):
  * ``starlet``: candidate c. Isotropic undecimated B3-spline wavelet (starlet, a trous; Starck & Murtagh,
    astronomical image processing) with a local Wiener shrinkage in every band: each coefficient is kept in
    proportion to how much local energy exceeds the known noise energy of that band. Coarse bands catch the
    low-frequency chroma blotches, fine bands protect luminance texture.
  * ``mppca``: candidate b. Local patch PCA with the Marchenko-Pastur threshold (Veraart et al. 2016, MRI). The
    noise level is known exactly after the VST, so the signal / noise cut is principled, not tuned.
  * ``nlm``: non-local means with the correct unit-noise statistics (the MRI "Rician NLM" row of section 14),
    the non-local baseline until a BM3D-class method is written (the ``bm3d`` package is not GPL-compatible).

Controls: luminance strength, chroma strength, keep-grain (adds back a fine monochrome fraction of what was
removed from luminance, film-like, never blotchy). Near the clip point the noise model no longer holds, so the
original values are blended back there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve1d, uniform_filter

from ... import log as slog
from ..calib.noise import NoiseModel, anscombe, inverse_anscombe_exact

log = slog.get("engine.denoise")

S2 = math.sqrt(2.0)
B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
P2_WIN_ADD = 2        # pass-2 energy window = pass-1 window + this
P2_NV = 1.0           # pass-2 noise variance factor (tuned on the harness, docs/RESULTS.md)
P2_MAX_SCALE = 1      # pass 2 only on the finest band(s), where isolated speckles live


@dataclass
class DenoiseParams:
    method: str = "starlet"
    luma: float = 1.0           # 0 = off, 1 = noise-level default, >1 stronger
    chroma: float = 1.0
    grain: float = 0.0          # 0..1 fraction of removed luminance noise put back (fine, monochrome)
    luma_scales: int = 4
    chroma_scales: int = 6
    luma_method: str | None = None     # override per component (e.g. "hybrid" = nlm luma + starlet chroma)
    chroma_method: str | None = None
    impulses: bool = False      # isolated-outlier filter; off: on real texture it cost 2 dB (RESULTS.md).
                                # Hot pixels will come from the calibration map (section 8.3) instead.


# ------------------------------------------------------------------------------------------ Y/C rotation
def plane_roles(pattern) -> dict:
    """Map 'R', 'G1', 'G2', 'B' to CFA positions (i, j) for a 2x2 pattern of 0/1/2."""
    p = np.asarray(pattern)
    roles, g = {}, []
    for i in (0, 1):
        for j in (0, 1):
            c = int(p[i, j])
            if c == 0:
                roles["R"] = (i, j)
            elif c == 2:
                roles["B"] = (i, j)
            else:
                g.append((i, j))
    roles["G1"], roles["G2"] = g[0], g[1]
    return roles


def to_yc(R, G1, G2, B):
    return (0.5 * (R + G1 + G2 + B), (R - B) / S2, 0.5 * (G1 + G2 - R - B), (G1 - G2) / S2)


def from_yc(Y, C1, C2, C3):
    R = 0.5 * Y + C1 / S2 - 0.5 * C2
    B = 0.5 * Y - C1 / S2 - 0.5 * C2
    G1 = 0.5 * Y + 0.5 * C2 + C3 / S2
    G2 = 0.5 * Y + 0.5 * C2 - C3 / S2
    return R, G1, G2, B


# ------------------------------------------------------------------------------------------ starlet
def _dilated(j: int) -> np.ndarray:
    step = 2 ** j
    k = np.zeros(4 * step + 1, np.float32)
    k[::step] = B3
    return k


def starlet(x: np.ndarray, scales: int) -> list:
    """Isotropic undecimated wavelet transform. Returns [w_1, ..., w_J, c_J]; sum of all = x exactly."""
    c = x.astype(np.float32, copy=False)
    out = []
    for j in range(scales):
        k = _dilated(j)
        s = convolve1d(convolve1d(c, k, axis=0, mode="mirror"), k, axis=1, mode="mirror")
        out.append(c - s)
        c = s
    out.append(c)
    return out


@lru_cache(maxsize=None)
def starlet_noise_sigma(scales: int = 8) -> tuple:
    """Standard deviation of each starlet band for unit white noise (measured once, then cached)."""
    rng = np.random.default_rng(7)
    n = rng.standard_normal((1024, 1024)).astype(np.float32)
    bands = starlet(n, scales)
    m = 64                                     # ignore the mirrored border
    return tuple(float(b[m:-m, m:-m].std()) for b in bands[:-1])


def _wiener_band(w: np.ndarray, sigma: float, strength: float, j: int) -> np.ndarray:
    """Two-pass empirical Wiener shrinkage of one band (unit-noise input).
    Pass 1: a pilot from local-energy shrinkage. Pass 2: gains from the pilot's smoothed energy,
    g = E_pilot / (E_pilot + s * sigma^2), as in the second stage of BM3D. A single strong noise sample can push
    local energy over the threshold in pass 1, but it cannot fake a smooth pilot energy, so it is not left behind
    as an isolated dot."""
    if strength <= 0:
        return w
    win = 3 + 2 * min(j, 3)                    # local energy window grows with scale
    nv = strength * sigma * sigma
    e = uniform_filter(w * w, size=win, mode="mirror")
    pilot = w * np.clip(1.0 - nv / np.maximum(e, 1e-12), 0.0, 1.0)
    if j >= P2_MAX_SCALE:
        return pilot
    ep = uniform_filter(pilot * pilot, size=win + P2_WIN_ADD, mode="mirror")
    return w * (ep / (ep + P2_NV * nv))


def denoise_starlet(x: np.ndarray, strength: float, scales: int) -> np.ndarray:
    """Denoise one unit-noise plane."""
    if strength <= 0:
        return x
    sig = starlet_noise_sigma()
    bands = starlet(x, scales)
    out = bands[-1].copy()
    for j in range(scales):
        out += _wiener_band(bands[j], sig[j], strength, j)
    return out


# ------------------------------------------------------------------------------------------ MP-PCA (candidate b)
def _shifts(x: np.ndarray, p: int) -> list:
    """The p*p shifted copies of a padded plane that make up every p x p patch (patch element k)."""
    r = p // 2
    xp = np.pad(x, r, mode="reflect")
    h, w = x.shape
    return [xp[dy:dy + h, dx:dx + w] for dy in range(p) for dx in range(p)]


def _mppca_tile(x: np.ndarray, strength: float, p: int = 5, win: int = 9, stride: int = 3) -> np.ndarray:
    """Local patch PCA with the Marchenko-Pastur threshold (Veraart et al. 2016, MRI), on a unit-noise plane.
    For each grid point, all p x p patches inside a win x win window form an N x M matrix (N patches,
    M = p^2 pixels). With unit noise, eigenvalues of pure noise stay below (1 + sqrt(M/N))^2 (the MP edge);
    components above it are signal and are kept, the rest are dropped. Each grid point's own patch is
    reconstructed and the overlapping patches are averaged (weights 1/(1+rank), as in the literature)."""
    from scipy.ndimage import uniform_filter
    h, w = x.shape
    M, N = p * p, win * win
    sh = _shifts(x.astype(np.float32), p)
    gy, gx = np.arange(0, h, stride), np.arange(0, w, stride)
    mean = np.stack([uniform_filter(s, win, mode="reflect")[np.ix_(gy, gx)] for s in sh], -1)      # (gh, gw, M)
    cov = np.empty((len(gy), len(gx), M, M), np.float32)
    for i in range(M):
        for j in range(i, M):
            c = uniform_filter(sh[i] * sh[j], win, mode="reflect")[np.ix_(gy, gx)] - mean[..., i] * mean[..., j]
            cov[..., i, j] = c
            cov[..., j, i] = c
    edge = (1.0 + np.sqrt(M / N)) ** 2 * max(strength, 1e-3)
    evals, evecs = np.linalg.eigh(cov)                         # ascending
    keep = (evals > edge).astype(np.float32)                   # (gh, gw, M)
    rank = keep.sum(-1)
    patch = np.stack([s[np.ix_(gy, gx)] for s in sh], -1) - mean
    coef = np.einsum("...mk,...m->...k", evecs, patch) * keep
    den = mean + np.einsum("...mk,...k->...m", evecs, coef)   # denoised central patch, (gh, gw, M)
    wgt = (1.0 / (1.0 + rank))[..., None]
    acc = np.zeros((h + p, w + p), np.float32)
    wsum = np.zeros_like(acc)
    r = p // 2
    for k in range(M):
        dy, dx = divmod(k, p)
        ys, xs = gy + dy, gx + dx
        acc[np.ix_(ys, xs)] += den[..., k] * wgt[..., 0]
        wsum[np.ix_(ys, xs)] += wgt[..., 0]
    acc, wsum = acc[r:r + h, r:r + w], wsum[r:r + h, r:r + w]
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), x).astype(np.float32)


def denoise_mppca(x: np.ndarray, strength: float, scales: int = 0, tile: int = 192, pad: int = 12) -> np.ndarray:
    """MP-PCA over a whole plane in overlapping tiles (memory stays small on 8 GB machines)."""
    if strength <= 0:
        return x
    h, w = x.shape
    out = np.empty_like(x, dtype=np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            a, b = max(0, y0 - pad), max(0, x0 - pad)
            c, d = min(h, y0 + tile + pad), min(w, x0 + tile + pad)
            res = _mppca_tile(x[a:c, b:d], strength)
            out[y0:min(h, y0 + tile), x0:min(w, x0 + tile)] = res[y0 - a:y0 - a + min(tile, h - y0),
                                                                  x0 - b:x0 - b + min(tile, w - x0)]
    return out


# ------------------------------------------------------------------------------------------ NLM (MRI-style baseline)
def denoise_nlm(x: np.ndarray, strength: float, scales: int = 0, search: int = 7, patch: int = 5) -> np.ndarray:
    """Non-local means with the correct statistics for unit Gaussian noise (the VST makes it so):
    for identical patches the mean squared difference is 2, so weights are exp(-max(d - 2, 0) / h^2).
    Vectorised over the search offsets."""
    from scipy.ndimage import uniform_filter
    if strength <= 0:
        return x
    x = x.astype(np.float32)
    r = search // 2
    xp = np.pad(x, r, mode="reflect")
    h, w = x.shape
    h2 = 0.55 * strength                       # ~1 std of d for identical 5x5 patches; tuned on the harness
    num = x.copy()
    den = np.ones_like(x)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dy, dx) == (0, 0):
                continue
            s = xp[r + dy:r + dy + h, r + dx:r + dx + w]
            d = uniform_filter((x - s) ** 2, patch, mode="reflect")
            wgt = np.exp(-np.maximum(d - 2.0, 0.0) / h2)
            num += wgt * s
            den += wgt
    return num / den


METHODS = {"starlet": denoise_starlet, "mppca": denoise_mppca, "nlm": denoise_nlm}


# ------------------------------------------------------------------------------------------ impulses
def remove_impulses(d: np.ndarray, k: float = 5.0) -> tuple[np.ndarray, int]:
    """Replace isolated outliers in a unit-noise plane (hot / cold / dead pixels, read-noise spikes).

    A pixel is an impulse when it differs from the median of its 8 same-colour neighbours by more than
    k noise sigmas AND by more than 3x the local spread of those neighbours. Real detail (a candle flame,
    a catchlight, a star) is blurred by the lens over several pixels, so its neighbours are raised too and the
    second test protects it (section 8.3; the lens-PSF version of this test comes with the lens wizard)."""
    p = np.pad(d, 1, mode="reflect")
    h, w = d.shape
    nb = np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy, dx) != (0, 0)], 0)
    med = np.median(nb, axis=0)
    spread = np.median(np.abs(nb - med[None]), axis=0) * 1.4826
    r = d - med
    bad = (np.abs(r) > k) & (np.abs(r) > 3.0 * np.maximum(spread, 1.0) + k - 1.0)
    out = np.where(bad, med, d).astype(np.float32)
    return out, int(bad.sum())


# ------------------------------------------------------------------------------------------ driver
def denoise_mosaic(mos: np.ndarray, pattern, model: NoiseModel, params: DenoiseParams | None = None,
                   clip_start: float = 0.85) -> np.ndarray:
    """Denoise a normalised mosaic (black-subtracted, 1.0 = clip, BEFORE white balance). Returns a new array."""
    p = params or DenoiseParams()
    lm, cm = p.luma_method, p.chroma_method
    if p.method == "hybrid":                      # detail-keeping NLM on luminance, starlet on colour blotches
        lm, cm = lm or "nlm", cm or "starlet"
    lm, cm = lm or p.method, cm or p.method
    for m in (lm, cm):
        if m not in METHODS:
            raise ValueError(f"unknown denoise method {m!r}; available: {sorted(METHODS) + ['hybrid']}")
    fn, fc = METHODS[lm], METHODS[cm]
    h, w = (mos.shape[0] // 2) * 2, (mos.shape[1] // 2) * 2
    roles = plane_roles(pattern)
    planes = {}
    n_imp = 0
    with slog.stage("denoise.vst+impulses", log):
        for name, (i, j) in roles.items():
            v = anscombe(mos[i:h:2, j:w:2].astype(np.float32), float(model.a[i, j]), float(model.b[i, j]))
            if p.impulses:
                v, n = remove_impulses(v)
                n_imp += n
            planes[name] = v
    if n_imp:
        log.debug("denoise: %d impulse pixels replaced (%.1f per MP)", n_imp, n_imp / (h * w / 1e6))
    Y, C1, C2, C3 = to_yc(planes["R"], planes["G1"], planes["G2"], planes["B"])
    with slog.stage("denoise." + p.method, log, luma=p.luma, chroma=p.chroma):
        Yd = fn(Y, p.luma, p.luma_scales)
        C1d = fc(C1, p.chroma, p.chroma_scales)
        C2d = fc(C2, p.chroma, p.chroma_scales)
        C3d = fn(C3, max(p.luma, p.chroma), p.luma_scales)
        if p.grain > 0:
            Yd = Yd + float(p.grain) * (Y - Yd)
    R, G1, G2, B = from_yc(Yd, C1d, C2d, C3d)
    out = mos.astype(np.float32, copy=True)
    with slog.stage("denoise.inverse_vst", log):
        for name, plane in (("R", R), ("G1", G1), ("G2", G2), ("B", B)):
            i, j = roles[name]
            den = inverse_anscombe_exact(plane, float(model.a[i, j]), float(model.b[i, j]))
            orig = mos[i:h:2, j:w:2]
            # near clip the model breaks down: fade back to the original
            t = np.clip((orig - clip_start) / (1.0 - clip_start), 0.0, 1.0)
            out[i:h:2, j:w:2] = den * (1 - t) + orig * t
    return out

"""Dust and scratches on scans and old photos: find them, then fill them in with the surrounding light and grain.

What a defect looks like. Dust on a print, a negative or the scanner glass, a hair, a scratch through the
emulsion, a crack in an old print: small or thin marks, much brighter or much darker than what is right round them,
with edges sharper than anything the lens drew. On a negative turned positive, dust (opaque on the film) comes out
white; on a print it can be either.

Finding them (classical morphology, Serra 1982; the same idea as the "dust & scratches" filters of scanner
software, made stricter):

  1. log brightness, so a mark counts by how many stops it stands out, in dark and light areas alike;
  2. a top-hat in both directions: brightness minus its opening (bright marks), its closing minus brightness (dark
     marks), with a square a little bigger than the largest mark to remove. What is left is everything narrower
     than that square: the marks, but also texture (hair, grass, fabric, film grain);
  3. marks stand far above the texture *around them*: the threshold is a multiple of the local spread of the
     residual (measured over a wide window), so a speck on a clear sky or a cheek is found at a low contrast and
     the same contrast inside foliage is left alone. Hysteresis (strong core, weaker body) gives each mark its
     whole shape;
  4. each mark is kept only if it is a speck (small) or a scratch (long and thin); anything else is texture.

Filling them: the light inside each mark is interpolated from its edge (a harmonic membrane in log RGB, solved on
the marked pixels only, so thousands of marks cost little), then grain like the surrounding grain is laid over it,
from a coordinate hash, so a zoomed tile and the whole photo get the same pixels.

Everything is local: a block rendered with ``PAD`` pixels of context gives the same result as the whole photo.
Sizes are in full-resolution pixels; ``step`` 2 is the half-size preview.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from ... import log as slog

log = slog.get("engine.scratches")

LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)       # ProPhoto luminance
EPS = 1.0 / 2048.0
DEFAULTS = {"on": False, "amount": 50.0, "size": 8.0, "bright": True, "dark": True}


def settings(d: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (d or {}).items() if k in DEFAULTS})
    return out


def active(d: dict | None) -> bool:
    return bool(d) and bool(d.get("on"))


def pad_for(d: dict | None) -> int:
    """Context (full-resolution pixels) a block needs so its marks are found and filled as in the whole photo."""
    s = settings(d)
    return int(math.ceil(6 * s["size"])) + 48


def _k(amount: float) -> float:
    """Amount 0..100 -> how many local spreads above the texture a mark must stand (8 = only the obvious)."""
    a = float(np.clip(amount, 0.0, 100.0)) / 100.0
    return 8.0 * (2.4 / 8.0) ** a


P_DAMAGE = 0.35                     # the detector's probability above which a place counts as damaged


def detect(Y: np.ndarray, d: dict | None, step: int = 1, prior: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Marks in a scene-linear luminance block (h, w). Returns (mask, stats).

    ``prior``: the learned detector's damage probability at each pixel of the block (scratchnet.py), or None when
    it is not installed. With it, a long or large mark is kept only where the detector sees damage (it tells a
    scratch from a collar, a wire or a strand of hair), and inside the places it marks the search is more
    sensitive, so a scratch's faint ends are found too. Small specks are below its resolution (it looks at the
    photo 256 pixels high), so they are judged here alone, by how far they stand above their surroundings."""
    s = settings(d)
    st = max(1, int(step))
    r = max(1.0, 0.5 * float(s["size"]) / st)        # "size" is the widest mark (a diameter)
    D = 2 * int(math.ceil(r)) + 1
    L = np.log2(np.maximum(Y, 0.0) + EPS).astype(np.float32)
    k = _k(s["amount"])
    win = max(9, int(round(48 / st)) | 1)
    masks, n_found = [], {"bright": 0, "dark": 0}
    for pol in ("bright", "dark"):
        if not s[pol]:
            continue
        R = (L - ndi.grey_opening(L, size=(D, D))) if pol == "bright" else (ndi.grey_closing(L, size=(D, D)) - L)
        # local spread of the residual, robust to the marks themselves: clip before averaging
        Rc = np.minimum(R, 0.75)                    # a fixed cap (in stops): a block-wide statistic would make a
                                                    # zoomed tile disagree with the whole photo
        spread = np.sqrt(ndi.uniform_filter(Rc * Rc, win, mode="reflect")) + 0.02
        kk = k if prior is None else np.where(prior >= P_DAMAGE, 0.6 * k, k)
        strong = R > np.maximum(kk * spread, 0.25)              # and at least a quarter stop
        weak = R > np.maximum(0.45 * kk * spread, 0.12)
        lab, n = ndi.label(weak)
        if n == 0:
            continue
        has_strong = ndi.maximum(strong, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = has_strong.astype(bool)
        # shape: specks (small) or scratches (long and thin); bigger blobs are texture or picture
        sl = ndi.find_objects(lab)
        area = ndi.sum(np.ones_like(L), lab, index=np.arange(1, n + 1))
        # the detector's verdict near each mark (its map is coarse: a few pixels either side count)
        pmax = ndi.maximum(ndi.maximum_filter(prior, size=int(2 * r) | 1), lab, index=np.arange(1, n + 1)) \
            if prior is not None else None
        rmax = ndi.maximum(R / spread, lab, index=np.arange(1, n + 1))     # how far it stands above its texture
        speck_max = math.pi * (r + 1) ** 2 * 1.6
        for i in np.nonzero(keep[1:])[0]:
            sly, slx = sl[i]
            hh, ww = sly.stop - sly.start, slx.stop - slx.start
            length = math.hypot(hh, ww)
            a = float(area[i])
            width = a / max(length, 1.0)
            # a long thin mark far above its texture is a scratch even where the detector (which sees the photo
            # only 256-384 pixels high) cannot see one that thin
            judged = pmax is None or pmax[i] >= P_DAMAGE or rmax[i] >= 2.0 * k
            if a <= speck_max and (judged or a <= 0.5 * speck_max):
                continue
            if length >= 5.0 * width and width <= 2.2 * r and judged:
                continue
            keep[i + 1] = False
        m = keep[lab]
        n_found[pol] = int(np.count_nonzero(np.unique(lab[m])))
        masks.append(m)
    if not masks:
        return np.zeros(Y.shape, bool), {"marks": 0, **n_found}
    mask = np.logical_or.reduce(masks)
    mask = ndi.binary_dilation(mask, iterations=1 if st > 1 else 2)          # the soft halo round a mark
    return mask, {"marks": n_found["bright"] + n_found["dark"], **n_found, "pixels": int(mask.sum())}


def _fill_log(V: np.ndarray, mask: np.ndarray, iters: int) -> np.ndarray:
    """Harmonic fill of the masked pixels of V (h, w, C) from their neighbours (Jacobi on the masked pixels only,
    started from a normalised blur of the unmasked light round them)."""
    h, w, C = V.shape
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return V
    out = V.reshape(-1, C).copy()
    known = (~mask).astype(np.float32)
    sig = 2.0
    den = ndi.gaussian_filter(known, sig, mode="nearest")
    for c in range(C):
        num = ndi.gaussian_filter(V[..., c] * known, sig, mode="nearest")
        init = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan).reshape(-1)
        out[idx, c] = init[idx]
    bad = ~np.isfinite(out[idx, 0])
    if bad.any():                                           # deep inside a wide mark: the block's mean to start
        out[idx[bad]] = np.nanmean(out[~mask.reshape(-1)], axis=0) if (~mask).any() else 0.0
    yy, xx = np.divmod(idx, w)
    # red-black successive over-relaxation: converges in a number of sweeps proportional to the mark's width
    # (plain Jacobi needs the square of it)
    groups = []
    for parity in (0, 1):
        sel = ((yy + xx) & 1) == parity
        ii, y1, x1 = idx[sel], yy[sel], xx[sel]
        nb = [np.clip(y1 + dy, 0, h - 1) * w + np.clip(x1 + dx, 0, w - 1)
              for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        groups.append((ii, nb))
    omega = np.float32(1.85)
    for _ in range(iters):
        for ii, nb in groups:
            avg = 0.25 * (out[nb[0]] + out[nb[1]] + out[nb[2]] + out[nb[3]])
            out[ii] += omega * (avg - out[ii])
    return out.reshape(h, w, C)


def repair(rgb: np.ndarray, mask: np.ndarray, ox: int, oy: int, step: int = 1, seed: int = 11,
           size: float = 8.0) -> np.ndarray:
    """Fill the marked pixels of a scene-linear block: light from the edges, grain like the surroundings.
    (ox, oy): the block's top-left in full-resolution pixels (for the grain hash); ``size``: the widest mark
    (full-resolution pixels), which sets how many relaxation sweeps the fill gets (the same for every block)."""
    if not mask.any():
        return rgb
    from ..detail.finish import hash_gauss
    V = np.log2(np.maximum(rgb, 0.0) + EPS).astype(np.float32)
    width = float(size) / max(1, step) + 6.0
    F = _fill_log(V, mask, iters=int(min(160, 10 + 4 * width)))
    # grain: the robust spread of the finest detail outside the marks, per channel
    hp = V - ndi.gaussian_filter(V, (1.0, 1.0, 0), mode="nearest")
    ok = ~ndi.binary_dilation(mask, iterations=2)
    h, w = mask.shape
    jj = (ox + np.arange(w) * step)[None, :]
    ii = (oy + np.arange(h) * step)[:, None]
    g = hash_gauss(jj, ii, seed)
    g = ndi.gaussian_filter(g, 0.6 / max(1, step) + 0.1)
    g /= max(float(g.std()), 1e-6)
    # the grain's strength where it is: mean absolute fine detail of the unmarked pixels nearby (normalised
    # box average), x sqrt(pi/2) for a Gaussian's spread; log brightness grain is stronger in the shadows
    win = max(7, int(round(24 / step)) | 1)
    okf = ok.astype(np.float32)
    den = ndi.uniform_filter(okf, win, mode="reflect")
    for c in range(3):
        num = ndi.uniform_filter(np.abs(hp[..., c]) * okf, win, mode="reflect")
        sig = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0) * np.float32(1.2533 * 1.15)
        F[..., c] += np.where(mask, g * sig, 0.0)
    out = np.where(mask[..., None], np.exp2(F) - EPS, rgb)
    return np.maximum(out, 0.0).astype(np.float32)


def clean(rgb: np.ndarray, d: dict | None, ox: int, oy: int, step: int = 1, stats: dict | None = None,
          protect: np.ndarray | None = None, prior: np.ndarray | None = None) -> np.ndarray:
    """Find and fill the dust and scratches of a scene-linear block. ``protect``: pixels never touched (eyes);
    ``prior``: the learned detector's damage probability on the block (see detect)."""
    Y = rgb.reshape(-1, 3) @ LUMA
    Y = Y.reshape(rgb.shape[:2])
    with slog.stage("scratches.find", log, w=rgb.shape[1], h=rgb.shape[0], step=step, learned=prior is not None):
        mask, st = detect(Y, d, step, prior)
    if protect is not None:
        mask &= ~protect
    if stats is not None:
        stats.update(st)
    with slog.stage("scratches.fill", log, marks=st.get("marks", 0)):
        return repair(rgb, mask, ox, oy, step, size=settings(d)["size"])


def damage(img: np.ndarray, n_specks: int = 120, n_lines: int = 12, seed: int = 3, bright: float = 0.5,
           dark: float = 0.5):
    """Synthetic dust and scratches on a scene-linear picture (for tests): returns (damaged, truth mask)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.copy()
    truth = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_specks):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(0.8, 3.5)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        lvl = 4.0 if rng.random() < bright / max(bright + dark, 1e-6) else 0.12
        out[m] = out[m] * lvl if lvl < 1 else np.maximum(out[m] * lvl, 0.6)
        truth |= m
    for _ in range(n_lines):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, math.pi)
        ln = rng.uniform(0.15, 0.5) * max(h, w)
        t = np.linspace(0, ln, int(ln * 2))
        px, py = (x0 + t * math.cos(ang)).astype(int), (y0 + t * math.sin(ang)).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        line = np.zeros((h, w), bool)
        line[py[ok], px[ok]] = True
        if rng.random() < 0.5:                      # (iterations=0 would mean "until nothing changes")
            line = ndi.binary_dilation(line, iterations=1)
        lvl = 3.0 if rng.random() < bright / max(bright + dark, 1e-6) else 0.2
        out[line] = out[line] * lvl if lvl < 1 else np.maximum(out[line] * lvl, 0.5)
        truth |= line
    return out.astype(np.float32), truth

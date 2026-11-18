"""The residual test and the detail guard (BUILD_PLAN_NOISE block E).

What Clean removed is ``removed = before - after``. The planes are variance-stabilised, so the noise in them has
strength one everywhere. If Clean did its job and nothing else, ``removed`` is white noise of strength one at
most: no edges, no texture, the same everywhere. Structure in it is detail that was taken.

White noise has a known strength in every wavelet band. So the residual is split into bands; where a band's local
energy stands clearly above what noise alone can put there, that part is **not noise** and is given back to the
photo (the same shrinkage Clean uses, pointed the other way and set far more strictly). What stays removed is
what cannot be told from noise. Then the **noise envelope**: at no scale may the result differ from the photo by
more than noise could account for, so no method, a learned one least of all, can add something the noise could
not have hidden.

Related: "method noise" (Buades, Coll, Morel 2005), iterative regularisation (Osher et al. 2005), SOS boosting
(Romano and Elad 2015), Tukey's twicing. The code is ours.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import uniform_filter

GUARD_K = 3.0               # a band's local energy must exceed this many times the noise energy to count as detail
GUARD_SCALES = 4
ENVELOPE_K = 5.0            # per scale, the result may not differ from the photo by more than this many sigmas
ENVELOPE_K_COLOUR = 6.0     # colour is cleaned harder than luminance (blotches), so its envelope is a little wider
PASSES = 2
BUDGET_WINDOW = 33          # the neighbourhood over which removed energy is weighed against the noise
BUDGET_MARGIN = 0.06        # the weighing is itself noisy (about 4% over this window): this much over is allowed


def _bands(x: np.ndarray, scales: int):
    from .denoise import starlet, starlet_noise_sigma
    return starlet(x, scales), starlet_noise_sigma()


def structure(removed: np.ndarray, k: float = GUARD_K, scales: int = GUARD_SCALES) -> tuple[np.ndarray, dict]:
    """The part of ``removed`` (unit-noise units) that is not noise, and what was measured.
    The finest band is judged twice as strictly: one strong noise sample lives there."""
    bands, sig = _bands(removed, scales)
    back = np.zeros_like(removed, dtype=np.float32)
    excess = []
    for j in range(scales):
        w = bands[j]
        win = 5 + 2 * min(j, 3)
        nv = sig[j] * sig[j]
        e = uniform_filter(w * w, size=win, mode="mirror")
        kk = k * (2.0 if j == 0 else 1.0)
        gain = np.clip(1.0 - kk * nv / np.maximum(e, 1e-12), 0.0, 1.0)
        # a second look at a broader area: lone specks of noise do not make a neighbourhood of energy
        e2 = uniform_filter(gain, size=win + 4, mode="mirror")
        gain = gain * np.clip(e2 * 4.0, 0.0, 1.0)
        back += gain * w
        excess.append(float(np.mean(e) / nv))
    return back, dict(band_energy=[round(v, 3) for v in excess])


def budget(removed: np.ndarray, scales: int = GUARD_SCALES, margin: float = BUDGET_MARGIN) -> np.ndarray:
    """The noise budget. Detail fainter than the noise cannot be picked out of what was removed, but its energy
    can be weighed: in any neighbourhood and band, noise can account for only so much. Where more than that was
    removed, the surplus was the picture. It cannot be told apart from the noise beside it, so that share of the
    removed band is handed back whole: the area keeps a little grain and all of its texture, instead of losing
    texture to look smooth. Returns what to give back."""
    bands, sig = _bands(removed, scales)
    back = np.zeros_like(removed, dtype=np.float32)
    for j in range(scales):
        w = bands[j]
        rho = uniform_filter(w * w, size=BUDGET_WINDOW, mode="mirror") / np.float32(sig[j] * sig[j])
        f = np.clip(1.0 - np.sqrt((1.0 + margin) / np.maximum(rho, 1e-6)), 0.0, 1.0)
        back += uniform_filter(f, size=5, mode="mirror") * w
    return back


def give_back(orig: np.ndarray, den: np.ndarray, k: float = GUARD_K, passes: int = PASSES) -> tuple[np.ndarray, dict]:
    """Return to ``den`` what was removed from ``orig`` (both unit-noise planes) that was not noise: first the
    structure that stands above the noise (``structure``), then whatever exceeds the noise budget (``budget``).
    Report: removed energy, the share of it that was returned, and the band energies of the residual before and
    after (1.0 = exactly what white noise gives; above = detail in it; below = noise was left in the photo)."""
    out = den.astype(np.float32, copy=True)
    removed0 = orig - den
    e0 = float(np.mean(removed0 * removed0)) or 1e-12
    rep = None
    returned = np.zeros_like(out)
    for _ in range(max(passes, 1)):
        back, r = structure(orig - out, k)
        rep = rep or r
        if float(np.mean(back * back)) < 1e-7 * e0:
            break
        out += back
        returned += back
    back = budget(orig - out)
    out += back
    returned += back
    after, _ = _energies(orig - out)
    rep.update(removed_energy=round(e0, 4), returned=round(float(np.mean(returned * returned)) / e0, 5),
               band_energy_after=after)
    return out, rep


def _energies(removed: np.ndarray, scales: int = GUARD_SCALES):
    bands, sig = _bands(removed, scales)
    return [round(float(np.mean(bands[j] ** 2) / (sig[j] ** 2)), 3) for j in range(scales)], sig


@lru_cache(maxsize=None)
def _smooth_sigma(scales: int) -> float:
    """Strength of what unit white noise leaves in the smooth remainder of the transform (very little)."""
    from .denoise import starlet
    n = np.random.default_rng(11).standard_normal((1024, 1024)).astype(np.float32)
    return float(starlet(n, scales)[-1][96:-96, 96:-96].std())


def envelope(orig: np.ndarray, den: np.ndarray, k: float = ENVELOPE_K, scales: int = GUARD_SCALES) -> tuple:
    """No scale of (orig - den) beyond k sigma of that scale's noise: from k to 1.5 k the photo is blended back,
    beyond it is kept. The smooth remainder is held to the same rule: noise has almost no energy there, so a
    method that shifts the brightness or colour of an area did not remove noise. Returns (result, share of
    coefficients held back)."""
    bands, sig = _bands(orig - den, scales)
    limits = [k * sig[j] for j in range(scales)] + [k * _smooth_sigma(scales)]
    keep = np.zeros_like(bands[0])
    held, n = 0, 0
    for w, lim in zip(bands, limits):
        lim = np.float32(lim)
        t = np.clip((np.abs(w) - lim) / (0.5 * lim), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        keep = keep + w * (1.0 - t)
        held += int(np.count_nonzero(t > 0.5))
        n += w.size
    return orig - keep, held / max(n, 1)


def whiteness(removed: np.ndarray) -> dict:
    """Numbers for the harness and the "What Clean removed" view: variance (1 = all the noise and only the noise),
    neighbour agreement (0 for noise) and band energies against white noise."""
    r = removed - float(np.mean(removed))
    v = float(np.mean(r * r)) or 1e-12
    bands, _ = _energies(r)
    return dict(variance=round(v, 4), neighbour_h=round(float(np.mean(r[:, :-1] * r[:, 1:])) / v, 4),
                neighbour_v=round(float(np.mean(r[:-1] * r[1:])) / v, 4), band_energy=bands)

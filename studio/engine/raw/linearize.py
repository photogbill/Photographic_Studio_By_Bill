"""Stage 2: linearise. Black level per channel, white level, normalise to 0..1 (1.0 = sensor clip).

Section 21 change 1: black level is a first-class accuracy target (small black errors become colour casts
in deep shadows). The black level is measured per frame, per CFA position, from the masked optical-black
columns with a sigma-clipped mean; LibRaw's value is the fallback and the sanity check.

Per-row black (row banding) is stage 5 and is not done here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ... import log as slog
from ..io.rawfile import RawImage

log = slog.get("engine.linearize")


def clipped_stats(x, k: float = 4.0, iters: int = 6, max_n: int = 2_000_000) -> tuple[float, float]:
    """Sigma-clipped mean and std (astronomy style). Robust to hot pixels, not quantised like the median."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size > max_n:
        x = x[:: int(math.ceil(x.size / max_n))]
    m, s = float(np.median(x)), float(x.std())
    for _ in range(iters):
        if s <= 0:
            break
        sel = np.abs(x - m) < k * s
        if sel.sum() < 10:
            break
        m2, s2 = float(x[sel].mean()), float(x[sel].std())
        done = abs(s2 - s) <= 1e-6 * max(s, 1e-12)
        m, s = m2, s2
        if done:
            break
    return m, s


@dataclass
class Levels:
    black: np.ndarray        # 2x2, per CFA position of the visible area
    white: float             # clip level (raw units)
    source: str              # 'optical-black' or 'libraw'
    read_noise_dn: np.ndarray | None = None   # 2x2 sigma of the masked pixels, when measured

    @property
    def range(self) -> np.ndarray:
        return self.white - self.black


def measure_optical_black(img: RawImage) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-CFA-position black level and read noise from the left masked columns. None if unusable."""
    lm = img.left
    if lm < 16:
        return None
    region = img.mosaic[img.top:img.top + img.height, :lm].astype(np.float32)
    # skip the first and last few masked columns (edge effects), and columns that are not dark
    cmed = np.median(region[::4], axis=0)
    g = float(np.median(cmed))
    _, s0 = clipped_stats(region[::8, lm // 4: 3 * lm // 4])
    good = np.where(np.abs(cmed - g) < max(3.0, 0.5 * s0))[0]
    good = good[(good >= 4) & (good <= lm - 5)]
    if len(good) < 8:
        return None
    black = np.zeros((2, 2), np.float32)
    sig = np.zeros((2, 2), np.float32)
    for i in (0, 1):
        for j in (0, 1):
            # CFA position (i, j) of the visible area corresponds to absolute row parity (top+i)
            # and absolute column parity (left+j); masked column c has the same parity rule.
            cols = good[(good % 2) == ((lm + j) % 2)]
            if len(cols) < 4:
                return None
            vals = region[i::2][:, cols]
            black[i, j], sig[i, j] = clipped_stats(vals)
    return black, sig


def levels(img: RawImage, use_optical_black: bool = True) -> Levels:
    lib = img.black_for_positions(img.black_libraw)
    white = img.clip_level()
    if use_optical_black:
        ob = measure_optical_black(img)
        if ob is not None:
            black, sig = ob
            rng = white - lib
            # sanity check: the optical-black estimate must sit close to LibRaw's (within 1% of range)
            if np.all(np.abs(black - lib) < 0.01 * rng) and np.all(np.isfinite(black)):
                log.debug("black from optical black %s (LibRaw %s), read noise %s DN", black.round(2).tolist(),
                          lib.tolist(), sig.round(2).tolist())
                return Levels(black=black, white=white, source="optical-black", read_noise_dn=sig)
            log.warning("optical-black estimate %s far from LibRaw %s: using LibRaw", black.tolist(), lib.tolist())
    return Levels(black=lib.astype(np.float32), white=white, source="libraw")


def normalise(img: RawImage, lv: Levels, region: tuple | None = None) -> np.ndarray:
    """Black-subtracted, scaled mosaic of the visible area (or a region of it), float32, 1.0 = clip.

    ``region`` is (x, y, w, h) in visible-area coordinates; x and y must be even so the pattern is unchanged.
    Values below zero are kept (noise around black is symmetric; clipping it lifts the shadows)."""
    vis = img.visible
    if region is not None:
        x, y, w, h = region
        if x % 2 or y % 2:
            raise ValueError("region origin must be even (CFA alignment)")
        vis = vis[y:y + h, x:x + w]
    out = vis.astype(np.float32)
    rng = lv.range
    for i in (0, 1):
        for j in (0, 1):
            out[i::2, j::2] -= lv.black[i, j]
            out[i::2, j::2] *= np.float32(1.0 / rng[i, j])
    return out

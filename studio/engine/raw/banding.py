"""Stage 5: row banding removal from the masked (optical-black) border (sections 8.2, 21 change 7, 22 change 3).

Canon sensors read out a row at a time; each row picks up a small random offset shared by the whole row. The masked
columns at the left edge see the same offset with no picture on top, so each row's offset can be measured there.

The catch (measured on Bill's 5D IV at ISO 32000): there are only ~128 masked columns, so each row estimate carries
noise of sigma / sqrt(128), about 9% of the read noise, which is as large as the banding itself on that frame.
Subtracting the raw estimate would swap one banding for another. So the correction is a Wiener estimate:
    applied offset = measured offset * B^2 / (B^2 + E^2)
where B^2 is the true row-banding variance (row-mean variance minus what white noise predicts) and E^2 the estimation
noise. No banding measured -> nothing applied, so bands can never be created on a clean frame (the plan's gate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import log as slog
from ..io.rawfile import RawImage
from .linearize import Levels

log = slog.get("engine.banding")


@dataclass
class RowBanding:
    offsets: np.ndarray        # per visible row, normalised units (already Wiener-weighted)
    true_sigma: float          # measured row-banding amplitude (normalised)
    est_sigma: float           # estimation noise per row (normalised)
    weight: float              # B^2 / (B^2 + E^2)


def measure_rows(img: RawImage, lv: Levels) -> RowBanding | None:
    lm = img.left
    if lm < 24:
        return None
    ob = img.mosaic[img.top:img.top + img.height, 4:lm - 4].astype(np.float32)
    rng = float(np.mean(lv.range))
    # per row: robust mean over the masked columns (per-column black differences removed first)
    ob = ob - np.median(ob, axis=0, keepdims=True)
    med = np.median(ob, axis=1, keepdims=True)
    mad = np.median(np.abs(ob - med), axis=1, keepdims=True) * 1.4826 + 1e-6
    keep = np.abs(ob - med) < 5 * mad
    row = np.where(keep, ob, 0).sum(1) / np.maximum(keep.sum(1), 1)
    n_eff = float(np.mean(keep.sum(1)))
    pix_sigma = float(np.median(mad))
    est2 = pix_sigma ** 2 / max(n_eff, 1.0)
    # row-to-row variance, per row parity (each colour row pair shares a readout pattern)
    row = row - np.mean(row)
    var = float(np.var(row))
    true2 = max(var - est2, 0.0)
    w = true2 / (true2 + est2) if (true2 + est2) > 0 else 0.0
    log.debug("row banding: true %.2f DN, estimate noise %.2f DN per row, weight %.2f", np.sqrt(true2),
              np.sqrt(est2), w)
    return RowBanding(offsets=(row * w / rng).astype(np.float32), true_sigma=float(np.sqrt(true2)) / rng,
                      est_sigma=float(np.sqrt(est2)) / rng, weight=float(w))


def remove_rows(mos: np.ndarray, rb: RowBanding, y0: int = 0) -> np.ndarray:
    """Subtract the weighted row offsets from a normalised mosaic whose first row is visible row y0 (in place)."""
    if rb is None or rb.weight <= 0.01:
        return mos
    h = mos.shape[0]
    mos -= rb.offsets[y0:y0 + h, None]
    return mos

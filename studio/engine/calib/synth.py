"""Photographs with a known right answer (BUILD_PLAN_NOISE A.1, A.2): clean truth plus this camera's real noise.

A clean mosaic (a base-ISO raw reduced 2x, or a generated chart) is turned into what the camera would have
recorded at a chosen ISO: photon noise drawn with the **measured gain**, then a patch of a **real dark frame** of
that ISO from the camera's noise bank (deep.py), with the camera's clipping. That carries true read noise,
banding, fixed pattern and warm pixels. With no bank, read noise comes from the profile's figure (formula noise),
and the result says so.

Used by the benchmark (tools/noise_bench.py) and by the training of a camera's own network (studio/lab).
"""
from __future__ import annotations

import numpy as np

from .noise import NoiseModel

RGGB = np.array([[0, 1], [1, 2]])


# ------------------------------------------------------------------------------------------------ clean truth
def reduce_planes(mos: np.ndarray, k: int = 2) -> np.ndarray:
    """Average k x k photosites **of the same colour** into one: a mosaic 1/k the size, still a true Bayer mosaic
    with the same pattern, its own noise cut k times. This is how a base-ISO raw becomes clean truth."""
    h, w = (mos.shape[0] // (2 * k)) * 2 * k, (mos.shape[1] // (2 * k)) * 2 * k
    out = np.empty((h // k, w // k), np.float32)
    for i in (0, 1):
        for j in (0, 1):
            p = mos[i:h:2, j:w:2].astype(np.float32)
            ph, pw = p.shape[0] // k, p.shape[1] // k
            out[i::2, j::2] = p[:ph * k, :pw * k].reshape(ph, k, pw, k).mean(axis=(1, 3))
    return out


def mosaic_of(rgb: np.ndarray, pattern=RGGB) -> np.ndarray:
    p = np.asarray(pattern)
    h, w = (rgb.shape[0] // 2) * 2, (rgb.shape[1] // 2) * 2
    m = np.empty((h, w), np.float32)
    for i in (0, 1):
        for j in (0, 1):
            m[i::2, j::2] = rgb[i:h:2, j:w:2, int(p[i, j])]
    return m


def dead_leaves(h: int = 768, w: int = 768, seed: int = 0, colour: float = 0.35) -> np.ndarray:
    """The dead-leaves chart (plan section 22): overlapping discs of every size, the standard target for "is
    texture kept?", because its detail has the statistics of natural texture and no orientation to favour.
    Linear RGB, 0.03 to 0.85, as the camera would see it (no white balance)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.3, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    n = int(h * w / 18)
    r = 1.5 * (1.0 / np.sqrt(rng.uniform(0.0006, 1.0, n)))            # radii: many small, few large
    for k in np.argsort(-r):                                           # large first, small on top
        cy, cx, rad = rng.uniform(0, h), rng.uniform(0, w), float(min(r[k], 0.2 * min(h, w)))
        y0, y1 = max(int(cy - rad), 0), min(int(cy + rad) + 1, h)
        x0, x1 = max(int(cx - rad), 0), min(int(cx + rad) + 1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        m = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2 <= rad * rad
        g = rng.uniform(0.03, 0.85)
        c = np.clip(g * (1.0 + colour * rng.uniform(-1, 1, 3)), 0.02, 0.9)
        img[y0:y1, x0:x1][m] = c
    return img


def flat_patches(h: int = 512, w: int = 768, levels: int = 12) -> np.ndarray:
    """Grey patches from deep shadow to near white: noise left and colour blotches are measured here."""
    img = np.empty((h, w, 3), np.float32)
    cols, rows = 4, int(np.ceil(levels / 4))
    v = np.geomspace(0.004, 0.8, levels)
    for k in range(rows * cols):
        y0, x0 = (k // cols) * h // rows, (k % cols) * w // cols
        img[y0:y0 + h // rows + 1, x0:x0 + w // cols + 1] = v[min(k, levels - 1)]
    return img


def slanted_edge(h: int = 512, w: int = 512, lo: float = 0.08, hi: float = 0.55, angle_deg: float = 6.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (xx - w / 2) * np.cos(np.radians(angle_deg)) + (yy - h / 2) * np.sin(np.radians(angle_deg))
    g = lo + (hi - lo) * np.clip(d / 1.2 + 0.5, 0.0, 1.0)           # an edge about one photosite wide
    return np.repeat(g[..., None], 3, axis=2).astype(np.float32)


# ------------------------------------------------------------------------------------------------ real noise
class NoiseSource:
    """One camera's noise, ready to be put on a clean mosaic at any ISO it was calibrated at."""

    def __init__(self, profile, bank=None):
        self.profile, self.bank = profile, bank
        k = [e.gain_e_per_dn * e.iso for e in profile.isos if e.gain_e_per_dn]
        self._gain_x_iso = float(np.median(k)) if k else None

    def isos(self) -> list:
        out = [e.iso for e in self.profile.isos if e.read_noise_dn is not None and self.gain(e.iso)]
        return sorted(out)

    def entry(self, iso: float):
        e = self.profile.entry_for(iso, tol_stops=0.01)
        if e is None:
            raise ValueError(f"ISO {iso:g} is not in this camera's profile")
        return e

    def gain(self, iso: float) -> float | None:
        e = self.profile.entry_for(iso, tol_stops=0.01)
        if e is not None and e.gain_e_per_dn:
            return float(e.gain_e_per_dn)
        return self._gain_x_iso / iso if self._gain_x_iso else None          # gain x ISO is constant (physics)

    def range_dn(self, iso: float) -> float:
        got = self.bank.patches(iso) if self.bank is not None else None
        e = self.entry(iso)
        black = float(np.mean(e.black_dn))
        if got is not None:
            return float(got[1]["clip_dn"]) - black
        if e.full_well_e and e.gain_e_per_dn:
            return float(e.full_well_e) / float(e.gain_e_per_dn)
        return 16383.0 - black

    def has_real_noise(self, iso: float) -> bool:
        return self.bank is not None and self.bank.patches(iso) is not None

    def model(self, iso: float) -> NoiseModel:
        """The Poisson-Gaussian model of what ``add`` makes, in normalised units (1.0 = clip)."""
        e, rng, g = self.entry(iso), self.range_dn(iso), self.gain(iso)
        if not g:
            raise ValueError("this camera's profile has no gain: shoot flat pairs (a Full profile) first")
        got = self.bank.patches(iso) if self.bank is not None else None
        rn = np.asarray(e.read_noise_dn, float)
        return NoiseModel(a=np.full((2, 2), 1.0 / (g * rng)), b=(rn / rng) ** 2,
                          source="calibrated", confidence="good", iso=iso,
                          notes=["real dark-frame noise" if got is not None else "formula read noise"])

    def dark(self, iso: float, shape: tuple, rng: np.random.Generator, cls: str = "short") -> np.ndarray:
        """Signal-free noise in DN above black for a mosaic of ``shape``: tiles of real dark-frame patches (CFA
        phase kept), or Gaussian read noise when the camera has no bank at this ISO."""
        h, w = shape
        got = self.bank.patches(iso, cls) if self.bank is not None else None
        if got is None and cls != "short":
            got = self.bank.patches(iso, "short") if self.bank is not None else None
        e = self.entry(iso)
        if got is None:
            rn = np.asarray(e.read_noise_dn, np.float32)
            out = rng.standard_normal((h, w)).astype(np.float32)
            for i in (0, 1):
                for j in (0, 1):
                    out[i::2, j::2] *= rn[i, j]
            return out
        patches, info = got
        P = patches.shape[1]
        black = np.asarray(info["black"], np.float32)
        out = np.empty((h, w), np.float32)
        for y in range(0, h, P):
            for x in range(0, w, P):
                p = patches[int(rng.integers(patches.shape[0]))]
                oy, ox = int(rng.integers(0, P // 2)) * 2, int(rng.integers(0, P // 2)) * 2
                p = np.roll(p, (-oy, -ox), axis=(0, 1))            # even shifts keep the colour pattern in step
                out[y:y + P, x:x + P] = p[:min(P, h - y), :min(P, w - x)]
        for i in (0, 1):
            for j in (0, 1):
                out[i::2, j::2] -= black[i, j]
        return out

    def add(self, clean: np.ndarray, iso: float, rng: np.random.Generator, cls: str = "short") -> np.ndarray:
        """``clean``: a normalised mosaic (black 0, clip 1), even height and width. Returns what the camera would
        have recorded at ``iso``, normalised the same way (values below black stay negative, as in a raw)."""
        g, span = self.gain(iso), self.range_dn(iso)
        if not g:
            raise ValueError("this camera's profile has no gain: shoot flat pairs (a Full profile) first")
        electrons = np.maximum(clean.astype(np.float64), 0.0) * span * g
        shot = rng.poisson(electrons).astype(np.float32) / np.float32(g)
        dn = shot + self.dark(iso, clean.shape, rng, cls)
        black = float(np.mean(self.entry(iso).black_dn))
        return (np.clip(np.rint(dn), -black, span) / np.float32(span)).astype(np.float32)

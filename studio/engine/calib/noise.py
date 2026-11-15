"""The camera noise model and the variance-stabilising transform (sections 7 and 8.1).

Model (Foi et al. 2008, Poisson-Gaussian), per CFA position, in normalised units (0..1 = black..clip):
    observed z = mu + n,   Var(n) = a * mu + b
    a = 1 / (gain_e_per_DN * range_DN)         (photon / shot noise)
    b = (read_noise_DN / range_DN) ** 2        (read noise)

Three sources, colour-coded in the UI (rule 0.3.11):
    calibrated on this body (cyan) > shared profile (violet) > estimated from this image (amber)
``estimate_from_image`` is the amber path; the calibration builder (calib/profile.py) is the cyan one.

VST: generalised Anscombe transform f(z) = 2 sqrt(z/a + 3/8 + b/a^2) makes the noise approximately unit
variance. Its inverse is the *exact unbiased* inverse (Makitalo and Foi): the mean of f(z) is tabulated as a
function of mu by summing the Poisson-Gaussian distribution numerically, and that table is inverted. Plain
algebraic inversion is biased at low signal; the bias shows up as lifted, tinted shadows (section 21 change 4).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.noise")


@dataclass
class NoiseModel:
    a: np.ndarray                       # 2x2, per CFA position (normalised units)
    b: np.ndarray                       # 2x2
    source: str = "estimated"           # calibrated | shared | estimated
    confidence: str = "low"             # good | fair | low
    iso: float | None = None
    notes: list = field(default_factory=list)

    def sigma(self, mu, pos=(0, 0)):
        i, j = pos
        return np.sqrt(np.maximum(self.a[i, j] * np.asarray(mu) + self.b[i, j], 1e-20))

    @property
    def gain_e_per_dn_rel(self) -> np.ndarray:
        """1/a: electrons per full range. Multiply by 1/range_DN for e-/DN."""
        return 1.0 / np.maximum(self.a, 1e-30)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["a"] = np.asarray(self.a).tolist()
        d["b"] = np.asarray(self.b).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseModel":
        return cls(a=np.asarray(d["a"], float), b=np.asarray(d["b"], float), source=d.get("source", "estimated"),
                   confidence=d.get("confidence", "low"), iso=d.get("iso"), notes=list(d.get("notes", [])))

    def scaled(self, k: float) -> "NoiseModel":
        """Model of k * z (e.g. after white-balance or exposure gain): Var(k z) = k a (k mu) + k^2 b."""
        return NoiseModel(a=self.a * k, b=self.b * k * k, source=self.source, confidence=self.confidence,
                          iso=self.iso, notes=list(self.notes))


# ------------------------------------------------------------------------------------------ estimation
_CHI2_CACHE: dict = {}


def _chi2_q(dof: int, q: float) -> float:
    key = (dof, q)
    if key not in _CHI2_CACHE:
        rng = np.random.default_rng(12345)
        _CHI2_CACHE[key] = float(np.percentile(rng.chisquare(dof, 400_000) / dof, q))
    return _CHI2_CACHE[key]


def _plane_points(p: np.ndarray, block: int = 8, nbins: int = 18, q: float = 25.0):
    """(mean, noise variance, count) points for one CFA plane from smooth blocks.
    Noise samples are 2x2 Haar diagonal details d = (x00 - x01 - x10 + x11)/2, which cancel flat areas and
    linear gradients and have the same variance as the noise."""
    h = (p.shape[0] // (2 * block)) * 2 * block
    w = (p.shape[1] // (2 * block)) * 2 * block
    p = p[:h, :w]
    d = 0.5 * (p[0::2, 0::2] - p[0::2, 1::2] - p[1::2, 0::2] + p[1::2, 1::2])
    m = 0.25 * (p[0::2, 0::2] + p[0::2, 1::2] + p[1::2, 0::2] + p[1::2, 1::2])
    hb, wb = d.shape[0] // block, d.shape[1] // block
    dv = d[:hb * block, :wb * block].reshape(hb, block, wb, block).var(axis=(1, 3), ddof=1).ravel()
    mm = m[:hb * block, :wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3)).ravel()
    corr = _chi2_q(block * block - 1, q)
    lo = max(1e-5, float(np.percentile(mm, 1)))
    hi = float(np.percentile(mm, 99.0))
    pts = []
    if hi <= lo * 1.2:
        return pts
    edges = np.concatenate([[-np.inf], np.geomspace(lo, hi, nbins)])
    for e0, e1 in zip(edges[:-1], edges[1:]):
        sel = (mm >= e0) & (mm < e1) & (mm < 0.9)
        if sel.sum() >= 40:
            pts.append((float(np.median(mm[sel])), float(np.percentile(dv[sel], q) / corr), int(sel.sum())))
    return pts


def estimate_from_image(mosaic: np.ndarray, read_noise: np.ndarray | None = None, iso: float | None = None
                        ) -> NoiseModel:
    """Estimate the noise model from one normalised mosaic (black-subtracted, 1.0 = clip, before WB).

    ``read_noise``: optional 2x2 read-noise sigma in normalised units from the optical-black border; when given,
    b is pinned to it and only a is fitted (two independent measurements, one answer).
    The gain (in electrons) is the same for every CFA position, so a is fitted jointly across the four planes."""
    planes = {(i, j): mosaic[i::2, j::2] for i in (0, 1) for j in (0, 1)}
    allpts = {k: _plane_points(v) for k, v in planes.items()}
    b = np.zeros((2, 2))
    notes = []
    if read_noise is not None and np.all(np.isfinite(read_noise)):
        b = np.asarray(read_noise, float) ** 2
    X, Y, W = [], [], []
    for (i, j), pts in allpts.items():
        for mu, var, n in pts:
            X.append(mu)
            Y.append(var - (b[i, j] if read_noise is not None else 0.0))
            W.append(math.sqrt(n) / max(var, 1e-12))
    X, Y, W = np.array(X), np.array(Y), np.array(W)
    conf = "low"
    if len(X) >= 6:
        if read_noise is not None:
            a = float(np.sum(W * W * X * Y) / max(np.sum(W * W * X * X), 1e-30))
        else:
            A = np.vstack([X, np.ones_like(X)]).T
            coef = np.linalg.lstsq(A * W[:, None], Y * W, rcond=None)[0]
            a, b0 = float(coef[0]), float(coef[1])
            b[:] = max(b0, 1e-12)
        span = X.max() / max(X.min(), 1e-9)
        conf = "good" if span >= 8 and len(X) >= 30 else ("fair" if span >= 3 else "low")
    else:
        a = float("nan")
    if not (np.isfinite(a) and a > 0):
        # fall back to the median local variance in the darkest smooth areas (read-noise dominated)
        notes.append("Too few flat areas to fit the signal-dependent part; using a conservative default.")
        a = 4e-5 if iso is None else 4e-5 * max(iso, 100) / 100.0
        conf = "low"
    model = NoiseModel(a=np.full((2, 2), a), b=np.maximum(b, 1e-12), source="estimated", confidence=conf, iso=iso,
                       notes=notes)
    log.debug("noise estimate: a=%.3e b=%s conf=%s (%d points)", a, np.sqrt(model.b).round(6).tolist(), conf, len(X))
    return model


# ------------------------------------------------------------------------------------------ VST
def anscombe(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Generalised Anscombe transform: unit-variance noise for Var = a*mu + b."""
    s2 = b / (a * a)
    return 2.0 * np.sqrt(np.maximum(z / a + 0.375 + s2, 0.0))


_INV_CACHE: dict = {}


def _exact_table(s2: float):
    """E[f(z)] as a function of the scaled mean lambda = mu/a, for f(s) = 2 sqrt(s + 3/8 + s2) with
    s = k + g, k ~ Poisson(lambda), g ~ N(0, s2). Returns (lambda grid, E[f] grid)."""
    key = round(float(s2), 6)
    if key in _INV_CACHE:
        return _INV_CACHE[key]
    lam = np.concatenate([np.linspace(0.0, 0.5, 26)[:-1], np.geomspace(0.5, 3000.0, 400)])
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(32)          # probabilists' Hermite: N(0,1) nodes
    gh_w = gh_w / gh_w.sum()
    sig = math.sqrt(max(s2, 0.0))
    ef = np.empty_like(lam)
    for n, L in enumerate(lam):
        kmax = int(L + 10 * math.sqrt(L + 1) + 15)
        k = np.arange(0, kmax + 1, dtype=float)
        if L == 0:
            p = (k == 0).astype(float)
        else:
            logp = -L + k * math.log(L) - np.array([math.lgamma(v + 1) for v in k])
            p = np.exp(logp - logp.max())
            p /= p.sum()
        s = k[:, None] + sig * gh_x[None, :]
        f = 2.0 * np.sqrt(np.maximum(s + 0.375 + s2, 0.0))
        ef[n] = float((p[:, None] * gh_w[None, :] * f).sum())
    _INV_CACHE[key] = (lam, ef)
    return lam, ef


def inverse_anscombe_exact(D: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exact unbiased inverse of ``anscombe`` (returns mu in the original units)."""
    s2 = b / (a * a)
    lam, ef = _exact_table(s2)
    D = np.asarray(D)
    out = np.interp(D, ef, lam, left=0.0)
    hi = D > ef[-1]
    if np.any(hi):                                  # large signal: the algebraic inverse is unbiased enough
        out = np.where(hi, (D / 2.0) ** 2 - 0.125 - s2, out)       # asymptotic unbiased inverse
    return (out * a).astype(np.float32)


def inverse_anscombe_algebraic(D: np.ndarray, a: float, b: float) -> np.ndarray:
    s2 = b / (a * a)
    return (((np.asarray(D) / 2.0) ** 2 - 0.375 - s2) * a).astype(np.float32)

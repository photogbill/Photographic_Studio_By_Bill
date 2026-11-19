"""Clean's strength, chosen by the photo itself, detail first (BUILD_PLAN_NOISE D.4).

After the variance-stabilising transform the noise in a plane has strength one, exactly. For noise that is known
that well, Stein's lemma lets two things be measured **without having the true picture**, from how the cleaner's
result moves when the photo is nudged by a little extra known noise (the Monte-Carlo form: Ramani, Blu and Unser
2008):

  * the error the cleaner makes against the true picture (Stein's unbiased risk estimate, SURE), and
  * the share of the true picture's **finest detail** that survives the cleaning (the same lemma applied to the
    finest band alone; this second use is the Studio's own).

The strength that makes the least error is not the one a photographer wants: on the Studio's charts it gives up
fine texture for a few percent less noise. So the two are put together: error that is *lost detail* counts
``1 + DETAIL_WEIGHT`` times as much as error that is *noise left in*, and the **lightest** strength within a few
percent of the best is taken. A photo of flat walls is cleaned hard, a photo full of fine texture gently, and the
share of fine detail kept is reported with the choice.

Stein's lemma in its plain form needs each photosite's noise to be its own. On some cameras, at some ISO settings,
the noise the camera itself adds (read noise: what a lens-cap frame shows) is shared a little between neighbouring
photosites of one colour. Measured on Bill's R5: 0.08 to 0.10 at ISO 400, nothing at ISO 3200 and up; and in his
flat frames, where the noise is the light's own, nothing at any ISO. So the sharing belongs to the read noise
only, and a photo's noise shares by the profile's figure times the part of its noise that is read noise
(``read_share``): next to nothing in a normal photo, more in a very dark one. The lemma still holds when the
nudge is given the same sharing as the noise (the generalised form: the nudge must have the noise's own
covariance), so that product shapes the nudge. Beyond ``MAX_NEIGHBOUR`` the sharing is too strong to be described
by one number and Clean keeps its sliders.
"""
from __future__ import annotations

import numpy as np

CANDIDATES = (0.35, 0.5, 0.7, 0.85, 1.0, 1.25, 1.6, 2.0, 2.6)
CANDIDATES_SLOW = (0.35, 0.5, 0.7, 1.0, 1.4)   # the slow methods (collaborative, MP-PCA) clean harder per step
NEAR_BEST = 0.03            # the lightest strength whose cost is within this of the best is chosen
DETAIL_WEIGHT = 4.0         # lost fine detail costs this much more than noise left in (0 = plain least error)
EPS = 0.05                  # size of the nudge (in noise sigmas)
MAX_NEIGHBOUR = 0.30        # above this agreement between neighbouring photosites, the estimates are not trusted
MIN_NEIGHBOUR = 0.02        # below this the noise is treated as each photosite's own
PIECE = 192
FINE_NOISE = 0.8            # energy unit noise puts in the finest band (1 - 1/5)^2 + 4 (1/5)^2
MIN_TEXTURE = 0.03          # fine-band energy (noise = 0.8) under which a piece has no fine detail to speak of


def shaped(rng: np.random.Generator, shape: tuple, corr: float = 0.0) -> np.ndarray:
    """Noise of strength one whose neighbours (along rows and along columns) agree by ``corr``: white noise passed
    through the smallest filter that gives that agreement."""
    n = rng.standard_normal(shape).astype(np.float32)
    if corr < MIN_NEIGHBOUR:
        return n
    c = float(min(corr, 0.45))
    a = (1.0 - np.sqrt(1.0 - 2.0 * c * c)) / (2.0 * c)        # [a, 1, a] has neighbour agreement 2a / (1 + 2a^2)
    for ax in (-1, -2):
        n = (n + np.float32(a) * (np.roll(n, 1, ax) + np.roll(n, -1, ax))) / np.float32(np.sqrt(1.0 + 2.0 * a * a))
    return n


_FINE_NOISE: dict = {}


def fine_noise(corr: float = 0.0) -> float:
    """Energy that noise of strength one puts in the finest band, for noise whose neighbours agree by ``corr``."""
    if corr < MIN_NEIGHBOUR:
        return FINE_NOISE
    key = round(float(corr), 3)
    if key not in _FINE_NOISE:
        f = fine(shaped(np.random.default_rng(11), (768, 768), key))
        _FINE_NOISE[key] = float(np.mean(f * f))
    return _FINE_NOISE[key]


def read_share(mosaics: list, model) -> float:
    """The part of these pieces' noise that is the camera's own read noise (the rest is the light's): 0 to 1.
    ``mosaics``: normalised mosaics (1.0 = clip); ``model``: their NoiseModel (variance = a x + b)."""
    parts = []
    for m in mosaics:
        for i in (0, 1):
            for j in (0, 1):
                a, b = float(model.a[i, j]), float(model.b[i, j])
                parts.append(float(np.mean(b / (a * np.maximum(m[i::2, j::2], 0.0) + b))))
    return float(np.mean(parts)) if parts else 0.0


def fine(v: np.ndarray) -> np.ndarray:
    """The finest band: each photosite against the mean of itself and its four neighbours (lab/data.texture_kept
    uses the same band, so the estimate here and the benchmark's measured figure are the same quantity)."""
    return v - (v + np.roll(v, 1, -1) + np.roll(v, -1, -1) + np.roll(v, 1, -2) + np.roll(v, -1, -2)) / 5.0


def measure(f: np.ndarray, f2: np.ndarray, y: np.ndarray, probe: np.ndarray, corr: float = 0.0) -> dict:
    """What one cleaning did to one piece, without the truth. ``f`` = cleaner(y), ``f2`` = cleaner(y + EPS probe);
    ``probe`` must be ``shaped`` with the same ``corr`` as the photo's noise.
    risk: estimated mean squared error against the clean piece (1 = the untouched photo's). texture: estimated
    fine-band energy of the clean piece. along: how much of that the result still holds (kept = along / texture)."""
    d = (f2 - f) / np.float32(EPS)
    risk = float(np.mean((f - y) ** 2)) - 1.0 + 2.0 * float(np.mean(probe * d))
    fy = fine(y)
    texture = float(np.mean(fy * fy)) - fine_noise(corr)
    along = float(np.mean(fine(f) * fy)) - float(np.mean(fine(probe) * fine(d)))
    return dict(risk=risk, texture=texture, along=along)


def cost(m: dict, weight: float = DETAIL_WEIGHT) -> float:
    """Error, with the part of it that is lost fine detail counted ``weight`` times more."""
    t = m["texture"]
    if t < MIN_TEXTURE or weight <= 0:
        return m["risk"]
    kept = float(np.clip(m["along"] / t, 0.0, 1.0))
    return m["risk"] + weight * (1.0 - kept) ** 2 * t


def risk(fn, y: np.ndarray, strength: float, scales: int, probe: np.ndarray, corr: float = 0.0) -> float:
    """Estimated mean squared error of ``fn(y, strength, scales)`` against the unknown clean plane (unit noise).
    0 would be perfect, 1 is what the untouched photo has."""
    return measure(fn(y, strength, scales), fn(y + np.float32(EPS) * probe, strength, scales), y, probe,
                   corr)["risk"]


def pieces(plane: np.ndarray, n: int = 5, size: int = PIECE) -> list:
    """A few pieces of a plane, always the same ones: the centre and four around it."""
    h, w = plane.shape
    size = min(size, h, w)
    spots = [(0.5, 0.5), (0.27, 0.27), (0.27, 0.73), (0.73, 0.27), (0.73, 0.73)][:n]
    out = []
    for fy, fx in spots:
        y0 = int(np.clip(fy * h - size / 2, 0, h - size))
        x0 = int(np.clip(fx * w - size / 2, 0, w - size))
        out.append(plane[y0:y0 + size, x0:x0 + size])
    return out


def _kept(ms: list) -> float | None:
    """Share of fine detail kept over the pieces that have any (None: no piece has fine detail above its noise)."""
    t = sum(m["texture"] for m in ms if m["texture"] >= MIN_TEXTURE)
    if t <= 0:
        return None
    return float(np.clip(sum(m["along"] for m in ms if m["texture"] >= MIN_TEXTURE) / t, 0.0, 1.0))


def choose(fn, planes: list, scales: int, candidates=CANDIDATES, seed: int = 5,
           weight: float = DETAIL_WEIGHT, corr: float = 0.0) -> dict:
    """The strength to use for these pieces of one kind of plane (luminance, or colour).
    Returns {strength, risk, kept, best, table: [(strength, estimated error)], kept_table: [(strength, kept)]}.
    ``kept``: estimated share of the finest detail that survives at the chosen strength (None: nothing fine here).
    ``weight=0`` is plain least error. ``corr``: how much neighbouring photosites of one colour share noise at
    this ISO on this camera (the profile's figure)."""
    rng = np.random.default_rng(seed)
    probes = [shaped(rng, p.shape, corr) for p in planes]
    table, kept_table, costs = [], [], []
    for s in candidates:
        ms = [measure(fn(p, s, scales), fn(p + np.float32(EPS) * b, s, scales), p, b, corr)
              for p, b in zip(planes, probes)]
        table.append((float(s), round(float(np.mean([m["risk"] for m in ms])), 4)))
        k = _kept(ms)
        kept_table.append((float(s), None if k is None else round(k, 3)))
        costs.append(float(np.mean([cost(m, weight) for m in ms])))
    best = min(costs)
    j = next(i for i, c in enumerate(costs) if c <= best + NEAR_BEST * max(abs(best), 0.05) + 1e-4)
    return dict(strength=table[j][0], risk=table[j][1], kept=kept_table[j][1], best=min(r for _, r in table),
                table=table, kept_table=kept_table)


def choose_blend(clean_fn, planes4: list, seed: int = 5, weight: float = DETAIL_WEIGHT, corr: float = 0.0) -> dict:
    """For a cleaner with no strength of its own (the learned network): how much of what it removes to take,
    0.5 to 1. ``planes4``: pieces of the four colour planes together, (4, h, w) each. With t of the removal taken,
    the estimated error is a parabola in t and the detail lost grows as t squared, so one nudge gives every t."""
    rng = np.random.default_rng(seed)
    a = b = lost = tex = along = 0.0
    for y in planes4:
        probe = shaped(rng, y.shape, corr)
        f = clean_fn(y)
        f2 = clean_fn(y + np.float32(EPS) * probe)
        d = f - y
        m = measure(f, f2, y, probe, corr)
        a += float(np.mean(d * d))
        b += (m["risk"] - float(np.mean(d * d)) - 1.0) / 2.0      # divergence of (f - y), per photosite
        if m["texture"] >= MIN_TEXTURE:
            k = float(np.clip(m["along"] / m["texture"], 0.0, 1.0))
            lost += (1.0 - k) ** 2 * m["texture"]
            tex += m["texture"]
            along += float(np.clip(m["along"], 0.0, m["texture"]))
    n = len(planes4)
    a, b, lost = a / n, b / n, lost / n
    # cost(t) = t^2 (a + weight lost) + 2 t b: least at t = -b / (a + weight lost)
    t = float(np.clip(-b / max(a + weight * lost, 1e-9), 0.5, 1.0))
    k1 = along / tex if tex > 0 else None
    return dict(strength=round(t, 2), risk=round(t * t * a + 2 * t * b, 4), best=round(-b * b / max(a, 1e-9), 4),
                kept=None if k1 is None else round(1.0 - t * (1.0 - k1), 3),
                table=[(round(x, 2), round(x * x * a + 2 * x * b, 4)) for x in (0.5, 0.75, 1.0)], kept_table=[])

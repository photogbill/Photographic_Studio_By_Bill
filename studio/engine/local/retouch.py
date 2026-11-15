"""Phase 9 (second part): retouching (plan section 9.6). Heal, clone, red eye and dust spots. No neural networks.

Spots are plain data in the edit, attached to the photo like masks (fractions of the camera-crop frame before
rotation, the "S frame"), so straightening or cropping later does not move them:

    {"kind": "heal" | "clone" | "redeye", "pts": [[x, y], ...], "size": radius (fraction of the S long edge),
     "feather": 0..1, "opacity": 0..100, "src": [dx, dy] (where the source is, as an offset in S fractions),
     "enabled": true, "auto": "" | "dust" | "map"}

One point is a round spot; several points are a brushed stroke. The source has the same shape, moved by ``src``.

* Heal lays in the source's texture and spreads the difference between source and destination, measured just
  outside the spot, smoothly across it: a membrane, the solution of Laplace's equation with those differences as
  boundary values. That is seamless cloning (Perez, Gangnet and Blake, "Poisson image editing", SIGGRAPH 2003),
  done here on log values so the lighting is matched by multiplication, as in the Photoshop Healing Brush
  (Georgiev, "Photoshop Healing Brush: a tool for seamless cloning", ECCV workshop 2004). Small spots are solved
  exactly with a sparse solver; very large strokes on a reduced grid (the membrane is smooth, and the feathered edge
  hides the difference).
* Clone is a feathered copy of the source.
* Red eye pulls red down to the other two channels where red dominates, and darkens the pupil a little.

Sources are always read from the photo before retouching, so the order of spots does not matter and a 100% region
renders exactly like the full photo. Every patch is computed from its own small reads (``fetch``) and cached by the
caller.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import splu

from ... import log as slog

log = slog.get("engine.retouch")

LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)
EPS = 2e-3                     # log offset (scene-linear, 1 = clip): about 9 stops below white
MAX_UNKNOWNS = 160_000         # membrane solved exactly up to this many pixels, on a reduced grid above
KINDS = ("heal", "clone", "redeye")
DEFAULT_FEATHER = {"heal": 0.25, "clone": 0.5, "redeye": 0.3}


def active(spots) -> list:
    out = []
    for s in spots or []:
        if not s.get("enabled", True) or s.get("kind") not in KINDS or not s.get("pts"):
            continue
        if float(s.get("opacity", 100)) <= 0:
            continue
        out.append(s)
    return out


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------ shape
class Shape:
    """A spot in full-resolution S-frame pixels: centre line, radius, source offset."""

    def __init__(self, spot: dict, sw: int, sh: int, step: int = 1):
        self.sw, self.sh = int(sw), int(sh)
        long = float(max(sw, sh))
        p = np.asarray(spot["pts"], float).reshape(-1, 2)
        self.px, self.py = p[:, 0] * sw, p[:, 1] * sh
        self.R = max(float(spot.get("size", 0.01)) * long, 1.0)
        self.f = min(max(float(spot.get("feather", DEFAULT_FEATHER.get(spot.get("kind"), 0.3))), 0.02), 1.0)
        self.kind = spot.get("kind", "heal")
        self.opacity = min(max(float(spot.get("opacity", 100)) / 100.0, 0.0), 1.0)
        dx, dy = spot.get("src") or (0.0, 0.0)
        # source offsets are whole pixels of the grid in use, so source and destination pixels line up
        self.ox = int(round(float(dx) * sw / step)) * step
        self.oy = int(round(float(dy) * sh / step)) * step

    def box(self, ring: float = 4.0):
        """(x0, y0, x1, y1) full-res S pixels: the spot plus a ring around it, clamped to the frame."""
        m = self.R + ring
        x0, x1 = int(math.floor(self.px.min() - m)), int(math.ceil(self.px.max() + m)) + 1
        y0, y1 = int(math.floor(self.py.min() - m)), int(math.ceil(self.py.max() + m)) + 1
        return max(0, x0), max(0, y0), min(self.sw, x1), min(self.sh, y1)

    def distance(self, X, Y):
        """Distance (full-res px) from the spot's centre line."""
        d = np.full(np.shape(X), np.inf)
        if len(self.px) == 1:
            return np.hypot(X - self.px[0], Y - self.py[0])
        for k in range(len(self.px) - 1):
            ax, ay, bx, by = self.px[k], self.py[k], self.px[k + 1], self.py[k + 1]
            dx, dy = bx - ax, by - ay
            n = dx * dx + dy * dy
            if n < 1e-9:
                dd = np.hypot(X - ax, Y - ay)
            else:
                t = np.clip(((X - ax) * dx + (Y - ay) * dy) / n, 0.0, 1.0)
                dd = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            d = np.minimum(d, dd)
        return d

    def alpha(self, X, Y):
        """(alpha 0..1 with the feathered edge, inside = alpha > 0)."""
        d = self.distance(X, Y) / self.R
        a = 1.0 - _smooth((d - (1.0 - self.f)) / self.f)
        return a.astype(np.float32), d < 1.0

    def source_inside_frame(self, ring: float = 4.0) -> bool:
        x0, y0, x1, y1 = self.box(ring)
        return x0 + self.ox >= 0 and y0 + self.oy >= 0 and x1 + self.ox <= self.sw and y1 + self.oy <= self.sh


def clamp_source(spot: dict, sw: int, sh: int) -> list:
    """The spot's source offset moved the least needed to keep the source inside the frame."""
    s = Shape(spot, sw, sh)
    x0, y0, x1, y1 = s.box(4.0)
    ox = min(max(s.ox, -x0), sw - x1)
    oy = min(max(s.oy, -y0), sh - y1)
    return [round(ox / float(sw), 6), round(oy / float(sh), 6)]


# ------------------------------------------------------------------------------------------ membrane
def _shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def membrane(inside: np.ndarray, bval: np.ndarray) -> np.ndarray:
    """Harmonic interpolation: M = bval outside ``inside`` (on its boundary), Laplace(M) = 0 inside. bval is
    (h, w, C); returns (h, w, C) with M inside and 0 outside. Box edges are free (Neumann)."""
    h, w = inside.shape
    C = bval.shape[2]
    n = int(inside.sum())
    out = np.zeros((h, w, C), np.float32)
    if n == 0:
        return out
    if n > MAX_UNKNOWNS:
        return _membrane_coarse(inside, bval)
    idx = np.full((h, w), -1, np.int64)
    idx[inside] = np.arange(n)
    rows, cols, vals = [], [], []
    deg = np.zeros(n, np.float64)
    rhs = np.zeros((n, C), np.float64)
    ii = idx[inside]
    valid_any = np.zeros(n, bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb_in = _shift(inside, dy, dx, False)
        nb_idx = _shift(idx, dy, dx, -1)
        nb_exists = _shift(np.ones((h, w), bool), dy, dx, False)     # neighbour inside the box
        nb_val = _shift(bval, dy, dx, 0.0)
        e = nb_exists[inside]
        deg += e
        a_in = nb_in[inside]
        rows.append(ii[a_in])
        cols.append(nb_idx[inside][a_in])
        vals.append(np.full(int(a_in.sum()), -1.0))
        b = e & ~a_in                                       # Dirichlet neighbour (outside the spot, in the box)
        rhs[b] += nb_val[inside][b]
        valid_any |= b
    if not valid_any.any():                                 # no boundary at all: just the mean difference
        out[inside] = bval[~inside].mean(axis=0) if (~inside).any() else 0.0
        return out
    rows.append(ii)
    cols.append(ii)
    vals.append(deg)
    A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    lu = splu(A)
    sol = lu.solve(rhs)
    out[inside] = sol.astype(np.float32)
    return out


def _membrane_coarse(inside: np.ndarray, bval: np.ndarray) -> np.ndarray:
    """The membrane on a grid reduced so it has at most MAX_UNKNOWNS pixels, interpolated back."""
    from scipy.ndimage import zoom
    h, w = inside.shape
    k = int(math.ceil(math.sqrt(inside.sum() / float(MAX_UNKNOWNS)))) + 1
    hs, ws = max(2, h // k), max(2, w // k)
    outside = (~inside).astype(np.float32)
    zy, zx = hs / float(h), ws / float(w)
    o_s = zoom(outside, (zy, zx), order=1)
    num = zoom(bval * outside[..., None], (zy, zx, 1), order=1)
    in_s = o_s < 0.999
    bv_s = num / np.maximum(o_s, 1e-6)[..., None]
    m_s = membrane(in_s, bv_s)
    m_s[~in_s] = bv_s[~in_s]
    m = zoom(m_s, (h / float(hs), w / float(ws), 1), order=1)
    m = m[:h, :w]
    if m.shape[:2] != (h, w):
        m = np.pad(m, ((0, h - m.shape[0]), (0, w - m.shape[1]), (0, 0)), mode="edge")
    out = np.zeros_like(bval, dtype=np.float32)
    out[inside] = m[inside]
    return out


def _log(a):
    return np.log(np.maximum(a, -EPS * 0.9) + EPS)


def _boundary_values(diff: np.ndarray, inside: np.ndarray, sigma: float) -> np.ndarray:
    """Edge differences smoothed along the outside only (normalised convolution), so one noisy pixel on the edge
    does not streak across the spot and nothing from inside the defect leaks in."""
    if sigma <= 0.3:
        return diff
    w = (~inside).astype(np.float32)
    num = np.stack([gaussian_filter(diff[..., c] * w, sigma, mode="nearest") for c in range(diff.shape[2])], -1)
    den = gaussian_filter(w, sigma, mode="nearest")
    return np.where(den[..., None] > 1e-4, num / np.maximum(den, 1e-6)[..., None], diff).astype(np.float32)


def heal(D: np.ndarray, S: np.ndarray, inside: np.ndarray, step: int = 1) -> np.ndarray:
    """The healed patch: source texture, destination lighting and colour at the edges (log-space membrane)."""
    lD, lS = _log(D), _log(S)
    b = _boundary_values(lD - lS, inside, 1.0 / step)
    M = membrane(inside, b)
    H = (np.maximum(S, -EPS * 0.9) + EPS) * np.exp(M) - EPS
    return np.where(inside[..., None], H, D).astype(np.float32)


def redeye(D: np.ndarray) -> np.ndarray:
    r, g, b = D[..., 0], D[..., 1], D[..., 2]
    gb = 0.5 * (g + b)
    red = r / np.maximum(gb, 1e-5)
    w = _smooth((red - 1.5) / 1.0) * _smooth(r / 0.002)
    out = D.copy()
    out[..., 0] = r + w * (gb - r)
    out *= (1.0 - 0.25 * w)[..., None]
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------ one spot
def patch(spot: dict, fetch, sw: int, sh: int, step: int = 1, parity=(0, 0)):
    """Compute one spot. ``fetch(x0, y0, x1, y1)`` returns the photo (scene-linear, before retouching) for that
    full-res S box at this grid's step; boxes passed to it start on the grid (``(x0 - parity[0]) % step == 0``).
    Returns (H, alpha, gx, gy): the retouched values and their blend weight on the grid starting at S (gx, gy), or
    None when the spot is off the frame."""
    shp = Shape(spot, sw, sh, step)
    x0, y0, x1, y1 = shp.box(4.0 * step)
    x0 -= (x0 - parity[0]) % step
    y0 -= (y0 - parity[1]) % step
    x1 += (-(x1 - x0)) % step
    y1 += (-(y1 - y0)) % step
    x1, y1 = min(x1, sw - ((sw - x0) % step)), min(y1, sh - ((sh - y0) % step))
    if x1 - x0 < step or y1 - y0 < step:
        return None
    if shp.kind != "redeye":
        if not shp.source_inside_frame(4.0 * step):
            spot = dict(spot, src=clamp_source(spot, sw, sh))
            shp = Shape(spot, sw, sh, step)
        sx0, sy0, sx1, sy1 = x0 + shp.ox, y0 + shp.oy, x1 + shp.ox, y1 + shp.oy
        if sx0 < 0 or sy0 < 0 or sx1 > sw or sy1 > sh:
            return None
    nh, nw = (y1 - y0) // step, (x1 - x0) // step
    if shp.kind != "redeye":
        ux0, uy0 = min(x0, sx0), min(y0, sy0)
        ux1, uy1 = max(x1, sx1), max(y1, sy1)
        area = (ux1 - ux0) * (uy1 - uy0)
        if area <= 2.5 * 2 * (x1 - x0) * (y1 - y0):           # close together: one read for both
            U = fetch(ux0, uy0, ux1, uy1)
            D = U[(y0 - uy0) // step:(y0 - uy0) // step + nh, (x0 - ux0) // step:(x0 - ux0) // step + nw]
            S = U[(sy0 - uy0) // step:(sy0 - uy0) // step + nh, (sx0 - ux0) // step:(sx0 - ux0) // step + nw]
        else:
            D, S = fetch(x0, y0, x1, y1), fetch(sx0, sy0, sx1, sy1)
        n = min(D.shape[0], S.shape[0]), min(D.shape[1], S.shape[1])
        D, S = D[:n[0], :n[1]], S[:n[0], :n[1]]
    else:
        D = fetch(x0, y0, x1, y1)
    h, w = D.shape[:2]
    c = 0.5 * (step - 1)                                         # half-size pixels sit at the 2x2 cell centre
    X = x0 + step * np.arange(w)[None, :] + c
    Y = y0 + step * np.arange(h)[:, None] + c
    a, inside = shp.alpha(X, Y)
    if not inside.any():
        return None
    if shp.kind == "heal":
        H = heal(D, S, inside, step)
    elif shp.kind == "clone":
        H = S.astype(np.float32)
    else:
        H = redeye(D)
    return H, (a * shp.opacity).astype(np.float32), x0, y0


def paste(base: np.ndarray, bgx: int, bgy: int, step: int, res) -> None:
    """Blend a patch (from ``patch``) into ``base``, an array whose pixel (0, 0) is the cell at S (bgx, bgy)."""
    if res is None:
        return
    H, a, gx, gy = res
    j0, i0 = (gx - bgx) // step, (gy - bgy) // step
    h, w = H.shape[:2]
    bh, bw = base.shape[:2]
    a0, b0 = max(0, i0), max(0, j0)
    a1, b1 = min(bh, i0 + h), min(bw, j0 + w)
    if a1 <= a0 or b1 <= b0:
        return
    sl = (slice(a0, a1), slice(b0, b1))
    ps = (slice(a0 - i0, a1 - i0), slice(b0 - j0, b1 - j0))
    A = a[ps][..., None]
    base[sl] += A * (H[ps] - base[sl])


def overlaps(spot: dict, box, sw: int, sh: int) -> bool:
    """Does the spot touch the S box (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = Shape(spot, sw, sh).box(4.0)
    return x0 < box[2] and x1 > box[0] and y0 < box[3] and y1 > box[1]


# ------------------------------------------------------------------------------------------ automatic source
def features(img: np.ndarray):
    """(log brightness, exposure-proof colour) used to compare places when choosing a source."""
    from ..color.grade import to_oklab
    Y = np.maximum(img @ LUMA, 0.0)
    L = np.log2(Y + EPS).astype(np.float32)
    lab = to_oklab(np.maximum(img, 0.0))
    ch = (lab[..., 1:] / np.maximum(lab[..., :1], 1e-3) * 4.0).astype(np.float32)   # ~1/4 stop per unit
    return L, ch


def best_source(img: np.ndarray | None, gx: int, gy: int, step: int, spot: dict, sw: int, sh: int,
                avoid: list | None = None, feats=None, penalty=None) -> list:
    """Pick where a spot takes its source from: around the spot, the place whose surroundings look most like the
    spot's own surroundings and whose inside is clean (no spot of its own). ``img`` is scene-linear RGB covering
    the neighbourhood, its pixel (0, 0) at S (gx, gy) with grid ``step``. Returns the offset in S fractions."""
    shp = Shape(spot, sw, sh)
    R = max(shp.R, 3.0 * step)                                   # tiny spots: compare at least a few pixels
    L, ch = feats if feats is not None else features(img)
    h, w = L.shape[:2]
    # sample the spot (inside) and a ring around it on a lattice, in array pixels
    rs = R / step
    k = max(1.0, rs / 6.0)                                       # ~12 samples across the spot
    c = 0.5 * (step - 1)
    ax0, ax1 = (shp.px.min() - 1.75 * R - gx - c) / step, (shp.px.max() + 1.75 * R - gx - c) / step
    ay0, ay1 = (shp.py.min() - 1.75 * R - gy - c) / step, (shp.py.max() + 1.75 * R - gy - c) / step
    GX, GY = np.meshgrid(np.arange(ax0, ax1 + 1e-6, k), np.arange(ay0, ay1 + 1e-6, k))
    d = shp.distance(GX * step + gx + c, GY * step + gy + c) / R
    ring = (d > 1.05) & (d < 1.7)
    inner = d < 0.95
    fallback = clamp_source(dict(spot, src=[2.5 * R / sw, 0.0]), sw, sh)
    if ring.sum() < 6 or inner.sum() < 1:
        return fallback
    rx, ry = GX[ring], GY[ring]
    ix, iy = GX[inner], GY[inner]

    def sample(arr, x, y):
        xi = np.clip(np.round(x).astype(int), 0, w - 1)
        yi = np.clip(np.round(y).astype(int), 0, h - 1)
        return arr[yi, xi]

    ringL, ringC = sample(L, rx, ry), sample(ch, rx, ry)
    clone = spot.get("kind") == "clone"
    tex = max(float(np.std(ringL)), 0.02)
    ext = max(np.ptp(shp.px), np.ptp(shp.py)) / step
    avoid = avoid or []
    best, best_off = np.inf, None
    for dist in (2.3, 3.0, 4.0, 5.5, 7.5):
        D = dist * rs + (ext * 0.5 if ext > 0 else 0.0)
        n_ang = max(16, int(2 * math.pi * D / max(rs, 1.0) * 0.8))
        for ang in np.linspace(0.0, 2 * math.pi, n_ang, endpoint=False):
            ox, oy = D * math.cos(ang), D * math.sin(ang)
            if (rx + ox).min() < 0 or (ry + oy).min() < 0 or (rx + ox).max() > w - 1 or (ry + oy).max() > h - 1:
                continue
            if (ix + ox).min() < 0 or (iy + oy).min() < 0 or (ix + ox).max() > w - 1 or (iy + oy).max() > h - 1:
                continue
            cL, cC = sample(L, rx + ox, ry + oy), sample(ch, rx + ox, ry + oy)
            iL = sample(L, ix + ox, iy + oy)
            if clone:
                e_ring = np.mean((cL - ringL) ** 2)
            else:
                e_ring = np.mean(((cL - cL.mean()) - (ringL - ringL.mean())) ** 2)
            e_col = np.mean(np.sum((cC - ringC) ** 2, axis=-1))
            # a clean source looks like its own surroundings: no dark or bright blob, no extra texture inside
            e_in = (iL.mean() - cL.mean()) ** 2 + max(0.0, float(np.std(iL)) - 1.3 * float(np.std(cL))) ** 2
            score = (e_ring + 0.5 * e_col + 2.0 * e_in) / (tex * tex) + 0.08 * dist
            sxp, syp = ox * step, oy * step
            for (ax, ay, ar) in avoid:                         # other spots' areas (full-res S px)
                if np.min(np.hypot(shp.px + sxp - ax, shp.py + syp - ay)) < ar + R:
                    score += 50.0
            if penalty is not None:
                score += penalty(sxp, syp)
            if score < best:
                best, best_off = score, (sxp, syp)
    if best_off is None:                                         # nothing fitted inside the frame
        return fallback
    return [round(best_off[0] / float(sw), 6), round(best_off[1] / float(sh), 6)]


def neighbourhood_box(spot: dict, sw: int, sh: int):
    """S box (x0, y0, x1, y1) that best_source needs around a spot."""
    shp = Shape(spot, sw, sh)
    m = shp.R * 9.5 + 8
    return (max(0, int(shp.px.min() - m)), max(0, int(shp.py.min() - m)), min(sw, int(shp.px.max() + m) + 1),
            min(sh, int(shp.py.max() + m) + 1))

"""Flat Mosaic, the solve (BUILD_PLAN_MOSAIC.md sections 1 and 2): hand-held views over a flat or gently curved
surface, each view's camera pose found, the whole set adjusted at once.

Why this works when a panorama cannot: the panorama (pano.py) assumes the camera turned on one spot. Walking along a
road with the camera pointing down moves the camera, and that is parallax to a panorama. But every view of a PLANE
maps onto every other view through one perspective warp, whatever the camera did (moved, tilted, rolled, rose or
dropped), so a hand-held set of a flat surface is joined exactly. A change of height between views is only a change
of scale inside that warp, so differences in distance are part of the model, never an error.

The model: the ground is z = s(x, y) (s = 0 for **Plane**, a quadratic for **Gentle curve**); every view has a pose
over it (rotation R, camera centre C) and its own lens (focal length, centre: from the View, with focus breathing
already applied); one radial distortion k1 shared by the set. ``p ~ K R (X - C)`` (R: world -> camera). A camera
pointing straight down from height 1 is R = diag(1, -1, -1), C = (0, 0, 1).

Methods (each says what it is best for and weak at, plan 23.1):
  poses   6 numbers per view: physically constrained, drifts least, gives each view's height and tilt.
  free    8 numbers per view (a plane warp each), no lens assumed: for a lens with no focal length.

The solve: corners and rotation-invariant descriptors per view (at two scales, for views shot at different
heights), pairs matched (neighbours in order first, then every pair whose footprints overlap after a first solve),
RANSAC homographies polished by Lucas-Kanade (features.py), a maximum spanning tree, first poses from the chained
warps (Zhang's homography decomposition), then ONE robust least-squares adjustment over every match of every pair,
so loop closures between rows remove drift instead of it building up along a chain. Pairs whose residual stays far
above the rest (something moved, a wrong match) are dropped and the set adjusted again.

Scale: a ruler marked in one view (cyan, measured), else the lenses' recorded focus distances (amber: the lens
reports its distance coarsely), else relative (the JSON says so plainly).

No Qt here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog
from . import features as FE

log = slog.get("engine.stack.mosaic")

DOWN = np.diag([1.0, -1.0, -1.0])        # a camera looking straight down (world -> camera)
MAX_MATCHES = 100                       # matches per pair used in the adjustment (spread over the overlap)
PAIR_DROP_PX = 4.0                       # a pair's residual above this and 3x the median: dropped, adjusted again
PAIR_WARN_PX = 1.5
GAUGE_W = 1e3                            # weight of the gauge (reference view fixed at (0, 0, 1), no yaw)
SCALES = (1.0, 0.7, 0.5)                 # feature scales tried for a pair (views shot up to 2x higher or lower)

METHOD_INFO = {
    "auto": {"name": "Auto", "does": "Camera poses when the lens's focal length is known, plane warps otherwise.",
             "best": "Most sets.", "weak": "Nothing of its own: it picks one of the two below."},
    "poses": {"name": "Camera poses", "does": "Finds where each camera was over the ground: its height, tilt and "
              "turn (6 numbers per view), with the lens known.",
              "best": "Any hand-held set shot with a lens that records its focal length; long strips (drifts least); "
                      "tells you each view's height and tilt.",
              "weak": "A zoom whose focal length changed during the set; a set where no view looks roughly down."},
    "free": {"name": "Plane warps", "does": "A free perspective warp per view (8 numbers), no lens assumed.",
             "best": "A contactless vintage lens with no focal length in the file; scans; phone pictures.",
             "weak": "Long strips drift more; no camera heights, no relief correction (orthorectification)."},
}
SURFACE_INFO = {
    "plane": {"name": "Plane", "does": "The ground is one flat plane.",
              "best": "Floors, walls, short patches of road or path.",
              "weak": "A crowned road or a long lawn over a few metres: the curve shows as misfit."},
    "curve": {"name": "Gentle curve", "does": "The ground is a plane with a gentle bow (a quadratic), solved with "
              "the cameras.",
              "best": "Crowned roads, long lawns, a slightly bowed floor.",
              "weak": "Needs a second row or a walk back to be sure of the bow; a short single row can trade the bow "
                      "against the cameras' tilt."},
}
SCALE_INFO = {
    "auto": {"name": "Auto", "does": "A ruler if you marked one, else the lenses' focus distances, else relative.",
             "best": "Most sets.", "weak": "Nothing of its own."},
    "ruler": {"name": "Ruler", "does": "Two points you marked in one view and the distance between them.",
              "best": "The most exact size (cyan, measured).", "weak": "Needs a tape or ruler in one frame."},
    "focus": {"name": "Focus distance", "does": "How far each lens said it was focused, against each camera's "
              "solved distance to the ground; the median over the set.",
              "best": "No ruler needed; any lens with electrical contacts.",
              "weak": "The lens reports its distance coarsely (amber): a few per cent, more on some lenses."},
    "relative": {"name": "Relative", "does": "No size: the texture's units are the reference camera's height.",
                 "best": "When the size does not matter.", "weak": "The game must be told the size by hand."},
}


class MosaicProblem(ValueError):
    """The views cannot make a mosaic (said plainly on the page)."""


@dataclass
class MosaicParams:
    method: str = "auto"            # auto | poses | free
    surface: str = "plane"          # plane | curve
    resolution: str = "game"        # game (1 mm/px) | fine (0.5 mm/px) | full (the views' own) | custom
    mm_per_px: float = 1.0          # custom resolution
    scale: str = "auto"             # auto | ruler | focus | relative
    ruler: str = ""                 # "VIEW:U1,V1,U2,V2,MM": two points (0..1 of the view's picture) MM apart
    exposure: bool = True           # gains per view and channel from the overlaps
    even_out: bool = True           # with exposure: level the broad brightness across the strip (a passing cloud)
    vignette: bool = True           # take out the lens's vignetting (a texture repeats it as a dark blotch)
    seams: str = "smart"            # smart | straight
    crop: str = "rectangle"         # rectangle | edge
    height: bool = True             # a height mosaic from the views' measured depth (Dual Pixel, focus)
    ortho: bool = True              # with height: each place lifted by its own height (relief parallax removed)
    nudge: bool = True              # small leftover parallax hidden by a local nudge where no height is known
    tileable: bool = True
    tile_method: str = "cut"
    normals_y: str = "up"
    maps: str = ""                  # as Focus Stack's --maps ("" = colour, height, normal, confidence, agreement)
    size: int = 0                   # game square (0 = as made)
    focal_mm: float = 0.0           # typed focal length when no view records one
    leave_out: list = field(default_factory=list)
    pieces: str = ""                # "x0,y0,x1,y1;..." boxes (0..1 of the result) cut out as decals

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MosaicParams":
        return MosaicParams(**{k: v for k, v in (d or {}).items() if k in MosaicParams.__dataclass_fields__})


# ------------------------------------------------------------------------------------------- rotations
def rodrigues(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    t = float(np.linalg.norm(w))
    if t < 1e-12:
        return np.eye(3) + np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    k = w / t
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(t) * Kx + (1 - math.cos(t)) * (Kx @ Kx)


def log_rotation(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R) - 1) / 2.0
    c = min(1.0, max(-1.0, c))
    t = math.acos(c)
    if t < 1e-9:
        return np.zeros(3)
    if abs(math.pi - t) < 1e-6:                       # 180 degrees: from the symmetric part
        M = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.maximum(np.diag(M), 0))
        i = int(np.argmax(k))
        if k[i] > 0:
            k = M[:, i] / k[i]
        return t * k / max(np.linalg.norm(k), 1e-12)
    return t / (2 * math.sin(t)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def yaw_of(R: np.ndarray) -> float:
    """The turn of a camera about the vertical: the direction its x axis points in the ground plane."""
    return math.atan2(R[0, 1], R[0, 0])


def tilt_deg(R: np.ndarray) -> float:
    """How far the camera's axis is from straight down."""
    a = R[2, :]                                       # the optical axis in world coordinates
    return math.degrees(math.acos(min(1.0, max(-1.0, -a[2]))))


# ------------------------------------------------------------------------------------------- the lens
@dataclass
class Lens:
    f: float
    cx: float
    cy: float
    rn: float

    def K(self, fs: float = 1.0) -> np.ndarray:
        return np.array([[self.f * fs, 0, self.cx], [0, self.f * fs, self.cy], [0, 0, 1.0]])


def undistort(px: np.ndarray, L: Lens, k1: float) -> np.ndarray:
    if not k1:
        return px
    u = px[..., 0] - L.cx
    v = px[..., 1] - L.cy
    g = 1.0 + k1 * (u * u + v * v) / (L.rn * L.rn)
    return np.stack([u * g + L.cx, v * g + L.cy], -1)


def distort(px: np.ndarray, L: Lens, k1: float, iters: int = 5) -> np.ndarray:
    if not k1:
        return px
    u = px[..., 0] - L.cx
    v = px[..., 1] - L.cy
    uo, vo = u, v
    for _ in range(iters):
        g = 1.0 + k1 * (uo * uo + vo * vo) / (L.rn * L.rn)
        uo, vo = u / g, v / g
    return np.stack([uo + L.cx, vo + L.cy], -1)


# ------------------------------------------------------------------------------------------- the surface
def surface_z(q, x, y):
    """Height of the ground surface at (x, y): 0 for the plane; q = (a, b, c): a x^2 + b x y + c y^2."""
    if q is None:
        return np.zeros_like(x)
    return q[0] * x * x + q[1] * x * y + q[2] * y * y


def rays(px: np.ndarray, R: np.ndarray, L: Lens, k1: float, fs: float = 1.0) -> np.ndarray:
    """World directions of the rays through pixels (N, 2) -> (N, 3)."""
    p = undistort(px, L, k1)
    d = np.stack([(p[..., 0] - L.cx) / (L.f * fs), (p[..., 1] - L.cy) / (L.f * fs), np.ones(p.shape[:-1])], -1)
    return d @ R                                      # R^T d (rows)


def hit(C: np.ndarray, d: np.ndarray, q=None, h=None, iters: int = 4) -> np.ndarray:
    """Where rays from C along d meet the surface z = s(x, y) (+ h): (N, 3). ``h`` an extra height (scalar or N)."""
    dz = np.where(np.abs(d[..., 2]) < 1e-9, -1e-9, d[..., 2])
    z0 = 0.0 if h is None else h
    lam = (z0 - C[2]) / dz
    if q is None and h is None:
        return C + lam[..., None] * d
    for _ in range(iters if q is not None else 1):
        X = C + lam[..., None] * d
        zs = surface_z(q, X[..., 0], X[..., 1]) + (0.0 if h is None else h)
        lam = (zs - C[2]) / dz
    return C + lam[..., None] * d


def project(X: np.ndarray, R: np.ndarray, C: np.ndarray, L: Lens, k1: float, fs: float = 1.0):
    """World points (..., 3) -> observed pixels (..., 2) and whether they are in front of the camera."""
    Xc = (X - C) @ R.T
    z = Xc[..., 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    p = np.stack([L.f * fs * Xc[..., 0] / zz + L.cx, L.f * fs * Xc[..., 1] / zz + L.cy], -1)
    return distort(p, L, k1), z > 1e-9


def plane_homography(R, C, L: Lens, fs: float = 1.0) -> np.ndarray:
    """Ground (x, y, 1) -> ideal pixel, for the plane z = 0."""
    t = -R @ C
    return L.K(fs) @ np.c_[R[:, 0], R[:, 1], t]


def pose_from_homography(H: np.ndarray, L: Lens) -> tuple:
    """(R, C) from a ground -> pixel homography of the plane z = 0 (Zhang 2000), the camera above the plane."""
    B = np.linalg.inv(L.K()) @ H
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    for sgn in (1.0, -1.0):
        r1, r2, t = sgn * lam * B[:, 0], sgn * lam * B[:, 1], sgn * lam * B[:, 2]
        R = nearest_rotation(np.c_[r1, r2, np.cross(r1, r2)])
        C = -R.T @ t
        if C[2] > 0:
            return R, C
    return R, C


# ------------------------------------------------------------------------------------------- a solution
@dataclass
class Solution:
    method: str                      # poses | free
    surface: str                     # plane | curve
    n: int
    lenses: list                     # Lens per view (working pixels)
    R: list                          # per view (None: not placed); poses only
    C: list
    H: list                          # free: ground -> ideal pixel per view
    q: object = None                 # curve coefficients (a, b, c) or None
    k1: float = 0.0
    fs: float = 1.0                  # focal-length factor solved when the focal length was guessed
    ref: int = 0
    pairs: list = field(default_factory=list)   # {"i","j","inliers","matches","rms","closure"}
    rms: float = 0.0
    problems: dict = field(default_factory=dict)  # view index -> why it was left out
    m_per_unit: float | None = None  # metres per ground unit (None: relative)
    scale_how: str = "relative: no ruler and no focus distance"
    scale_colour: str = "violet"
    scale_spread: float | None = None
    drift: dict = field(default_factory=dict)
    focal_how: str = ""

    @property
    def placed(self) -> list:
        if self.method == "poses":
            return [r is not None for r in self.R]
        return [h is not None for h in self.H]

    # -- the maps every later step uses
    def to_ground(self, i: int, px: np.ndarray) -> np.ndarray:
        """Observed pixels of view i (working size) -> ground (x, y)."""
        L = self.lenses[i]
        px = np.asarray(px, float)
        if self.method == "poses":
            d = rays(px, self.R[i], L, self.k1, self.fs)
            X = hit(self.C[i], d, self.q)
            return X[..., :2]
        p = undistort(px, L, self.k1)
        Hi = np.linalg.inv(self.H[i])
        ph = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], -1) @ Hi.T
        w = np.where(np.abs(ph[..., 2:3]) < 1e-12, 1e-12, ph[..., 2:3])
        return ph[..., :2] / w

    def to_view(self, i: int, xy: np.ndarray, h=None) -> tuple:
        """Ground (x, y) (+ a height above the surface) -> observed pixels of view i and whether they are in front."""
        L = self.lenses[i]
        xy = np.asarray(xy, float)
        if self.method == "poses":
            z = surface_z(self.q, xy[..., 0], xy[..., 1])
            if h is not None:
                z = z + h
            X = np.concatenate([xy, z[..., None]], -1)
            return project(X, self.R[i], self.C[i], L, self.k1, self.fs)
        ph = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1) @ self.H[i].T
        w = ph[..., 2]
        ww = np.where(np.abs(w) < 1e-12, 1e-12, w)
        p = np.stack([ph[..., 0] / ww, ph[..., 1] / ww], -1)
        return distort(p, L, self.k1), w > 0

    def cam_height(self, i: int) -> float | None:
        if self.method != "poses" or self.R[i] is None:
            return None
        return float(self.C[i][2] - surface_z(self.q, self.C[i][0], self.C[i][1]))

    def axis_distance(self, i: int) -> float | None:
        """How far along its own axis view i is from the ground (ground units)."""
        if self.method != "poses" or self.R[i] is None:
            return None
        a = self.R[i][2, :]
        X = hit(self.C[i], a[None, :], self.q)[0]
        return float(np.linalg.norm(X - self.C[i]))

    def gsd(self, i: int, at_scale: float = 1.0) -> float:
        """Ground units per pixel at view i's centre (pixels ``at_scale`` times the working size)."""
        L = self.lenses[i]
        c = np.array([[L.cx, L.cy], [L.cx + 1.0, L.cy], [L.cx, L.cy + 1.0]])
        g = self.to_ground(i, c)
        a = np.linalg.norm(g[1] - g[0])
        b = np.linalg.norm(g[2] - g[0])
        return float(math.sqrt(max(a * b, 1e-30))) / at_scale

    def footprint(self, i: int, shape, n: int = 12) -> np.ndarray:
        """The view's outline on the ground (N, 2); ``shape`` its working (h, w)."""
        h, w = shape
        t = np.linspace(0, 1, n, endpoint=False)
        e = np.concatenate([np.c_[t * (w - 1), np.zeros(n)], np.c_[np.full(n, w - 1.0), t * (h - 1)],
                            np.c_[(1 - t) * (w - 1), np.full(n, h - 1.0)], np.c_[np.zeros(n), (1 - t) * (h - 1)]])
        return self.to_ground(i, e)

    def summary(self) -> dict:
        out = {"method": self.method, "surface": self.surface, "placed": int(sum(self.placed)), "views": self.n,
               "rms_px": round(self.rms, 3), "k1": round(self.k1, 6), "focal_how": self.focal_how,
               "scale": {"metres_per_unit": self.m_per_unit, "how": self.scale_how, "colour": self.scale_colour,
                         "spread": self.scale_spread},
               "drift": self.drift}
        if self.q is not None:
            out["curve"] = [round(float(v), 8) for v in self.q]
        return out


# ------------------------------------------------------------------------------------------- matching
class ViewFeatures:
    """Features of one view: at full scale at once, at the other scales in SCALES when first asked for."""

    def __init__(self, grey: np.ndarray, scales=SCALES):
        self.grey = np.asarray(grey, np.float32)
        self.scales = [s for s in scales if min(self.grey.shape) * s >= 64]
        self._lv = {1.0: FE.FrameFeatures(self.grey, oriented=True)}

    def get(self, s: float):
        if s not in self._lv:
            from scipy.ndimage import zoom
            self._lv[s] = FE.FrameFeatures(zoom(self.grey, s, order=1, grid_mode=True, mode="nearest"),
                                           oriented=True)
        return s, self._lv[s]

    @property
    def levels(self) -> list:
        return [self.get(s) for s in self.scales]


def _level(vf: ViewFeatures, s: float):
    return vf.get(min(vf.scales, key=lambda t: abs(math.log(t / s))))


def match_views(a: ViewFeatures, b: ViewFeatures, rng=None, ratio: float | None = None,
                scales: bool = False) -> dict | None:
    """Match two views; points come back in each view's own (working) pixels.

    ``ratio``: b's ground units per pixel over a's (known after a first solve): the pair is matched once, at the
    scales that make their pixels the same size on the ground. Otherwise at full scale, and with ``scales`` also
    with b at the other scales (views shot at different heights) when that fails or is weak."""
    tries = []
    if ratio is not None and ratio > 0:
        if ratio < 1:                                 # b is finer (nearer): shrink b
            tries.append((1.0, ("b", ratio)))
        else:
            tries.append((("a", 1.0 / ratio), 1.0))
        if abs(math.log(ratio)) > 0.15:
            tries.append((1.0, 1.0))
    else:
        tries.append((1.0, 1.0))
        if scales:                                    # b nearer (shrink b), then a nearer (shrink a)
            tries += [(1.0, s) for s in b.scales if s != 1.0] + [(s, 1.0) for s in a.scales if s != 1.0]
    best = None
    for ta, tb in tries:
        sa, fa = _level(a, ta[1]) if isinstance(ta, tuple) else a.get(ta)
        sb, fb = _level(b, tb[1]) if isinstance(tb, tuple) else b.get(tb)
        m = FE.match_pair(fa, fb, rng=rng)
        if m is not None and (best is None or m["inliers"] > best["inliers"]):
            best = dict(m, pa=(m["pa"] + 0.5) / sa - 0.5, pb=(m["pb"] + 0.5) / sb - 0.5)
        if best is not None and best["inliers"] >= 60:
            break
    if best is None:
        return None
    best["H"] = FE.dlt(best["pa"], best["pb"])
    return best


def _thin(pa: np.ndarray, pb: np.ndarray, k: int, rng) -> tuple:
    if len(pa) <= k:
        return pa, pb
    # spread over the overlap: one point per cell of a coarse grid first, then random
    g = 6
    x0, y0 = pa.min(0)
    x1, y1 = pa.max(0)
    cx = np.clip(((pa[:, 0] - x0) / max(x1 - x0, 1e-6) * g).astype(int), 0, g - 1)
    cy = np.clip(((pa[:, 1] - y0) / max(y1 - y0, 1e-6) * g).astype(int), 0, g - 1)
    cell = cy * g + cx
    order = rng.permutation(len(pa))
    seen, first, rest = set(), [], []
    for t in order:
        (first if cell[t] not in seen else rest).append(t)
        seen.add(cell[t])
    take = np.array((first + rest)[:k])
    return pa[take], pb[take]


def candidate_pairs(n: int, near: int = 4) -> list:
    if n <= 10:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + near + 1))]


# ------------------------------------------------------------------------------------------- the solve
def _tree(n: int, pairs: list) -> tuple:
    adj = {i: [] for i in range(n)}
    deg = np.zeros(n)
    for p in pairs:
        adj[p["i"]].append((p["inliers"], p["j"], p))
        adj[p["j"]].append((p["inliers"], p["i"], p))
        deg[p["i"]] += p["inliers"]
        deg[p["j"]] += p["inliers"]
    root = int(np.argmax(deg))
    import heapq
    seen, tree = {root}, []
    heap = [(-w, root, j, id(p), p) for w, j, p in adj[root]]
    heapq.heapify(heap)
    while heap:
        _, par, c, _, p = heapq.heappop(heap)
        if c in seen:
            continue
        seen.add(c)
        tree.append((par, c, p))
        for w, k, pp in adj[c]:
            if k not in seen:
                heapq.heappush(heap, (-w, c, k, id(pp), pp))
    return root, tree, seen


def _chain(n: int, pairs: list, lenses: list, k1: float) -> tuple:
    """Every view's warp to the root view (ideal pixels): (root, M (view -> root), connected set)."""
    root, tree, seen = _tree(n, pairs)
    M = [None] * n
    M[root] = np.eye(3)
    for par, c, p in tree:
        a = undistort(p["pa"], lenses[p["i"]], k1)
        b = undistort(p["pb"], lenses[p["j"]], k1)
        Hij = FE.dlt(a, b)                            # view j <- view i
        if p["i"] == c:                               # child is i: i -> j=parent -> root
            M[c] = M[par] @ Hij
        else:                                         # child is j: j -> i=parent -> root
            M[c] = M[par] @ np.linalg.inv(Hij)
        M[c] = M[c] / M[c][2, 2]
    return root, M, seen


def initial(n: int, pairs: list, lenses: list, k1: float = 0.0, fs: float = 1.0) -> dict:
    root, M, seen = _chain(n, pairs, lenses, k1)
    Lr = lenses[root]
    Href = plane_homography(DOWN, np.array([0.0, 0.0, 1.0]), Lr, fs)
    R, C, H = [None] * n, [None] * n, [None] * n
    for i in seen:
        Hi = np.linalg.inv(M[i]) @ Href                # ground -> view i
        Hi = Hi / Hi[2, 2]
        H[i] = Hi
        Li = Lens(lenses[i].f * fs, lenses[i].cx, lenses[i].cy, lenses[i].rn)
        R[i], C[i] = pose_from_homography(Hi, Li)
    R[root], C[root] = DOWN.copy(), np.array([0.0, 0.0, 1.0])
    return {"root": root, "R": R, "C": C, "H": H, "seen": seen}


def _pack(method, R, C, H, use, q, k1, fs, solve_q, solve_k1, solve_f) -> np.ndarray:
    x = []
    for i in use:
        if method == "poses":
            x.extend(log_rotation(R[i]))
            x.extend(C[i])
        else:
            x.extend((H[i] / H[i][2, 2]).ravel()[:8])
    if solve_q:
        x.extend(q if q is not None else (0.0, 0.0, 0.0))
    if solve_k1:
        x.append(k1)
    if solve_f:
        x.append(math.log(fs))
    return np.array(x, float)


def _rot_batch(W: np.ndarray) -> np.ndarray:
    """Rodrigues for many rotation vectors (N, 3) -> (N, 3, 3)."""
    t = np.linalg.norm(W, axis=1)
    k = W / np.where(t < 1e-12, 1.0, t)[:, None]
    Kx = np.zeros((len(W), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx[:, 1, 0], Kx[:, 2, 0], Kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(t)[:, None, None], np.cos(t)[:, None, None]
    R = np.eye(3)[None] + s * Kx + (1 - c) * (Kx @ Kx)
    small = t < 1e-12
    if small.any():
        w = W[small]
        R[small] = np.eye(3)[None] + np.stack([np.stack([0 * w[:, 0], -w[:, 2], w[:, 1]], 1),
                                               np.stack([w[:, 2], 0 * w[:, 0], -w[:, 0]], 1),
                                               np.stack([-w[:, 1], w[:, 0], 0 * w[:, 0]], 1)], 1)
    return R


def adjust(method: str, n: int, pairs: list, lenses: list, init: dict, ref: int, solve_q: bool = False,
           solve_k1: bool = False, solve_f: bool = False, k10: float = 0.0, fs0: float = 1.0, q0=None,
           max_nfev: int = 150, loss_scale: float = 1.5, rng=None) -> dict:
    """One robust least-squares adjustment of every view over every pair's matches (symmetric transfer error),
    worked on all matches at once."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    rng = rng if rng is not None else np.random.default_rng(0)
    use = sorted(i for i in range(n) if (init["R"][i] if method == "poses" else init["H"][i]) is not None)
    col = {v: t for t, v in enumerate(use)}
    per = 6 if method == "poses" else 8
    nv = len(use)
    base = nv * per
    iq = base if solve_q else None
    ik = base + (3 if solve_q else 0) if solve_k1 else None
    jf = base + (3 if solve_q else 0) + (1 if solve_k1 else 0) if solve_f else None
    nparam = base + (3 if solve_q else 0) + (1 if solve_k1 else 0) + (1 if solve_f else 0)
    src, dst, ps, pd, kept = [], [], [], [], []
    for p in pairs:
        if p["i"] not in col or p["j"] not in col:
            continue
        a, b = _thin(p["pa"], p["pb"], MAX_MATCHES, rng)
        m = len(a)
        src += [np.full(m, col[p["i"]]), np.full(m, col[p["j"]])]
        dst += [np.full(m, col[p["j"]]), np.full(m, col[p["i"]])]
        ps += [a, b]
        pd += [b, a]
        kept.append(p)
    if not kept:
        raise MosaicProblem("No two views share enough detail to be joined. Do they overlap by about half?")
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    ps = np.concatenate(ps).astype(float)
    pd = np.concatenate(pd).astype(float)
    LF = np.array([lenses[v].f for v in use])
    LX = np.array([lenses[v].cx for v in use])
    LY = np.array([lenses[v].cy for v in use])
    LR = np.array([lenses[v].rn for v in use])
    x0 = _pack(method, init["R"], init["C"], init["H"], use, q0, k10, fs0, solve_q, solve_k1, solve_f)
    if method == "free":
        Href = init["H"][ref] / init["H"][ref][2, 2]

    def unpack(x):
        q = tuple(x[iq:iq + 3]) if solve_q else q0
        k1 = float(x[ik]) if solve_k1 else k10
        fs = math.exp(float(x[jf])) if solve_f else fs0
        blk = x[:base].reshape(nv, per)
        if method == "poses":
            return _rot_batch(blk[:, :3]), blk[:, 3:6], None, q, k1, fs
        return None, None, np.concatenate([blk, np.ones((nv, 1))], 1).reshape(nv, 3, 3), q, k1, fs

    def und(p, v, k1):
        if not k1:
            return p
        u, w = p[:, 0] - LX[v], p[:, 1] - LY[v]
        g = 1.0 + k1 * (u * u + w * w) / (LR[v] ** 2)
        return np.stack([u * g + LX[v], w * g + LY[v]], 1)

    def dis(p, v, k1):
        if not k1:
            return p
        u, w = p[:, 0] - LX[v], p[:, 1] - LY[v]
        uo, wo = u, w
        for _ in range(5):
            g = 1.0 + k1 * (uo * uo + wo * wo) / (LR[v] ** 2)
            uo, wo = u / g, w / g
        return np.stack([uo + LX[v], wo + LY[v]], 1)

    def transfer_all(R, C, H, q, k1, fs, s_, d_, p_):
        p = und(p_, s_, k1)
        if method == "poses":
            f = LF[s_] * fs
            dc = np.stack([(p[:, 0] - LX[s_]) / f, (p[:, 1] - LY[s_]) / f, np.ones(len(p))], 1)
            d = np.einsum("nk,nkj->nj", dc, R[s_])
            X = hit_rows(C[s_], d, q)
            Xc = np.einsum("nij,nj->ni", R[d_], X - C[d_])
            z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
            fd = LF[d_] * fs
            out = np.stack([fd * Xc[:, 0] / z + LX[d_], fd * Xc[:, 1] / z + LY[d_]], 1)
        else:
            Hi = np.linalg.inv(H)
            G = H[d_] @ Hi[s_]
            ph = np.einsum("nij,nj->ni", G, np.c_[p, np.ones(len(p))])
            w = np.where(np.abs(ph[:, 2]) < 1e-12, 1e-12, ph[:, 2])
            out = ph[:, :2] / w[:, None]
        return dis(out, d_, k1)

    cr = col[ref]

    def fun(x):
        R, C, H, q, k1, fs = unpack(x)
        res = [(transfer_all(R, C, H, q, k1, fs, src, dst, ps) - pd).ravel()]
        if method == "poses":
            res.append(GAUGE_W * np.array([C[cr][0], C[cr][1], C[cr][2] - 1.0, yaw_of(R[cr])]))
        else:
            res.append(GAUGE_W * (H[cr] / H[cr][2, 2] - Href).ravel()[:8] /
                       np.maximum(np.abs(Href.ravel()[:8]), 1e-3))
        if solve_q:
            res.append(np.asarray(q) * 1e-2)          # a faint pull toward the plane
        return np.concatenate(res)

    # sparsity: each residual row depends on its source and destination views (+ the shared numbers)
    M = len(src)
    rows_, cols_ = [], []
    r2 = np.repeat(np.arange(2 * M), 1).reshape(M, 2)
    for vv in (src, dst):
        cc = (vv[:, None] * per + np.arange(per)[None, :])                     # (M, per)
        for k in (0, 1):
            rows_.append(np.repeat(r2[:, k], per))
            cols_.append(cc.ravel())
    shared = [c for c in ([iq, iq + 1, iq + 2] if solve_q else []) + ([ik] if solve_k1 else []) +
              ([jf] if solve_f else [])]
    for c in shared:
        rows_.append(np.arange(2 * M))
        cols_.append(np.full(2 * M, c))
    g = 4 if method == "poses" else 8
    gr = 2 * M + np.arange(g)
    rows_.append(np.repeat(gr, per))
    cols_.append(np.tile(cr * per + np.arange(per), g))
    nres = 2 * M + g
    if solve_q:
        rows_.append(nres + np.arange(3))
        cols_.append(np.array([iq, iq + 1, iq + 2]))
        nres += 3
    rr = np.concatenate(rows_)
    cc = np.concatenate(cols_)
    S = coo_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(nres, nparam)).tocsr()
    S.data[:] = 1
    sol = least_squares(fun, x0, jac_sparsity=S, loss="soft_l1", f_scale=loss_scale, x_scale="jac",
                        max_nfev=max_nfev, method="trf")
    R, C, H, q, k1, fs = unpack(sol.x)
    pair_rms = {}
    allr = []
    for p in kept:
        i, j = col[p["i"]], col[p["j"]]
        a, b = np.asarray(p["pa"], float), np.asarray(p["pb"], float)
        e1 = transfer_all(R, C, H, q, k1, fs, np.full(len(a), i), np.full(len(a), j), a) - b
        e2 = transfer_all(R, C, H, q, k1, fs, np.full(len(b), j), np.full(len(b), i), b) - a
        e = np.r_[np.hypot(e1[:, 0], e1[:, 1]), np.hypot(e2[:, 0], e2[:, 1])]
        pair_rms[(p["i"], p["j"])] = float(np.sqrt(np.mean(e * e)))
        allr.append(e)
    rms = float(np.sqrt(np.mean(np.concatenate(allr) ** 2))) if allr else 0.0
    Ro, Co, Ho = [None] * n, [None] * n, [None] * n
    for t, v in enumerate(use):
        if method == "poses":
            Ro[v], Co[v] = R[t], np.asarray(C[t], float)
            Ho[v] = plane_homography(R[t], Co[v], lenses[v], fs)
        else:
            Ho[v] = H[t]
    return {"R": Ro, "C": Co, "H": Ho, "q": q if solve_q else None, "k1": k1, "fs": fs, "rms": rms,
            "pair_rms": pair_rms, "cost": float(sol.cost)}


def hit_rows(C: np.ndarray, d: np.ndarray, q=None, iters: int = 4) -> np.ndarray:
    """``hit`` with one camera centre per row (N, 3)."""
    dz = np.where(np.abs(d[:, 2]) < 1e-9, -1e-9, d[:, 2])
    lam = -C[:, 2] / dz
    if q is None:
        return C + lam[:, None] * d
    for _ in range(iters):
        X = C + lam[:, None] * d
        lam = (surface_z(q, X[:, 0], X[:, 1]) - C[:, 2]) / dz
    return C + lam[:, None] * d


def _overlap_share(fa: np.ndarray, fb: np.ndarray) -> float:
    """Rough share of the smaller footprint that the two footprints share (their boxes)."""
    ax0, ay0 = fa.min(0)
    ax1, ay1 = fa.max(0)
    bx0, by0 = fb.min(0)
    bx1, by1 = fb.max(0)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    sa = (ax1 - ax0) * (ay1 - ay0)
    sb = (bx1 - bx0) * (by1 - by0)
    return float(ix * iy / max(min(sa, sb), 1e-12))


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
        inside ^= cond
        j = i
    return inside


def poly_overlap(pa: np.ndarray, pb: np.ndarray, grid: int = 48) -> float:
    x0, y0 = np.minimum(pa.min(0), pb.min(0))
    x1, y1 = np.maximum(pa.max(0), pb.max(0))
    xs = np.linspace(x0, x1, grid)
    ys = np.linspace(y0, y1, grid)
    X, Y = np.meshgrid(xs, ys)
    pts = np.c_[X.ravel(), Y.ravel()]
    ia, ib = _inside(pa, pts), _inside(pb, pts)
    small = min(ia.sum(), ib.sum())
    return float((ia & ib).sum() / max(small, 1))


def footprint_overlaps(sol: Solution, shapes: list) -> dict:
    fps = {i: sol.footprint(i, shapes[i]) for i in range(sol.n) if sol.placed[i]}
    out = {}
    keys = sorted(fps)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = keys[a], keys[b]
            if _overlap_share(fps[i], fps[j]) > 0.02:
                s = poly_overlap(fps[i], fps[j])
                if s > 0:
                    out[(i, j)] = s
    return out


def solve(views: list, params: MosaicParams | None = None, progress=None, cancel=None) -> Solution:
    """Poses (or plane warps) for every view. ``views``: scan.views.View (grey ``image``, ``f``, ``cx``, ``cy``)."""
    from .focus import Cancelled
    params = params or MosaicParams()
    say = progress or (lambda *_: None)
    n = len(views)
    if n < 2:
        raise MosaicProblem("A mosaic needs at least two views.")
    lenses = []
    guessed = []
    for v in views:
        h, w = v.shape
        f = float(v.f)
        if params.focal_mm and str(getattr(v, "f_from", "")).startswith("guessed"):
            from ..scan.views import focal_px
            fp = focal_px(params.focal_mm, str(v.meta.get("model") or ""), w, w)
            if fp:
                f = fp
        lenses.append(Lens(f, float(v.cx), float(v.cy), math.hypot(v.cx, v.cy)))
        guessed.append(str(getattr(v, "f_from", "")).startswith("guessed") and not params.focal_mm)
    method = params.method
    if method not in ("poses", "free"):
        method = "free" if all(guessed) else "poses"
    solve_f = method == "poses" and all(guessed)
    focal_how = ("not used (plane warps need no lens)" if method == "free" else
                 "guessed and found by the solve (no focal length in the files)" if solve_f else
                 "from the files (with focus breathing from the recorded distance where known)")
    feats = []
    for i, v in enumerate(views):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        say(0.05 + 0.3 * i / n, f"Finding points in view {i + 1} of {n}")
        feats.append(ViewFeatures(v.image))
    rng = np.random.default_rng(0)
    tried, pairs = set(), []

    def try_pairs(cands, lo, hi, ratios=None, scales=False):
        for t, (i, j) in enumerate(cands):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if (i, j) in tried:
                continue
            tried.add((i, j))
            say(lo + (hi - lo) * t / max(len(cands), 1), f"Matching views {i + 1} and {j + 1}")
            m = match_views(feats[i], feats[j], rng=rng, ratio=(ratios or {}).get((i, j)), scales=scales)
            if m is not None:
                m.update(i=i, j=j)
                pairs.append(m)

    try_pairs(candidate_pairs(n), 0.35, 0.5)
    # neighbours in order that did not match at one scale: try the others (shot at different heights)
    got = {(p["i"], p["j"]) for p in pairs}
    retry = [(i, i + 1) for i in range(n - 1) if (i, i + 1) not in got]
    if retry:
        for k in retry:
            tried.discard(k)
        try_pairs(retry, 0.5, 0.55, scales=True)
    if not pairs:
        raise MosaicProblem("No two views share enough detail to be joined. Do they overlap by about half?")
    surface = params.surface if params.surface in ("plane", "curve") else "plane"
    k1, fs = 0.0, 1.0
    say(0.58, "Solving where each camera was")
    init = initial(n, pairs, lenses, k1, fs)
    ref = init["root"]
    sol = adjust(method, n, pairs, lenses, init, ref, solve_f=solve_f, rng=rng)
    # second pass: every pair whose footprints overlap after the first solve
    tmp = Solution(method, surface, n, lenses, sol["R"], sol["C"], sol["H"], None, sol["k1"], sol["fs"], ref)
    shapes = [v.shape for v in views]
    ov = footprint_overlaps(tmp, shapes)
    more = [(i, j) for (i, j), s in sorted(ov.items()) if s > 0.12 and (i, j) not in tried]
    if more:
        ratios = {(i, j): tmp.gsd(j) / max(tmp.gsd(i), 1e-30) for i, j in more}
        try_pairs(more, 0.62, 0.75, ratios=ratios)
        init2 = dict(init, R=sol["R"], C=sol["C"], H=sol["H"])
        sol = adjust(method, n, pairs, lenses, init2, ref, solve_f=solve_f, fs0=sol["fs"], rng=rng)
    if len(pairs) >= 3 and method == "poses":         # one shared distortion (vintage lenses have no profile)
        sk = adjust(method, n, pairs, lenses, dict(init, R=sol["R"], C=sol["C"], H=sol["H"]), ref,
                    solve_k1=True, solve_f=solve_f, fs0=sol["fs"], rng=rng)
        if sk["rms"] < 0.97 * sol["rms"] and abs(sk["k1"]) < 0.2:
            sol = sk
    # drop pairs whose residual stays far above the rest, then adjust again
    for _ in range(2):
        pr = sol["pair_rms"]
        if len(pr) < 3:
            break
        med = float(np.median(list(pr.values())))
        bad = {k for k, v in pr.items() if v > max(PAIR_DROP_PX, 3 * med)}
        if not bad:
            break
        keep = [p for p in pairs if (p["i"], p["j"]) not in bad]
        if not keep:
            break
        log.info("mosaic: %d pairs dropped (residual above %.1f px)", len(bad), max(PAIR_DROP_PX, 3 * med))
        pairs = keep
        init2 = dict(init, R=sol["R"], C=sol["C"], H=sol["H"])
        sol = adjust(method, n, pairs, lenses, init2, ref, solve_k1=bool(sol["k1"]), k10=sol["k1"],
                     solve_f=solve_f, fs0=sol["fs"], rng=rng)
    if surface == "curve" and method == "poses":
        say(0.85, "Solving the bow of the ground")
        init2 = dict(init, R=sol["R"], C=sol["C"], H=sol["H"])
        sc = adjust(method, n, pairs, lenses, init2, ref, solve_q=True, solve_k1=bool(sol["k1"]), k10=sol["k1"],
                    solve_f=solve_f, fs0=sol["fs"], rng=rng)
        sol = sc
    elif surface == "curve":
        surface = "plane"                                # plane warps cannot carry a bow
    # which views are tied in at all
    _, _, seen = _tree(n, pairs)
    out = Solution(method, surface, n, lenses, sol["R"] if method == "poses" else [None] * n,
                   sol["C"] if method == "poses" else [None] * n, sol["H"], sol["q"], sol["k1"], sol["fs"], ref,
                   focal_how=focal_how)
    for i in range(n):
        if i not in seen:
            if method == "poses":
                out.R[i] = out.C[i] = None
            out.H[i] = None
            out.problems[i] = "no overlap found with the other views"
    in_tree = {(min(a, b), max(a, b)) for a, b, _ in _tree(n, pairs)[1]}
    out.pairs = [{"i": p["i"], "j": p["j"], "inliers": p["inliers"], "matches": p["matches"],
                  "rms": round(sol["pair_rms"].get((p["i"], p["j"]), 0.0), 3),
                  "closure": (p["i"], p["j"]) not in in_tree} for p in pairs]
    out.rms = sol["rms"]
    if method == "free":
        rectify_free(out, shapes)
    out.drift = drift_report(out, views)
    say(0.9, "Working out the size")
    set_scale(out, views, params)
    log.info("mosaic solve: %s/%s, %d of %d views placed, rms %.2f px, k1 %.4f, scale %s",
             method, surface, sum(out.placed), n, out.rms, out.k1, out.scale_how)
    return out


def rectify_free(sol: Solution, shapes: list) -> np.ndarray:
    """Plane warps have no lens, so their ground is the reference view's own picture: if that view was tilted, the
    whole layout is keystoned. On average a hand-held view looks straight down, so the ground is taken as the
    perspective (and shear) that makes every view's own pixels as square as possible on it: least squares over the
    views of how far each view's local warp is from a similarity. Applied to the solution in place; returns the
    correction P (ground -> corrected ground)."""
    from scipy.optimize import least_squares
    pts = []
    for i in range(sol.n):
        if not sol.placed[i]:
            continue
        h, w = shapes[i]
        for fx, fy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
            c = np.array([[fx * (w - 1), fy * (h - 1)], [fx * (w - 1) + 1, fy * (h - 1)],
                          [fx * (w - 1), fy * (h - 1) + 1]])
            pts.append(sol.to_ground(i, c))
    if len(pts) < 6:
        return np.eye(3)
    G = np.array(pts)                                  # (K, 3, 2)
    sc = float(np.median(np.linalg.norm(G[:, 1] - G[:, 0], axis=1)))
    ctr = G[:, 0].mean(0)

    def P_of(x):
        a, b, c, p1, p2 = x
        T = np.array([[1, 0, -ctr[0]], [0, 1, -ctr[1]], [0, 0, 1.0]])
        M = np.array([[1 + a, b, 0], [c, 1 - a, 0], [p1 / sc / 1e3, p2 / sc / 1e3, 1.0]])
        return np.linalg.inv(T) @ M @ T

    def fun(x):
        P = P_of(x)
        g = np.concatenate([G, np.ones(G.shape[:2] + (1,))], -1) @ P.T
        g = g[..., :2] / g[..., 2:3]
        u = g[:, 1] - g[:, 0]
        v = g[:, 2] - g[:, 0]
        n = 0.5 * (np.linalg.norm(u, axis=1) + np.linalg.norm(v, axis=1))
        # a similarity: v is u turned a quarter (image y down, ground y up: v = (u_y, -u_x))
        return np.r_[((v[:, 0] - u[:, 1]) / n), ((v[:, 1] + u[:, 0]) / n), 1e-3 * np.asarray(x)]

    r = least_squares(fun, np.zeros(5), method="lm")
    P = P_of(r.x)
    Pi = np.linalg.inv(P)
    for i in range(sol.n):
        if sol.H[i] is not None:
            sol.H[i] = sol.H[i] @ Pi
            sol.H[i] = sol.H[i] / sol.H[i][2, 2]
    return P


def drift_report(sol: Solution, views: list) -> dict:
    """Whether drift could be checked: loop closures (pairs not in the spanning tree) and their residual."""
    clos = [p for p in sol.pairs if p["closure"]]
    seq = all(abs(p["i"] - p["j"]) == 1 for p in clos) if clos else True
    n = int(sum(sol.placed))
    if clos and not seq:
        r = float(np.sqrt(np.mean([p["rms"] ** 2 for p in clos])))
        return {"checked": True, "closures": len(clos), "closure_rms_px": round(r, 3),
                "text": f"Drift checked: {len(clos)} loop closures (second row or walk back), {r:.2f} px."}
    # a single row: drift grows like a random walk of the pair errors along the chain
    pr = [p["rms"] for p in sol.pairs] or [0.0]
    est_px = float(np.median(pr)) * math.sqrt(max(n - 1, 1))
    return {"checked": False, "closures": len(clos), "estimate_px": round(est_px, 2),
            "text": "This set has no second row or walk back, so drift could not be checked "
                    f"(estimated about {est_px:.1f} px over the strip)."}


# ------------------------------------------------------------------------------------------- panorama or mosaic?
MOVED_SPREAD = 0.04              # a pair's warp this far from a pure turn of the camera: the camera moved


def moved_share(pairs: list, f: float, cx: float, cy: float) -> float:
    """Share of matched pairs whose warp is not a pure turn of the camera. A camera that only turned (a panorama)
    joins its frames by K R K^-1, whose singular values (scaled) are all 1; a camera that moved over a surface joins
    them by R + t n^T / d, whose spread grows with how far it moved (half a metre at 1.3 m: about 0.4).
    ``pairs``: dicts with "H" (b <- a) in pixels with that centre and focal length."""
    if not pairs or f <= 0:
        return 0.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    moved = 0
    for p in pairs:
        M = Ki @ np.asarray(p["H"], float) @ K
        sv = np.linalg.svd(M, compute_uv=False)
        sv = sv / np.cbrt(max(np.prod(sv), 1e-30))
        if float(sv.max() - sv.min()) > MOVED_SPREAD:
            moved += 1
    return moved / len(pairs)


def kind_of(greys: list, f: float | None = None) -> dict:
    """Auto (plan 5.1): "mosaic" when the camera moved between frames (most pairs are not pure turns), else
    "panorama". ``greys``: small grey copies of the frames, all one size; ``f`` the focal length in their pixels
    (None: a normal lens is assumed, 1.2 x the long side)."""
    h, w = greys[0].shape
    ff = f or 1.2 * max(h, w)
    feats = [FE.FrameFeatures(np.asarray(g, np.float32), oriented=True) for g in greys]
    pairs = []
    rng = np.random.default_rng(0)
    for i in range(len(feats) - 1):
        m = FE.match_pair(feats[i], feats[i + 1], rng=rng)
        if m is not None:
            pairs.append(m)
    share = moved_share(pairs, ff, (w - 1) / 2.0, (h - 1) / 2.0)
    kind = "mosaic" if pairs and share >= 0.5 else "panorama"
    text = ("These were shot while moving over a surface; using Mosaic." if kind == "mosaic" else
            "These were shot turning the camera on one spot; using Panorama.")
    if not pairs:
        text = "No two neighbouring frames matched, so the kind could not be told: using Panorama."
    return {"kind": kind, "moved_share": round(share, 3), "pairs": len(pairs), "text": text}


# ------------------------------------------------------------------------------------------- the size
def parse_ruler(text: str):
    """'VIEW:U1,V1,U2,V2,MM' -> (view index or name, (u1, v1), (u2, v2), mm). ValueError for anything else."""
    t = str(text or "").strip()
    if not t:
        return None
    if ":" not in t:
        raise ValueError("a ruler is VIEW:U1,V1,U2,V2,MM (two points 0..1 of the view's picture and the distance "
                         "between them in mm)")
    v, rest = t.split(":", 1)
    nums = [float(x) for x in rest.replace(" ", "").split(",")]
    if len(nums) != 5 or nums[4] <= 0 or not all(0 <= x <= 1 for x in nums[:4]):
        raise ValueError("a ruler is VIEW:U1,V1,U2,V2,MM with U and V between 0 and 1 and MM above 0")
    vv = v.strip()
    return (int(vv) if vv.isdigit() else vv), (nums[0], nums[1]), (nums[2], nums[3]), nums[4]


def set_scale(sol: Solution, views: list, params: MosaicParams) -> None:
    mode = params.scale
    if mode in ("auto", "ruler") and params.ruler:
        r = parse_ruler(params.ruler)
        vi = r[0]
        if not isinstance(vi, int):
            names = [v.name for v in views]
            vi = names.index(vi) if vi in names else -1
        if 0 <= vi < sol.n and sol.placed[vi]:
            h, w = views[vi].shape
            px = np.array([[r[1][0] * (w - 1), r[1][1] * (h - 1)], [r[2][0] * (w - 1), r[2][1] * (h - 1)]])
            g = sol.to_ground(vi, px)
            d = float(np.linalg.norm(g[1] - g[0]))
            if d > 0:
                sol.m_per_unit = r[3] / 1000.0 / d
                sol.scale_how = f"measured: a ruler marked in {views[vi].name}, {r[3]:g} mm"
                sol.scale_colour = "cyan"
                sol.scale_spread = None
                return
        if mode == "ruler":
            raise MosaicProblem("The ruler's view was not placed in the mosaic.")
    if mode in ("auto", "focus") and sol.method == "poses":
        ratios = []
        for i, v in enumerate(views):
            d = (v.meta or {}).get("focus_distance_m")
            t = sol.axis_distance(i)
            if d and t and float(d) > 0:
                ratios.append(float(d) / t)
        if ratios:
            r = np.array(ratios)
            med = float(np.median(r))
            spread = float(1.4826 * np.median(np.abs(r - med)) / med) if len(r) > 2 else None
            sol.m_per_unit = med
            sol.scale_spread = round(spread, 4) if spread is not None else None
            sp = f", spread {100 * spread:.0f}% between views" if spread is not None else ""
            sol.scale_how = (f"the focus distance {len(r)} view{'s' * (len(r) != 1)} recorded, against each camera's "
                             f"solved distance to the ground (median{sp}; the lens reports its distance coarsely)")
            sol.scale_colour = "amber"
            return
        if mode == "focus":
            raise MosaicProblem("No view records a focus distance (a lens with electrical contacts does).")
    sol.m_per_unit = None
    sol.scale_colour = "violet"
    sol.scale_spread = None
    sol.scale_how = ("relative: no ruler and no focus distance (the units are the reference camera's height)"
                     if sol.method == "poses" else "relative: plane warps (the units are the reference view's pixels)")

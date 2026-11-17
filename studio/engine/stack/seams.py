"""Seams for the panorama (BUILD_PLAN_PANO_IR.md 2.6): which frame each place of the canvas comes from.

1. **Voronoi start**: each canvas pixel goes to the valid frame whose own centre it is nearest (distance measured in
   that frame's pixels, so the least distorted, least vignetted part of every frame is used).
2. **Smart seams**: every boundary between two frames is re-cut through their overlap by dynamic programming, the
   whole boundary at once (Efros and Freeman 2001's image-quilting cut, extended along the boundary; a graph cut would
   need a max-flow library the Studio does not have, and this gives most of the benefit). The seam runs where the two
   frames agree (difference of log brightness and of gradients), so it goes around moving people, water and leaves.
   Each side of the cut also carries a per-pixel cost, so the cut prefers, area by area:
   * the frame that is **not clipped** where the other is (overexposed frames);
   * with the **flare guard**, the darker frame where one is brighter in a soft patch: flare and the infrared hot spot
     only add light and move with the camera, so the darker frame is the one without them there;
   * a little, the frame whose centre is nearer.
3. **Ghosts**: where frames still disagree after alignment and exposure matching (shown amber, so the person sees what
   was cut).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, uniform_filter

from ... import log as slog

log = slog.get("engine.stack.seams")

FLARE_WEIGHT = 6.0
CLIP_WEIGHT = 8.0
CENTRE_WEIGHT = 0.05
EDGE_MARGIN = 0.035     # a seam keeps this share of the frame's size away from a frame's edge: the edge pixels are
                        # the least trustworthy (demosaic border, the strongest vignetting correction), and a seam on
                        # the edge leaves the blend no room on one side, so a step or a thin line shows there
GHOST_LOG = 0.18                     # disagreement (log brightness) that counts as a ghost


class Layer:
    """One frame warped into the (seam-scale) canvas: its box, pixels, validity, centre distance, clipping."""

    def __init__(self, idx, box, rgb, valid, rdist, clipped):
        self.i = idx
        self.box = box                   # (y0, y1, x0, x1) in canvas pixels
        self.rgb = rgb
        self.valid = valid
        self.r = rdist
        self.clipped = clipped

    def full(self, shape, what: str, fill=0):
        a = getattr(self, what)
        out = np.full(shape[:2] + a.shape[2:], fill, a.dtype)
        y0, y1, x0, x1 = self.box
        out[y0:y1, x0:x1] = a
        return out


def lum(rgb):
    return (0.2880 * rgb[..., 0] + 0.7119 * rgb[..., 1] + 0.0001 * rgb[..., 2]).astype(np.float32)


def voronoi(layers: list, shape) -> np.ndarray:
    """Labels (int16, -1 where no frame) from the nearest frame centre."""
    best = np.full(shape[:2], np.inf, np.float32)
    lab = np.full(shape[:2], -1, np.int16)
    for L in layers:
        y0, y1, x0, x1 = L.box
        r = np.where(L.valid, L.r, np.inf).astype(np.float32)
        take = r < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1] = np.where(take, r, best[y0:y1, x0:x1])
        lab[y0:y1, x0:x1] = np.where(take, L.i, lab[y0:y1, x0:x1])
    return lab


def _region(La: Layer, Lb: Layer):
    """The intersection box of two layers (canvas coordinates) or None."""
    y0 = max(La.box[0], Lb.box[0])
    y1 = min(La.box[1], Lb.box[1])
    x0 = max(La.box[2], Lb.box[2])
    x1 = min(La.box[3], Lb.box[3])
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    return y0, y1, x0, x1


def _crop(L: Layer, box, what):
    y0, y1, x0, x1 = box
    by, _, bx, _ = L.box
    return getattr(L, what)[y0 - by:y1 - by, x0 - bx:x1 - bx]


def pair_costs(A: Layer, B: Layer, box, flare_guard: bool = True):
    """Edge cost of cutting between A and B at each pixel, and the per-pixel cost of giving it to A or to B."""
    ra, rb = _crop(A, box, "rgb").astype(np.float32), _crop(B, box, "rgb").astype(np.float32)
    la = np.log(np.maximum(lum(ra), 1e-4))
    lb = np.log(np.maximum(lum(rb), 1e-4))
    d = np.abs(la - lb)
    gya, gxa = np.gradient(la)
    gyb, gxb = np.gradient(lb)
    edge = d + 0.5 * (np.abs(gxa - gxb) + np.abs(gya - gyb))
    edge = uniform_filter(edge, 3)
    ca = CENTRE_WEIGHT * _crop(A, box, "r").astype(np.float32)
    cb = CENTRE_WEIGHT * _crop(B, box, "r").astype(np.float32)
    clip_a = _crop(A, box, "clipped")
    clip_b = _crop(B, box, "clipped")
    ca += CLIP_WEIGHT * (clip_a & ~clip_b)
    cb += CLIP_WEIGHT * (clip_b & ~clip_a)
    if flare_guard:
        diff = gaussian_filter(la - lb, 3.0)          # soft, broad brightening = added light
        ca += FLARE_WEIGHT * np.maximum(diff - 0.08, 0.0)
        cb += FLARE_WEIGHT * np.maximum(-diff - 0.08, 0.0)
    return edge, ca, cb, d


def _dp_cut(edge, ca, cb, U, step: int = 2, C=None):
    """Vertical cut: for each row y a column s(y); pixels of U left of it go to "first", right of it to "second".
    Minimises sum of edge cost at the cut + the area costs, with |s(y) - s(y-1)| <= step. Rows without U are free.
    ``C``: where the cut may run (default U); rows where C is empty fall back to U."""
    H, W = edge.shape
    big = np.float32(1e6)
    if C is None:
        C = U
    else:
        C = np.where(C.any(axis=1)[:, None], C & U, U)
    e = np.where(C, edge, big).astype(np.float32)
    caU = np.where(U, ca, 0).astype(np.float32)
    cbU = np.where(U, cb, 0).astype(np.float32)
    left = np.concatenate([np.zeros((H, 1), np.float32), np.cumsum(caU, axis=1)], axis=1)   # A on [0, s)
    right = np.cumsum(cbU[:, ::-1], axis=1)[:, ::-1]                                          # B on [s, W)
    right = np.concatenate([right, np.zeros((H, 1), np.float32)], axis=1)
    data = left + right                                   # (H, W+1): cut before column s
    cut_e = np.concatenate([e, np.full((H, 1), big, np.float32)], axis=1)     # a cut must fall inside the overlap
    cost = data + cut_e
    has = U.any(axis=1)
    M = np.empty_like(cost)
    back = np.zeros(cost.shape, np.int8)
    M[0] = cost[0] if has[0] else data[0] * 0
    for y in range(1, H):
        prev = M[y - 1]
        cand = np.stack([np.roll(prev, k) if k else prev for k in range(-step, step + 1)])
        for t, k in enumerate(range(-step, step + 1)):
            if k > 0:
                cand[t, :k] = np.inf
            elif k < 0:
                cand[t, k:] = np.inf
        a = np.argmin(cand, axis=0)
        M[y] = cand[a, np.arange(cand.shape[1])] + (cost[y] if has[y] else 0)
        back[y] = (a - step).astype(np.int8)
    s = np.empty(H, np.int64)
    s[-1] = int(np.argmin(M[-1]))
    for y in range(H - 1, 0, -1):
        s[y - 1] = s[y] - back[y, s[y]]
    return s


def core_mask(L: Layer) -> np.ndarray:
    """The frame's valid area less a margin at its edges (EDGE_MARGIN of its size): where a seam may run."""
    v = L.valid
    n = int(v.sum())
    if n == 0:
        return v
    m = max(2, int(round(EDGE_MARGIN * math.sqrt(n))))
    return binary_erosion(v, iterations=m, border_value=0)


def refine(labels: np.ndarray, layers: list, flare_guard: bool = True, order=None) -> np.ndarray:
    """Re-cut every boundary between two frames by dynamic programming (module text). Returns new labels."""
    lab = labels.copy()
    by = {L.i: L for L in layers}
    core = {L.i: core_mask(L) for L in layers}
    pairs = []
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            A, B = layers[a], layers[b]
            box = _region(A, B)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            sub = lab[y0:y1, x0:x1]
            va, vb = _crop(A, box, "valid"), _crop(B, box, "valid")
            both = va & vb & ((sub == A.i) | (sub == B.i))
            if both.sum() < 20:
                continue
            pairs.append((int(both.sum()), A.i, B.i, box))
    pairs.sort(reverse=True)
    for _, ia, ib, box in pairs:
        A, B = by[ia], by[ib]
        y0, y1, x0, x1 = box
        sub = lab[y0:y1, x0:x1]
        va, vb = _crop(A, box, "valid"), _crop(B, box, "valid")
        U = va & vb & ((sub == ia) | (sub == ib))
        if U.sum() < 20:
            continue
        # which way the boundary runs: a cut down the rows for a side-by-side pair, across the columns otherwise
        ya, xa = np.nonzero(sub == ia)
        yb, xb = np.nonzero(sub == ib)
        if len(ya) == 0 or len(yb) == 0:
            continue
        dx = abs(xa.mean() - xb.mean())
        dy = abs(ya.mean() - yb.mean())
        edge, ca, cb, _ = pair_costs(A, B, box, flare_guard)
        by_, bx_ = A.box[0], A.box[2]
        ca_core = core[ia][y0 - by_:y1 - by_, x0 - bx_:x1 - bx_]
        by_, bx_ = B.box[0], B.box[2]
        cb_core = core[ib][y0 - by_:y1 - by_, x0 - bx_:x1 - bx_]
        C = U & ca_core & cb_core
        if dx >= dy:
            first_a = xa.mean() < xb.mean()
            s = _dp_cut(edge, ca if first_a else cb, cb if first_a else ca, U, C=C)
            cols = np.arange(U.shape[1])[None, :]
            left = cols < s[:, None]
        else:
            first_a = ya.mean() < yb.mean()
            s = _dp_cut(edge.T, (ca if first_a else cb).T, (cb if first_a else ca).T, U.T, C=C.T)
            rows = np.arange(U.shape[0])[:, None]
            left = rows < s[None, :]
        first, second = (ia, ib) if first_a else (ib, ia)
        new = np.where(left, first, second).astype(np.int16)
        sub[U] = new[U]
    return lab


def ghosts(layers: list, shape, labels: np.ndarray | None = None) -> np.ndarray:
    """0..1: where two frames still disagree after alignment and exposure matching (amber on the page)."""
    g = np.zeros(shape[:2], np.float32)
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            A, B = layers[a], layers[b]
            box = _region(A, B)
            if box is None:
                continue
            va, vb = _crop(A, box, "valid"), _crop(B, box, "valid")
            both = va & vb
            if both.sum() < 20:
                continue
            la = np.log(np.maximum(lum(_crop(A, box, "rgb").astype(np.float32)), 1e-4))
            lb = np.log(np.maximum(lum(_crop(B, box, "rgb").astype(np.float32)), 1e-4))
            d = gaussian_filter(np.abs(la - lb), 1.0)
            v = np.clip((d - GHOST_LOG) / GHOST_LOG, 0, 1) * both
            y0, y1, x0, x1 = box
            g[y0:y1, x0:x1] = np.maximum(g[y0:y1, x0:x1], v)
    return g


def largest_rectangle(mask: np.ndarray) -> tuple:
    """(y0, y1, x0, x1) of the largest axis-aligned rectangle of True in a binary mask (histogram stack, O(HW))."""
    H, W = mask.shape
    heights = np.zeros(W, np.int64)
    best = (0, 0, 0, 0, 0)
    for y in range(H):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []
        for x in range(W + 1):
            h = heights[x] if x < W else 0
            start = x
            while stack and stack[-1][1] >= h:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, y - sh + 1, y + 1, sx, x)
                start = sx
            stack.append((start, h))
    _, y0, y1, x0, x1 = best
    return y0, y1, x0, x1

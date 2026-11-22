"""The model's texture (BUILD_PLAN_DP3D E.8): each face takes its colour from the view that sees it most squarely
and nearest (the stacks are all in focus, so sharpness is mostly the view angle), checked for being in sight by
rasterising the mesh into every view. Faces that take the same view form charts; each chart is cut from that view's
picture and packed into an atlas (power of two, at most ``max_size``); a colour gain per chart is solved over the
seams (outdoor light changes between stacks) and the empty texels are bled. A second texture says how each texel was
made: cyan measured, violet filled (faces of the hull no view measured).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import raster as RA

CYAN = (0x22, 0xD3, 0xEE)
VIOLET = (0xC0, 0x84, 0xFC)


@dataclass
class Textured:
    V: np.ndarray            # (n, 3) world (vertices split along chart edges)
    T: np.ndarray            # (m, 3)
    uv: np.ndarray           # (n, 2) glTF convention (0, 0) = top left of the atlas
    atlas: np.ndarray        # uint8 (S, S, 3) sRGB
    made: np.ndarray         # uint8 (S, S, 3): cyan measured, violet filled
    source: np.ndarray       # (n,) original vertex index of each split vertex
    face_view: np.ndarray    # (m,) view index per face (-1: none)
    measured_face: np.ndarray    # (m,) bool
    charts: int
    gains: dict
    vchart: np.ndarray = None    # (n,) chart of each split vertex


def _face_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _colour_size(view):
    if view.colour is not None:
        return view.colour.shape[0] / float(view.shape[0])
    return 1.0


def choose_views(V, T, sol, views, work: int = 640):
    """(view per face, score per face): the placed view that sees the face, facing it most squarely and nearest."""
    m = len(T)
    best = np.full(m, -1)
    score = np.zeros(m)
    allsc = np.zeros((m, len(sol.names)), np.float32)
    fn = _face_normals(V, T)
    fc = V[T].mean(1)
    for v in range(len(sol.names)):
        if not sol.placed[v]:
            continue
        vw = views[v]
        s = min(1.0, work / float(max(vw.shape)))
        H, W = int(round(vw.shape[0] * s)), int(round(vw.shape[1] * s))
        P, z = RA.project(V, sol.R[v], sol.t[v], sol.f[v] * s, (vw.cx + 0.5) * s - 0.5, (vw.cy + 0.5) * s - 0.5,
                          sol.k1)
        face, _, _ = RA.rasterize(P, z, T, H, W)
        seen = np.bincount(face[face >= 0], minlength=m) > 0
        C = sol.centre(v)
        d = C - fc
        dist = np.linalg.norm(d, axis=1)
        cosang = np.sum(fn * d, 1) / np.maximum(dist, 1e-12)
        sc = np.where(seen & (cosang > 0.05), cosang / np.maximum(dist, 1e-9) ** 0.5, 0.0)
        allsc[:, v] = sc
        better = sc > score
        best[better] = v
        score[better] = sc[better]
    return best, allsc


def _edges(T):
    m = len(T)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    f = np.tile(np.arange(m), 3)
    key = np.sort(e, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, fs = key[order], f[order]
    same = np.all(ks[1:] == ks[:-1], 1)
    return fs[:-1][same], fs[1:][same]


def smooth_labels(T, scores, best, rounds: int = 12, lam: float = 0.45, min_faces: int = 60):
    """Fewer, larger charts (fewer seams, a tidier atlas): iterated conditional modes on a Potts energy (Besag 1986):
    each face takes the view minimising (1 - its score for that view / its best score) + lam x (neighbours with
    another view); a view that does not see the face is not allowed (faces no view sees allow every view). Then any
    chart under ``min_faces`` faces joins its largest neighbouring chart."""
    a, b = _edges(T)
    m, nv = scores.shape
    sees = scores > 0
    none = ~sees.any(1)
    top = np.maximum(scores.max(1, keepdims=True), 1e-12)
    data = np.where(sees, 1.0 - scores / top, np.inf)
    data[none] = 0.0
    best = best.copy()
    for _ in range(rounds):
        nb = np.zeros((m, nv), np.float32)
        np.add.at(nb, (a, best[b]), 1.0)
        np.add.at(nb, (b, best[a]), 1.0)
        deg = nb.sum(1, keepdims=True)
        energy = data + lam * (deg - nb)
        new = np.argmin(energy, 1)
        if np.array_equal(new, best):
            break
        best = new
    for _ in range(6):
        chart, _ = _face_graph(T, best)
        sizes = np.bincount(chart)
        small_chart = sizes < min_faces
        if not small_chart.any():
            break
        # every small chart takes the view of its largest neighbouring chart
        ca, cb = chart[a], chart[b]
        diff = ca != cb
        pairs = np.concatenate([np.stack([ca[diff], cb[diff]], 1), np.stack([cb[diff], ca[diff]], 1)])
        pairs = pairs[small_chart[pairs[:, 0]]]
        if not len(pairs):
            break
        nbsize = sizes[pairs[:, 1]]
        order = np.lexsort((-nbsize, pairs[:, 0]))
        pairs = pairs[order]
        first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
        target = dict(zip(pairs[first, 0].tolist(), pairs[first, 1].tolist()))
        view_of_chart = np.zeros(len(sizes), int)
        view_of_chart[chart] = best
        src = np.array(list(target.keys()))
        dst = np.array(list(target.values()))
        newview = view_of_chart.copy()
        newview[src] = view_of_chart[dst]
        best = newview[chart]
    return best


def _fill_unseen(T, best):
    """Faces no view sees take the view of a neighbouring face (repeated until all have one)."""
    m = len(T)
    if (best >= 0).all() or (best < 0).all():
        return best
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    f = np.tile(np.arange(m), 3)
    key = np.sort(e, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, fs = key[order], f[order]
    same = np.all(ks[1:] == ks[:-1], 1)
    a, b = fs[:-1][same], fs[1:][same]
    best = best.copy()
    for _ in range(200):
        miss = best < 0
        if not miss.any():
            break
        take_a = miss[a] & ~miss[b]
        take_b = miss[b] & ~miss[a]
        best[a[take_a]] = best[b[take_a]]
        best[b[take_b]] = best[a[take_b]]
    best[best < 0] = 0
    return best


def _face_graph(T, label):
    m = len(T)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    f = np.tile(np.arange(m), 3)
    key = np.sort(e, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, fs = key[order], f[order]
    same = np.all(ks[1:] == ks[:-1], 1)
    a, b = fs[:-1][same], fs[1:][same]
    keep = label[a] == label[b]
    g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(m, m))
    return connected_components(g, directed=False)[1], (a[~keep], b[~keep])


def _pack(sizes, max_size, pad=4):
    from ..stack.foliage import shelf_pack
    area = sum((w + pad) * (h + pad) for w, h in sizes)
    S = 256
    while S * S < area * 1.15 and S < max_size:
        S *= 2
    scale = 1.0
    while True:
        sz = [(max(2, int(np.ceil(w * scale))), max(2, int(np.ceil(h * scale)))) for w, h in sizes]
        pos = shelf_pack(sz, S, pad)
        if pos is not None:
            return S, scale, pos, sz
        if S < max_size:
            S *= 2
        else:
            scale *= 0.85


def build(V, T, measured_vertex, sol, views, max_size: int = 4096, texel_scale: float = 1.0) -> Textured:
    V = np.asarray(V, np.float64)
    T = np.asarray(T, np.int64)
    best, scores = choose_views(V, T, sol, views)
    measured_face = measured_vertex[T].mean(1) >= 0.5
    unseen = best < 0
    best = _fill_unseen(T, best)
    best = smooth_labels(T, scores, best)
    chart, seams = _face_graph(T, best)
    nch = int(chart.max()) + 1 if len(chart) else 0
    chart_view = np.zeros(nch, int)
    chart_view[chart] = best
    # vertices split per chart (a vertex on a chart edge gets one copy per chart), all at once
    key = T.ravel().astype(np.int64) * nch + np.repeat(chart, 3)
    uk, inv = np.unique(key, return_inverse=True)
    newT = inv.reshape(-1, 3)
    src = uk // nch
    vch = (uk % nch).astype(int)
    pix = np.zeros((len(uk), 2))
    for v in np.unique(chart_view):
        sel = chart_view[vch] == v
        vw = views[v]
        cs = _colour_size(vw)
        P, _ = RA.project(V[src[sel]], sol.R[v], sol.t[v], sol.f[v] * cs, (vw.cx + 0.5) * cs - 0.5,
                          (vw.cy + 0.5) * cs - 0.5, sol.k1)
        Hc, Wc = (vw.colour.shape[:2] if vw.colour is not None else vw.shape)
        P[:, 0] = np.clip(P[:, 0], 0, Wc - 1)
        P[:, 1] = np.clip(P[:, 1], 0, Hc - 1)
        pix[sel] = P
    lo_all = np.full((nch, 2), np.inf)
    hi_all = np.full((nch, 2), -np.inf)
    np.minimum.at(lo_all, vch, pix)
    np.maximum.at(hi_all, vch, pix)
    rects = [(np.floor(lo_all[c]).astype(int), np.ceil(hi_all[c]).astype(int)) for c in range(nch)]
    sizes = [((hi - lo)[0] * texel_scale + 1, (hi - lo)[1] * texel_scale + 1) for lo, hi in rects]
    S, scale, pos, sz = _pack(sizes, max_size)
    sc = scale * texel_scale
    # colour gain per chart over the seams (log space, one per channel, held near 1)
    gains = _gains(V, T, seams, chart, best, sol, views, nch)
    atlas = np.zeros((S, S, 3), np.float32)
    filled = np.zeros((S, S), bool)
    uv = np.zeros((len(src), 2))
    for c in range(nch):
        lo, hi = rects[c]
        x0, y0 = pos[c]
        w, h = sz[c]
        v = chart_view[c]
        col = views[v].colour if views[v].colour is not None else \
            np.dstack([np.clip(views[v].image * 255, 0, 255)] * 3).astype(np.uint8)
        ys = lo[1] + (np.arange(h) + 0.5) / sc - 0.5
        xs = lo[0] + (np.arange(w) + 0.5) / sc - 0.5
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        for k in range(3):
            atlas[y0:y0 + h, x0:x0 + w, k] = map_coordinates(col[..., k].astype(np.float32), [gy, gx], order=1,
                                                             mode="nearest") * gains[c][k]
        sel = vch == c
        uv[sel, 0] = (x0 + (pix[sel, 0] - lo[0] + 0.5) * sc - 0.5 + 0.5) / S
        uv[sel, 1] = (y0 + (pix[sel, 1] - lo[1] + 0.5) * sc - 0.5 + 0.5) / S
    # only the texels the faces cover are kept (the rest bled from them); the "how made" map likewise
    face_px, _, _ = RA.rasterize(uv * S - 0.5, np.zeros(len(uv)), newT, S, S)
    filled = face_px >= 0
    made = np.zeros((S, S, 3), np.uint8)
    lab = np.where(measured_face & ~unseen, 1, 0)
    mm = filled & (lab[np.maximum(face_px, 0)] == 1)
    made[mm] = CYAN
    made[filled & ~mm] = VIOLET
    from ..stack.foliage import bleed
    atlas = bleed(np.clip(atlas, 0, 255), filled.astype(np.float32))
    return Textured(V=V[src], T=newT, uv=uv, atlas=np.clip(np.rint(atlas), 0, 255).astype(np.uint8), made=made,
                    source=src, face_view=best, measured_face=measured_face & ~unseen, charts=nch,
                    gains={int(c): [round(float(g), 4) for g in gains[c]] for c in range(nch)}, vchart=vch)


def _gains(V, T, seams, chart, best, sol, views, nch):
    a, b = seams
    if nch <= 1 or not len(a):
        return {c: np.ones(3) for c in range(nch)}
    rng = np.random.default_rng(0)
    if len(a) > 4000:
        k = rng.choice(len(a), 4000, replace=False)
        a, b = a[k], b[k]
    # a point on the seam between faces a and b: the midpoint of their centres
    X = 0.5 * (V[T[a]].mean(1) + V[T[b]].mean(1))

    def colour_at(view_idx, pts):
        out = np.full((len(pts), 3), np.nan)
        for v in np.unique(view_idx):
            s = view_idx == v
            vw = views[v]
            cs = _colour_size(vw)
            P, z = RA.project(pts[s], sol.R[v], sol.t[v], sol.f[v] * cs, (vw.cx + 0.5) * cs - 0.5,
                              (vw.cy + 0.5) * cs - 0.5, sol.k1)
            col = vw.colour if vw.colour is not None else np.dstack([vw.image * 255] * 3)
            for k in range(3):
                out[s, k] = map_coordinates(col[..., k].astype(np.float32), [P[:, 1], P[:, 0]], order=1,
                                            mode="nearest")
        return out
    ca = colour_at(best[a], X)
    cb = colour_at(best[b], X)
    ok = np.isfinite(ca).all(1) & np.isfinite(cb).all(1) & (ca.min(1) > 8) & (cb.min(1) > 8)
    if ok.sum() < 10:
        return {c: np.ones(3) for c in range(nch)}
    ia, ib = chart[a[ok]], chart[b[ok]]
    gains = {c: np.ones(3) for c in range(nch)}
    n = len(ia)
    rows = np.concatenate([np.arange(n), np.arange(n), n + np.arange(nch)])
    cols = np.concatenate([ia, ib, np.arange(nch)])
    vals = np.concatenate([np.ones(n), -np.ones(n), 0.3 * np.ones(nch)])
    A = coo_matrix((vals, (rows, cols)), shape=(n + nch, nch)).tocsr()
    from scipy.sparse.linalg import lsqr
    for k in range(3):
        rhs = np.concatenate([np.log(cb[ok, k]) - np.log(ca[ok, k]), np.zeros(nch)])
        g = lsqr(A, rhs, atol=1e-8, btol=1e-8, iter_lim=500)[0]
        g = np.clip(np.exp(g - np.median(g)), 0.6, 1.6)
        for c in range(nch):
            gains[c][k] = g[c]
    return gains

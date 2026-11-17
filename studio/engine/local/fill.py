"""Fill: rebuild an area from what surrounds it (Retouch > Fill, and the flare tool's Rebuild).

Two engines, chosen automatically (``method="auto"``) or by the person, each with what it is good and weak at:

* **LaMa** (Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions", WACV 2022; Samsung
  AI Center, Apache-2.0; the ONNX export Carve/LaMa-ONNX, Apache-2.0; trained on Places2, CC-BY 4.0). A generative
  network that sees the whole crop at once (its Fourier convolutions reach across the picture), so it continues
  structure (a horizon, a branch, a wall's edge) through large holes. It runs at 512 x 512 on this PC through the
  Studio's own NumPy runner, after Convert Models.bat has converted lama_fp32.onnx (put the download in models/).
  At 512 px it is soft on a big photo, so its answer is used as the plan and the detail comes from the next engine.
* **PatchMatch** (Barnes, Shechtman, Finkelstein and Goldman, SIGGRAPH 2009, with the multi-scale completion of
  Wexler, Shechtman and Irani, PAMI 2007: what Photoshop's Content-Aware Fill is built on). Built in, no model: every
  patch in the hole is replaced, scale by scale, by the most similar patch of the photo outside it, and overlapping
  patches vote. It brings real texture from the photo at full resolution; on its own it cannot invent structure that
  is not elsewhere in the picture.

"auto" = LaMa for the layout when it is installed, PatchMatch at full resolution for the texture, guided by LaMa's
plan (the patches are matched to what LaMa drew, not only to their neighbours). Then the grain of the surroundings is
matched so the fill does not look smoother than the photo. Everything is seeded: the same edit gives the same fill.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, zoom

from ... import log as slog

log = slog.get("engine.fill")

METHODS = {
    "auto": {"name": "Auto", "best": "LaMa when it is installed (with fine texture from your photo added on big "
             "areas); PatchMatch alone otherwise.", "weak": "The slowest; results matter more than speed here."},
    "lama": {"name": "LaMa (generative)", "best": "Large holes and continuing structure: horizons, branches, edges.",
             "weak": "Made at 512 px: soft on a large area unless PatchMatch adds the detail (Auto does)."},
    "patchmatch": {"name": "PatchMatch (from the photo)", "best": "Texture: grass, sky, foliage, walls; small and "
                   "medium areas. Built in, no model.",
                   "weak": "Cannot invent structure that is not elsewhere in the photo; can repeat a patch."},
}
PATCH = 7
LAMA_SIZE = 512
DETAIL_SCALE = 0.75          # LaMa shrank the hole less than this: its answer is used as it is


def model_dir() -> str:
    from ... import paths
    return os.path.join(str(paths.APP_ROOT), "models", "inpaint")


def model_path() -> str:
    return os.path.join(model_dir(), "lama.npz")


def lama_available() -> bool:
    try:
        return os.path.isfile(model_path())
    except Exception:                                   # noqa: BLE001 - no models folder: not installed
        return False


# ------------------------------------------------------------------------------------------- encodings
def encode(rgb: np.ndarray) -> np.ndarray:
    """Scene-linear -> a perceptual 0..1 encoding (1/2.2), where patch differences mean what the eye sees."""
    return np.power(np.clip(rgb, 0.0, None), 1 / 2.2).astype(np.float32)


def decode(enc: np.ndarray) -> np.ndarray:
    return np.power(np.clip(enc, 0.0, None), 2.2).astype(np.float32)


# ------------------------------------------------------------------------------------------- PatchMatch
def _patch_offsets(p: int):
    r = p // 2
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    return dy.ravel(), dx.ravel()


def _dist(img_t, img_s, ty, tx, sy, sx, oy, ox, wmask=None):
    """Sum of squared differences between target patches (centres ty, tx in img_t) and source patches."""
    T = img_t[ty[:, None] + oy[None, :], tx[:, None] + ox[None, :]]
    S = img_s[sy[:, None] + oy[None, :], sx[:, None] + ox[None, :]]
    d = ((T - S) ** 2).sum(-1)
    if wmask is not None:
        d = d * wmask
    return d.sum(1)


def _scale_fill(img: np.ndarray, hole: np.ndarray, nnf, rng, iters: int, guide=None, guide_w: float = 0.0,
                search: int | None = None):
    """One scale of completion: PatchMatch search + voting, ``iters`` times. img (h, w, 3) encoded, hole bool.
    nnf: (ys, xs) source centre for every pixel, or None. ``search``: the random search's first radius (default
    the whole crop; finer scales start from the coarser answer and look nearer). Returns (img, nnf)."""
    h, w = hole.shape
    r = PATCH // 2
    oy, ox = _patch_offsets(PATCH)
    pad = ((r, r), (r, r), (0, 0))
    known_patch = binary_erosion(~hole, iterations=r, border_value=0)     # source centres: their patch is all known
    known_patch[:r] = known_patch[-r:] = False
    known_patch[:, :r] = known_patch[:, -r:] = False
    sy_all, sx_all = np.nonzero(known_patch)
    if sy_all.size < 4:
        return img, nnf
    tgt = binary_dilation(hole, iterations=r)                            # target centres: their patch touches it
    ty, tx = np.nonzero(tgt)
    n = ty.size
    if nnf is None:
        pick = rng.integers(0, sy_all.size, n)
        sy, sx = sy_all[pick], sx_all[pick]
    else:
        sy, sx = nnf[0][ty, tx].copy(), nnf[1][ty, tx].copy()
        bad = ~known_patch[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)]
        pick = rng.integers(0, sy_all.size, int(bad.sum()))
        sy[bad], sx[bad] = sy_all[pick], sx_all[pick]
    pimg = np.pad(img, pad, mode="reflect")
    pg = np.pad(guide, pad, mode="reflect") if guide is not None else None
    TY, TX = ty + r, tx + r
    idx = -np.ones((h, w), np.int64)
    idx[ty, tx] = np.arange(n)

    def cost(sy_, sx_, sel=None):
        T = Tall if sel is None else Tall[sel]
        S = pimg[(sy_ + r)[:, None] + oy[None, :], (sx_ + r)[:, None] + ox[None, :]]
        d = ((T - S) ** 2).sum(-1).sum(1)
        if Gall is not None:
            G = Gall if sel is None else Gall[sel]
            d = d + guide_w * ((G - S) ** 2).sum(-1).sum(1)
        return d
    for it in range(iters):
        Tall = pimg[TY[:, None] + oy[None, :], TX[:, None] + ox[None, :]]        # target patches, this pass
        Gall = pg[TY[:, None] + oy[None, :], TX[:, None] + ox[None, :]] if pg is not None and guide_w > 0 else None
        best = cost(sy, sx)
        for sweep in range(2):                                           # propagation from the neighbours
            for dyy, dxx in (((0, 1), (1, 0)) if (it + sweep) % 2 == 0 else ((0, -1), (-1, 0))):
                ny, nx = ty - dyy, tx - dxx
                ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                j = np.where(ok, idx[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
                has = j >= 0
                cy = np.where(has, sy[np.maximum(j, 0)] + dyy, sy)
                cx = np.where(has, sx[np.maximum(j, 0)] + dxx, sx)
                valid = has & (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
                valid &= known_patch[np.clip(cy, 0, h - 1), np.clip(cx, 0, w - 1)]
                if not valid.any():
                    continue
                dc = np.full(n, np.inf)
                dc[valid] = cost(cy[valid], cx[valid], valid)
                take = dc < best
                sy, sx, best = np.where(take, cy, sy), np.where(take, cx, sx), np.where(take, dc, best)
        rad = max(h, w) if search is None else max(1, int(search))
        while rad >= 1:                                                  # random search, shrinking
            cy = np.clip(sy + rng.integers(-rad, rad + 1, n), 0, h - 1)
            cx = np.clip(sx + rng.integers(-rad, rad + 1, n), 0, w - 1)
            valid = known_patch[cy, cx]
            if valid.any():
                dc = np.full(n, np.inf)
                dc[valid] = cost(cy[valid], cx[valid], valid)
                take = dc < best
                sy, sx, best = np.where(take, cy, sy), np.where(take, cx, sx), np.where(take, dc, best)
            rad //= 2
        # voting (Wexler): every hole pixel = weighted mean of the source pixels the covering patches put there
        sig2 = max(float(np.percentile(best, 75)), 1e-6)
        wt = np.exp(-best / (2 * sig2)).astype(np.float32)
        acc = np.zeros((h + 2 * r, w + 2 * r, 3), np.float32)
        wsum = np.zeros((h + 2 * r, w + 2 * r), np.float32)
        for k in range(oy.size):
            yy, xx = TY + oy[k], TX + ox[k]
            src = pimg[sy + r + oy[k], sx + r + ox[k]]
            acc[yy, xx] += src * wt[:, None]          # one offset: every target pixel at most once, so += is exact
            wsum[yy, xx] += wt
        new = acc[r:h + r, r:w + r] / np.maximum(wsum[r:h + r, r:w + r], 1e-9)[..., None]
        img = np.where(hole[..., None] & (wsum[r:h + r, r:w + r] > 0)[..., None], new, img).astype(np.float32)
        pimg = np.pad(img, pad, mode="reflect")
    full_y = np.zeros((h, w), np.int64)
    full_x = np.zeros((h, w), np.int64)
    full_y[ty, tx], full_x[ty, tx] = sy, sx
    return img, (full_y, full_x)


def _smooth_start(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """A first guess for the coarsest scale: the hole filled by a membrane from its edge."""
    from .retouch import membrane
    out = img.copy()
    m = membrane(hole, img.astype(np.float32))
    out[hole] = m[hole]
    return out


def patchmatch_fill(enc: np.ndarray, hole: np.ndarray, seed: int = 0, guide: np.ndarray | None = None,
                    iters: int = 4, min_size: int = 24) -> np.ndarray:
    """Multi-scale PatchMatch completion of an encoded image. ``guide`` (same size, encoded): a plan of the fill
    (LaMa's answer) the patches are matched to as well; with it only the finest scales are searched."""
    rng = np.random.default_rng(seed)
    h, w = hole.shape
    levels = 0
    hh = int(hole.any(axis=1).sum())
    ww = int(hole.any(axis=0).sum())
    while min(h, w) / (2 ** (levels + 1)) >= 3 * PATCH and max(hh, ww) / (2 ** (levels + 1)) >= min_size / 2:
        levels += 1
    if guide is not None:
        levels = min(levels, 1)
    pyr_i, pyr_h, pyr_g = [enc], [hole], [guide]
    for _ in range(levels):
        i2 = zoom(pyr_i[-1], (0.5, 0.5, 1), order=1)
        h2 = zoom(pyr_h[-1].astype(np.float32), 0.5, order=1) > 0.01
        pyr_i.append(i2.astype(np.float32))
        pyr_h.append(h2)
        pyr_g.append(None if guide is None else zoom(pyr_g[-1], (0.5, 0.5, 1), order=1).astype(np.float32))
    cur = pyr_i[-1].copy()
    cur = pyr_g[-1].copy() if guide is not None else _smooth_start(cur, pyr_h[-1])
    cur = np.where(pyr_h[-1][..., None], cur, pyr_i[-1])
    nnf = None
    for lv in range(levels, -1, -1):
        img = np.where(pyr_h[lv][..., None], cur, pyr_i[lv]).astype(np.float32)
        gw = 0.5 if guide is not None else 0.0
        if lv == levels:                                   # coarsest: search everywhere, a little longer
            n_it, rad = iters + 2, None
        else:                                              # finer: refine what the coarser scale found
            n_it = max(2, iters - (1 if lv == 0 else 0) - (1 if pyr_h[lv].size > 1.5e6 else 0))
            rad = max(8, max(pyr_h[lv].shape) // 4)
        img, nnf = _scale_fill(img, pyr_h[lv], nnf, rng, n_it, pyr_g[lv], gw, rad)
        if lv > 0:
            hn, wn = pyr_h[lv - 1].shape
            cur = zoom(img, (hn / img.shape[0], wn / img.shape[1], 1), order=1)[:hn, :wn]
            if cur.shape[:2] != (hn, wn):
                cur = np.pad(cur, ((0, hn - cur.shape[0]), (0, wn - cur.shape[1]), (0, 0)), mode="edge")
            ny = np.clip(np.repeat(np.repeat(nnf[0] * 2, 2, 0), 2, 1)[:hn, :wn], 0, hn - 1)
            nx = np.clip(np.repeat(np.repeat(nnf[1] * 2, 2, 0), 2, 1)[:hn, :wn], 0, wn - 1)
            if ny.shape != (hn, wn):
                ny = np.pad(ny, ((0, hn - ny.shape[0]), (0, wn - ny.shape[1])), mode="edge")
                nx = np.pad(nx, ((0, hn - nx.shape[0]), (0, wn - nx.shape[1])), mode="edge")
            nnf = (ny, nx)
        else:
            cur = img
    return np.where(hole[..., None], cur, enc).astype(np.float32)


# ------------------------------------------------------------------------------------------- LaMa
_LAMA = None


def _lama():
    global _LAMA
    if _LAMA is None:
        from .onnxnp import Graph
        _LAMA = Graph.load(model_path())
    return _LAMA


def lama_fill(enc: np.ndarray, hole: np.ndarray, return_scale: bool = False):
    """LaMa on a crop (encoded 0..1). A window around the hole (2.4 x the hole, at least 512 px) is padded to a
    square, resized to 512, run, resized back; only the hole is taken. ``return_scale``: also return the resize
    factor used (below 1: LaMa saw the hole smaller than it is, so its answer is softer than the photo)."""
    H0, W0 = hole.shape
    ys, xs = np.nonzero(hole)
    if ys.size == 0:
        return (enc, 1.0) if return_scale else enc
    side = max(LAMA_SIZE, int(2.4 * max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)))
    cy, cx = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
    wy0 = min(max(0, cy - side // 2), max(0, H0 - side))
    wx0 = min(max(0, cx - side // 2), max(0, W0 - side))
    wy1, wx1 = min(H0, wy0 + side), min(W0, wx0 + side)
    sub, sh_ = enc[wy0:wy1, wx0:wx1], hole[wy0:wy1, wx0:wx1]
    h, w = sh_.shape
    s = max(h, w)
    pi = np.pad(sub, ((0, s - h), (0, s - w), (0, 0)), mode="reflect")
    pm = np.pad(sh_, ((0, s - h), (0, s - w)))
    k = LAMA_SIZE / float(s)
    x = zoom(pi, (k, k, 1), order=1)[:LAMA_SIZE, :LAMA_SIZE]
    m = zoom(pm.astype(np.float32), k, order=1)[:LAMA_SIZE, :LAMA_SIZE] > 0.01
    m = binary_dilation(m, iterations=2)
    # LaMa learnt natural colours (Places2): an infrared or other false-colour photo can be pulled toward green
    # leaves and blue sky inside the hole. So it sees the photo with its colour cast taken out (grey world over the
    # known part), and the cast is put back on its answer.
    known = ~m
    cm = x[known].mean(0) if known.any() else x.reshape(-1, 3).mean(0)
    cast = np.clip(cm / max(float(cm.mean()), 1e-6), 0.2, 5.0).astype(np.float32)
    xin = np.clip(x / cast, 0, 1)
    img = xin.transpose(2, 0, 1)[None].astype(np.float32)
    msk = m[None, None].astype(np.float32)
    g = _lama()
    names = g.inputs
    feed = {names[0]: img, names[1]: msk} if len(names) > 1 else {names[0]: np.concatenate([img, msk], 1)}
    out = g(**feed)
    o = out[g.outputs[0]][0].transpose(1, 2, 0).astype(np.float32)
    if known.any():                                     # Carve's export answers 0..255: judged against the input
        if float(np.mean(o[known]) + 1e-6) / float(np.mean(xin[known]) + 1e-6) > 16.0:
            o = o / 255.0
    elif o.max() > 2.0:
        o = o / 255.0
    o = o * cast
    back = zoom(o, (s / float(LAMA_SIZE), s / float(LAMA_SIZE), 1), order=1)[:h, :w]
    if back.shape[:2] != (h, w):
        back = np.pad(back, ((0, h - back.shape[0]), (0, w - back.shape[1]), (0, 0)), mode="edge")
    res = enc.astype(np.float32).copy()
    res[wy0:wy1, wx0:wx1] = np.where(sh_[..., None], np.clip(back, 0, 1), sub)
    return (res, k) if return_scale else res


# ------------------------------------------------------------------------------------------- grain
def match_grain(enc: np.ndarray, filled: np.ndarray, hole: np.ndarray, seed: int = 0) -> np.ndarray:
    """Add back fine grain where the fill is smoother than the photo around it (noise measured just outside)."""
    ring = binary_dilation(hole, iterations=8) & ~hole
    if ring.sum() < 50 or hole.sum() == 0:
        return filled
    hp_o = enc - np.stack([gaussian_filter(enc[..., c], 1.0) for c in range(3)], -1)
    hp_f = filled - np.stack([gaussian_filter(filled[..., c], 1.0) for c in range(3)], -1)
    so = float(np.std(hp_o[ring]))
    sf = float(np.std(hp_f[hole]))
    if sf >= so * 0.9:
        return filled
    add = math.sqrt(max(so * so - sf * sf, 0.0))
    rng = np.random.default_rng(seed + 7)
    nz = gaussian_filter(rng.normal(0, 1, hole.shape), 0.6)
    nz = nz / max(float(nz.std()), 1e-9) * add
    out = filled.copy()
    out[hole] += nz[hole][:, None]
    return out


def fill(rgb: np.ndarray, hole: np.ndarray, method: str = "auto", seed: int = 0) -> tuple:
    """Fill ``hole`` in a scene-linear crop. Returns (rgb, report {"method", "text"})."""
    hole = np.asarray(hole, bool)
    if not hole.any():
        return rgb, {"method": "none", "text": "Nothing to fill."}
    scale = max(float(np.percentile(rgb[~hole], 99.5)) if (~hole).any() else 1.0, 1e-6)
    enc = encode(rgb / scale)
    use_lama = method in ("auto", "lama") and lama_available()
    plan = None
    used = []
    k = 1.0
    if use_lama:
        try:
            with slog.stage("fill.lama", log, px=int(hole.sum())):
                plan, k = lama_fill(enc, hole, return_scale=True)
            used.append("LaMa")
        except Exception:                               # noqa: BLE001 - PatchMatch alone is still a good answer
            slog.log_exception("fill: LaMa failed", log)
            slog.save_error_report("Fill: running LaMa")
            plan = None
    if plan is not None and (method == "lama" or k >= DETAIL_SCALE):
        out = plan                                      # LaMa saw the hole at (about) full size: its answer stands
    elif plan is not None:
        # LaMa saw the hole smaller than it is: keep its layout (the soft part) and take the fine texture from
        # the photo itself (PatchMatch, guided by the layout), so a big fill is not softer than the photo
        with slog.stage("fill.patchmatch", log, px=int(hole.sum()), guided=True):
            pm = patchmatch_fill(enc, hole, seed, guide=plan)
        sig = 0.6 / max(k, 1e-3)
        lo = gaussian_filter(plan, (sig, sig, 0))
        hi = pm - gaussian_filter(pm, (sig, sig, 0))
        out = np.where(hole[..., None], np.clip(lo + hi, 0, None), enc).astype(np.float32)
        used.append("PatchMatch detail")
    else:
        with slog.stage("fill.patchmatch", log, px=int(hole.sum()), guided=False):
            out = patchmatch_fill(enc, hole, seed)
        used.append("PatchMatch")
    out = match_grain(enc, out, hole, seed)
    res = decode(np.clip(out, 0, None)) * scale
    res = np.where(hole[..., None], res, rgb).astype(np.float32)
    txt = " + ".join(used) + (" (LaMa is not installed: Convert Models.bat adds it)" if method in ("auto", "lama")
                              and not use_lama else "")
    return res, {"method": "+".join(used).lower(), "text": txt}

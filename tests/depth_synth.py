"""A synthetic hand-held focus stack of Dual Pixel RAWs with known depth, for the Depth Stack tests
(BUILD_PLAN_DEPTH_STACK, Tests).

A textured surface with a known height field (two bumps, a ramp and a step edge) and one plain patch. Depth is in
focus steps. Frame i has its plane of focus at depth p_i: i on a rail, i plus up to +-0.3 of a step by hand
(``uneven``). Defocus is a disc (a real lens's blur, not a Gaussian) of radius R = S_TRUE * |z - p_i| / 0.849
pixels; each half of a Dual Pixel photo sees half the lens, so its blur is a half disc, on opposite sides for the two
halves and flipping side between in front of and behind focus. The centroids of the two half discs are 0.849 R apart,
so the offset of half 2 against half 1 is d = S_TRUE * (p_i - z) full-size pixels (the Studio's convention), and it
grows with the blur as on a real sensor. Each frame is then moved as a hand does: shifted, turned by up to 0.9 degree
and magnified by up to 2 % (focus breathing), the halves exactly like the frame. Noise at the R5's ISO 100 or ISO 800
level (variance a * v + b, 1.0 = clip; each half has half the light). A frame can be given a patch that moved in depth
(``moved``).

Frames and halves are served to the engine through monkeypatched frames.scene_rgb / dual_halves (there is no raw
writer); the files on disk are empty stand-ins with the right names.
"""
from __future__ import annotations

import os
import types

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from conftest import ROOT, scene
from studio.engine.stack import align as A
from studio.engine.stack import frames as F

OUT = ROOT / "outputs" / "pytest_depthstack"
S_TRUE = 1.6                 # offset (full-size px) per focus step
HALF_SEP = 8.0 / (3.0 * np.pi)   # centroid distance of the two half discs, as a share of the disc's radius (0.849)
R_STEP = 0.5                 # the bank of blurs: one entry per half pixel of radius
NOISE = {"iso100": (2.1e-5, 1.1e-8), "iso800": (1.7e-4, 9e-8)}      # R5: variance = a * v + b, 1.0 = clip
MARGIN = 48


def _similarity(scale, angle_deg, dy, dx, h, w):
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    a = np.radians(angle_deg)
    C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    R = np.array([[scale * np.cos(a), -scale * np.sin(a), 0], [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1.0]])
    Tr = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return A.Transform(Tr @ C @ R @ np.linalg.inv(C), "similarity")


def height_field(H, W, n):
    """Depth in steps (0 = frame 0's plane): a gentle ramp, two bumps, a raised block with a step edge."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    z = 1.2 + 4.2 * x / (W - 1)                                   # ramp across the stack
    for cy, cx, r, a in ((0.35, 0.3, 0.12, 1.6), (0.7, 0.62, 0.1, -1.4)):
        z += a * np.exp(-(((y / H - cy) ** 2) + ((x / W - cx) ** 2)) / (2 * r * r))
    z[int(0.58 * H):int(0.86 * H), int(0.06 * W):int(0.3 * W)] -= 1.3                       # a step edge
    return np.clip(z, 0.6, n - 1.6).astype(np.float32)


def _disc(r: float, side: int = 0) -> np.ndarray:
    """A normalised disc of radius r (anti-aliased by 8 x 8 supersampling); side +1 / -1: only its x > 0 / x < 0
    half, with half the mass (the light one half of the lens lets through), moved by a fraction of a pixel so its
    centre of mass sits exactly at +-4 r / (3 pi) (pixel sampling alone would lose that below about 2 px)."""
    R = int(np.ceil(r)) + 2
    o = (np.arange(8) + 0.5) / 8.0 - 0.5
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    k = np.zeros_like(x)
    for dy in o:
        for dx in o:
            yy, xx = y + dy, x + dx
            inside = yy * yy + xx * xx <= r * r
            if side > 0:
                inside &= xx > 0
            elif side < 0:
                inside &= xx < 0
            k += inside
    if k.sum() == 0:
        k[R, R] = 1.0
    k = k / k.sum()
    if side != 0:
        cx = float((k * x).sum())
        sh = side * 4.0 * r / (3.0 * np.pi) - cx
        n0 = int(np.floor(sh))
        f = sh - n0
        k = (1 - f) * np.roll(k, n0, axis=1) + f * np.roll(k, n0 + 1, axis=1)
        k = 0.5 * k
    return k.astype(np.float32)


def _blur(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    if k.shape == (1, 1) or k.size <= 9 and k[k.shape[0] // 2, k.shape[1] // 2] > 0.99:
        return img.astype(np.float32) * float(k.sum())
    if img.ndim == 3:
        return np.stack([_blur(img[..., c], k) for c in range(img.shape[2])], 2)
    return fftconvolve(img, k, mode="same").astype(np.float32)


def make_stack(n=9, h=768, w=1152, iso="iso100", handheld=True, uneven=True, moved=None, flat=True, seed=0,
               bright_step=False):
    """(frames (full size RGB), halves [(la, lb) half size], truth). ``moved``: {frame: dz} - the moving patch's
    depth is off by dz in those frames only. Truth "t" is in the Studio's units: steps from frame 0's plane, one
    step = the median step between neighbouring frames."""
    rng = np.random.default_rng(seed + 1)
    m = MARGIN
    Hh, Ww = h + 2 * m, w + 2 * m
    tex = gaussian_filter(scene(Hh, Ww, seed), (0.6, 0.6, 0)).astype(np.float32)
    z = height_field(Hh, Ww, n)
    flat_sl = (slice(int(0.12 * Hh), int(0.3 * Hh)), slice(int(0.66 * Ww), int(0.9 * Ww)))
    if flat:
        tex[flat_sl] = np.array([0.42, 0.38, 0.33], np.float32)
    if bright_step:                  # the raised block (nearer, a step edge) bright with a little texture of its own
        st = (slice(int(0.58 * Hh), int(0.86 * Hh)), slice(int(0.06 * Ww), int(0.3 * Ww)))
        tex[st] = 0.85 + 0.08 * (tex[st] - tex[st].mean())
    move_sl = (slice(int(0.42 * Hh), int(0.56 * Hh)), slice(int(0.4 * Ww), int(0.52 * Ww)))
    planes = np.arange(n, dtype=np.float64)
    if uneven:
        planes = planes + rng.uniform(-0.3, 0.3, n)
    L0 = A.luma(tex)
    r_per_step = S_TRUE / HALF_SEP
    r_max = r_per_step * (n + 1.0)
    radii = np.arange(0.0, r_max + R_STEP, R_STEP)
    bank_rgb = np.stack([_blur(tex, _disc(r)) if r > 0 else tex for r in radii])
    bank_p = np.stack([_blur(L0, 2 * _disc(r, +1)) if r > 0 else L0 for r in radii])     # half discs, x 2 = the
    bank_m = np.stack([_blur(L0, 2 * _disc(r, -1)) if r > 0 else L0 for r in radii])     # brightness of the whole
    a_n, b_n = NOISE[iso]
    O = np.array([[1, 0, m], [0, 1, m], [0, 0, 1.0]])
    S = []
    for i in range(n):
        if handheld:
            S.append(_similarity(1 + 0.0022 * i, 0.22 * (i - n // 2), 0.41 * i - 1.3, -0.33 * i + 0.9, h, w))
        else:
            S.append(_similarity(1.0, 0.0, 0.0, 0.0, h, w))

    def pick(bank, r):
        x = np.clip(r / R_STEP, 0, len(radii) - 1.001)
        k = np.floor(x).astype(int)
        f = (x - k).astype(np.float32)
        lo = np.take_along_axis(bank, k[None, ..., None] if bank.ndim == 4 else k[None], 0)[0]
        hi = np.take_along_axis(bank, (k + 1)[None, ..., None] if bank.ndim == 4 else (k + 1)[None], 0)[0]
        return lo * (1 - (f[..., None] if bank.ndim == 4 else f)) + hi * (f[..., None] if bank.ndim == 4 else f)

    frames, halves = [], []
    for i in range(n):
        zi = z.copy()
        if moved and i in moved:
            zi[move_sl] += moved[i]
        dz = zi - planes[i]                                          # > 0: behind this frame's plane
        r = (r_per_step * np.abs(dz)).astype(np.float32)
        blurred = pick(bank_rgb, r)
        # half 1's half disc lies on the +x side behind focus and on the -x side in front: content moves by
        # +0.42 R (behind) or -0.42 R (in front) in half 1, the other way in half 2; offset of 2 against 1 = d
        hp, hm = pick(bank_p, r), pick(bank_m, r)
        h1 = np.where(dz > 0, hp, hm)
        h2 = np.where(dz > 0, hm, hp)
        Tb = A.Transform(O @ S[i].matrix @ np.linalg.inv(O), "similarity")
        fr = A.warp(blurred.astype(np.float32), Tb.inverse(), order=3)[m:m + h, m:m + w]
        g1 = A.warp(h1.astype(np.float32), Tb.inverse(), order=3)[m:m + h, m:m + w]
        g2 = A.warp(h2.astype(np.float32), Tb.inverse(), order=3)[m:m + h, m:m + w]
        fr = fr + rng.normal(0, 1, fr.shape).astype(np.float32) * np.sqrt(b_n + a_n * np.maximum(fr, 0))
        # one half has half the light: its noise, at the brightness of the whole photo, is 2a v + 4b
        g1 = g1 + rng.normal(0, 1, g1.shape).astype(np.float32) * np.sqrt(4 * b_n + 2 * a_n * np.maximum(g1, 0))
        g2 = g2 + rng.normal(0, 1, g2.shape).astype(np.float32) * np.sqrt(4 * b_n + 2 * a_n * np.maximum(g2, 0))
        frames.append(np.ascontiguousarray(fr, np.float32))
        halves.append((A.bin_by(g1, 2), A.bin_by(g2, 2)))
    del bank_rgb, bank_p, bank_m
    ref = n // 2
    Ob = A.Transform(O @ S[ref].matrix @ np.linalg.inv(O), "similarity")
    z_ref = A.warp(z, Ob.inverse(), order=1)[m:m + h, m:m + w]
    tex_ref = A.warp(tex, Ob.inverse(), order=1)[m:m + h, m:m + w]
    mk = np.zeros((Hh, Ww), np.float32)
    mk[flat_sl] = 1.0
    flat_ref = A.warp(mk, Ob.inverse(), order=1)[m:m + h, m:m + w] > 0.5
    mk[:] = 0.0
    mk[move_sl] = 1.0
    move_ref = A.warp(mk, Ob.inverse(), order=1)[m:m + h, m:m + w] > 0.5
    med = float(np.median(np.diff(planes)))
    return frames, halves, {"z": z_ref, "t": (z_ref - planes[0]) / med, "tex": tex_ref,
                            "flat": flat_ref if flat else np.zeros_like(flat_ref), "moved": move_ref, "S": S,
                            "ref": ref, "scale_ref": S[ref].scale(), "n": n, "planes": planes,
                            "s_ref": S_TRUE * med * S[ref].scale(), "median_step": med}


def install(monkeypatch, frames, halves, tag, names=None, dual=True):
    """Serve the frames (and halves) under empty stand-in files; returns the paths."""
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
    names = names or [f"3C9A{9000 + i:04d}.CR3" for i in range(len(frames))]
    paths, table, htable = [], {}, {}
    for i, (name, fr) in enumerate(zip(names, frames)):
        p = str(d / name)
        open(p, "wb").close()
        paths.append(p)
        key = os.path.normcase(os.path.abspath(p))
        table[key] = fr
        if halves is not None and halves[i] is not None:     # None: an ordinary frame in a mixed stack
            htable[key] = halves[i]

    def scene_rgb(path, edit, half, region=None):
        fr = table[os.path.normcase(os.path.abspath(path))]
        if region is not None:
            x, y, w, h = region
            fr = fr[y:y + h, x:x + w]
        return A.bin_by(fr, 2) if half else fr.copy()

    def open_frame(path):
        raw = types.SimpleNamespace(wb_as_shot=np.array([2.0, 1.0, 1.5]), flip=0, is_scan=False)
        meta = types.SimpleNamespace(make="Canon", model="Canon EOS R5", iso=100.0, exposure_s=0.01,
                                     fnumber=2.0, focal_mm=50.0, lens="Makro-Planar 2/50", datetime="2026:11:21 04:00:00")
        return types.SimpleNamespace(raw=raw, meta=meta)

    def dual_halves(path):
        la, lb = htable[os.path.normcase(os.path.abspath(path))]
        return la.copy(), lb.copy()

    monkeypatch.setattr(F, "scene_rgb", scene_rgb)
    monkeypatch.setattr(F, "open_frame", open_frame)
    monkeypatch.setattr(F, "is_dual", lambda p: bool(dual and halves is not None and
                                                     os.path.normcase(os.path.abspath(p)) in htable))
    def develop_and_halves(path, edit, state=None):
        la, lb = dual_halves(path)
        return scene_rgb(path, edit, True), la, lb

    monkeypatch.setattr(F, "dual_halves", dual_halves)
    monkeypatch.setattr(F, "develop_and_halves", develop_and_halves)
    return paths


def half_truth(tr, an):
    """Truth depth (steps) and masks at half size in the analysis's box frame."""
    y0, y1, x0, x1 = an.box_half
    z = A.bin_by(tr["t"], 2)[y0:y1, x0:x1]
    flat = A.bin_by(tr["flat"].astype(np.float32), 2)[y0:y1, x0:x1] > 0.5
    moved = A.bin_by(tr["moved"].astype(np.float32), 2)[y0:y1, x0:x1] > 0.5
    from scipy.ndimage import binary_dilation
    step = (np.abs(np.gradient(z, axis=0)) + np.abs(np.gradient(z, axis=1))) > 0.15
    edge = binary_dilation(step, iterations=16)
    away = np.zeros_like(flat)
    away[12:-12, 12:-12] = True
    measurable = away & ~edge & ~binary_dilation(flat, iterations=12) & ~binary_dilation(moved, iterations=12)
    return z, flat, moved, measurable

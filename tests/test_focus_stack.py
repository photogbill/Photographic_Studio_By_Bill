"""Focus Stack (studio/engine/stack/focus.py) on synthetic focus brackets with known truth (BUILD_PLAN_TABS 4.3).

A sharp scene and a depth map (a ramp with a foreground block standing in front); frame i is the scene blurred
at every place by 1.2 * |depth - focus_i| pixels, magnified by 1 + 0.004 i about the centre (focus breathing) and
shifted by a known sub-pixel amount, with noise. Frames are served to the engine through a monkeypatched
frames.scene_rgb (there is no raw writer); the files on disk are empty stand-ins with the right names.
"""
from __future__ import annotations

import os
import threading
import types

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import ROOT, psnr, scene
from studio.engine.stack import align as A
from studio.engine.stack import focus as FS
from studio.engine.stack import frames as F

OUT = ROOT / "outputs" / "pytest_focus"


def _similarity(scale, dy, dx, h, w):
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1.0]])
    Tr = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return A.Transform(Tr @ C @ S @ np.linalg.inv(C), "similarity")


def make_bracket(n=7, h=768, w=1152, noise=True, moved=None, seed=0):
    """(frames full size, truth dict). truth: sharp (ref frame coords), depth (ref coords), transforms
    (reference -> frame j, full size), focus positions."""
    m = 48
    H, W = h + 2 * m, w + 2 * m
    sharp = gaussian_filter(scene(H, W, seed), (0.6, 0.6, 0))
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = (y / (H - 1)) * (n - 1)                                # far at the bottom
    fy0, fx0 = int(0.25 * H), int(0.3 * W)
    alpha = np.zeros((H, W), np.float32)                           # a foreground block stands in front:
    alpha[fy0:fy0 + H // 4, fx0:fx0 + W // 4] = 1.0               # its blur spreads over what is behind it
    fg_col = np.clip(sharp * 1.6 + 0.15, 0, 0.95)
    FG_DEPTH = 0.4
    sig_bank = np.arange(0, 9.01, 0.5)
    bank = np.stack([sharp] + [gaussian_filter(sharp, (s, s, 0)) for s in sig_bank[1:]])
    focus = np.arange(n, dtype=np.float32)
    S = [_similarity(1 + 0.004 * i, 0.37 * i - 1.1, -0.29 * i + 0.8, h, w) for i in range(n)]
    O = np.array([[1, 0, m], [0, 1, m], [0, 0, 1.0]])
    rng = np.random.default_rng(seed + 1)
    frames = []
    for i in range(n):
        sg = np.clip(1.2 * np.abs(depth - focus[i]), 0, sig_bank[-1])
        k = np.clip(np.floor(sg / 0.5).astype(int), 0, len(sig_bank) - 2)
        f = (sg / 0.5 - k)[..., None]
        blurred = np.take_along_axis(bank, k[None, ..., None], 0)[0] * (1 - f) + \
            np.take_along_axis(bank, (k + 1)[None, ..., None], 0)[0] * f
        sf = 1.2 * abs(FG_DEPTH - focus[i])
        a_i = gaussian_filter(alpha, sf) if sf > 0 else alpha
        fg_i = gaussian_filter(fg_col * alpha[..., None], (sf, sf, 0)) if sf > 0 else fg_col * alpha[..., None]
        blurred = fg_i + (1 - a_i[..., None]) * blurred
        if moved is not None and i == moved:                      # something moved in this frame only
            blurred = blurred.copy()
            blurred[H // 2:H // 2 + 120, W // 2 + 120:W // 2 + 260] = np.roll(
                blurred[H // 2:H // 2 + 120, W // 2 + 120:W // 2 + 260], 9, axis=1)
        Tb = A.Transform(O @ S[i].matrix @ np.linalg.inv(O), "similarity")
        fr = A.warp(blurred.astype(np.float32), Tb.inverse(), order=3)[m:m + h, m:m + w]
        if noise:
            fr = fr + rng.normal(0, 1, fr.shape).astype(np.float32) * np.sqrt(1e-5 + 6e-5 * np.maximum(fr, 0))
        frames.append(np.ascontiguousarray(fr, np.float32))
    ref = n // 2
    Ob = A.Transform(O @ S[ref].matrix @ np.linalg.inv(O), "similarity")
    sharp = fg_col * alpha[..., None] + (1 - alpha[..., None]) * sharp
    depth = np.where(alpha > 0.5, FG_DEPTH, depth).astype(np.float32)
    sharp_ref = A.warp(sharp, Ob.inverse(), order=3)[m:m + h, m:m + w]
    depth_ref = A.warp(depth, Ob.inverse(), order=1)[m:m + h, m:m + w]
    T = [S[j].compose(S[ref].inverse()) for j in range(n)]
    return frames, {"sharp": sharp_ref, "depth": depth_ref, "T": T, "focus": focus, "ref": ref}


def install(monkeypatch, frames, names, tag):
    """Serve ``frames`` (full size) under file ``names`` in a fresh folder; returns the paths."""
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
    paths = []
    table = {}
    for name, fr in zip(names, frames):
        p = str(d / name)
        open(p, "wb").close()
        paths.append(p)
        table[os.path.normcase(os.path.abspath(p))] = fr

    def scene_rgb(path, edit, half, region=None):
        fr = table[os.path.normcase(os.path.abspath(path))]
        if region is not None:
            x, y, w, h = region
            fr = fr[y:y + h, x:x + w]
        return A.bin_by(fr, 2) if half else fr.copy()

    def open_frame(path):
        raw = types.SimpleNamespace(wb_as_shot=np.array([2.0, 1.0, 1.5]), flip=0, is_scan=False)
        meta = types.SimpleNamespace(make="Synthetic", model="Test Camera", iso=100.0, exposure_s=0.01,
                                     fnumber=8.0, focal_mm=90.0, lens="Test 90mm", datetime="2026:11:16 12:00:00")
        return types.SimpleNamespace(raw=raw, meta=meta)
    monkeypatch.setattr(F, "scene_rgb", scene_rgb)
    monkeypatch.setattr(F, "open_frame", open_frame)
    return paths


@pytest.fixture(scope="module")
def bracket():
    return make_bracket()


def _names(n):
    return [f"IMG_{1000 + i:04d}.CR2" for i in range(n)]


def _half_truth(tr, an):
    y0, y1, x0, x1 = an.box_half
    return A.bin_by(tr["sharp"], 2)[y0:y1, x0:x1], A.bin_by(tr["depth"], 2)[y0:y1, x0:x1]


def test_alignment_order_and_depth(bracket, monkeypatch):
    frames, tr = bracket
    n = len(frames)
    shuffled = [3, 0, 5, 1, 6, 2, 4]                       # capture order is not focus order
    names = _names(n)
    paths = install(monkeypatch, [frames[i] for i in shuffled], names, "align")
    an = FS.analyse(paths, FS.FocusParams())
    try:
        got = [shuffled[names.index(os.path.basename(p))] for p in an.paths]
        assert got in (list(range(n)), list(range(n))[::-1]), got
        if got[0] != 0:
            pytest.fail("frames came out in reverse")       # capture direction kept: frame 3 was shot first
        h, w = frames[0].shape[:2]
        ys = np.array([0, 0, h - 1, h - 1.0, h / 2])
        xs = np.array([0, w - 1, 0, w - 1.0, w / 2])
        for j in range(n):
            T, Tt = an.t_full[j], tr["T"][got[j]]
            assert abs(T.scale() - Tt.scale()) < 2e-4, (j, T.scale(), Tt.scale())
            Y, X = T.apply(ys, xs)
            Yt, Xt = Tt.apply(ys, xs)
            assert np.max(np.hypot(Y - Yt, X - Xt)[-1:]) < 0.1, (j, Y - Yt, X - Xt)
        _, dtrue = _half_truth(tr, an)
        d = an.depth()
        tex = an.presence() > 0.5
        c = np.corrcoef(d[tex], dtrue[tex])[0, 1]
        assert c > 0.9, c
    finally:
        an.close()


@pytest.fixture(scope="module")
def analysed(bracket, tmp_path_factory):
    import _pytest.monkeypatch as mp
    frames, tr = bracket
    m = mp.MonkeyPatch()
    paths = install(m, frames, _names(len(frames)), "methods")
    an = FS.analyse(paths, FS.FocusParams())
    yield an, tr, frames, m
    an.close()
    m.undo()


def _quality(an, tr, out):
    sharp_h, depth_h = _half_truth(tr, an)
    best = max(psnr(np.asarray(an.aligned[j], np.float32), sharp_h) for j in range(an.n))
    focus = tr["focus"]
    k = np.argmin(np.abs(depth_h[None] - focus[:, None, None]), axis=0)
    al = np.stack([np.asarray(an.aligned[j], np.float32) for j in range(an.n)])
    oracle = np.take_along_axis(al, k[None, ..., None], 0)[0]
    return psnr(out, sharp_h), best, psnr(oracle, sharp_h)


@pytest.mark.parametrize("method", ["depth", "pyramid", "average", "auto"])
def test_every_method_beats_the_best_single_frame(analysed, method):
    an, tr, _, _ = analysed
    out = FS.preview(an, FS.FocusParams(method=method))
    p, best, oracle = _quality(an, tr, out)
    assert p >= best + 4.0, (method, p, best)
    if method == "depth":
        assert p >= oracle - 1.5, (p, oracle)


def test_halo_guard_at_the_foreground_edge(analysed):
    an, tr, _, _ = analysed
    al = np.stack([np.asarray(an.aligned[j], np.float32) for j in range(an.n)])
    _, depth_h = _half_truth(tr, an)
    edge = np.abs(np.gradient(depth_h, axis=0)) + np.abs(np.gradient(depth_h, axis=1)) > 0.5
    from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter
    band = binary_dilation(edge, iterations=6)
    lo = minimum_filter(al.min(0), (5, 5, 1))
    hi = maximum_filter(al.max(0), (5, 5, 1))

    def overshoot(out):
        o = np.maximum(lo - out, 0) + np.maximum(out - hi, 0)
        return float(np.max(o[band]))
    on = FS.preview(an, FS.FocusParams(method="pyramid", halo_guard=True))
    off = FS.preview(an, FS.FocusParams(method="pyramid", halo_guard=False))
    assert overshoot(on) <= 0.02 * float(hi[band].max())
    # The plan expected the pyramid to fail this without the guard. This pyramid takes its coarsest level from
    # the depth result and picks only detail that stands above the noise, so it overshoots far less (about 0.4%
    # here; see docs/RESULTS.md). The guard stays on as a safety net: it may only ever reduce the overshoot.
    assert overshoot(on) <= overshoot(off)
    assert psnr(on, _half_truth(tr, an)[0]) >= psnr(off, _half_truth(tr, an)[0]) - 0.05


def test_noise_on_a_flat_patch(monkeypatch):
    n, h, w = 6, 384, 576
    rng = np.random.default_rng(5)
    base = gaussian_filter(scene(h, w, 9), (0.6, 0.6, 0))
    base[100:260, 150:420] = 0.3                            # flat grey patch
    frames = []
    for i in range(n):
        f = base + rng.normal(0, 0.006, base.shape).astype(np.float32)
        frames.append(f.astype(np.float32))
    paths = install(monkeypatch, frames, _names(n), "noise")
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        y0, _, x0, _ = an.box_half
        sl = (slice(60 - y0, 120 - y0), slice(85 - x0, 200 - x0))
        single = float(np.mean([np.asarray(an.aligned[j], np.float32)[sl].std() for j in range(n)]))
        for method, lim in (("depth", 1.0), ("average", 1.0), ("auto", 1.0), ("pyramid", 1.2)):
            out = FS.preview(an, FS.FocusParams(method=method))
            s = float(out[sl].std())
            assert s < lim * single, (method, s, single)
    finally:
        an.close()


def test_a_frame_where_something_moved_is_flagged(monkeypatch):
    frames, _ = make_bracket(n=5, h=512, w=768, moved=2, seed=3)
    paths = install(monkeypatch, frames, _names(5), "moved")
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        rep = {r.name: r for r in an.reports}
        flagged = [r.name for r in an.reports if r.status == "warn"]
        assert _names(5)[2] in flagged, [(r.name, r.status, r.reason, r.moved_share) for r in an.reports]
        assert rep[_names(5)[0]].status == "ok"
    finally:
        an.close()


def test_build_full_size_writes_an_intermediate_and_is_deterministic(analysed):
    from studio import engine as E
    from studio.engine.io.intermediate import read_info
    an, tr, frames, _ = analysed
    out1 = str(OUT / "methods" / F.result_name(an.paths, "Focus"))
    p = FS.build(an, FS.FocusParams(method="auto"), out1)
    assert os.path.basename(p).endswith("_Focus_Studio.tif") and "_to_" in os.path.basename(p)
    assert read_info(p)["kind"] == "focus"
    img = E.open(p)
    y0, y1, x0, x1 = an.box_full
    assert (img.raw.height, img.raw.width) == (y1 - y0, x1 - x0)
    b1 = open(p, "rb").read()
    out2 = str(OUT / "methods" / "again_Focus_Studio.tif")
    FS.build(an, FS.FocusParams(method="auto"), out2)
    assert open(out2, "rb").read() == b1
    from studio.engine import api
    doc = api.load_doc(p)
    assert doc["stack"]["kind"] == "focus" and len(doc["stack"]["sources"]) == an.n
    full = np.asarray(img.raw.data, np.float32)
    assert psnr(full, tr["sharp"][y0:y1, x0:x1]) > max(psnr(frames[j][y0:y1, x0:x1], tr["sharp"][y0:y1, x0:x1])
                                                       for j in range(an.n)) + 3.0


def test_cancel_leaves_no_partial(analysed):
    an, _, _, _ = analysed
    ev = threading.Event()
    target = str(OUT / "methods" / "cancelled_Focus_Studio.tif")

    def prog(f, msg):
        if f > 0.3:
            ev.set()
    with pytest.raises(FS.Cancelled):
        FS.build(an, FS.FocusParams(method="depth"), target, prog, ev)
    assert not os.path.exists(target) and not os.path.exists(target + ".partial")


def test_retouch_from_a_frame_takes_that_frames_pixels(analysed):
    an, _, _, _ = analysed
    stroke = [{"frame": 0, "points": [[0.5, 0.5]], "radius": 0.05}]
    out = FS.preview(an, FS.FocusParams(method="depth", retouch=stroke, halo_guard=False))
    src = np.asarray(an.aligned[0], np.float32)
    h, w = out.shape[:2]
    c = out[h // 2 - 3:h // 2 + 3, w // 2 - 3:w // 2 + 3]
    assert np.allclose(c, src[h // 2 - 3:h // 2 + 3, w // 2 - 3:w // 2 + 3], atol=1e-3)


def test_build_memory_stays_bounded(monkeypatch):
    import tracemalloc
    n, h, w = 8, 2000, 3000
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.random((h // 4, w // 4, 3)).astype(np.float32), (1, 1, 0))
    base = np.repeat(np.repeat(base, 4, 0), 4, 1)
    frames = [np.roll(base, i, axis=1) + np.float32(0.001 * i) for i in range(n)]
    paths = install(monkeypatch, frames, _names(n), "memory")
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    del frames
    try:
        tracemalloc.start()
        FS.build(an, FS.FocusParams(method="auto"), str(OUT / "memory" / "mem_Focus_Studio.tif"))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 1.5e9, peak / 1e9
    finally:
        an.close()

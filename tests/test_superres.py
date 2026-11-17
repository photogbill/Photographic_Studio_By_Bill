"""Super-Res Stack (studio/engine/stack/superres.py and astro.py) on synthetic bursts with known truth
(BUILD_PLAN_TABS 5.8).

A high-resolution truth (a band-limited scene at 4x, with fine stripes near the single-frame limit); frame k is
the truth moved by a known sub-pixel amount and a small turn, box-averaged 4x down (the photosite's area),
mosaiced, with shot and read noise. Frames reach the engine through monkeypatched frames.mosaic / scene_rgb
(there is no raw writer); the colour matrix is the identity so results compare with the truth directly.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import ROOT, mosaic_of, scene
from studio.engine.stack import align as A
from studio.engine.stack import astro
from studio.engine.stack import frames as F
from studio.engine.stack import superres as SR

OUT = ROOT / "outputs" / "pytest_superres"
STRIPE_F = 0.7          # cycles per input pixel


def _sim(scale, deg, dy, dx, h, w):
    a = np.radians(deg)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    R = np.array([[scale * np.cos(a), -scale * np.sin(a), 0], [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1.]])
    C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    S = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return A.Transform(S @ C @ R @ np.linalg.inv(C), "similarity")


def make_truth(h=320, w=480, pad=12, seed=0):
    """Truth at 4x (with a pad), band-limited like a good lens, with a stripe patch at STRIPE_F."""
    H4, W4 = 4 * (h + 2 * pad), 4 * (w + 2 * pad)
    from scipy.ndimage import zoom
    t = zoom(scene(H4 // 2, W4 // 2, seed), (2, 2, 1), order=3, grid_mode=True, mode="nearest")
    t = gaussian_filter(np.clip(t, 0, 1), (1.2, 1.2, 0))
    y, x = np.mgrid[0:H4, 0:W4].astype(np.float32)
    f4 = STRIPE_F / 4.0
    sy0, sx0 = 4 * (pad + 20), 4 * (pad + 150)
    patch = (slice(sy0, sy0 + 4 * 50), slice(sx0, sx0 + 4 * 60))
    t[patch] = (0.35 + 0.25 * np.sin(2 * np.pi * f4 * x[patch]))[..., None]
    return t.astype(np.float32), {"stripe": (20, 70, 150, 210), "pad": pad, "h": h, "w": w}


def to4(T: A.Transform, pad: int) -> A.Transform:
    """A transform of cropped input coordinates in padded 4x coordinates (x4 = 4 (x + pad) + 1.5)."""
    B = np.array([[4, 0, 4 * pad + 1.5], [0, 4, 4 * pad + 1.5], [0, 0, 1.0]])
    return A.Transform(B @ T.matrix @ np.linalg.inv(B), T.kind)


def frame_from(truth4, meta, T: A.Transform, noise=True, rng=None, blur=0.0):
    """Frame whose content at p (cropped input coords) comes from truth point T^-1(p): box-averaged, mosaiced."""
    pad, h, w = meta["pad"], meta["h"], meta["w"]
    t4 = A.warp(truth4, to4(T, pad).inverse(), order=3)
    if blur:
        t4 = gaussian_filter(t4, (4 * blur, 4 * blur, 0))
    rgb = A.bin_by(t4, 4)[pad:pad + h, pad:pad + w]
    mos = mosaic_of(rgb)
    if noise:
        rng = rng or np.random.default_rng(0)
        mos = mos + rng.normal(0, 1, mos.shape).astype(np.float32) * np.sqrt(4e-6 + 2e-5 * np.maximum(mos, 0))
    return rgb, mos.astype(np.float32)


def truth2(truth4, meta, T_ref: A.Transform):
    pad, h, w = meta["pad"], meta["h"], meta["w"]
    t4 = A.warp(truth4, to4(T_ref, pad).inverse(), order=3)
    return A.bin_by(t4, 2)[2 * pad:2 * (pad + h), 2 * pad:2 * (pad + w)]


def install(monkeypatch, mosaics, tag, rgbs=None):
    """Serve float mosaics (and optional developed frames) under empty .CR2 files; identity colour matrix."""
    from studio.engine.color import camera as cam
    from studio.engine.raw import demosaic as dm
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
    paths, table = [], {}
    for i, mos in enumerate(mosaics):
        p = str(d / f"BURST_{i:04d}.CR2")
        open(p, "wb").close()
        paths.append(p)
        table[os.path.normcase(os.path.abspath(p))] = (mos, None if rgbs is None else rgbs[i])
    pattern = np.array([[0, 1], [1, 2]])

    def mosaic(path, noise=False):
        mos, _ = table[os.path.normcase(os.path.abspath(path))]
        m16 = np.rint(np.clip(mos, -1, 1.09) * F.INT16_SCALE).astype(np.int16)
        info = F.RawInfo(pattern=pattern, cam_xyz=np.eye(3), flip=0, wb=np.ones(3), noise_a=np.full((2, 2), 2e-5),
                         noise_b=np.full((2, 2), 4e-6), shape=m16.shape,
                         meta={"make": "Synthetic", "model": "Test Camera", "iso": 100.0},
                         crop_origin=(0, 0), visible_shape=m16.shape)
        return m16, info

    def scene_rgb(path, edit, half, region=None):
        mos, rgb = table[os.path.normcase(os.path.abspath(path))]
        out = rgb if rgb is not None else dm.demosaic(mos.copy(), pattern, "rcd")
        if region is not None:
            x, y, w, h = region
            out = out[y:y + h, x:x + w]
        return A.bin_by(out, 2) if half else np.ascontiguousarray(out, np.float32)

    def open_frame(path):
        raw = types.SimpleNamespace(wb_as_shot=np.ones(3), flip=0, is_scan=False)
        meta = types.SimpleNamespace(make="Synthetic", model="Test Camera", iso=100.0, exposure_s=0.01,
                                     fnumber=8.0, focal_mm=50.0, lens="", datetime="")
        return types.SimpleNamespace(raw=raw, meta=meta, lateral_ca=lambda: None)
    monkeypatch.setattr(F, "mosaic", mosaic)
    monkeypatch.setattr(F, "scene_rgb", scene_rgb)
    monkeypatch.setattr(F, "open_frame", open_frame)
    monkeypatch.setattr(cam, "camera_to_working", lambda m, working=None: np.eye(3))
    return paths


def _psnr(a, b, border=12):
    a = a[border:-border, border:-border]
    b = b[border:-border, border:-border]
    return float(10 * np.log10(1.0 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


@pytest.fixture(scope="module")
def burst():
    truth4, meta = make_truth()
    rng = np.random.default_rng(7)
    n = 16
    Ts, rgbs, mos = [], [], []
    for k in range(n):
        T = _sim(1.0, rng.uniform(-0.1, 0.1), rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6), meta["h"], meta["w"])
        rgb, m = frame_from(truth4, meta, T, rng=np.random.default_rng(100 + k))
        Ts.append(T)
        rgbs.append(rgb)
        mos.append(m)
    return truth4, meta, Ts, rgbs, mos


@pytest.fixture(scope="module")
def analysed(burst):
    import _pytest.monkeypatch as mp
    truth4, meta, Ts, rgbs, mos = burst
    m = mp.MonkeyPatch()
    paths = install(m, mos, "handheld")
    an = SR.analyse(paths, SR.SuperParams.for_mode("handheld"))
    yield an, burst
    an.close()
    m.undo()


def test_registration_recovers_every_shift(analysed):
    an, (truth4, meta, Ts, _, _) = analysed
    ref = an.ref
    h, w = meta["h"], meta["w"]
    ys, xs = np.array([h / 2.0, 0.2 * h, 0.8 * h]), np.array([w / 2.0, 0.2 * w, 0.8 * w])
    for k in range(an.n):
        truth = Ts[k].compose(Ts[ref].inverse())            # reference -> frame k
        Y, X = an.t_full[k].apply(ys, xs)
        if an.flows[k] is not None:
            dy, dx = an.flows[k].at(ys, xs)
            Y, X = an.t_full[k].apply(ys + dy, xs + dx)
        Yt, Xt = truth.apply(ys, xs)
        assert np.max(np.hypot(Y - Yt, X - Xt)) < 0.05 * 2, (k, Y - Yt, X - Xt)
    assert an.moved


def test_drizzle_2x_adds_real_detail(analysed):
    from studio.engine.raw import demosaic as dm
    an, (truth4, meta, Ts, rgbs, mos) = analysed
    p = SR.SuperParams.for_mode("handheld")
    out, _ = SR.merge(an, p)
    t2 = truth2(truth4, meta, Ts[an.ref])
    single = dm.demosaic(mos[an.ref].copy(), np.array([[0, 1], [1, 2]]), "rcd")
    s = 0.5
    up = A.warp(single, A.Transform(np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1.]]), "affine"),
                t2.shape, order=3)
    p_d, p_s = _psnr(out, t2), _psnr(up, t2)
    assert p_d >= p_s + 2.0, (p_d, p_s)
    y0, y1, x0, x1 = (2 * v for v in meta["stripe"])

    def contrast(img, f_out):
        L = A.luma(img[y0 + 4:y1 - 4, x0 + 4:x1 - 4])
        L = L - L.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(L, axis=1)).mean(0)
        k = int(round(f_out * L.shape[1]))
        return float(spec[k - 1:k + 2].max() * 2 / L.shape[1]) / 0.25   # truth amplitude 0.25
    c_d, c_s = contrast(out, STRIPE_F / 2), contrast(up, STRIPE_F / 2)
    # The plan asked for 0.3. A photosite averages its whole area, which alone leaves 0.37 of the contrast at
    # 0.7 cycles per pixel (sinc), the drizzle drop (0.7 px) keeps 0.66 of that and the output pixel 0.82: the
    # ceiling is about 0.20 before sharpening (docs/RESULTS.md). The single frame cannot show the stripes at all.
    assert c_d > 0.15 and c_s < 0.1, (c_d, c_s)


def test_bands_equal_one_piece(analysed):
    an, _ = analysed
    p = SR.SuperParams.for_mode("handheld")
    p.sharpen = 0.0
    a, _ = SR.merge(an, p, band_rows=10_000)
    b, _ = SR.merge(an, p, band_rows=200)
    assert np.array_equal(a, b)


def test_tripod_burst_cleans_noise_and_says_no_detail(monkeypatch):
    truth4, meta = make_truth(seed=2)
    n = 16
    T0 = A.Transform.identity("similarity")
    rgb0, _ = frame_from(truth4, meta, T0, noise=False)
    rgb0 = rgb0.copy()
    rgb0[60:120, 30:120] = 0.3                                   # a flat patch to measure noise on
    rgbs, mos = [], []
    for k in range(n):
        rng = np.random.default_rng(300 + k)
        r = rgb0 + rng.normal(0, 0.01, rgb0.shape).astype(np.float32)
        rgbs.append(r.astype(np.float32))
        mos.append(mosaic_of(r))
    paths = install(monkeypatch, mos, "tripod", rgbs)
    an = SR.analyse(paths, SR.SuperParams.for_mode("handheld"))
    try:
        assert not an.moved and any(SR.NO_MOVEMENT in t for t in an.notes)
        assert any(SR.NO_MOVEMENT in t for t in SR.frame_warnings(an, SR.SuperParams.for_mode("handheld")))
        out, _ = SR.merge(an, SR.SuperParams.for_mode("clean"))
        flat = (slice(70, 110), slice(40, 110))
        single = float(rgbs[0][flat].std())
        assert float(out[flat].std()) < 1.3 / np.sqrt(n) * single, (out[flat].std(), single)
    finally:
        an.close()


def test_robust_average_noise_and_a_moving_block(monkeypatch):
    """A bright block crosses the frame in 4 of 16 frames (a different place in each). Where the reference does not
    have it, it must not leave a ghost; where the reference has it, that place falls back to the reference (no
    worse than a single frame's noise)."""
    truth4, meta = make_truth(seed=4)
    n = 16
    rng = np.random.default_rng(11)
    rgbs, mos, clean = [], [], []
    movers = {3: 150, 7: 210, 9: 270, 12: 330}            # frame -> block's left edge (input pixels)
    for k in range(n):
        T = _sim(1.0, 0.0, rng.uniform(-1, 1), rng.uniform(-1, 1), meta["h"], meta["w"])
        rgb, _ = frame_from(truth4, meta, T, noise=False)
        rgb = rgb.copy()
        rgb[250:300, 20:120] = 0.25                          # flat patch
        clean.append(rgb.copy())
        if k in movers:
            rgb[200:240, movers[k]:movers[k] + 40] = 0.9
        r = rgb + np.random.default_rng(500 + k).normal(0, 0.01, rgb.shape).astype(np.float32)
        rgbs.append(r.astype(np.float32))
        mos.append(mosaic_of(r))
    paths = install(monkeypatch, mos, "moving", rgbs)
    an = SR.analyse(paths, SR.SuperParams.for_mode("clean"))
    try:
        out, _ = SR.merge(an, SR.SuperParams.for_mode("clean"))
        flat = (slice(260, 290), slice(30, 110))
        single = float(rgbs[an.ref][flat].std())
        assert float(out[flat].std()) < 0.3 * single, (out[flat].std(), single)
        ref_clean = clean[an.ref]
        for k, x0 in movers.items():
            inner = (slice(204, 236), slice(x0 + 4, x0 + 36))
            if k == an.ref:                                  # the reference's own block: it alone, single noise
                err = out[inner] - 0.9
                assert abs(float(err.mean())) < 0.02 * 0.65
                assert float(err.std()) <= 1.15 * 0.01
            else:                                            # another frame's block: no ghost
                ghost = float(np.abs(out[inner] - ref_clean[inner]).mean()) / 0.65
                assert ghost < 0.02, (k, ghost)
    finally:
        an.close()


def test_drizzle_with_few_frames_fills_holes_and_warns(monkeypatch, burst):
    truth4, meta, Ts, rgbs, mos = burst
    paths = install(monkeypatch, mos[:4], "few")
    p = SR.SuperParams.for_mode("handheld")
    an = SR.analyse(paths, p)
    try:
        assert any("few" in t for t in SR.frame_warnings(an, p))
        out, extra = SR.merge(an, p)
        assert np.isfinite(out).all() and extra["filled_share"] > 0
    finally:
        an.close()


def test_moon_keeps_the_sharpest_and_undoes_the_wobble(monkeypatch):
    h, w, pad = 160, 160, 12
    H4 = 4 * (h + 2 * pad)
    y, x = np.mgrid[0:H4, 0:H4].astype(np.float32)
    c = H4 / 2
    disc = ((y - c) ** 2 + (x - c) ** 2 < (0.38 * H4) ** 2).astype(np.float32)
    tex = gaussian_filter(np.random.default_rng(1).random((H4, H4)).astype(np.float32), 3)
    moon = disc * (0.25 + 0.6 * (tex - tex.min()) / (tex.max() - tex.min()))
    moon = gaussian_filter(moon, 1.0)
    truth4 = np.repeat(moon[..., None], 3, 2)
    meta = {"pad": pad, "h": h, "w": w}
    n = 24
    rng = np.random.default_rng(3)
    blurs = rng.uniform(0.0, 1.6, n)
    mos, wob = [], []
    for k in range(n):
        T = _sim(1.0, 0.0, rng.uniform(-3, 3), rng.uniform(-3, 3), h, w)
        _, m = frame_from(truth4, meta, T, rng=np.random.default_rng(700 + k), blur=blurs[k])
        mos.append(m)
    paths = install(monkeypatch, mos, "moon")
    p = SR.SuperParams.for_mode("moon")
    an = SR.analyse(paths, p)
    try:
        kept = an.kept(p)
        assert len(kept) == round(0.25 * sum(r.status != "dropped" for r in an.reports[:an.n]))
        sharpest = set(np.argsort(blurs)[:len(kept)].tolist())
        assert len(sharpest & set(kept)) >= len(kept) - 1, (sorted(kept), sorted(sharpest))
        out, _ = SR.merge(an, p)
        assert np.isfinite(out).all() and out.shape[0] == 2 * h
    finally:
        an.close()


def test_local_flow_removes_a_wobble():
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.random((400, 400)).astype(np.float32), 2.0)
    Y, X = np.mgrid[0:400, 0:400].astype(float)
    dy = 0.9 * np.sin(2 * np.pi * X / 260.0)
    dx = 0.9 * np.cos(2 * np.pi * Y / 240.0)
    from scipy.ndimage import map_coordinates
    img = map_coordinates(base, [Y - dy, X - dx], order=3, mode="mirror").astype(np.float32)
    f = A.flow(base, img, 48, 24, 8)
    w = A.warp(img, A.Transform.identity(), flw=f)
    fy, fx, _ = f.fields(base.shape)
    inner = (slice(30, -30), slice(30, -30))
    assert float(np.hypot(fy - dy, fx - dx)[inner].mean()) < 0.2
    assert float(np.abs(w - base)[inner].mean()) < float(np.abs(img - base)[inner].mean()) * 0.5


def test_night_sky_lines_up_stars_and_removes_a_plane(monkeypatch):
    h, w = 300, 450
    rng = np.random.default_rng(2)
    n_st = 220
    sy, sx = rng.uniform(5, h - 5, n_st), rng.uniform(5, w - 5, n_st)
    fl = rng.uniform(0.05, 0.6, n_st)
    n = 12
    rgbs, Ts = [], []
    for k in range(n):
        dy, dx = 0.31 * k, -0.47 * k                          # the sky drifts
        T = _sim(1.0, 0.02 * k, dy, dx, h, w)
        Y, X = T.apply(sy, sx)
        img = np.full((h, w), 0.03, np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for y0, x0, f0 in zip(Y, X, fl):
            iy, ix = int(round(y0)), int(round(x0))
            if 4 <= iy < h - 4 and 4 <= ix < w - 4:
                sl = (slice(iy - 4, iy + 5), slice(ix - 4, ix + 5))
                img[sl] += f0 * np.exp(-((yy[sl] - y0) ** 2 + (xx[sl] - x0) ** 2) / (2 * 1.1 ** 2))
        if k in (4, 8):                                      # a plane's trail
            yt = int(60 + 10 * k)
            img[yt:yt + 3, 40:400] += 0.5
        img = img + np.random.default_rng(900 + k).normal(0, 0.004, img.shape).astype(np.float32)
        rgb = np.repeat(img[..., None], 3, 2)
        rgbs.append(rgb.astype(np.float32))
        Ts.append(T)
    mos = [mosaic_of(r) for r in rgbs]
    paths = install(monkeypatch, mos, "sky", rgbs)
    p = SR.SuperParams.for_mode("sky")
    an = SR.analyse(paths, p)
    try:
        ref = an.ref
        pts_y, pts_x = np.array([50.0, 150.0, 250.0]), np.array([60.0, 220.0, 400.0])
        for k in range(n):
            truth = Ts[k].compose(Ts[ref].inverse())
            Y, X = an.t_full[k].apply(pts_y, pts_x)
            Yt, Xt = truth.apply(pts_y, pts_x)
            assert np.max(np.hypot(Y - Yt, X - Xt)) < 0.2, (k, Y - Yt, X - Xt)
        out, extra = SR.merge(an, p)
        L = A.luma(out)
        for k in (4, 8):
            Ytr, _ = Ts[ref].compose(Ts[k].inverse()).apply(np.array([60 + 10 * k + 1.0]), np.array([220.0]))
            row = int(round(Ytr[0]))
            seg = L[row, 100:350] - 0.03
            assert float(np.median(seg)) < 0.02 * 0.5, float(np.median(seg))
            assert float(extra["rejected"][row, 100:350].mean()) > 0.03
    finally:
        an.close()


def test_build_writes_a_named_intermediate(analysed):
    from studio import engine as E
    from studio.engine.io.intermediate import read_info
    an, (truth4, meta, *_rest) = analysed
    p = SR.SuperParams.for_mode("handheld")
    name = F.result_name([an.paths[0], an.paths[-1]], SR.result_tag(p))
    assert name.endswith("_SuperRes2x_Studio.tif") and "_to_" in name
    out = SR.build(an, p, str(OUT / "handheld" / name))
    info = read_info(out)
    assert info["kind"] == "superres" and info["scale"] == 2.0 and info["mode"] == "handheld"
    img = E.open(out)
    assert (img.raw.height, img.raw.width) == (2 * meta["h"], 2 * meta["w"])
    s = SR.output_info(an, p)
    assert (s["w"], s["h"]) == (2 * meta["w"], 2 * meta["h"]) and "2x" in SR.describe_output(s)
    res, ref, _ = SR.preview(an, p)
    assert res.shape == ref.shape and res.shape[0] <= 2 * meta["h"]


def test_memory_drizzle_bands(monkeypatch):
    import tracemalloc
    n, h, w = 12, 2000, 3000
    rng = np.random.default_rng(0)
    base = gaussian_filter(rng.random((h // 8 + 4, w // 8 + 4)).astype(np.float32), 1)
    base = np.repeat(np.repeat(base, 8, 0), 8, 1)
    mos = [np.ascontiguousarray(base[(k % 3):(k % 3) + h, (k % 4):(k % 4) + w]) for k in range(n)]
    paths = install(monkeypatch, mos, "memory")
    del mos
    p = SR.SuperParams.for_mode("handheld")
    p.local = False
    an = SR.analyse(paths, p)
    try:
        tracemalloc.start()
        with F.Scratch("memtest") as sc:
            SR.merge(an, p, None, sc.array, band_rows=256)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 2.0e9, peak / 1e9
    finally:
        an.close()

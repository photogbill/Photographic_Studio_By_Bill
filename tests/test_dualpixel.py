"""Dual Pixel Lab engine (studio/engine/dualpixel) on a synthetic split-aperture photo with known truth
(BUILD_PLAN_TABS 3.7).

A scene and a depth map (a ramp plus two blocks, one of them flat); the offset between the halves is
d = k * (depth - focus). Half 1 = the scene moved by +d/2 along x and blurred by |d|, half 2 = moved by -d/2; each
multiplied by opposite left-right shading ramps (1.6 to 0.6, with a mild colour cast); A+B = their sum; both
mosaiced, independent shot and read noise in each half, black level added, A+B clipped at the white level.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from conftest import ROOT, mosaic_of, scene, synthetic_raw

OUT = ROOT / "outputs" / "pytest_dualpixel"
BLACK, WHITE = 512.0, 15000.0
WB = np.array([2.0, 1.0, 1.5])
K = 1.6                     # offset (full-size px) per unit of depth from focus
FOCUS = 2.0


def make_dual(h=384, w=576, seed=0, bright=False):
    from studio.engine.dualpixel.decode import make_dual as mk
    from studio.engine.pipeline.pipeline import Image
    rng = np.random.default_rng(seed)
    sc = gaussian_filter(scene(h, w, seed), (0.7, 0.7, 0)).astype(np.float32)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = 1.0 + 2.0 * x / (w - 1)                         # ramp: 1 (in front) .. 3 (behind), focus at 2
    depth[30:150, 60:200] = 0.6                              # a block in front (bigger than a 48 px tile)
    flat = (slice(200, 360), slice(30, 200))
    sc[flat] = 0.35                                          # a plain patch: nothing to measure there
    depth[flat] = 3.0
    if bright:                                               # a white patch at 1.8x the clip, on the right
        sc[60:130, 400:520] = 1.8
    d = K * (depth - FOCUS)                                  # full-size px
    shade1 = np.linspace(1.6, 0.6, w, dtype=np.float32)[None, :, None] * np.array([1.0, 0.97, 0.92], np.float32)
    shade2 = 2.0 - shade1                                    # together they make the full photo's own light
    halves = []
    for sgn, shade in ((1.0, shade1), (-1.0, shade2)):
        moved = np.stack([map_coordinates(sc[..., c], [y, x - sgn * d / 2.0], order=3, mode="nearest")
                          for c in range(3)], 2)
        blur = np.stack([gaussian_filter(moved[..., c], 0.5 * float(np.median(np.abs(d)))) for c in range(3)], 2)
        sig = np.abs(d)[..., None]
        moved = np.where(sig > 0.6, blur, moved)
        halves.append(0.5 * moved * shade)                   # one photodiode: half the light
    raw_h = []
    for hv in halves:
        m = mosaic_of(hv / WB)                               # undo white balance: raw camera values
        m = m + rng.normal(0, 1, m.shape).astype(np.float32) * np.sqrt(2e-6 + 1.5e-5 * np.maximum(m, 0))
        raw_h.append(m)
    top, left = 16, 32
    full_ab = np.full((h + top, w + left), BLACK, np.float32)
    full_h1 = np.full((h + top, w + left), BLACK, np.float32)
    full_ab[top:, left:] = BLACK + (raw_h[0] + raw_h[1]) * (WHITE - BLACK)
    full_h1[top:, left:] = BLACK + raw_h[0] * (WHITE - BLACK)
    full_ab = np.clip(np.rint(full_ab), 0, WHITE).astype(np.uint16)
    full_h1 = np.clip(np.rint(full_h1), 0, WHITE).astype(np.uint16)
    base = synthetic_raw(np.zeros((h, w, 3), np.float32), black=BLACK, white=WHITE, wb=tuple(WB), top=top, left=left)
    raw = dataclasses.replace(base, mosaic=full_ab, path=str(OUT / "IMG_DUAL.CR2"))
    dp = mk(Image(raw), full_h1, None, "synthetic", str(OUT / "IMG_DUAL.CR2"))
    truth = {"disp": -d, "shade1": shade1, "shade2": shade2, "flat": flat, "scene": sc, "depth": depth}
    # where an answer exists: texture in the scene, and away from depth steps (a tile there sees two depths)
    from scipy.ndimage import binary_dilation, uniform_filter
    L = sc.mean(2)
    tex = np.sqrt(np.maximum(uniform_filter(L * L, 9) - uniform_filter(L, 9) ** 2, 0)) > 0.02
    step = (np.abs(np.gradient(depth, axis=0)) + np.abs(np.gradient(depth, axis=1))) > 0.2
    truth["measurable"] = tex & ~binary_dilation(step, iterations=24)
    return dp, truth


@pytest.fixture(scope="module")
def dual():
    OUT.mkdir(parents=True, exist_ok=True)
    return make_dual()


def test_halves_and_masks(dual):
    dp, _ = dual
    ok = ~dp.clipped_ab
    assert np.array_equal(dp.b[ok], np.maximum(dp.ab - dp.a, 0)[ok])
    assert 1.9 < dp.level_ratio < 2.4
    assert dp.a.shape == dp.ab.shape == dp.b.shape


def test_shading_is_recovered(dual):
    from studio.engine.dualpixel.decode import shading
    dp, tr = dual
    ga, gb = shading(dp)
    h, w = dp.shape
    s1, s2 = tr["shade1"][0], tr["shade2"][0]                # (w, 3)
    true_ga = (s1 + s2) / (2 * s1)
    true_gb = (s1 + s2) / (2 * s2)
    for i in (0, 1):
        for j in (0, 1):
            c = int(dp.pattern[i, j])
            pa = ga.plane(i, j, (h, w))
            pb = gb.plane(i, j, (h, w))
            xs = j + 2 * np.arange(pa.shape[1])
            inner = slice(8, -8)
            ra = pa[:, inner].mean(0) / true_ga[xs[inner], c]
            rb = pb[:, inner].mean(0) / true_gb[xs[inner], c]
            assert np.max(np.abs(ra - 1)) < 0.03, (i, j, np.max(np.abs(ra - 1)))
            assert np.max(np.abs(rb - 1)) < 0.03, (i, j, np.max(np.abs(rb - 1)))
    # after correction the halves agree on the plain patch
    fl = tr["flat"]
    A = (dp.a * ga.full())[fl].mean()
    B = (dp.b * gb.full())[fl].mean()
    assert abs(A / B - 1) < 0.03


@pytest.fixture(scope="module")
def depth_maps(dual):
    from studio.engine.dualpixel import depth as D
    from studio.engine.dualpixel.decode import shading
    dp, tr = dual
    gains = shading(dp)
    la, lb = D.halves_luma(dp, gains)
    disp, conf = D.disparity(la, lb)
    ref = D.refine(disp, conf, la + lb)
    rho = D.detail_map(la, lb, ref)
    return gains, la, lb, disp, conf, ref, rho


def test_disparity_matches_the_true_offset(dual, depth_maps):
    from studio.engine.stack.align import bin_by
    dp, tr = dual
    gains, la, lb, disp, conf, ref, rho = depth_maps
    truth = bin_by(tr["disp"], 2)[:ref.shape[0], :ref.shape[1]]
    meas = bin_by(tr["measurable"].astype(np.float32), 2)[:ref.shape[0], :ref.shape[1]] > 0.99
    m = np.zeros(meas.shape, bool)
    m[12:-12, 12:-12] = True
    sel = (conf > 0.3) & m & meas
    assert sel.mean() > 0.1
    c = np.corrcoef(ref[sel], truth[sel])[0, 1]
    err = float(np.abs(ref[sel] - truth[sel]).mean())
    assert c > 0.9 and err < 0.15, (c, err)
    fy, fx = tr["flat"]
    flat = (slice(fy.start // 2 + 26, fy.stop // 2 - 26), slice(fx.start // 2 + 26, fx.stop // 2 - 26))
    assert float(conf[flat].mean()) < 0.3, float(conf[flat].mean())
    band = (np.abs(truth) < 0.05) & sel
    if band.sum() > 20:
        assert float(np.abs(ref[band]).mean()) < 0.1


def test_detail_map_separates_noise_from_detail(dual, depth_maps):
    dp, tr = dual
    gains, la, lb, disp, conf, ref, rho = depth_maps
    fy, fx = tr["flat"]
    flat = (slice(fy.start // 2 + 6, fy.stop // 2 - 6), slice(fx.start // 2 + 6, fx.stop // 2 - 6))
    assert float(rho[flat].mean()) < 0.2
    tex = (slice(100, 160), slice(140, 200))                 # textured, near focus
    assert float(rho[tex].mean()) > 0.8, float(rho[tex].mean())


def test_focus_report_and_air(dual, depth_maps):
    from studio.engine.dualpixel import depth as D
    gains, la, lb, disp, conf, ref, rho = depth_maps
    rep = D.focus_report(ref, conf)
    assert 0 < rep["in_focus_share"] < 1 and isinstance(rep["sentence"], str)
    air = D.air_steadiness(ref, conf)
    assert air["spread_px"] >= 0 and air["map"].shape == ref.shape


def test_highlight_recovery():
    from studio.engine.dualpixel import hdr
    from studio.engine.dualpixel.decode import shading
    from studio.engine.pipeline.pipeline import linear_rgb
    dp, tr = make_dual(seed=2, bright=True)
    gains = shading(dp)
    out, rep = hdr.recover(dp, gains)
    normal = linear_rgb(dp.image, hdr.hdr_edit(), None, cache=False)
    w = hdr.weight_map(dp)
    keep = (w == 0)
    assert np.array_equal(out[keep], normal[keep])           # untouched below the blend: bit for bit
    patch = (slice(70, 120), slice(420, 500))
    good = (~np.repeat(np.repeat(dp.clipped_a, 1, 0), 1, 1))[patch]
    rec = out[patch][good]
    assert abs(float(np.median(rec)) / 1.8 - 1) < 0.05, float(np.median(rec))
    assert 0.8 <= rep["stops"] <= 1.0, rep
    assert "Recovered" in rep["sentence"]


def test_a_normal_raw_is_not_dual_pixel(monkeypatch, tmp_path):
    import rawpy
    from studio.engine import api
    from studio.engine.dualpixel import decode
    from studio.engine.io.rawfile import UnsupportedRaw
    from studio.engine.pipeline.pipeline import Image
    p = OUT / "plain.tif"
    OUT.mkdir(parents=True, exist_ok=True)
    import tifffile
    tifffile.imwrite(str(p), np.zeros((8, 8, 3), np.uint16), photometric="rgb")
    assert not decode.is_dual_pixel(str(p))
    img = Image(synthetic_raw(scene(64, 96)))
    monkeypatch.setattr(api, "open", lambda path: img)

    def no_second(path, shot_select=0):
        raise rawpy.LibRawRequestForNonexistentImageError("Request for nonexisting image number")
    monkeypatch.setattr(rawpy, "imread", no_second)
    with pytest.raises(UnsupportedRaw) as e:
        decode.open_dual("normal.cr2")
    assert decode.NOT_DUAL in str(e.value) and "Enable" in str(e.value)


def test_views_and_3d_outputs(dual, depth_maps):
    from PIL import Image as PI
    from studio.engine.dualpixel import views as V
    from studio.engine.io.intermediate import read_info
    dp, tr = dual
    gains, la, lb, disp, conf, ref, rho = depth_maps
    maps = {"disp": ref, "conf": conf, "rho": rho}
    for name in V.VIEWS:
        v = V.view(dp, gains, name, 0.5, maps)
        assert v.dtype == np.uint8 and v.ndim == 3 and abs(v.shape[0] - dp.shape[0] // 2) <= 1, (name, v.shape)
    base = str(OUT / "IMG_DUAL.CR2")
    p = V.export_half(dp, gains, 1, str(OUT / V.out_name(base, "half1")))
    assert p.endswith("IMG_DUAL_DP-A_Studio.tif") and read_info(p)["kind"] == "dualpixel"
    g = V.wigglegram(dp, gains, str(OUT / V.out_name(base, "wiggle")), 300, boost=2.0, disp=ref)
    with PI.open(g) as im:
        assert im.n_frames == 2
    an = V.anaglyph(dp, gains, 0.25)
    assert an.shape[2] == 3
    sbs = V.side_by_side(dp, gains, 0.25)
    assert sbs.shape[1] > 2 * an.shape[1]
    from studio.engine.stack.focus import _up2
    d = V.depth_png(_up2(ref, dp.shape), conf, str(OUT / V.out_name(base, "depth")))
    with PI.open(d) as im:
        assert im.mode.startswith("I") and im.size == (dp.shape[1], dp.shape[0])
    assert V.out_name(base, "hdr") == "IMG_DUAL_DP-HDR_Studio.tif"


REAL = sorted((ROOT / "test_raws" / "dual_pixel").glob("*.CR*")) if (ROOT / "test_raws" / "dual_pixel").is_dir() \
    else []


@pytest.mark.skipif(not REAL, reason="no Dual Pixel RAW files in test_raws/dual_pixel yet")
def test_real_dual_pixel_files():
    from studio.engine.dualpixel import decode
    for p in REAL:
        assert decode.is_dual_pixel(str(p))
        dp = decode.open_dual(str(p))
        assert 1.9 <= dp.level_ratio <= 2.1, dp.level_ratio

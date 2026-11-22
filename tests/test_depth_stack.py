"""Depth Stack (BUILD_PLAN_DEPTH_STACK): depth measured two ways in a hand-held focus stack of Dual Pixel RAWs,
fused, with a map of where the two agree, and the surface files for games. Synthetic stacks with known truth
(tests/depth_synth.py: a disc blur, half-disc halves, uneven hand-held steps, shift, turn and focus breathing,
R5 noise at ISO 100 and 800). Nothing is written outside outputs/.

What "alone" means in the comparisons:
  focus alone          Focus Stack's own depth (which frame is sharpest, edge-aware smoothed: an.depth()) and the
                       sharper sub-frame estimate this build adds (depthfuse.focus_peak)
  one Dual Pixel frame the reference frame's own offset (the single-photo method), on the stack's step
  Dual Pixel alone     every frame's Dual Pixel votes on the stack's measured steps, without focus
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

import depth_synth as DS
from studio.engine.stack import depthfuse as DF
from studio.engine.stack import focus as FS
from studio.engine.stack import surface as SURF


def _rms(a, truth, mask):
    e = (np.asarray(a, np.float64) - truth)[mask]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e * e)))


def _analyse(monkeypatch, tag, **kw):
    frames, halves, tr = DS.make_stack(**kw)
    paths = DS.install(monkeypatch, frames, halves, tag)
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    return an, tr, paths


@pytest.fixture(scope="module", params=["iso100", "iso800"])
def handheld(request):
    import _pytest.monkeypatch as mp
    m = mp.MonkeyPatch()
    an, tr, paths = _analyse(m, "hand_" + request.param, iso=request.param)
    res = DF.fuse(an)
    yield an, tr, res, request.param
    an.close()
    m.undo()


def test_fused_beats_each_measure_alone(handheld):
    an, tr, res, iso = handheld
    assert an.dual_pixel and res.method == "focus+dual_pixel"
    z, flat, moved, meas = DS.half_truth(tr, an)
    fused = _rms(res.depth, z, meas)
    focus_sub = _rms(res.focus_depth, z, meas)
    focus_old = _rms(an.depth(), z, meas)
    dp_all = _rms(np.where(np.isfinite(res.dp_depth), res.dp_depth, res.depth), z, meas)
    # one Dual Pixel frame: the reference frame's offset, filled where it says nothing
    q, s = res.positions, res.step_px
    d0, c0 = np.asarray(an.dp_disp[an.ref], np.float32), np.asarray(an.dp_conf[an.ref], np.float32)
    ok = (c0 > DF.DP_CONF_MIN) & (np.abs(d0) < DF.DP_LIMIT_PX)
    one = np.where(ok, q[an.ref] - d0 / s, np.nan)
    one = np.where(ok, one, DF.push_pull(np.nan_to_num(one), ok.astype(np.float32)))
    single = _rms(one, z, meas)
    print(f"\n{iso}: fused {fused:.4f}  dp-alone {dp_all:.4f}  focus sub-frame {focus_sub:.4f}  "
          f"Focus Stack depth {focus_old:.4f}  one DP frame {single:.4f}")
    assert fused < dp_all, (fused, dp_all)
    assert fused < 0.35 * focus_sub, (fused, focus_sub)
    assert fused < 0.2 * focus_old, (fused, focus_old)
    assert fused < 0.6 * single, (fused, single)
    assert fused < (0.05 if iso == "iso100" else 0.065)


def test_the_focus_step_is_measured_from_the_stack(handheld):
    """s within 5 % of the truth (in the reference frame's pixels), and the uneven hand-held steps followed."""
    an, tr, res, _ = handheld
    assert abs(res.step_px / tr["s_ref"] - 1) < 0.05, (res.step_px, tr["s_ref"])
    true_q = (tr["planes"] - tr["planes"][0]) / tr["median_step"]
    assert np.max(np.abs(np.asarray(res.positions) - true_q)) < 0.15, (res.positions, true_q)
    assert res.step_spread > 0.15                       # a hand does not step evenly, and the spread says so
    assert res.sigma_dp is not None and res.sigma_focus is not None


def test_a_plain_area_is_never_invented(handheld):
    an, tr, res, _ = handheld
    z, flat, moved, meas = DS.half_truth(tr, an)
    inner = binary_erosion(flat, iterations=24)      # the measuring windows reach about this far in
    assert inner.sum() > 1000
    assert np.all(res.label[inner] == DF.LABEL_FILLED)
    assert float(res.confidence[inner].max()) == 0.0
    assert np.all(~np.isfinite(res.agreement[inner]))
    sh = res.shares()
    assert sh["filled"] > 0.01 and sh["measured_both"] > 0.9


def test_a_moving_patch_shows_in_the_agreement_map(monkeypatch):
    an, tr, _ = _analyse(monkeypatch, "moved", n=9, h=640, w=960, moved={4: 1.5, 5: 1.5, 6: 1.5})
    try:
        res = DF.fuse(an)
        z, flat, moved, meas = DS.half_truth(tr, an)
        inner = binary_erosion(moved, iterations=10)
        ag = res.agreement
        inside = np.nanmedian(ag[inner])
        outside = np.nanmedian(ag[meas])
        assert inside > 4 * outside, (inside, outside)
        assert np.mean(np.nan_to_num(ag[inner]) > DF.DISAGREE_STEPS) > 0.25
        assert np.mean(np.nan_to_num(ag[meas]) > DF.DISAGREE_STEPS) < 0.01
        assert np.median(res.confidence[inner]) < 0.6 < np.median(res.confidence[meas])
    finally:
        an.close()


def test_even_steps_on_a_rail_fuse_at_least_as_well_as_the_better_measure(monkeypatch):
    """On a rail with even steps focus is good on its own; the fusion weighs itself by what it measures."""
    an, tr, _ = _analyse(monkeypatch, "rail", n=9, h=640, w=960, handheld=False, uneven=False)
    try:
        res = DF.fuse(an)
        z, _, _, meas = DS.half_truth(tr, an)
        fused = _rms(res.depth, z, meas)
        focus = _rms(res.focus_depth, z, meas)
        dp = _rms(np.where(np.isfinite(res.dp_depth), res.dp_depth, res.depth), z, meas)
        assert fused <= 1.02 * min(focus, dp), (fused, focus, dp)
        assert res.step_spread < 0.05
    finally:
        an.close()


def test_ordinary_raws_give_the_same_files_from_focus_alone(monkeypatch, tmp_path):
    frames, _, tr = DS.make_stack(n=7, h=512, w=768)
    paths = DS.install(monkeypatch, frames, None, "ordinary", names=[f"IMG_{7000 + i}.CR3" for i in range(7)])
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        assert not an.dual_pixel and an.dp_disp is None
        res = DF.fuse(an)
        assert res.method == "focus" and res.units == "frames" and res.step_px is None
        assert set(np.unique(res.label)) <= {DF.LABEL_FILLED, DF.LABEL_ONE}
        assert not np.isfinite(res.agreement).any()
        z, _, _, meas = DS.half_truth(tr, an)
        assert np.corrcoef(res.depth[meas], z[meas])[0, 1] > 0.95
        r = SURF.stack_surface(an, str(tmp_path), "ordinary", SURF.SurfaceParams())
        assert set(r["paths"]) == set(SURF.KINDS) | {"json"}
        doc = json.load(open(r["paths"]["json"], encoding="utf-8"))
        assert doc["method"] == "focus" and doc["dual_pixel"] is None and not doc["agreement"]["available"]
    finally:
        an.close()


def test_one_dual_pixel_photo_gives_its_own_surface(tmp_path):
    import test_dualpixel as TD
    TD.OUT.mkdir(parents=True, exist_ok=True)
    dp, tr = TD.make_dual()
    r = SURF.dual_surface(dp, str(tmp_path), "IMG_DUAL", SURF.SurfaceParams())
    doc = r["json"]
    assert set(r["paths"]) == set(SURF.KINDS) | {"json"}
    assert doc["method"] == "dual_pixel" and doc["height"]["units"] == "relative"
    from PIL import Image
    h = np.asarray(Image.open(r["paths"]["height"])).astype(np.float64)
    assert h.shape == dp.shape                          # the photo's own size (upright: flip 0 here)
    from studio.engine.stack.align import bin_by
    hs = bin_by(h, 4)
    dt = bin_by(tr["depth"], 4)
    m = bin_by(tr["measurable"].astype(np.float32), 4) > 0.9
    c = np.corrcoef(hs[m], dt[m])[0, 1]
    assert abs(c) > 0.8, c
    r2 = SURF.dual_surface(dp, str(tmp_path), "IMG_DUAL", SURF.SurfaceParams(swap=True))
    h2 = np.asarray(Image.open(r2["paths"]["height"])).astype(np.float64)
    c2 = np.corrcoef(bin_by(h2, 4)[m], dt[m])[0, 1]
    assert np.sign(c2) == -np.sign(c) and abs(c2) > 0.8           # swap turns near and far round
    assert r2["paths"]["height"] == r["paths"]["height"]           # its own earlier file is replaced, not -2


def test_tileable_repeats_without_a_join():
    rng = np.random.default_rng(2)
    H, W = 240, 360
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(rng.random((H, W)), 3).astype(np.float32)
    img += np.linspace(0, 0.5, W, dtype=np.float32)[None, :]          # a gradient: the worst case for a seam
    w = SURF.tile_weights(H, W)
    tot = w["A"] + w["B"] + w["C"] + w["D"]
    assert np.allclose(tot, 1.0, atol=1e-5)
    t = SURF.tile(img, w)

    def jump(a, axis):
        inner = np.abs(np.diff(a, axis=axis)).mean()
        wrap = np.abs(np.take(a, 0, axis) - np.take(a, -1, axis)).mean()
        return wrap / inner
    assert jump(img, 1) > 10                                         # the picture as it is has a big join
    assert jump(t, 1) < 2.0 and jump(t, 0) < 2.0                     # tiled: no join either way
    c = (slice(H // 2 - 10, H // 2 + 10), slice(W // 2 - 10, W // 2 + 10))
    assert np.allclose(t[c], img[c], atol=1e-6)                      # the middle is the picture itself


def test_normals_follow_the_height_and_the_convention():
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)
    h = x / 63.0                                                     # rises to the right (near on the right)
    n_up = SURF.normals(h, 63.0, True)
    assert np.allclose(n_up[10:-10, 10:-10, 0], -np.sqrt(0.5), atol=1e-3)   # 45 degrees, facing left
    h2 = y / 63.0                                                    # rises downwards in the picture
    up, down = SURF.normals(h2, 63.0, True), SURF.normals(h2, 63.0, False)
    assert up[32, 32, 1] > 0.5 and down[32, 32, 1] < -0.5          # OpenGL green up, DirectX green down


def test_ratio_and_step_give_millimetres(monkeypatch, tmp_path):
    an, tr, _ = _analyse(monkeypatch, "mm", n=7, h=512, w=768)
    try:
        r = SURF.stack_surface(an, str(tmp_path), "mm", SURF.SurfaceParams(ratio="1:2", step_mm=0.2),
                               SURF.DEPTH_KINDS + ("normal",))
        doc = r["json"]
        assert doc["height"]["units"] == "mm"
        assert doc["height"]["range"]["mm"] == pytest.approx(0.2 * doc["height"]["range"]["in_depth_units"],
                                                             rel=1e-3)
        assert doc["lateral"]["mm_per_pixel"] == pytest.approx(36.0 / 8192 / 0.5, rel=1e-3)
        assert doc["normal"]["scale"].startswith("physical")
        assert doc["displacement"]["height_range_over_width"] is not None
        assert "colour" not in r["paths"]
        with pytest.raises(ValueError):
            SURF.parse_ratio("banana")
        assert SURF.parse_ratio("2:1") == 2.0 and SURF.parse_ratio("") is None
    finally:
        an.close()


def test_flip_near_far_turns_the_height_round(monkeypatch, tmp_path):
    from PIL import Image
    from studio.engine.dualpixel import depth as DEP
    monkeypatch.delitem(DEP.NEAR_SIGN, "Canon EOS R5", raising=False)   # a camera not yet checked (the synthetic
    #                                                                   stack's halves read the other way round)
    an, tr, _ = _analyse(monkeypatch, "flip", n=7, h=512, w=768)
    try:
        a = SURF.stack_surface(an, str(tmp_path / "a"), "f", SURF.SurfaceParams(near="first"), ("height",))
        b = SURF.stack_surface(an, str(tmp_path / "b"), "f", SURF.SurfaceParams(near="last"), ("height",))
        ha = np.asarray(Image.open(a["paths"]["height"])).astype(np.float64) / 65535
        hb = np.asarray(Image.open(b["paths"]["height"])).astype(np.float64) / 65535
        assert np.corrcoef(ha.ravel(), hb.ravel())[0, 1] < -0.98
        assert a["json"]["height"]["near_end"].startswith("set by you")
        # auto, with this camera's Dual Pixel reading not yet checked: the first frame, said plainly
        c = SURF.stack_surface(an, str(tmp_path / "c"), "f", SURF.SurfaceParams(), ("height",))
        assert c["json"]["height"]["near_end"].startswith("assumed") and not c["json"]["height"]["near_measured"]
        hc = np.asarray(Image.open(c["paths"]["height"])).astype(np.float64) / 65535
        assert np.allclose(hc, ha, atol=2 / 65535)
    finally:
        an.close()


def test_a_checked_camera_reads_near_from_its_halves(monkeypatch, tmp_path):
    """Once a camera's Dual Pixel reading is in the table, near comes from the halves, whatever the frame order."""
    from studio.engine.dualpixel import depth as DEP
    an, tr, _ = _analyse(monkeypatch, "checked", n=7, h=512, w=768)
    try:
        res = DF.fuse(an)
        for sign in (-1, 1):
            monkeypatch.setitem(DEP.NEAR_SIGN, "Canon EOS R5", sign)
            near_high, how, measured, _ = SURF.decide_near(res, "Canon EOS R5", SURF.SurfaceParams())
            assert measured and how.startswith("measured")
            assert near_high == (np.sign(res.step_px) != sign)
    finally:
        an.close()


def test_depth_view_colours_say_how_each_place_was_measured(handheld):
    an, tr, res, _ = handheld
    rgb = SURF.depth_view(res, False, an.flip, overlay=False).astype(np.float32)
    lab = res.label
    for k, col in SURF.VIEW_RGB.items():
        m = lab == k
        if m.sum() < 50:
            continue
        mean = rgb[m].mean(0)
        hue = mean / max(mean.max(), 1e-6)
        ref = np.asarray(col, np.float32) / max(col)
        assert np.allclose(hue, ref, atol=0.08), (k, hue, ref)
    assert "Measured two ways" in SURF.summary(res)


def test_shading_gain_fast_path_is_exact():
    """GainField.plane works the quartic as (powers of v) @ C @ (powers of u): the same numbers as on the grid."""
    from studio.engine.dualpixel.decode import GainField, _TERMS, _poly_eval
    rng = np.random.default_rng(0)
    coef = {(i, j): rng.normal(0, 0.3, len(_TERMS)).tolist() for i in (0, 1) for j in (0, 1)}
    h, w = 130, 197
    g = GainField(coef, (h, w))
    for (i, j), c in coef.items():
        ph, pw = (h - i + 1) // 2, (w - j + 1) // 2
        u = (j + 2 * np.arange(pw) - (w - 1) / 2.0) / w
        v = (i + 2 * np.arange(ph) - (h - 1) / 2.0) / h
        U, V = np.meshgrid(u, v)
        assert np.allclose(g.plane(i, j), np.exp(_poly_eval(U, V, c)), rtol=1e-5)


def test_surface_names_stay_a_set(tmp_path):
    """A file the Studio did not write is never replaced, and then every file of the set moves to -2 together."""
    d = str(tmp_path)
    p = SURF.targets(d, "IMG_1", SURF.KINDS)
    assert os.path.basename(p["height"]) == "IMG_1_height_Studio.png"
    with open(p["normal"], "wb") as fh:
        fh.write(b"somebody else's file")
    p2 = SURF.targets(d, "IMG_1", SURF.KINDS)
    assert all(os.path.basename(v).startswith("IMG_1-2_") for v in p2.values())

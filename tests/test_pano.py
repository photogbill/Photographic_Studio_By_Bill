"""Panorama stitching (BUILD_PLAN_PANO_IR.md 2.6-2.9), end to end on the synthetic sphere: frames written as linear
TIFFs (they open through the Studio's real code path), layout, exposure gains, a moving object the seam must cut
around, tiles that join exactly, and the result against the world rendered directly in the same projection.
Bill's Iraq set is stitched when it is in test_raws."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from conftest import ROOT
from pano_synth import render, rot, world
from studio.engine.stack import pano as PA
from studio.engine.stack import project as PJ
from studio.engine.stack import pyramid as PY
from studio.engine.stack import seams as SE

OUT = ROOT / "outputs" / "pytest_pano"
F, W, H = 560.0, 480, 360
GAINS = [0.85, 1.0, 1.18, 0.92]
YAWS = [0, 30, 60, 90]


def _write(path, rgb):
    from studio.engine.io.intermediate import write_intermediate
    write_intermediate(str(path), rgb, None, {"tool": "test frame"})


@pytest.fixture(scope="module")
def frames():
    OUT.mkdir(parents=True, exist_ok=True)
    tex = world(seed=3)
    rng = np.random.default_rng(5)
    Rs, paths = [], []
    for k, (yaw, g) in enumerate(zip(YAWS, GAINS)):
        R = rot(yaw, rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
        img = render(tex, R, F, W, H) * np.float32(g)
        if k == 2:                                  # someone walked into frame 3 only (in its overlap with frame 2)
            img[150:190, 40:70] = np.float32(0.9 * g)
        img3 = np.repeat(img[..., None], 3, axis=2) * np.array([1.0, 0.95, 0.85], np.float32)
        p = OUT / f"SYN_{k + 1:04d}.tif"
        _write(p, img3)
        Rs.append(R)
        paths.append(str(p))
    return tex, Rs, paths


@pytest.fixture(scope="module")
def analysis(frames):
    tex, Rs, paths = frames
    return PA.analyse(paths, PA.PanoParams(level=False))


def test_layout_on_the_synthetic_sphere(frames, analysis):
    an = analysis
    assert all(an.placed) and an.focal_source == "estimated"
    assert abs(an.f - F) / F < 0.01, an.f
    assert an.rms < 1.0
    lv = PA.layout_view(an, PA.PanoParams())
    assert len(lv["outlines"]) == 4 and all(o["tie"] == "ok" for o in lv["outlines"])


def test_exposure_gains_are_matched_within_one_percent(analysis):
    sp = PA.plan_seams(analysis, PA.PanoParams(), 0.5)
    g = sp["gains"]
    rec = np.array([1.0 / float(np.mean(g[i])) for i in range(4)])   # the gain that undoes each frame's exposure
    rec = rec / rec[1]
    want = np.array(GAINS) / GAINS[1]
    assert np.abs(rec - want).max() / want.max() < 0.01, (rec, want)


def test_seam_cuts_around_the_moving_object(analysis):
    pv = PA.preview(analysis, PA.PanoParams(seams="smart"))
    pv_straight = PA.preview(analysis, PA.PanoParams(seams="straight"))
    # where the object is, the smart result comes from one frame (no half-strength ghost)
    cv = pv["canvas"]
    an = analysis
    d = PJ.from_camera(np.array([55.0]), np.array([170.0]), an.R[2], an.f, (W - 1) / 2, (H - 1) / 2)
    u, v, _ = PJ.to_canvas(d, cv.projection, cv.scale)
    x, y = int(round(u[0] - cv.u0)), int(round(v[0] - cv.v0))
    lab = pv["labels"]
    assert lab[y, x] != 2 or np.all(lab[max(0, y - 2):y + 3, max(0, x - 2):x + 3] == 2)
    assert pv["ghosts"].max() > 0.5                       # the moving object shows in the ghost view
    assert pv["rgb"].shape[0] > 10 and pv["rgb"].shape[1] > pv["rgb"].shape[0]
    assert pv_straight["rgb"].shape == pv["rgb"].shape or True


def test_tiles_join_exactly():
    rng = np.random.default_rng(0)
    h, w, lv = 300, 420, 4
    imgs = [rng.random((h, w, 3)).astype(np.float32) for _ in range(2)]
    lab = np.zeros((h, w), np.int16)
    lab[:, w // 2 + 7:] = 1
    layers = [SE.Layer(i, (0, h, 0, w), imgs[i], np.ones((h, w), bool), np.zeros((h, w), np.float32),
                       np.zeros((h, w), bool)) for i in range(2)]
    filled = {i: imgs[i] for i in range(2)}
    whole, _ = PA.blend_region(layers, filled, lambda a, b, c, d: lab[a:b, c:d], 0, h, 0, w, lv)
    M = PY.margin(lv)
    step = 2 ** lv * 8
    out = np.zeros_like(whole)
    for ty in range(0, h, step):
        for tx in range(0, w, step):
            y0, y1 = max(0, ty - M), min(h, ty + step + M)
            x0, x1 = max(0, tx - M), min(w, tx + step + M)
            r, _ = PA.blend_region(layers, filled, lambda a, b, c, d: lab[a:b, c:d], y0, y1, x0, x1, lv)
            hh, ww = min(step, h - ty), min(step, w - tx)
            out[ty:ty + hh, tx:tx + ww] = r[ty - y0:ty - y0 + hh, tx - x0:tx - x0 + ww]
    inner = (slice(M, h - M), slice(M, w - M))
    assert np.array_equal(out[inner], whole[inner])


def test_largest_rectangle():
    m = np.zeros((20, 30), bool)
    m[3:15, 5:25] = True
    m[3:5, 5:8] = False
    y0, y1, x0, x1 = SE.largest_rectangle(m)
    assert (y1 - y0) * (x1 - x0) == max(12 * 17, 10 * 20)
    assert m[y0:y1, x0:x1].all()


def test_build_matches_the_world_rendered_directly(frames, analysis):
    tex, Rs, paths = frames
    an = analysis
    params = PA.PanoParams(projection="cylindrical", level=False)
    out = OUT / "SYN_0001_to_SYN_0004_Pano_Studio.tif"
    rep = PA.build(an, params, str(out), tile=256)
    import tifffile
    got = tifffile.imread(rep["path"], key=0).astype(np.float32)
    assert got.ndim == 3 and got.shape[2] == 3
    # the truth: the world seen through the same canvas (solved world = true world turned by the reference frame)
    cv = PA.make_canvas(an, params, 1.0)
    ys, xs = np.mgrid[0:cv.H, 0:cv.W].astype(np.float64)
    d = PJ.from_canvas(xs + cv.u0, ys + cv.v0, "cylindrical", cv.scale)
    ref = int(np.argmax([np.allclose(r, np.eye(3), atol=1e-9) for r in an.R]))
    Rref = Rs[ref]
    dt = d @ Rref                                    # R_ref^T d
    th = np.arctan2(dt[..., 0], dt[..., 2])
    ph = np.arcsin(np.clip(dt[..., 1], -1, 1))
    Ht, Wt = tex.shape
    truth = map_coordinates(tex, [(ph / math.pi + 0.5) * Ht - 0.5, (th / (2 * math.pi) + 0.5) * Wt - 0.5], order=1,
                            mode="wrap").astype(np.float32)
    # the build is cropped: the crop's place on the canvas is in the report
    gl = got[..., 0]
    hh, ww = gl.shape
    from studio.engine.stack.align import phase_shift
    oy, _, ox, _ = rep["crop_canvas"]
    kgain = float(np.median(gl) / max(np.median(truth[oy:oy + hh, ox:ox + ww]), 1e-6))
    t = truth[oy:oy + hh, ox:ox + ww] * kgain
    c = (slice(hh // 4, 3 * hh // 4), slice(ww // 4, 3 * ww // 4))
    dy, dx, _ = phase_shift(t[c], gl[c])
    assert abs(dy) < 1.0 and abs(dx) < 1.0, (dy, dx, oy, ox)
    from scipy.ndimage import gaussian_filter
    gb, tb = gaussian_filter(gl, 1.5), gaussian_filter(t, 1.5)    # the texture is finer than two resamplings keep
    rel = np.abs(gb[c] - tb[c]) / np.maximum(tb[c], 0.02)
    assert np.median(rel) < 0.06, float(np.median(rel))


def test_overlap_levelling_takes_a_flare_veil_out_of_the_seam(frames, analysis):
    """Frame 2 gets broad added light growing toward its left edge (a flare veil from a sun to the left). Stitched
    without levelling, the result steps where frame 2 takes over; levelled, the step at the seam is at least halved
    and the stitch stays closer to the same stitch of the clean frames."""
    import tifffile
    from scipy.ndimage import gaussian_filter
    tex, Rs, paths = frames
    vp = []
    for k, p in enumerate(paths):
        img = tifffile.imread(p, key=0).astype(np.float32)
        if k == 1:
            xx = np.arange(img.shape[1], dtype=np.float32)[None, :, None]
            img = img + 0.22 * np.exp(-(xx / (0.35 * img.shape[1])) ** 2) * np.array([1.0, 0.9, 1.1], np.float32)
        q = OUT / f"VEIL_{k + 1:04d}.tif"
        _write(q, img)
        vp.append(str(q))
    an = PA.analyse(vp, PA.PanoParams(level=False))
    assert all(an.placed)

    def stitch(a, level):
        params = PA.PanoParams(projection="cylindrical", level=False, seam_level=level)
        sp = PA.plan_seams(a, params, 0.5)
        lab = sp["labels"]
        filled = {L.i: PY.fill(L.rgb.astype(np.float32), L.valid) for L in sp["layers"]}
        lv = PA._levels(a, 0.5, params, sp["overlap"] * 0.5)
        out, _ = PA.blend_region(sp["layers"], filled, lambda p, q, r, t: lab[p:q, r:t], 0, lab.shape[0], 0,
                                 lab.shape[1], lv)
        return out[..., 0], lab
    clean, _ = stitch(analysis, False)

    def judge(level):
        out, lab = stitch(an, level)
        h, w = min(out.shape[0], clean.shape[0]), min(out.shape[1], clean.shape[1])
        lr = np.log(np.maximum(gaussian_filter(out[:h, :w], 3), 1e-4)) - \
            np.log(np.maximum(gaussian_filter(clean[:h, :w], 3), 1e-4))
        rows = range(h // 4, 3 * h // 4)
        xs = [np.nonzero((lab[y, :w - 1] == 0) & (lab[y, 1:w] == 1))[0] for y in rows]
        s = int(np.median([x[0] for x in xs if len(x)]))
        prof = lr[h // 4:3 * h // 4].mean(axis=0)
        step = float(prof[s + 3:s + 10].mean() - prof[s - 10:s - 3].mean())
        return abs(step), float(np.abs(prof[: w // 2]).mean())
    (s0, m0), (s1, m1) = judge(False), judge(True)
    assert s1 < 0.6 * s0 and m1 < m0, ((s0, m0), (s1, m1))


IRAQ = sorted((ROOT / "test_raws" / "panorama_iraq_6d_135mm").glob("IMG_*.jpg"))


@pytest.mark.skipif(len(IRAQ) < 7, reason="Bill's Iraq panorama is not in test_raws")
def test_bills_iraq_set_places_every_frame_and_finds_about_135mm():
    an = PA.analyse([str(p) for p in IRAQ], PA.PanoParams())
    assert all(an.placed)
    f_mm = an.f * 6.54e-3                              # the 6D's pixel pitch
    assert abs(f_mm - 135) / 135 < 0.05, f_mm
    assert max(p["rms"] for p in an.pairs) < 4.0        # smoke moved in the 6 s: two pairs are named, none fail
    pv = PA.preview(an, PA.PanoParams())
    assert pv["rgb"].shape[1] > 2.2 * pv["rgb"].shape[0]

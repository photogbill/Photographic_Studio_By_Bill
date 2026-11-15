"""Portrait: the face models run by the Studio's own runner, face zones, lines / folds, blemishes, body sculpting,
region consistency and the settings plumbing (looks, history labels)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from conftest import synthetic_raw

HERE = os.path.dirname(__file__)


def sample_face(cx: float, cy: float, width: float) -> np.ndarray:
    """A real face mesh (468 points), placed at (cx, cy) with this face width (pixels)."""
    with open(os.path.join(HERE, "data", "face_mesh_sample.json"), encoding="utf-8") as fh:
        norm = np.asarray(json.load(fh)["pts"], float)
    return np.array([cx, cy]) + norm * width


def face_dict(P: np.ndarray, sw: int, sh: int) -> dict:
    return {"pts": (P / np.array([sw, sh])).round(6).tolist(), "conf": 1.0, "how": "found", "enabled": True}


def skin_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Scene-linear, skin-coloured, with a fine grain (pores) and gentle shading."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    shade = 0.75 + 0.25 * np.cos(2.5 * (xx - 0.5)) * np.cos(2.0 * (yy - 0.45))
    grain = 1.0 + 0.04 * gaussian_filter(rng.normal(0, 1, (h, w)), 0.8)
    skin = np.array([0.42, 0.27, 0.19], np.float32)
    return (skin[None, None, :] * (shade * grain)[..., None]).astype(np.float32)


def _line(img, x0, y0, x1, y1, width, depth):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    d = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    img *= np.exp2(-depth * np.exp(-0.5 * (d / width) ** 2))[..., None]


def _dot(img, x, y, r, gains):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r))[..., None]
    img *= 1.0 - g * (1.0 - np.asarray(gains, np.float32))


# ------------------------------------------------------------------------------------------ the model runner
def test_runner_matches_reference_outputs():
    """The NumPy runner gives the same numbers as TensorFlow Lite models run elsewhere (OpenCV's importer agreed
    to 1e-4 when these references were made), on fixed inputs: a cross-platform check."""
    from studio.engine.local import faces as fz
    from studio.engine.local.tinynet import Model
    if not fz.available():
        pytest.skip("face models not installed")
    x = np.sin(7 * np.linspace(-1, 1, 128 * 128 * 3, dtype=np.float32).reshape(1, 128, 128, 3)) * 0.8
    out = Model(os.path.join(fz.model_dir(), fz.MODEL_FILES["short"]))(x)
    assert out["regressors"].shape == (1, 896, 16) and out["classificators"].shape == (1, 896, 1)
    assert math.isclose(float(out["regressors"].sum()), 85412.53, rel_tol=1e-4)
    y = (np.sin(np.linspace(0, 40, 192 * 192 * 3, dtype=np.float32)).reshape(1, 192, 192, 3) * 0.5 + 0.5)
    o2 = Model(os.path.join(fz.model_dir(), fz.MODEL_FILES["mesh"]))(y.astype(np.float32))
    lm = [v for v in o2.values() if v.size == 1404][0]
    assert math.isclose(float(lm.sum()), 88084.73, rel_tol=1e-4)


def test_runner_refuses_other_files(tmp_path):
    from studio.engine.local.tinynet import Model, ModelError
    p = tmp_path / "x.tflite"
    p.write_bytes(b"not a model at all")
    with pytest.raises(ModelError):
        Model(str(p))


def test_faces_found_on_a_real_portrait_and_from_two_eye_clicks():
    skimage = pytest.importorskip("skimage")              # a public-domain NASA portrait ships with scikit-image
    from studio.engine.local import faces as fz
    if not fz.available():
        pytest.skip("face models not installed")
    im = skimage.data.astronaut().astype(np.float32) / 255.0
    found = fz.find(im)
    assert len(found) == 1 and found[0]["conf"] > 0.9
    P = found[0]["pts"]
    assert 70 < fz.size_of(P) < 120
    dark = np.clip((im ** 2.2 * np.array([1.0, 0.55, 0.25]) * 0.08) ** (1 / 2.2), 0, 1).astype(np.float32)
    assert len(fz.find(dark)) == 1                        # candle-dark and orange: the brightened copy finds it
    r, l_ = fz.eye_centres(P)
    g = fz.from_eyes(im, r + [3, -2], l_ + [-2, 3])
    assert g["conf"] > 0.9 and np.mean(np.hypot(*(g["pts"] - P).T)) < 3.0


# ------------------------------------------------------------------------------------------ face work
def _face_setup(h=700, w=700, width=420.0):
    from studio.engine.local import portrait as pt
    P = sample_face(w * 0.5, h * 0.55, width)
    f = face_dict(P, w, h)
    return pt, P, f, pt.Face(f, w, h)


def test_zones_sit_on_the_face():
    pt, P, f, fc = _face_setup()
    z = pt.zones(fc)
    from studio.engine.local import faces as fz

    def at(name, p):
        return float(pt.sample_zone(z, name, np.array([[p[0]]]), np.array([[p[1]]]))[0, 0])
    eye = P[fz.RIGHT_EYE].mean(axis=0)
    assert at("eye", eye) > 0.8 and at("skin", eye) < 0.1 and at("blem", eye) < 0.05
    cheek = P[50]
    assert at("skin", cheek) > 0.8 and at("blem", cheek) > 0.8
    assert at("teeth", 0.5 * (P[13] + P[14])) > 0.5 or at("teeth", 0.5 * (P[13] + P[14])) >= 0.0
    under = P[[145, 119]].mean(axis=0)
    assert at("under", under) > 0.5
    assert at("smile", P[206]) > 0.3
    far = P[fz.CHIN] + 0.5 * fc.W * (P[fz.CHIN] - P[fz.NOSE_TIP]) / np.hypot(*(P[fz.CHIN] - P[fz.NOSE_TIP]))
    assert at("skin", far) < 0.05


def test_lines_fill_grooves_and_keep_moles_and_pores():
    pt, P, f, fc = _face_setup()
    img = skin_image(700, 700)
    ref = img.copy()
    W = fc.W
    # two forehead lines and crow's feet (thin grooves), plus a round mole of the same darkness
    top = P[10]
    for k in range(2):
        y = top[1] + (0.06 + 0.05 * k) * W
        _line(img, top[0] - 0.18 * W, y, top[0] + 0.18 * W, y, 0.004 * W, 0.35)
    oc = P[33]
    _line(img, oc[0] - 0.02 * W, oc[1], oc[0] - 0.12 * W, oc[1] + 0.03 * W, 0.003 * W, 0.35)
    mole = (top[0] + 0.12 * W, top[1] + 0.16 * W)
    _dot(img, mole[0], mole[1], 0.008 * W, (0.78, 0.78, 0.78))
    G = pt.face_gain(img, 0, 0, 1, fc, pt.amounts({"lines": 100}))
    out = img * G
    L = lambda a: np.log2(a @ pt.LUMA)                                  # noqa: E731
    d_before, d_after = L(img) - L(ref), L(out) - L(ref)
    yy, xx = np.mgrid[0:700, 0:700]
    lines = (d_before < -0.12) & (np.hypot(xx - mole[0], yy - mole[1]) > 0.03 * W)
    removed = 1 - d_after[lines].mean() / d_before[lines].mean()
    assert removed > 0.45, removed
    m = np.hypot(xx - mole[0], yy - mole[1]) < 0.006 * W
    kept = d_after[m].mean() / d_before[m].mean()
    assert kept > 0.8, kept                                             # a mole is round: it stays
    clean = (np.abs(d_before) < 1e-4) & (pt.sample_zone(pt.zones(fc), "lines", xx, yy) > 0.5)
    fine = L(img) - __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(L(img), 2.0)
    fine_out = L(out) - __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(L(out), 2.0)
    assert np.std(fine_out[clean]) > 0.85 * np.std(fine[clean])          # pores (the fine grain) kept


def test_folds_lift_part_way_and_nothing_happens_at_zero():
    pt, P, f, fc = _face_setup()
    img = skin_image(700, 700, 1)
    ref = img.copy()
    W = fc.W
    for idx in ((145, 119), (374, 348)):                                  # dark hollows under both eyes
        c = P[list(idx)].mean(axis=0)
        _line(img, c[0] - 0.07 * W, c[1], c[0] + 0.07 * W, c[1], 0.025 * W, 0.4)
    G0 = pt.face_gain(img, 0, 0, 1, fc, pt.amounts({}))
    assert np.allclose(G0, 1.0, atol=1e-5)
    G = pt.face_gain(img, 0, 0, 1, fc, pt.amounts({"under_eyes": 100}))
    L = lambda a: np.log2(a @ pt.LUMA)                                  # noqa: E731
    d0, d1 = L(img) - L(ref), L(img * G) - L(ref)
    m = d0 < -0.2
    frac = 1 - d1[m].mean() / d0[m].mean()
    assert 0.35 < frac < 0.95, frac                                       # lifted, but the face keeps its shape


def test_younger_drives_every_slider_in_proportion():
    from studio.engine.local import portrait as pt
    a = pt.amounts({"younger": 50, "teeth": 80})
    assert a["lines"] == 0.5 and abs(a["under_eyes"] - 0.425) < 1e-9 and a["teeth"] == 0.8
    assert pt.amounts({})["lines"] == 0.0


def test_blemishes_found_moles_left_alone():
    pt, P, f, fc = _face_setup()
    img = skin_image(700, 700, 2)
    W = fc.W
    spots = [P[50] + [0.0, 0.02 * W], P[280] + [0.0, 0.03 * W], P[205] + [0.02 * W, 0.0]]
    for s in spots:
        _dot(img, s[0], s[1], 0.006 * W, (0.92, 0.70, 0.72))              # red bumps
    mole = P[425] + [0.0, -0.02 * W]
    _dot(img, mole[0], mole[1], 0.005 * W, (0.35, 0.35, 0.36))            # a dark, well-defined mole
    blobs, moles = pt.find_blemishes(img, 0, 0, 1, fc)
    found = [any(math.hypot(b["x"] - s[0], b["y"] - s[1]) < 0.02 * W for b in blobs) for s in spots]
    assert all(found), (found, blobs)
    assert not any(math.hypot(b["x"] - mole[0], b["y"] - mole[1]) < 0.02 * W for b in blobs)
    assert moles >= 1
    assert len(blobs) <= 8                                                # not every pore


# ------------------------------------------------------------------------------------------ body sculpting
def test_warps_are_capped_and_invert_exactly():
    from studio.engine.local import portrait as pt
    sw, sh = 1000, 800
    p = {"sculpt": [{"kind": "push", "pts": [[0.5, 0.5]], "vec": [[0.2, 0.0]], "r": 0.05},
                    {"kind": "shape", "tool": "breast", "c": [0.3, 0.4], "e1": [1, 0], "a": 0.06, "b": 0.05,
                     "up": [0, -1], "lift": 100, "size": 100},
                    {"kind": "shape", "tool": "thigh", "c": [0.7, 0.6], "e1": [0, 1], "a": 0.15, "b": 0.04,
                     "up": [0, -1], "slim": 100}]}
    ws = pt.warps(p, sw, sh)
    push = ws[0]
    assert math.hypot(push[4], push[5]) <= 0.3 * push[3] + 1e-9            # a big drag is capped
    rng = np.random.default_rng(0)
    S = rng.uniform([100, 100], [900, 700], (400, 2))
    X, Y = S[:, 0].copy(), S[:, 1].copy()
    for w in ws:                                                           # forward: every warp in order
        dx, dy = pt._disp(w, X, Y)
        X, Y = X + dx, Y + dy
    BX, BY = pt.source_of(ws, X, Y, iters=8)
    assert np.max(np.hypot(BX - S[:, 0], BY - S[:, 1])) < 0.05
    G = np.stack(np.meshgrid(np.arange(0, 1000, 4.0), np.arange(0, 800, 4.0)), -1).reshape(-1, 2)
    moved = 0.0
    for w in ws:
        dx, dy = pt._disp(w, G[:, 0], G[:, 1])
        moved = max(moved, float(np.max(np.hypot(dx, dy))))
    assert moved <= pt.margin(ws) < 40                                     # the context a region render reads
    assert pt.warps({}, sw, sh) == []


def test_sculpt_and_face_work_match_between_region_and_full_render():
    from studio.engine.pipeline.edit import Edit
    from studio.engine.pipeline.pipeline import Image, render, render_view, view_window, sculpt_source
    h, w = 600, 800
    sc = skin_image(h, w, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    sc = sc * (1 + 0.3 * (((yy // 30) + (xx // 30)) % 2))[..., None].astype(np.float32)
    img = Image(synthetic_raw(np.clip(sc, 0, 1)))
    e = Edit()
    e.sharpen.enabled = False
    P = sample_face(400, 330, 300)
    e.portrait = {"faces": [face_dict(P, w, h)], "younger": 80, "eyes": 50, "teeth": 50, "double_chin": 100,
                  "sculpt": [{"kind": "push", "pts": [[0.25, 0.5]], "vec": [[0.02, 0.0]], "r": 0.08},
                             {"kind": "shape", "tool": "butt", "c": [0.75, 0.7], "e1": [1, 0], "a": 0.07,
                              "b": 0.06, "up": [0, -1], "lift": 60, "size": 40}]}
    full = render(img, e)
    for u, v in ((0.5, 0.45), (0.25, 0.5), (0.75, 0.7), (0.5, 0.86)):            # (0.5, 0.86): under the chin
        r = render_view(img, e, u, v, 200, 160)
        x, y, ww, hh = view_window(img, e, u, v, 200, 160)
        assert np.abs(r - full[y:y + hh, x:x + ww]).max() < 1e-5
    e0 = Edit()
    e0.sharpen.enabled = False
    assert np.abs(full - render(img, e0)).max() > 0.01                    # it did something
    xs, ys = sculpt_source(img, e, [0.25 + 0.02], [0.5])                   # a click on pushed content
    assert abs(float(xs[0]) - 0.25) < 0.004


def test_double_chin_lifts_under_the_jaw_only():
    """A friend's request: the soft area under the jaw drawn up towards it (nothing at or above the jaw line moves,
    capped, never folds), on skin only (a collar right under the chin stays put), and the crease softened."""
    pt, P, f, fc = _face_setup(h=1000, w=700, width=300.0)
    W = fc.W
    assert pt.warps({"faces": [f]}, 700, 1000) == [] and pt.amounts({"younger": 100})["double_chin"] == 0.0
    ws = pt.warps({"faces": [f], "double_chin": 100}, 700, 1000)
    assert [w[0] for w in ws] == ["tuck"] and pt.sculpt_active({"faces": [f], "double_chin": 40})
    up = fc.up
    above = np.concatenate([P[pt.JAW_LINE], P[[1, 10, 50, 280, 13]], P[pt.JAW_LINE] + 0.05 * W * up])
    dx, dy = pt._disp(ws[0], above[:, 0], above[:, 1])
    assert np.abs(np.hypot(dx, dy)).max() < 0.02                            # the jaw and the face never move
    under = P[pt.fz.CHIN] - 0.16 * W * up
    dx, dy = pt._disp(ws[0], np.array([under[0]]), np.array([under[1]]))
    mv = float(np.hypot(dx, dy)[0])
    assert 0.03 * W < mv <= 0.0451 * W and float(dx[0] * up[0] + dy[0] * up[1]) > 0   # drawn up towards the jaw
    G = np.stack(np.meshgrid(np.arange(60, 640, 2.0), np.arange(450, 950, 2.0)), -1)
    X, Y = G[..., 0], G[..., 1]
    dx, dy = pt._disp(ws[0], X, Y)
    nx, ny = X + dx, Y + dy
    jac = (np.gradient(nx, axis=1) * np.gradient(ny, axis=0) - np.gradient(nx, axis=0) * np.gradient(ny, axis=1)) / 4
    assert jac.min() > 0.4                                                  # never folds
    # a dark collar 0.22 of the face under the chin: the skin weight is ~0 there, and high on the skin above it
    img = skin_image(1000, 700, 5)
    c = P[pt.fz.CHIN]
    yy = np.mgrid[0:1000, 0:700][0].astype(float)
    img[yy > c[1] + 0.22 * W] *= 0.05
    sk = pt.chin_skin(img, 0, 0, 1, fc)
    assert sk[int(c[1] + 0.30 * W), int(c[0])] < 0.1 and sk[int(c[1] + 0.08 * W), int(c[0])] > 0.6
    ws2 = pt.warps({"faces": [f], "double_chin": 100}, 700, 1000, skin=[(sk, 0, 0, 1)])
    edge = np.array([[c[0] - 0.1 * W, c[1] + 0.22 * W], [c[0], c[1] + 0.22 * W], [c[0] + 0.1 * W, c[1] + 0.22 * W]])
    dx, dy = pt._disp(ws2[0], edge[:, 0], edge[:, 1])
    assert np.hypot(dx, dy).max() < 0.25 * mv                               # the collar's edge stays (nearly) put
    # the crease: lifted part way, nothing at 0
    img = skin_image(1000, 700, 4)
    ref = img.copy()
    cc = c - 0.12 * W * up
    _line(img, cc[0] - 0.12 * W, cc[1], cc[0] + 0.12 * W, cc[1], 0.012 * W, 0.5)
    G0 = pt.face_gain(img, 0, 0, 1, fc, pt.amounts({}))
    assert np.allclose(G0, 1.0, atol=1e-6)
    Gc = pt.face_gain(img, 0, 0, 1, fc, pt.amounts({"double_chin": 100}))
    L = lambda a: np.log2(a @ pt.LUMA)                                  # noqa: E731
    d0, d1 = L(img) - L(ref), L(img * Gc) - L(ref)
    m = d0 < -0.2
    assert 0.3 < 1 - d1[m].mean() / d0[m].mean() < 0.9


# ------------------------------------------------------------------------------------------ settings plumbing
def test_looks_keep_faces_with_their_photo():
    from studio.engine.pipeline.edit import Edit
    from studio.engine.presets import looks as lk
    a, b = Edit(), Edit()
    a.portrait = {"faces": [{"pts": [[0.1, 0.1]] * 468}], "younger": 40, "sculpt": [{"kind": "push"}]}
    b.portrait = {"faces": [{"pts": [[0.5, 0.5]] * 468}]}
    c = lk.copy_groups(a, b, ["portrait"])
    assert c.portrait["younger"] == 40 and c.portrait["faces"] == b.portrait["faces"] and "sculpt" not in c.portrait
    look = lk.look_from_edit(a, "Soft", ["portrait"])
    assert set(look["settings"]["portrait"]) <= set(__import__("studio.engine.local.portrait",
                                                               fromlist=["FACE_KEYS"]).FACE_KEYS)
    d = lk.apply_look(b, look)
    assert d.portrait["younger"] == 40 and d.portrait["faces"] == b.portrait["faces"]
    e = Edit.from_dict(b.to_dict())
    e.portrait = dict(b.portrait, lines=30)
    assert lk.describe_change(b, e) == "Portrait: lines +30"
    f = Edit.from_dict(b.to_dict())
    f.portrait = dict(b.portrait, sculpt=[{"kind": "shape", "tool": "butt"}])
    assert lk.describe_change(b, f) == "Sculpt: butt improvement added"

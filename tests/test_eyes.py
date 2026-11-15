"""Eyes (session 11): the iris found (MediaPipe Iris through tinynet, or the classical fit), eye colour, the iris
tools, red eye and blown pupils, and the plumbing (pipeline, retouch spots, old edits unchanged)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from conftest import synthetic_raw

HERE = os.path.dirname(__file__)


# ------------------------------------------------------------------------------------------ test eyes
def mesh(cx, cy, width):
    with open(os.path.join(HERE, "data", "face_mesh_sample.json"), encoding="utf-8") as fh:
        norm = np.asarray(json.load(fh)["pts"], float)
    return np.array([cx, cy]) + norm * width


def eye_scene(H=900, W=900, iris=(0.08, 0.035, 0.012), pupil=(0.004, 0.004, 0.004), catch=True, face_w=600,
              pupil_frac=0.38):
    """A scene-linear skin image with both eyes drawn in a real face mesh's eye outlines: brown irises with fibres,
    a dark pupil and a catchlight. Returns (img, P, [(cx, cy, r) right, left])."""
    from PIL import Image, ImageDraw
    from studio.engine.local import faces as fz
    P = mesh(W / 2, H * 0.55, face_w)
    img = np.zeros((H, W, 3), np.float32) + np.array([0.42, 0.27, 0.19], np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    truth = []
    for idx, c1, c2 in ((fz.RIGHT_EYE, 33, 133), (fz.LEFT_EYE, 362, 263)):
        poly = P[idx]
        m = Image.new("L", (W * 2, H * 2), 0)
        ImageDraw.Draw(m).polygon([(x * 2, y * 2) for x, y in poly], fill=255)
        lid = np.asarray(m.resize((W, H), Image.Resampling.BILINEAR)) / 255.0
        cx, cy = poly.mean(axis=0)
        r = 0.21 * np.hypot(*(P[c1] - P[c2]))
        d = np.hypot(xx - cx, yy - cy) / r
        ang = np.arctan2(yy - cy, xx - cx)
        fib = 1 + 0.25 * np.sin(ang * 45 + 0.3 * np.sin(ang * 7))
        eye = np.where((d < 1)[..., None], np.asarray(iris, np.float32) * fib[..., None],
                       np.array([0.55, 0.52, 0.5], np.float32))
        eye = np.where((d < pupil_frac)[..., None], np.asarray(pupil, np.float32), eye)
        if catch:
            eye = np.where((np.hypot(xx - cx - 0.3 * r, yy - cy + 0.35 * r) < 0.09 * r)[..., None],
                           np.array([0.9, 0.9, 0.9], np.float32), eye)
        img = (img * (1 - lid[..., None]) + eye * lid[..., None]).astype(np.float32)
        truth.append((cx, cy, r))
    return img, P, truth


def face_with_irises(P, truth, W, H, how="model"):
    return {"pts": (P / [W, H]).round(6).tolist(), "conf": 1.0, "how": "found", "enabled": True,
            "irises": [{"side": s, "c": [cx / W, cy / H], "e": [(cx + r) / W, cy / H], "how": how}
                       for s, (cx, cy, r) in zip("rl", truth)]}


def gain(img, f, p):
    from studio.engine.local import eyes as ey
    from studio.engine.local import portrait as pt
    H, W = img.shape[:2]
    fc = pt.Face(f, W, H)
    p = dict(p, faces=[f])
    return img * pt.face_gain(img, 0, 0, 1, fc, pt.amounts(p), ey.colour_settings(p))


def oklch(img, mask):
    from studio.engine.color.grade import to_oklab
    lab = to_oklab(np.maximum(img, 0))[mask]
    return (math.degrees(math.atan2(np.median(lab[:, 2]), np.median(lab[:, 1]))) % 360,
            float(np.median(np.hypot(lab[:, 1], lab[:, 2]) / np.maximum(lab[:, 0], 1e-4))), float(np.median(lab[:, 0])))


def display(img_lin):
    return np.clip(img_lin, 0, 1) ** (1 / 2.2)


# ------------------------------------------------------------------------------------------ finding the iris
def test_iris_found_by_model_and_by_the_fit():
    from studio.engine.local import eyes as ey
    img, P, truth = eye_scene()
    view = display(img).astype(np.float32)
    modes = [True, False] if ey.model_available() else [False]
    for use_model in modes:
        orig = ey.model_available
        ey.model_available = (lambda: True) if use_model else (lambda: False)
        try:
            found = ey.find_irises(view, P)
        finally:
            ey.model_available = orig
        assert [e["side"] for e in found] == ["r", "l"]
        for e, (cx, cy, r) in zip(found, truth):
            assert math.hypot(e["c"][0] - cx, e["c"][1] - cy) < 0.15 * r
            assert 0.85 < e["r"] / r < 1.15
            assert e["how"] == ("model" if use_model else "fit")


def test_iris_seeded_by_the_newer_mesh_and_a_bad_seed_ignored():
    """The newer mesh's own iris points (centre + 4 edge points per eye) are used first and finished by the edge
    search; a seed outside the eye is not trusted and the model / fit takes over."""
    from studio.engine.local import eyes as ey
    img, P, truth = eye_scene()
    view = display(img).astype(np.float32)
    rng = np.random.default_rng(3)
    seed = []
    for cx, cy, r in truth:                              # off by a few pixels and 10% too small, as a mesh may be
        c = np.array([cx, cy]) + rng.normal(0, 0.08 * r, 2)
        seed += [c] + [c + 0.9 * r * np.array(v) for v in ((1, 0), (0, -1), (-1, 0), (0, 1))]
    found = ey.find_irises(view, P, np.array(seed))
    assert [e["how"] for e in found] == ["mesh", "mesh"]
    for e, (cx, cy, r) in zip(found, truth):
        assert math.hypot(e["c"][0] - cx, e["c"][1] - cy) < 0.15 * r and 0.85 < e["r"] / r < 1.15
    bad = np.array(seed)
    bad[:5] += [0, 4 * truth[0][2]]                      # the right seed below the eye
    found = ey.find_irises(view, P, bad)
    assert found[0]["how"] in ("model", "fit") and found[1]["how"] == "mesh"
    cx, cy, r = truth[0]
    assert math.hypot(found[0]["c"][0] - cx, found[0]["c"][1] - cy) < 0.15 * r


def test_irises_on_a_real_portrait():
    skimage = pytest.importorskip("skimage")              # a public-domain NASA portrait ships with scikit-image
    from studio.engine.local import eyes as ey
    from studio.engine.local import faces as fz
    if not fz.available():
        pytest.skip("face models not installed")
    im = skimage.data.astronaut().astype(np.float32) / 255.0
    f = fz.find(im)[0]
    P = f["pts"]
    found = ey.find_irises(im, P, f.get("iris"))
    assert len(found) == 2
    if f.get("iris") is not None:                         # the newer mesh's irises agree with the iris model's
        assert [e["how"] for e in found] == ["mesh", "mesh"]
        if ey.model_available():
            for e, m in zip(found, ey.find_irises(im, P)):
                assert math.hypot(e["c"][0] - m["c"][0], e["c"][1] - m["c"][1]) < 0.25 * m["r"]
                assert 0.85 < e["r"] / m["r"] < 1.15
    for e in found:
        idx = fz.RIGHT_EYE if e["side"] == "r" else fz.LEFT_EYE
        assert ey._polygon_mask(P[idx], np.array([[e["c"][0]]]), np.array([[e["c"][1]]]))[0, 0]
        d = ey.eye_roi(P, e["side"])[4]
        assert 0.15 < e["r"] / d < 0.28
        # the iris is darker than the eye whites beside it
        y, x = int(round(e["c"][1])), int(round(e["c"][0]))
        assert im[y, x].mean() < 0.45


# ------------------------------------------------------------------------------------------ eye colour and tools
def test_eye_colour_hits_its_hue_and_touches_nothing_else():
    from studio.engine.local import eyes as ey
    img, P, truth = eye_scene()
    H, W = img.shape[:2]
    f = face_with_irises(P, truth, W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    (cx, cy, r), (lx, ly, lr) = truth
    d = np.hypot(xx - cx, yy - cy) / r
    dl = np.hypot(xx - lx, yy - ly) / lr
    core = (d > 0.48) & (d < 0.85) & (np.abs(img[..., 0] - 0.42) > 0.05)
    for name in ("blue", "green", "grey", "violet"):
        out = gain(img, f, {"eye_colour": {"preset": name, "amount": 100}})
        h, c, _ = oklch(out, core)
        assert abs((h - ey.COLOURS[name][0] + 180) % 360 - 180) < 4 and abs(c - ey.COLOURS[name][1]) < 0.03
    out = gain(img, f, {"eye_colour": {"preset": "blue", "amount": 100}})
    skin = (np.minimum(d, dl) > 1.3) & (np.minimum(d, dl) < 3)
    assert np.abs(out[skin] - img[skin]).max() < 1e-5                       # the skin and eyelids are untouched
    pupil = d < 0.25
    assert np.abs(out[pupil] - img[pupil]).max() < 0.004                    # the pupil stays black
    catch = np.hypot(xx - cx - 0.3 * r, yy - cy + 0.35 * r) < 0.05 * r
    assert np.abs(out[catch] - img[catch]).max() < 0.03                     # and the catchlight white
    # one eye only
    one = gain(img, f, {"eye_colour": {"preset": "blue", "amount": 100, "which": "right"}})
    lcore = (dl > 0.48) & (dl < 0.85)
    assert np.abs(one[lcore] - img[lcore]).max() < 1e-5 and np.abs(one[core] - img[core]).max() > 0.01
    # texture kept: the fibres still vary
    lum = (out @ np.array([0.2880, 0.7119, 0.0001], np.float32))[core]
    assert lum.std() / lum.mean() > 0.08


def test_iris_tools_and_zero_does_nothing():
    img, P, truth = eye_scene()
    H, W = img.shape[:2]
    f = face_with_irises(P, truth, W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, r = truth[0]
    d = np.hypot(xx - cx, yy - cy) / r
    lum = lambda a: a @ np.array([0.2880, 0.7119, 0.0001], np.float32)       # noqa: E731
    iris_px = np.abs(img[..., 0] - 0.42) > 0.05                              # not the eyelid skin
    mid = (d > 0.5) & (d < 0.7) & iris_px
    outer = (d > 0.85) & (d < 0.95) & iris_px
    zero = gain(img, f, {"iris_light": 0, "ring": 0})
    assert np.abs(zero - img).max() < 1e-6
    light = gain(img, f, {"iris_light": 100})
    assert lum(light[mid]).mean() > 1.3 * lum(img[mid]).mean()
    pupil = d < 0.25
    assert np.abs(light[pupil] - img[pupil]).max() < 0.003
    ring = gain(img, f, {"ring": 100})
    assert lum(ring[outer]).mean() < 0.85 * lum(img[outer]).mean()
    assert abs(lum(ring[mid]).mean() / lum(img[mid]).mean() - 1) < 0.08
    catch = np.hypot(xx - cx - 0.3 * r, yy - cy + 0.35 * r) < 0.05 * r
    cl = gain(img, f, {"catch": 100})
    assert lum(cl[catch]).mean() > 1.15 * lum(img[catch]).mean()
    clar = gain(img, f, {"iris_clarity": 100})
    from studio.engine.local import eyes as ey
    from studio.engine.local import faces as fz
    inside = ey._polygon_mask(P[fz.RIGHT_EYE], xx.astype(float), yy.astype(float))
    band = (d > 0.55) & (d < 0.65) & inside & iris_px                        # the fibres run round this band
    cv = lambda a: lum(a[band]).std() / lum(a[band]).mean()                  # noqa: E731
    assert cv(clar) > 1.15 * cv(img)
    assert abs(lum(clar[band]).mean() / lum(img[band]).mean() - 1) < 0.1     # detail, not a brightness change
    # without irises the eye tools do nothing (faces mapped before them)
    f0 = dict(f)
    f0.pop("irises")
    assert np.abs(gain(img, f0, {"iris_light": 100}) - img).max() < 1e-6


# ------------------------------------------------------------------------------------------ red eye and pupils
def test_red_and_blown_pupils_are_found_and_normal_ones_are_not():
    from studio.engine.local import eyes as ey
    from studio.engine.local import portrait as pt
    for pupil, want in (((0.35, 0.02, 0.02), "redeye"), ((0.9, 0.85, 0.75), "pupil"), ((0.004, 0.004, 0.004), None)):
        img, P, truth = eye_scene(pupil=pupil, catch=False, pupil_frac=0.5)
        H, W = img.shape[:2]
        fc = pt.Face(face_with_irises(P, truth, W, H), W, H)
        from studio.engine.local import faces as fz
        found = ey.find_red_pupils(img, 0, 0, 1, fc.irises, {"r": fc.P[fz.RIGHT_EYE], "l": fc.P[fz.LEFT_EYE]})
        if want is None:
            assert found == []
        else:
            assert [b["kind"] for b in found] == [want, want]
            for b, (cx, cy, r) in zip(found, truth):
                assert math.hypot(b["x"] - cx, b["y"] - cy) < 0.2 * r and 0.35 * r <= b["r"] <= 0.96 * r


def test_pupil_fix_makes_a_dark_pupil_and_keeps_the_catchlight():
    from studio.engine.local import eyes as ey
    n = 60
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot(xx - 30, yy - 30)
    D = np.zeros((n, n, 3), np.float32) + np.array([0.08, 0.05, 0.02], np.float32)   # the iris around it
    glow = d < 15
    D[glow] = [0.3, 0.8, 0.4]                                                   # a green animal-eye glow
    core = np.hypot(xx - 34, yy - 25) < 2.5
    D[core] = [1.0, 1.0, 1.0]
    out = ey.pupil_fix(D, d < 15 * 0.85, (d > 16) & (d < 24))
    body = (d < 11) & ~(np.hypot(xx - 34, yy - 25) < 4.5)
    assert out[body].max() < 0.02 and np.allclose(out[body][:, 0], out[body][:, 1], atol=1e-4)
    assert out[np.hypot(xx - 34, yy - 25) < 1.0].min() > 0.8
    flat = np.zeros((n, n, 3), np.float32) + 0.05
    flat[glow] = 0.9                                                            # blown evenly: a small catchlight
    out2 = ey.pupil_fix(flat, d < 15 * 0.85, (d > 16) & (d < 24))
    bright = (out2[d < 12].mean(axis=-1) > 0.5).sum()
    assert 0 < bright < 0.15 * (d < 12).sum()


def test_pupil_spot_through_retouch_and_the_pipeline():
    from studio import engine as E
    from studio.engine.local import retouch as rt
    from studio.engine.pipeline.pipeline import Image, render
    img, P, truth = eye_scene(H=600, W=600, face_w=400, pupil=(0.85, 0.85, 0.8), catch=False, pupil_frac=0.5)
    raw = synthetic_raw(np.clip(img, 0, 0.95))
    im = Image(raw)
    H, W = img.shape[:2]
    cx, cy, r = truth[0]
    e = E.Edit()
    e.sharpen.enabled = False
    base = render(im, e)
    e.retouch = [{"kind": "pupil", "pts": [[cx / W, cy / H]], "size": 0.6 * r / max(W, H), "feather": 0.25,
                  "opacity": 100, "src": [0, 0], "enabled": True, "auto": ""}]
    assert rt.active(e.retouch)
    fixed = render(im, e)
    y, x = int(cy), int(cx)
    assert fixed[y - 3:y + 3, x - 3:x + 3].mean() < 0.3 * base[y - 3:y + 3, x - 3:x + 3].mean()
    far = (slice(20, 60), slice(20, 60))
    assert np.abs(fixed[far] - base[far]).max() < 1e-6


def test_find_red_eyes_and_irises_through_the_engine():
    from studio import engine as E
    from studio.engine.pipeline.pipeline import Image
    img, P, truth = eye_scene(H=600, W=600, face_w=400, pupil=(0.35, 0.02, 0.02), catch=False, pupil_frac=0.5)
    im = Image(synthetic_raw(np.clip(img, 0, 0.95)))
    H, W = img.shape[:2]
    e = E.Edit()
    e.portrait = {"faces": [face_with_irises(P, truth, W, H)]}
    res = E.find_red_eyes(im, e)
    assert res["red"] == 2 and all(s["kind"] == "redeye" and s["auto"] == "redeye" for s in res["spots"])
    e.retouch = res["spots"]
    assert E.find_red_eyes(im, e)["spots"] == []                               # not added twice
    f0 = face_with_irises(P, truth, W, H)
    f0.pop("irises")
    e.portrait = {"faces": [f0]}
    faces = E.add_irises(im, e)
    assert len(faces[0]["irises"]) == 2
    for got, (cx, cy, r) in zip(faces[0]["irises"], truth):
        assert math.hypot(got["c"][0] * W - cx, got["c"][1] * H - cy) < 0.25 * r


def test_eye_settings_round_trip_and_old_edits_render_the_same():
    from studio.engine.local import eyes as ey
    from studio.engine.local import portrait as pt
    assert ey.colour_settings({"eye_colour": {"preset": "blue", "amount": 0}}) is None
    assert ey.colour_settings({"eye_colour": {"preset": "none", "amount": 50}}) is None
    c = ey.colour_settings({"eye_colour": {"preset": "custom", "hue": 200, "sat": 0.9, "light": 2, "amount": 40}})
    assert c["sat"] == 0.4 and c["light"] == 0.95 and c["amount"] == 0.4
    h, s, L = ey.custom_from_srgb((60, 110, 170))
    assert 200 < h < 270 and 0.05 < s < 0.4
    assert all(pt.amounts({"younger": 100})[k] == 0 for k in ("iris_light", "iris_clarity", "iris_colour", "ring",
                                                               "catch"))                  # not part of Younger
    img, P, truth = eye_scene()
    H, W = img.shape[:2]
    f = face_with_irises(P, truth, W, H)
    f_old = dict(f)
    f_old.pop("irises")
    a = gain(img, f_old, {"eyes": 60, "teeth": 40})
    b = gain(img, f, {"eyes": 60, "teeth": 40})                                # irises found, no eye tool used
    assert np.abs(a - b).max() < 1e-6

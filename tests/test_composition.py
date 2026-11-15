"""Composition overlays (true geometry) and suggested crops (the rules they follow)."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from studio.engine.local import composition as C


@pytest.mark.parametrize("aspect", [1.5, 1.0, 0.8, 16 / 9])
def test_every_overlay_stays_in_the_frame(aspect):
    for name, _ in C.OVERLAYS:
        for o in range(C.ORIENTATIONS.get(name, 1)):
            L = C.lines(name, aspect, o)
            assert L, name
            for pl in L:
                assert len(pl) >= 2
                assert all(-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6 for x, y in pl), name
            assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in C.points(name, aspect, o)), name


def test_the_geometry_is_true():
    # the golden spiral converges where the theory says: (5 + sqrt 5) / 10 across and down
    (x, y), = C.points("spiral", 1.5, 0)
    assert abs(x - (5 + 5 ** 0.5) / 10) < 1e-6 and abs(y - (5 + 5 ** 0.5) / 10) < 1e-6
    assert C.points("spiral", 1.5, 1)[0][0] == pytest.approx(1 - x)               # turned: mirrored
    # golden triangles: the short lines meet the diagonal at right angles (in pixels, on a 3:2 frame)
    a = 1.5
    diag, l1, l2 = C.lines("triangles", a, 0)
    d = np.subtract(diag[1], diag[0]) * [a, 1]
    for ln in (l1, l2):
        v = np.subtract(ln[1], ln[0]) * [a, 1]
        assert abs(float(np.dot(d, v))) < 1e-9
    # the diagonal method's lines really are 45 degrees on the photo
    for ln in C.lines("diag45", a):
        dx, dy = (ln[1][0] - ln[0][0]) * a, ln[1][1] - ln[0][1]
        assert abs(abs(dx) - abs(dy)) < 1e-9
    # rabatment: the squares on the short sides
    assert C.lines("rabatment", 2.0) == [[(0.5, 0.0), (0.5, 1.0)], [(0.5, 0.0), (0.5, 1.0)]]
    # the harmonic armature's corner-to-midpoint lines cross the diagonals at the thirds
    assert (1 / 3, 1 / 3) in C.points("armature", 1.5)


def _portrait_subject(gaze=0.0):
    return {"main": (0.5, 0.3), "keep": [(0.4, 0.12, 0.6, 0.45)], "box": (0.4, 0.12, 0.6, 0.45), "head_top": 0.12,
            "gaze": gaze, "portrait": True, "what": "the eyes", "others": [], "joints": [(0.3, 0.8)], "necks": []}


def test_suggestions_keep_the_face_and_follow_the_method():
    sub = _portrait_subject()
    out = C.suggest(sub, 1.5, 0.8, "thirds")
    assert 1 <= len(out) <= 3
    for s in out:
        x0, y0, x1, y1 = s["crop"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert abs((x1 - x0) * 1.5 / (y1 - y0) - 0.8) < 1e-3                        # the crop shape asked for
        assert x0 <= 0.4 and x1 >= 0.6 and y0 <= 0.12 and y1 >= 0.45                 # the face is whole
        q = ((0.5 - x0) / (x1 - x0), (0.3 - y0) / (y1 - y0))
        assert min(math.hypot(q[0] - px, q[1] - py) for px, py in C.points("thirds", 0.8)) < 0.08
        assert s["method"] == "thirds" and "Rule of thirds" in s["text"]
    assert len({s["crop"] for s in out}) == len(out)                                # different from each other


def test_space_in_front_of_a_turned_face():
    for g in (0.6, -0.6):
        best = C.suggest(_portrait_subject(gaze=g), 1.5, 1.5, "thirds")[0]
        x0, _, x1, _ = best["crop"]
        qx = (0.5 - x0) / (x1 - x0)
        assert (qx < 0.5) if g > 0 else (qx > 0.5)                                 # more room in front


def test_no_edge_on_a_joint():
    sub = _portrait_subject()
    sub["joints"] = [(0.5, 0.62), (0.45, 0.70)]
    pen_cut, why = C.score((0.25, 0.05, 0.75, 0.62), sub, "thirds", 1.0)
    pen_ok, _ = C.score((0.25, 0.05, 0.75, 0.70 - 0.08), sub, "thirds", 1.0)
    assert "an edge cuts at a joint" in why
    assert all("joint" not in w for w in C.score((0.25, 0.05, 0.75, 0.55), sub, "thirds", 1.0)[1])


def test_saliency_finds_the_subject():
    h, w = 240, 360
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 0.25, np.float32) + 0.02 * np.random.default_rng(0).normal(size=(h, w, 3))
    blob = ((xx - 250) ** 2 + (yy - 80) ** 2) < 25 ** 2
    img[blob] = (0.9, 0.3, 0.2)
    sub = C.salient_subject(np.clip(img, 0, 1))
    assert abs(sub["main"][0] - 250 / w) < 0.06 and abs(sub["main"][1] - 80 / h) < 0.08


def test_crop_suggestions_on_a_real_portrait():
    """End to end on the synthetic astronaut raw: the face finder and person finder run, and every suggestion keeps
    her head whole."""
    from conftest import CAM_XYZ_5D4, synthetic_raw
    from studio.engine.color import camera as cam
    from studio.engine.color.spaces import PROPHOTO, SRGB, conversion_matrix
    from studio.engine.local import faces as fz
    from studio.engine.pipeline.edit import Edit
    from studio.engine.pipeline.pipeline import Image, composition_subject, suggest_crops
    if not fz.available():
        pytest.skip("face models missing")
    skimage = pytest.importorskip("skimage.data")
    from PIL import Image as PI
    im = np.asarray(PI.fromarray(skimage.astronaut()).resize((768, 768), PI.LANCZOS)).astype(np.float32) / 255
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    camrgb = np.clip((lin @ conversion_matrix(SRGB, PROPHOTO).T) @ np.linalg.inv(cam.camera_to_working(CAM_XYZ_5D4)).T
                     * 0.85, 0, 1).astype(np.float32)
    img = Image(synthetic_raw(camrgb))
    e = Edit()
    e.sharpen.enabled = False
    sub = composition_subject(img, e)
    assert sub["what"] == "the eyes" and sub["keep"]
    out = suggest_crops(img, e, 1.5, None)
    assert out
    hx0, hy0, hx1, hy1 = sub["keep"][0]
    for s in out:
        x0, y0, x1, y1 = s["crop"]
        assert x0 <= hx0 + 1e-6 and y0 <= max(hy0, 0) + 1e-6 and x1 >= hx1 - 1e-6 and y1 >= hy1 - 1e-6

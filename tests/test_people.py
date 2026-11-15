"""People: the person and pose models run by the Studio's own runner, and the Person mask's parts."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest


def _have_models():
    from studio.engine.local import people as pp
    if not pp.available():
        pytest.skip("people models not installed")
    return pp


def test_runner_matches_reference_on_people_models():
    """Same numbers as Google's own TensorFlow Lite runtime (LiteRT) gave on fixed inputs when these references were
    made: transposed convolution, hard swish, average pooling, logistic, depth-to-space and sparse weights."""
    pp = _have_models()
    from studio.engine.local.tinynet import Model
    yy, xx = np.mgrid[0:256, 0:256] / 256.0
    blob = np.exp(-(((xx - 0.5) / 0.18) ** 2 + ((yy - 0.55) / 0.3) ** 2))
    x = np.stack([0.2 + 0.6 * blob, 0.15 + 0.45 * blob, 0.1 + 0.35 * blob], -1)[None].astype(np.float32)
    seg = next(iter(Model(os.path.join(pp.model_dir(), pp.MODEL_FILES["seg"]))(x).values()))
    assert math.isclose(float(seg.sum()), 5.0434, rel_tol=1e-3)
    for key, sz, ref_sum in (("detect", 224, -8799.12), ("pose", 256, 7198.43)):
        y = (np.sin(np.linspace(0, 31, sz * sz * 3, dtype=np.float32)).reshape(1, sz, sz, 3) * 0.5 + 0.5)
        out = Model(os.path.join(pp.model_dir(), pp.MODEL_FILES[key]))(y.astype(np.float32))
        first = out["Identity"]
        assert math.isclose(float(first.astype(np.float64).sum()), ref_sum, rel_tol=1e-4)


def test_densify_expands_csr_weights():
    from studio.engine.local import tinynet as tn

    class Fake:                                           # the parts of a FlatBuffer table densify reads
        def __init__(self, scalars=None, vectors=None, tables=None):
            self.s, self.v, self.t = scalars or {}, vectors or {}, tables or {}

        def scalar(self, i, fmt, default=0):
            return self.s.get(i, default)

        def vector(self, i, fmt=None):
            return self.v.get(i, np.zeros(0, int) if fmt else [])

        def table(self, i):
            return self.t.get(i)
    seg = Fake(vectors={0: np.array([0, 2, 2, 3])})
    idx = Fake(vectors={0: np.array([1, 3, 0])})
    dims = [Fake(scalars={0: 0, 1: 3}), Fake(scalars={0: 1, 2: 3, 4: 3}, tables={3: seg, 5: idx})]
    sp = Fake(vectors={0: np.array([0, 1]), 1: np.zeros(0, int), 2: dims})
    d = tn.densify(np.array([5.0, 6.0, 7.0]), (3, 4), sp)
    assert np.array_equal(d, [[0, 5, 0, 6], [0, 0, 0, 0], [7, 0, 0, 0]])


def test_people_found_on_a_real_portrait():
    skimage = pytest.importorskip("skimage")
    pp = _have_models()
    im = skimage.data.astronaut().astype(np.float32) / 255.0
    r = pp.find(im)
    assert len(r["people"]) == 1 and r["people"][0]["conf"] > 0.8
    P = r["people"][0]["pts"]
    assert P[pp.L_SH, 2] > 0.5 and P[pp.R_SH, 2] > 0.5
    assert P[pp.NOSE, 1] < P[pp.L_SH, 1] and P[pp.NOSE, 1] < P[pp.R_SH, 1]   # the head is above the shoulders
    m = r["mask"]
    assert m[int(P[pp.NOSE, 1]), int(P[pp.NOSE, 0])] > 0.8                    # the face is in the person
    assert m[20, 480] < 0.2                                                   # the sky behind is not


def _synthetic_person(h=400, w=300):
    """A drawn 'person': skin face, brown hair on top, blue clothes, an arm; with body points and a person mask."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rgb = np.full((h, w, 3), 0.55, np.float32)                               # grey wall
    person = np.zeros((h, w))
    face = ((xx - 150) / 32) ** 2 + ((yy - 90) / 42) ** 2 < 1
    hair = (((xx - 150) / 42) ** 2 + ((yy - 70) / 50) ** 2 < 1) & ~face & (yy < 95)
    body = (np.abs(xx - 150) < 70) & (yy > 140)
    arm = (np.abs(xx - 240) < 16) & (yy > 150) & (yy < 330)
    for mk, col in ((body, (0.08, 0.12, 0.35)), (arm, (0.42, 0.26, 0.18)), (hair, (0.12, 0.07, 0.04)),
                    (face, (0.45, 0.28, 0.20))):
        rgb[mk] = col
        person[mk] = 1.0
    pts = np.zeros((33, 3))
    pts[:, 2] = 0.0
    for i, (x, y) in {0: (150, 95), 2: (138, 82), 5: (162, 82), 7: (120, 88), 8: (180, 88), 9: (142, 110),
                      10: (158, 110), 11: (230, 160), 12: (70, 160), 13: (240, 240), 15: (240, 320), 19: (240, 335),
                      23: (195, 330), 24: (105, 330)}.items():
        pts[i] = (x, y, 0.99)
    return rgb, person, pts, (face, hair, body, arm)


def test_person_parts_land_on_the_right_places():
    from studio.engine.local import masks as mk
    from studio.engine.local import people as pp
    h, w = 400, 300
    rgb, person, pts, (face, hair, body, arm) = _synthetic_person(h, w)
    shape = {"type": "person", "op": "add", "part": "person", "w": w, "h": h,
             "data": mk.encode_bitmap((person * 255).astype(np.uint8)),
             "people": [[[x / w, y / h, v] for x, y, v in pts]], "faces": []}
    xs, ys = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    c = mk.Ctx(rgb, xs, ys, w, h, 1.0, lambda y: np.clip(y, 0, 1))
    got = {p: pp.part_alpha(dict(shape, part=p), c) for p in pp.PARTS}
    inner = lambda m: m & (np.abs(np.gradient(m.astype(float))[0]) + np.abs(np.gradient(m.astype(float))[1]) == 0)  # noqa
    assert got["person"][inner(body)].mean() > 0.9 and got["person"][inner(~(person > 0))].mean() < 0.1
    assert np.allclose(got["background"], 1.0 - got["person"], atol=1e-5)
    assert got["face"][inner(face)].mean() > 0.7 and got["face"][inner(body)].mean() < 0.05
    assert got["hair"][inner(hair)].mean() > 0.6 and got["hair"][inner(face)].mean() < 0.15
    assert got["clothes"][inner(body)].mean() > 0.7 and got["clothes"][inner(face)].mean() < 0.1
    assert got["skin"][inner(face)].mean() > 0.7 and got["skin"][inner(body)].mean() < 0.15
    assert got["arms"][inner(arm)].mean() > 0.6 and got["arms"][inner(face)].mean() < 0.05
    assert got["legs"].max() < 0.05                                          # no legs in view
    assert got["torso"][(np.abs(xs - 150) < 40) & (ys > 200) & (ys < 300)].mean() > 0.6


def test_runner_matches_reference_on_the_parts_model():
    """The optional multiclass model (softmax, transpose, sum, nearest resize, transposed convolution)."""
    pp = _have_models()
    if not pp.multiclass_available():
        pytest.skip("multiclass model not downloaded")
    from studio.engine.local.tinynet import Model
    yy, xx = np.mgrid[0:256, 0:256] / 256.0
    blob = np.exp(-(((xx - 0.5) / 0.18) ** 2 + ((yy - 0.55) / 0.3) ** 2))
    x = (np.stack([0.2 + 0.6 * blob, 0.15 + 0.45 * blob, 0.1 + 0.35 * blob], -1)[None] * 2 - 1).astype(np.float32)
    v = next(iter(Model(os.path.join(pp.model_dir(), pp.MULTICLASS))(x).values()))
    assert v.shape == (1, 256, 256, 6)
    assert math.isclose(float(v.astype(np.float64).sum()), -157441.08, rel_tol=1e-5)


def test_parts_from_the_parts_model_labels():
    """When the multiclass labels are stored, hair / skin / face / clothes come from them."""
    from studio.engine.local import masks as mk
    from studio.engine.local import people as pp
    h, w = 400, 300
    rgb, person, pts, (face, hair, body, arm) = _synthetic_person(h, w)
    enc = lambda m: mk.encode_bitmap((m * 255).astype(np.uint8))          # noqa: E731
    shape = {"type": "person", "op": "add", "part": "person", "w": w, "h": h, "data": enc(person),
             "people": [[[x / w, y / h, v] for x, y, v in pts]], "faces": [], "cw": w, "ch": h,
             "classes": {"hair": enc(hair.astype(float)), "face_skin": enc(face.astype(float)),
                         "body_skin": enc(arm.astype(float)), "clothes": enc(body.astype(float))}}
    xs, ys = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    c = mk.Ctx(rgb, xs, ys, w, h, 1.0, lambda y: np.clip(y, 0, 1))
    got = {p: pp.part_alpha(dict(shape, part=p), c) for p in ("hair", "face", "skin", "clothes")}
    core = lambda m: m & (np.abs(np.gradient(m.astype(float))[0]) + np.abs(np.gradient(m.astype(float))[1]) == 0)  # noqa
    assert got["hair"][core(hair)].mean() > 0.8 and got["hair"][core(body)].mean() < 0.05
    assert got["face"][core(face)].mean() > 0.8
    assert got["skin"][core(arm)].mean() > 0.7 and got["skin"][core(face)].mean() > 0.8
    assert got["clothes"][core(body)].mean() > 0.8 and got["clothes"][core(face)].mean() < 0.05

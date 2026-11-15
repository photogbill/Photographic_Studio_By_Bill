"""Phase 9 (first part): masks and local adjustments."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.local import masks as mk
from studio.engine.pipeline.pipeline import Image

LUM = np.array([0.288, 0.712, 0.0], np.float32)


def _img(h=200, w=300, sc=None, flip=0):
    return Image(synthetic_raw(scene(h, w) if sc is None else sc, flip=flip))


def _radial(cx=0.5, cy=0.5, r=0.2, feather=0.3, **adj):
    return {"name": "m", "enabled": True, "opacity": 100,
            "shapes": [{"type": "radial", "cx": cx, "cy": cy, "rx": r, "ry": r, "angle": 0, "feather": feather}],
            "adjust": adj}


def test_masks_off_or_empty_change_nothing():
    img = _img()
    base = E.render(img, E.Edit())
    assert np.array_equal(base, E.render(img, E.Edit(masks=[])))
    assert np.array_equal(base, E.render(img, E.Edit(masks=[_radial()])))           # no adjustment = inactive
    off = _radial(exposure=1.0)
    off["enabled"] = False
    assert np.array_equal(base, E.render(img, E.Edit(masks=[off])))


def test_radial_exposure_brightens_inside_only():
    img = _img(sc=np.full((200, 300, 3), 0.1, np.float32))
    y0 = E.render(img, E.Edit()) @ LUM
    y1 = E.render(img, E.Edit(masks=[_radial(exposure=1.0)])) @ LUM
    assert y1[100, 150] > 1.6 * y0[100, 150]                                       # centre: about +1 stop
    assert abs(y1[5, 5] / y0[5, 5] - 1) < 1e-3                                     # corner untouched
    a = E.render_mask(img, E.Edit(masks=[_radial(exposure=1.0)]), 0, scale=1.0)
    assert a[100, 150] > 0.99 and a[5, 5] < 1e-3


def test_linear_gradient_fades_from_start_to_end():
    img = _img()
    m = {"shapes": [{"type": "linear", "x0": 0.2, "y0": 0.5, "x1": 0.8, "y1": 0.5}], "adjust": {"exposure": 1}}
    a = E.render_mask(img, E.Edit(masks=[m]), 0, scale=1.0)
    row = a[100]
    assert row[:55].min() > 0.99 and row[245:].max() < 0.01 and np.all(np.diff(row) <= 1e-6)


def test_region_render_matches_full_frame_with_masks():
    img = _img(240, 360)
    stroke = {"pts": [[0.2, 0.3], [0.5, 0.4], [0.7, 0.7]], "size": 0.05, "feather": 0.5, "flow": 1.0}
    masks = [_radial(0.4, 0.5, 0.25, exposure=0.7, highlights=-50, saturation=-40, clarity=40),
             {"shapes": [{"type": "brush", "strokes": [stroke]}], "adjust": {"texture": 50, "sharpness": 40,
                                                                           "temperature": 30}}]
    e = E.Edit(masks=masks)
    full = E.render(img, e)
    part = E.render(img, e, region=(120, 80, 120, 90))
    assert np.abs(part - full[80:170, 120:240]).max() < 3e-3


def test_masks_stay_on_their_subject_when_straightened_and_rotated_raw():
    img = _img(240, 360, flip=6)
    e = E.Edit(masks=[_radial(0.3, 0.4, 0.05, feather=0.2, exposure=1.0)])
    e.geometry.rotate = 5.0
    a = E.render_mask(img, e, 0, scale=1.0)
    i, j = np.unravel_index(np.argmax(a), a.shape)
    ii, jj = np.nonzero(a > 0.5 * a.max())
    u, v = (jj.mean() + 0.5) / a.shape[1], (ii.mean() + 0.5) / a.shape[0]
    xs, ys = E.view_to_s(img, e, u, v)
    assert abs(float(xs) - 0.3) < 0.01 and abs(float(ys) - 0.4) < 0.01


def test_brush_flow_erase_and_auto_mask():
    img = _img()
    stroke = {"pts": [[0.2, 0.5], [0.8, 0.5]], "size": 0.04, "feather": 0.3, "flow": 0.6}
    m = {"shapes": [{"type": "brush", "strokes": [stroke]}], "adjust": {"exposure": 1}}
    a = E.render_mask(img, E.Edit(masks=[m]), 0, scale=1.0)
    assert abs(a[100, 150] - 0.6) < 0.02 and a[40, 150] < 1e-3
    erase = dict(stroke, pts=[[0.45, 0.3], [0.45, 0.7]], flow=1.0, erase=True)
    m2 = {"shapes": [{"type": "brush", "strokes": [stroke, erase]}], "adjust": {"exposure": 1}}
    a2 = E.render_mask(img, E.Edit(masks=[m2]), 0, scale=1.0)
    assert a2[100, 135] < 0.02 and abs(a2[100, 210] - 0.6) < 0.02
    # auto mask: a stroke across a green and a red half stays on the colour it was started on
    sc = np.zeros((200, 300, 3), np.float32)
    sc[:, :150] = [0.1, 0.3, 0.08]
    sc[:, 150:] = [0.35, 0.08, 0.06]
    im2 = _img(sc=sc)
    ref = E.mask_features(im2, E.Edit(), 0.25, 0.5)
    auto = {"shapes": [{"type": "brush", "auto": True, "strokes": [dict(stroke, flow=1.0, ref=ref)]}],
            "adjust": {"exposure": 1}}
    a3 = E.render_mask(im2, E.Edit(masks=[auto]), 0, scale=1.0)
    assert a3[100, 90] > 0.8 and a3[100, 220] < 0.1


def test_range_masks_by_brightness_and_colour():
    sc = np.zeros((120, 200, 3), np.float32)
    sc[:, :100] = 0.02
    sc[:, 100:] = 0.5
    img = _img(sc=sc)
    m = {"shapes": [{"type": "luminance", "lo": 0.6, "hi": 1.0, "soft": 0.1}], "adjust": {"exposure": -1}}
    a = E.render_mask(img, E.Edit(masks=[m]), 0, scale=1.0)
    assert a[60, 150] > 0.95 and a[60, 50] < 0.01
    sc2 = np.zeros((120, 200, 3), np.float32)
    sc2[:, :100] = [0.1, 0.3, 0.08]                                                 # green
    sc2[:, 100:] = [0.1, 0.12, 0.35]                                                # blue
    img2 = _img(sc=sc2)
    from studio.engine.color.grade import to_oklab
    lab = to_oklab(E.render(img2, E.Edit())[60:61, 150:151])[0, 0]
    hue = float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)
    m2 = {"shapes": [{"type": "colour", "hue": hue, "width": 30, "soft": 20, "chroma": 0.03}],
          "adjust": {"saturation": -100}}
    out = E.render(img2, E.Edit(masks=[m2]))
    blue, green = out[60, 150], out[60, 50]
    assert np.ptp(blue) < 0.15 * np.ptp(E.render(img2, E.Edit())[60, 150]) and np.ptp(green) > 0.02


def test_combining_shapes_subtract_intersect_invert_opacity():
    img = _img()
    r1 = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.3, "ry": 0.3, "feather": 0.05}
    r2 = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.1, "ry": 0.1, "feather": 0.05}
    e = lambda shapes, **kw: E.Edit(masks=[dict({"shapes": shapes, "adjust": {"exposure": 1}}, **kw)])  # noqa
    sub = E.render_mask(img, e([r1, dict(r2, op="subtract")]), 0, scale=1.0)
    inter = E.render_mask(img, e([r1, dict(r2, op="intersect")]), 0, scale=1.0)
    inv = E.render_mask(img, e([r2], invert=True, opacity=50), 0, scale=1.0)
    assert sub[100, 150] < 0.01 and sub[100, 210] > 0.95
    assert inter[100, 150] > 0.95 and inter[100, 210] < 0.01
    assert inv[100, 150] < 0.01 and abs(inv[10, 10] - 0.5) < 0.01


def _leaves():
    from scipy.ndimage import gaussian_filter
    h, w = 240, 360
    yy, xx = np.mgrid[0:h, 0:w]
    sc = np.zeros((h, w, 3), np.float32) + np.array([0.20, 0.12, 0.06], np.float32)          # brown ground
    masks = []
    for cy, cx in ((80, 90), (90, 220), (180, 150)):
        m = ((yy - cy) / 45.0) ** 2 + ((xx - cx) / 70.0) ** 2 < 1
        masks.append(m)
        sc[m] = [0.08, 0.30, 0.06]
    # a dark shadow line where leaves touch
    sc[(np.abs(xx - 160) < 2) & (yy < 140)] = 0.02
    rng = np.random.default_rng(1)
    sc = sc * (1 + 0.03 * rng.standard_normal(sc.shape)).astype(np.float32)
    return np.clip(sc, 0, 1).astype(np.float32), masks


def test_click_an_object_selects_one_leaf():
    sc, leaves = _leaves()
    img = _img(sc=sc)
    shape = E.object_mask(img, E.Edit(), [[90 / 360, 80 / 240]])
    m = {"shapes": [shape], "adjust": {"exposure": 1}}
    a = E.render_mask(img, E.Edit(masks=[m]), 0, scale=1.0)
    from scipy.ndimage import binary_erosion
    inside = a[binary_erosion(leaves[0], iterations=4)].mean()                    # edges are soft by design
    others = max(a[leaves[1]].mean(), a[leaves[2]].mean())
    ground = a[~(leaves[0] | leaves[1] | leaves[2])].mean()
    assert inside > 0.95 and others < 0.1 and ground < 0.05
    bm = mk.decode_bitmap(shape["data"], shape["w"], shape["h"])
    assert bm.shape == (shape["h"], shape["w"]) and set(np.unique(bm)) <= {0, 255}


def test_local_tone_colour_and_detail():
    img = _img()
    base = E.render(img, E.Edit())
    m = _radial(0.5, 0.5, 0.3, feather=0.1, highlights=-80, contrast=40, vibrance=30, noise=60)
    out = E.render(img, E.Edit(masks=[m]))
    a = E.render_mask(img, E.Edit(masks=[m]), 0, scale=1.0)
    inside, outside = a > 0.99, a == 0
    assert np.abs(out[outside] - base[outside]).max() < 1e-5
    assert np.abs(out[inside] - base[inside]).mean() > 1e-3
    yb, yo = base @ LUM, out @ LUM
    bright = inside & (yb > np.percentile(yb[inside], 90))
    assert yo[bright].mean() < yb[bright].mean()


def test_view_to_s_is_the_inverse_of_the_display():
    img = _img(240, 360)
    e = E.Edit()
    xs, ys = E.view_to_s(img, e, np.array([0.25, 0.5]), np.array([0.5, 0.75]))
    assert np.allclose(xs, [0.25 - 0.5 / 360, 0.5 - 0.5 / 360], atol=1e-6)
    assert np.allclose(ys, [0.5 - 0.5 / 240, 0.75 - 0.5 / 240], atol=1e-6)

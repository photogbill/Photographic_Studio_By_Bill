"""Phase 5: stage 13 geometry (distortion, perspective, straighten, crop), lensfun, falloff, straight lines,
lens tagging and export metadata. Synthetic data with known answers."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.lens import falloff as F
from studio.engine.lens import geometry as G
from studio.engine.lens import library as LIB
from studio.engine.lens import lines as LN
from studio.engine.pipeline.pipeline import Image, apply_orientation


# ------------------------------------------------------------------------------------------ frames and warps
@pytest.mark.parametrize("flip", [0, 3, 5, 6])
def test_orientation_maps_agree_with_the_renderer(flip):
    S = np.arange(6 * 10).reshape(6, 10).astype(float)
    D = apply_orientation(S, flip)
    for yd in range(D.shape[0]):
        for xd in range(D.shape[1]):
            xs, ys = G.d_to_s(xd, yd, 10, 6, flip)
            assert D[yd, xd] == S[int(ys), int(xs)]
            assert G.s_to_d(xs, ys, 10, 6, flip) == (xd, yd)


def test_straighten_and_perspective_directions():
    w = G.Warp(1000, 600, 0, G.GeomParams(rotate=10, constrain=False))
    t = math.radians(10)
    x, y = w.canvas_to_d(w.cx + 100 * math.cos(t), w.cy - 100 * math.sin(t))
    assert abs(x - w.cx - 100) < 1e-6 and abs(y - w.cy) < 1e-6      # + = counter-clockwise
    p = G.Warp(1000, 600, 0, G.GeomParams(vertical=50, constrain=False))
    top = p.canvas_to_d(999, 0)[0] - p.canvas_to_d(0, 0)[0]
    bottom = p.canvas_to_d(999, 599)[0] - p.canvas_to_d(0, 599)[0]
    assert top < bottom                                              # + = the top is stretched wider


def test_constrain_zoom_for_a_rotation():
    w = G.Warp(6000, 4000, 0, G.GeomParams(rotate=10))
    assert abs(w.zoom - (math.cos(math.radians(10)) + 1.5 * math.sin(math.radians(10)))) < 1e-3
    assert w.output_size[0] < 6000 and w.output_size[1] < 4000      # smaller, never upsampled


def test_distortion_models_and_slider_sign():
    r = np.linspace(0, 1.8, 10)
    assert np.allclose(G.radial_scale({"model": "poly3", "terms": [0.0]}, r), 1.0)
    pt = G.radial_scale({"model": "ptlens", "terms": [0.01, -0.03, 0.03]}, np.array([1.0]))
    assert abs(pt[0] - 1.0) < 1e-12                                  # radius 1 is fixed by construction
    assert G.manual_model(50)["terms"][0] < 0                        # + fixes barrel = models barrel (k < 0)
    assert G.manual_model(0) is None


def test_crop_for_aspect():
    assert G.crop_for_aspect(6000, 4000, "original") is None
    c = G.crop_for_aspect(6000, 4000, "1:1")
    assert abs((c[2] - c[0]) * 6000 - (c[3] - c[1]) * 4000) < 1
    c = G.crop_for_aspect(4000, 6000, "4:5")                         # portrait frame, portrait ratio
    assert abs((c[2] - c[0]) * 4000 / ((c[3] - c[1]) * 6000) - 0.8) < 1e-6


def test_distortion_matches_lensfun():
    lf = pytest.importorskip("lensfunpy")
    db = lf.Database()
    cams = db.find_cameras("Canon", "Canon EOS 5D Mark IV")
    lenses = db.find_lenses(cams[0], None, "Canon EF 50mm f/1.8 STM") if cams else []
    if not lenses:
        pytest.skip("lens not in this lensfun database")
    cam, lens = cams[0], lenses[0]
    W, H = 6720, 4480
    mod = lf.Modifier(lens, cam.crop_factor, W, H)
    mod.initialize(50.0, 2.8, 1000.0, scale=1.0, pixel_format=np.float32, flags=lf.ModifyFlags.DISTORTION)
    ref = mod.apply_geometry_distortion(0, 0, W, 1)[0]
    d = lens.interpolate_distortion(50.0)
    model = {"model": "ptlens", "terms": list(d.terms)[:3], "norm": cam.crop_factor / lens.crop_factor}
    w = G.Warp(W, H, 0, G.GeomParams(distortion=model, constrain=False))
    xs, ys = w.canvas_to_d(np.arange(W, dtype=float), np.zeros(W))
    assert max(np.abs(xs - ref[:, 0]).max(), np.abs(ys - ref[:, 1]).max()) < 0.3


# ------------------------------------------------------------------------------------------ in the pipeline
def _img(h=240, w=360, flip=0):
    return Image(synthetic_raw(scene(h, w), flip=flip))


@pytest.mark.parametrize("flip", [0, 6])
def test_crop_only_is_a_plain_rectangle(flip):
    img = _img(flip=flip)
    full = E.render(img, E.Edit())
    e = E.Edit()
    e.geometry.crop = [0.25, 0.2, 0.75, 0.9]
    out = E.render(img, e)
    W, H = img.output_size
    x0, y0 = round(0.25 * W), round(0.2 * H)
    assert out.shape[:2] == (round(0.9 * H) - y0, round(0.75 * W) - x0)
    assert np.allclose(out, full[y0:y0 + out.shape[0], x0:x0 + out.shape[1]], atol=1e-5)   # no resampling


def test_rotation_leaves_no_empty_corners():
    img = _img()
    e = E.Edit()
    e.geometry.rotate = 7.0
    out = E.render(img, e)
    assert out.shape[1] < 360 and out.shape[0] < 240
    for c in (out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]):
        assert c.max() > 0.005


def test_view_window_matches_the_full_render():
    img = _img(320, 480)
    e = E.Edit()
    e.geometry.rotate = 3.0
    e.geometry.vertical = 20
    full = E.render(img, e)
    ow, oh = E.output_size(img, e)
    assert full.shape[:2] == (oh, ow)
    v = E.render_view(img, e, 0.5, 0.5, 120, 80)
    x, y = round(0.5 * ow - 60), round(0.5 * oh - 40)
    assert np.abs(v - full[y:y + 80, x:x + 120]).max() < 1e-4


def test_render_window_matches_the_full_render_at_any_scale():
    """Develop's zoomed view renders only the part on screen: it must be the same pixels as that part of the whole
    photo rendered at the same scale (warped, crop-only and plain), and render_view at 100%."""
    img = _img(320, 480)
    for setup in ("plain", "crop", "warp"):
        e = E.Edit()
        if setup == "crop":
            e.geometry.crop = [0.1, 0.05, 0.9, 0.95]
        elif setup == "warp":
            e.geometry.rotate = 3.0
            e.geometry.vertical = 20
        ow, oh = E.output_size(img, e)
        full = E.render(img, e)
        x, y, w, h = ow // 4, oh // 3, ow // 2, oh // 3
        win = E.render_window(img, e, (x, y, w, h), 1.0)
        assert win.shape[:2] == (h, w)
        assert np.abs(win - full[y:y + h, x:x + w]).max() < 1e-4, setup
        assert np.abs(win - E.render_view(img, e, (x + w / 2) / ow, (y + h / 2) / oh, w, h)).max() < 1e-4
        sc = 0.75                                                     # on a shared pixel grid: x, y, w, h by 4
        x, y, w, h = x // 4 * 4, y // 4 * 4, w // 4 * 4, h // 4 * 4
        part = E.render_window(img, e, (x, y, w, h), sc)
        whole = E.render(img, e, scale=sc)
        X, Y = round(x * sc), round(y * sc)
        hh, ww = part.shape[:2]
        assert abs(ww - w * sc) <= 1.01 and abs(hh - h * sc) <= 1.01
        inner = (slice(4, hh - 4), slice(4, ww - 4))                  # resampling edges aside
        assert np.abs(part[inner] - whole[Y:Y + hh, X:X + ww][inner]).mean() < 0.01, setup


def test_half_size_region_is_cut_from_the_cached_whole_frame():
    """A zoomed view up to 60% renders half-size: its region comes out of the whole frame the preview already
    worked out (no second decode), and matches a region worked out on its own."""
    from studio.engine.pipeline import pipeline as P
    img = _img(320, 480)
    e = E.Edit()
    region = (37, 51, 120, 90)
    fresh = P.linear_rgb(img, e, region, half_size=True, cache=False)
    P.linear_rgb(img, e, None, half_size=True)                    # what the preview leaves behind
    calls = []
    real = P.lin.normalise
    P.lin.normalise = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        cut = P.linear_rgb(img, e, region, half_size=True)
    finally:
        P.lin.normalise = real
    assert not calls                                              # no decoding: cut from the cache
    assert cut.shape == fresh.shape and np.abs(cut - fresh).max() < 1e-4
    ow, oh = E.output_size(img, e)
    a = E.render_window(img, e, (40, 60, 160, 120), 0.4)
    img.drop_cache()
    b = E.render_window(img, e, (40, 60, 160, 120), 0.4)
    assert a.shape == b.shape and np.abs(a - b).max() < 1e-3


def test_lens_tag_does_not_invalidate_the_raw_cache():
    a, b = E.Edit(), E.Edit()
    b.lens.tag = "Laack 13.5cm Anastigmat"
    assert a.raw_key() == b.raw_key()


# ------------------------------------------------------------------------------------------ falloff
def _textured(h=800, w=1200, seed=5):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.2, np.float32)
    for _ in range(400):
        y, x = rng.integers(0, h - 40), rng.integers(0, w - 40)
        hh, ww = rng.integers(10, 120, 2)
        img[y:y + hh, x:x + ww] = rng.uniform(0.02, 0.7)
    return img * np.exp2(gaussian_filter(rng.standard_normal((h, w)), 3) * 0.3)


def test_falloff_is_estimated_from_a_photo():
    sc = _textured()
    h, w = sc.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (math.hypot(h, w) / 2)
    est = F.estimate(sc * np.exp2(-1.0 * r ** 2.2))
    assert est is not None and abs(est["corner_ev"] + 1.0) < 0.2
    assert F.estimate(_textured(seed=6)) is None                     # no vignetting, none invented


def test_falloff_gain_map_and_off_by_default():
    m = {"model": "pa", "terms": [-0.3, -0.2, 0.0], "norm": 1.0}
    g = F.gain_map(m, 1.0, 101, 151, (0, 0), (151, 101))
    assert abs(g[50, 75] - 1.0) < 1e-3
    assert abs(math.log2(g[0, 0]) + F.corner_ev(m)) < 0.05
    img = _img()
    e = E.Edit()
    e.lens.vignette_model = m                                        # measured, but correction left off
    assert np.array_equal(E.render(img, e), E.render(img, E.Edit()))
    e.lens.vignette = True
    on = E.render(img, e)
    assert on[2, 2].sum() > E.render(img, E.Edit())[2, 2].sum()     # corners brighter when switched on


# ------------------------------------------------------------------------------------------ straight lines
def _grid(rot=0.0, k=0.0, h=1000, w=1500):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    t = math.radians(rot)
    xr = (xx - cx) * math.cos(t) - (yy - cy) * math.sin(t)
    yr = (xx - cx) * math.sin(t) + (yy - cy) * math.cos(t)
    img = np.full((h, w), 0.6)
    for g in np.arange(-900, 900, 110):
        img[np.abs(xr - g) < 2.5] = 0.1
        img[np.abs(yr - g) < 2.5] = 0.1
    img = gaussian_filter(img, 0.8)
    if k:
        P = np.stack([xx.ravel(), yy.ravel()], 1)
        U = LN._undistort_points(P, k, cx, cy, math.hypot(w, h) / 2 / math.sqrt(3.25))
        img = map_coordinates(img, [U[:, 1].reshape(h, w), U[:, 0].reshape(h, w)], order=1, mode="nearest")
    return img


def test_auto_level():
    for rot in (2.0, -3.5):
        lv = LN.auto_level(LN.segments(_grid(rot)), (1500, 1000))
        assert lv is not None and abs(lv["rotate"] + rot) < 0.15


def test_distortion_measured_from_straight_lines():
    for k in (-0.05, 0.04):
        d = LN.measure_distortion(LN.segments(_grid(k=k)), (1500, 1000))
        assert d is not None and 0.7 * abs(k) < abs(d["terms"][0]) < 1.3 * abs(k) and np.sign(d["terms"][0]) == np.sign(k)


def test_named_never_guess_distortion_on_organic_subjects():
    rng = np.random.default_rng(1)
    org = gaussian_filter(rng.standard_normal((1000, 1500)), 12)
    assert LN.measure_distortion(LN.segments(org), (1500, 1000)) is None
    yy, xx = np.mgrid[0:1000, 0:1500]
    rings = np.full((1000, 1500), 0.6)
    for cy, cx, r in ((300, 400, 250), (600, 1000, 300), (500, 700, 420)):
        rings[np.abs(np.hypot(yy - cy, xx - cx) - r) < 2.5] = 0.1
    assert LN.measure_distortion(LN.segments(gaussian_filter(rings, 0.8)), (1500, 1000)) is None


def test_fix_lens_on_an_organic_frame_reports_and_changes_no_geometry():
    from studio.engine.lens.fix import fix_lens
    img = _img(400, 600)
    e, rep = fix_lens(img, E.Edit())
    assert e.geometry.distortion is None and e.lens.auto_ca and e.lens.defringe and not e.lens.vignette
    assert {r["what"] for r in rep} >= {"Lateral colour", "Distortion", "Vignetting", "Lens"}


# ------------------------------------------------------------------------------------------ tagging and export
def test_lens_list_and_tagging(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    names = [e.name for e in LIB.list_lenses()]
    assert "Laack 13.5cm Anastigmat" in names and LIB.is_character("Laack 13.5cm Anastigmat")
    LIB.add_lens(LIB.LensEntry(name="My Petzval 85", focal_mm=85, character=True))
    assert LIB.get("my petzval 85").focal_mm == 85
    assert json.load(open(tmp_path / "my_lenses.json"))["lenses"][0]["name"] == "My Petzval 85"
    e = E.Edit()
    e.lens.tag = "My Petzval 85"
    md = type("M", (), {"lens": "", "manual_lens": True, "fnumber": None})()
    assert LIB.effective_lens(md, e) == ("My Petzval 85", "tagged by you")


def test_export_writes_lens_and_true_iso(tmp_path):
    import tifffile
    from PIL import Image as PI
    img = _img()
    img.raw.meta.iso = 102400.0
    e = E.Edit()
    e.lens.tag = "Laack 13.5cm Anastigmat"
    t = E.export(img, e, E.ExportSpec(format="tiff"), out_path=str(tmp_path / "a.tif"))
    xmp = tifffile.TiffFile(t).pages[0].tags["XMP"].value
    assert b"Laack 13.5cm Anastigmat" in xmp and b"102400" in xmp
    j = E.export(img, e, E.ExportSpec(format="jpeg"), out_path=str(tmp_path / "a.jpg"))
    ex = PI.open(j).getexif().get_ifd(0x8769)
    assert ex[0xA434] == "Laack 13.5cm Anastigmat" and ex[0x8832] == 102400


def test_sidecar_round_trip_with_geometry(tmp_path):
    e = E.Edit()
    e.geometry.rotate = -1.25
    e.geometry.crop = [0.1, 0.1, 0.9, 0.95]
    e.geometry.distortion = {"model": "poly3", "terms": [-0.02], "norm": 1.0, "source": "manual"}
    e.sharpen.enabled = True
    e.microcontrast.amount = 0.4
    e.lens.tag = "Helios 44-2 58mm f/2"
    back = E.Edit.from_json(e.to_json())
    assert back.to_dict() == e.to_dict()

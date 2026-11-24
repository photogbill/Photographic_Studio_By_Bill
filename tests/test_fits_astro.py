"""FITS in and out (io/fits.py, astropy) and the star tools (stack/astro.py, photutils): synthetic data, known truth.

Skipped when the optional astronomy packages are not installed (the Studio starts without them)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import ROOT, mosaic_of

astropy = pytest.importorskip("astropy")
from astropy.io import fits as AF  # noqa: E402

from studio.engine.io import fits as FX  # noqa: E402
from studio.engine.stack import astro  # noqa: E402

OUT = ROOT / "outputs" / "pytest_fits"


def _out(name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    for q in (p, p.with_name(p.name + ".partial")):
        if q.exists():
            q.unlink()
    return str(p)


def _star_field(h=240, w=320, n=60, sigma=1.3, seed=1, trail=0.0, bg=0.02, noise=0.002):
    rng = np.random.default_rng(seed)
    sy, sx = rng.uniform(12, h - 12, n), rng.uniform(12, w - 12, n)
    fl = rng.uniform(0.1, 0.5, n)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w), bg, np.float32)
    steps = np.linspace(-trail / 2, trail / 2, 9) if trail else [0.0]
    for y0, x0, f0 in zip(sy, sx, fl):
        sl = (slice(int(y0) - 8, int(y0) + 9), slice(int(x0) - 8, int(x0) + 9))
        for s in steps:
            img[sl] += f0 / len(steps) * np.exp(-((yy[sl] - y0) ** 2 + (xx[sl] - x0 - s) ** 2) / (2 * sigma ** 2))
    img += np.random.default_rng(seed + 100).normal(0, noise, img.shape).astype(np.float32)
    return img, np.stack([sy, sx], 1)


def test_rgb_round_trip_keeps_values_upright_and_above_one():
    rng = np.random.default_rng(0)
    a = rng.random((37, 53, 3)).astype(np.float32) * 0.8
    a[0, 0] = (2.5, 0.1, 0.3)                       # top-left, brighter than white: must survive, in place
    p = _out("rgb_Studio.fits")
    FX.write(p, a, {"model": "EOS R5", "make": "Canon", "exposure_s": 30.0, "iso": 1600,
                    "datetime": "2026:11:23 21:04:05"})
    with AF.open(p) as h:
        assert h[0].data.shape == (3, 37, 53)       # planes first, what Siril and PixInsight expect
        assert h[0].header["ROWORDER"] == "BOTTOM-UP"
        assert h[0].header["DATE-OBS"] == "2026-11-23T21:04:05"
        assert np.allclose(h[0].data[:, -1, 0], (2.5, 0.1, 0.3))   # FITS row 0 is the bottom
    back, cards, info = FX.read(p)
    assert info["planes"] == 3 and np.array_equal(back, a)
    m = FX.metadata(cards)
    assert (m.model, m.exposure_s, m.iso, m.datetime) == ("EOS R5", 30.0, 1600.0, "2026:11:23 21:04:05")
    from studio.engine.io.export import is_own_export
    assert is_own_export(p) and FX.is_own(p)


def test_foreign_integer_files_scale_and_row_order():
    a = (np.arange(20 * 30).reshape(20, 30) * 100 % 65536).astype(np.uint16)
    p = _out("foreign_bottom.fits")
    AF.PrimaryHDU(a).writeto(p)                     # astropy stores uint16 as BITPIX 16 + BZERO 32768
    back, _, _ = FX.read(p)
    assert np.allclose(back, a[::-1] / 65535.0, atol=1e-7)
    q = _out("foreign_top.fits")
    h = AF.PrimaryHDU(a)
    h.header["ROWORDER"] = "TOP-DOWN"
    h.writeto(q)
    back, _, _ = FX.read(q)
    assert np.allclose(back, a / 65535.0, atol=1e-7)
    from studio.engine.io.export import is_own_export
    assert not is_own_export(p)                     # somebody else's file is never replaced
    f = _out("dss_float.fits")                      # DeepSkyStacker: floats on a 0..65535 scale
    AF.PrimaryHDU(np.full((8, 8), 32767.5, np.float32)).writeto(f)
    back, _, info = FX.read(f)
    assert info["scale"] == 65535.0 and np.allclose(back, 0.5, atol=1e-6)


@pytest.mark.parametrize("height", [40, 41])
def test_one_shot_colour_raw_is_demosaiced_with_the_right_pattern(height):
    colour = np.array([0.6, 0.3, 0.1], np.float32)
    rgb = np.broadcast_to(colour, (height, 48, 3)).copy()
    up = mosaic_of(rgb)                             # upright RGGB mosaic (conftest)
    p = _out(f"cfa_{height}.fits")
    hdu = AF.PrimaryHDU(up[::-1].astype(np.float32))   # stored bottom-up: FITS row 0 = the picture's last row
    pat = "RGGB" if height % 2 else "GBRG"          # the pattern as FITS row 0 sees it
    hdu.header["BAYERPAT"] = pat
    hdu.writeto(p)
    back, _, info = FX.read(p)
    assert info["bayer"] is not None and back.shape == (height, 48, 3)
    assert np.allclose(back[4:-4, 4:-4].mean(axis=(0, 1)), colour, atol=0.01)


def test_fits_opens_as_a_dark_scene_linear_photo_and_previews_stretched():
    from studio import engine as E
    img_a, _ = _star_field()
    p = _out("mono_stars.fits")
    FX.write(p, img_a, {"model": "ZWO ASI2600MM"})
    img = E.open(p)
    assert img.raw.scene_linear and img.raw.grey and img.meta.model == "ZWO ASI2600MM"
    out = E.render(img, E.Edit())
    assert np.isfinite(out).all() and out.shape[:2] == img_a.shape
    pv = FX.preview(p)
    assert pv.dtype == np.uint8 and pv.shape == img_a.shape + (3,)
    from studio.engine.color import spaces as sp
    bg = float(np.median(pv[..., 1])) / 255.0
    assert abs(float(sp.srgb_decode(np.array([bg]))[0]) - 0.25) < 0.06   # the sky lands near the stretch target
    from studio.engine.io.thumbs import embedded_info
    assert embedded_info(p)["width"] == img_a.shape[1]


def test_export_as_fits_writes_linear_float_in_the_chosen_space():
    from studio.engine.color import spaces as sp
    from studio.engine.io.export import ExportSpec, export_name, write
    pp = np.full((10, 12, 3), 0.4, np.float32)
    pp[0, 0] = 1.7
    spec = ExportSpec(format="fits", space="srgb")
    assert export_name("IMG_0001.CR3", spec) == "IMG_0001_Studio.fits"
    p = write(_out("IMG_0001_Studio.fits"), pp, spec, {"model": "EOS R5"})
    back, cards, _ = FX.read(p)
    want = sp.apply_matrix(pp, sp.conversion_matrix(sp.PROPHOTO, sp.SRGB))
    assert np.allclose(back, want, atol=1e-5) and back[0, 0, 0] > 1.0
    assert "sRGB" in str(cards.get("COLORSPC"))


@pytest.mark.parametrize("method", ["own", "photutils"])
def test_star_finders_centre_to_a_tenth_of_a_pixel(method):
    if method == "photutils" and not astro.photutils_ok():
        pytest.skip("photutils not installed")
    img, truth = _star_field(seed=4)
    st = astro.stars(img, method=method)
    assert st.shape[0] >= 40
    from scipy.spatial import cKDTree
    d, _ = cKDTree(truth).query(st[:, :2])
    assert float(np.median(d)) < 0.1 and float(np.percentile(d, 90)) < 0.25, (method, np.median(d))


def test_star_quality_measures_width_and_trailing():
    sharp, _ = _star_field(sigma=1.2, seed=5)
    soft, _ = _star_field(sigma=2.2, seed=5)
    trailed, _ = _star_field(sigma=1.2, seed=5, trail=7.0)
    qs, qb, qt = (astro.star_quality(x) for x in (sharp, soft, trailed))
    assert abs(qs["fwhm"] - 2.3548 * 1.2) < 0.5 and qs["elongation"] < 1.25
    assert qb["fwhm"] > 1.5 * qs["fwhm"]
    assert qt["elongation"] > 1.6


def test_night_sky_leaves_out_a_trailed_frame(monkeypatch):
    from test_superres import install
    from studio.engine.stack import superres as SR
    h, w = 240, 320
    n = 8
    rgbs = []
    for k in range(n):
        img, _ = _star_field(h, w, n=90, sigma=1.1, seed=11, trail=8.0 if k == 5 else 0.0, noise=0.003)
        img = img + np.random.default_rng(300 + k).normal(0, 0.002, img.shape).astype(np.float32)
        rgbs.append(np.repeat(img[..., None], 3, 2).astype(np.float32))
    paths = install(monkeypatch, [mosaic_of(r) for r in rgbs], "sky_trail", rgbs)
    p = SR.SuperParams.for_mode("sky")
    an = SR.analyse(paths, p)
    try:
        r = an.reports[5]
        assert r.status == "dropped" and "trailed" in r.reason, (r.status, r.reason)
        assert sum(x.status == "dropped" for x in an.reports[:an.n]) == 1
    finally:
        an.close()


def test_cli_opens_fits_and_writes_fits(tmp_path, capsys):
    import json
    from studio import cli
    img_a, _ = _star_field(seed=7)
    src = str(tmp_path / "M42.fits")
    FX.write(src, np.repeat(img_a[..., None], 3, 2), {"model": "ZWO ASI533MC"})
    capsys.readouterr()
    code = cli.main(["edit", src, "--out", str(tmp_path / "o"), "--set", "exposure_ev=3", "--format", "fits",
                     "--json"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and res["ok"], res
    out = res["outputs"][0]["output"]
    assert out.endswith("M42_Studio.fits")
    back, cards, _ = FX.read(out)
    assert back.shape == img_a.shape + (3,) and float(np.median(back)) > float(np.median(img_a))
    assert FX.is_own(out)

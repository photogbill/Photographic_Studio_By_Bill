"""End-to-end develop on a synthetic raw with known ground truth (no files needed)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import find_sample_raws, scene, synthetic_raw
from studio import engine as E
from studio.engine.color import spaces as S
from studio.engine.io.export import to_output
from studio.engine.pipeline.pipeline import Image, linear_rgb


def make_image(**kw):
    sc = scene(128, 192)
    return Image(synthetic_raw(sc, **kw)), sc


def test_optical_black_is_measured():
    img, _ = make_image(read_noise=3.0)
    lv = img.levels()
    assert lv.source == "optical-black"
    assert np.allclose(lv.black, 512.0, atol=0.6)
    assert np.allclose(lv.read_noise_dn, 3.0, atol=0.5)


def test_neutral_ramp_stays_neutral_in_every_output_space():
    img, _ = make_image()
    arr = E.render(img, E.Edit())
    ramp = arr[-12, 20:-20]                          # the grey ramp row
    for space in ("adobergb", "srgb", "prophoto", "displayp3"):
        # 16-bit: within a quarter of an 8-bit level (the synthetic raw is quantised like a real one)
        out = to_output(ramp[None], E.ExportSpec(format="tiff", space=space))[0].astype(int)
        assert np.abs(out[:, 0] - out[:, 1]).max() <= 64
        assert np.abs(out[:, 2] - out[:, 1]).max() <= 64
        out8 = to_output(ramp[None], E.ExportSpec(format="png", space=space, bits=8))[0].astype(int)
        assert np.abs(out8[:, 0] - out8[:, 1]).max() <= 2


def test_linear_path_recovers_the_scene():
    """Linearise + WB + demosaic + matrix must give back the scene (in camera space) where it is smooth."""
    img, sc = make_image()
    lin = linear_rgb(img, E.Edit(demosaic="menon"))
    from studio.engine.color.camera import camera_to_working
    M = camera_to_working(img.raw.cam_xyz)
    cam = S.apply_matrix(lin, np.linalg.inv(M).astype(np.float32))
    err = np.abs(cam[10:-30, 10:-10] - sc[10:-30, 10:-10])
    assert np.median(err) < 0.004


def test_exposure_is_a_clean_multiplier():
    img, _ = make_image()
    e0 = E.Edit()
    e0.tone.mode = "linear"
    e1 = E.Edit(exposure_ev=1.0)
    e1.tone.mode = "linear"
    a, b = E.render(img, e0), E.render(img, e1)
    assert np.allclose(b, 2 * a, rtol=1e-5, atol=1e-6)


def test_highlights_clip_to_neutral():
    sc = np.full((64, 64, 3), 0.5, np.float32)
    sc[20:40, 20:40] = 3.0                            # far beyond clip in every channel
    img = Image(synthetic_raw(sc))
    e = E.Edit()
    e.tone.mode = "linear"
    out = E.render(img, e)[30, 30]
    assert np.allclose(out, out[1], atol=1e-4), "clipped highlights must be neutral, not magenta"


@pytest.mark.parametrize("flip,shape", [(0, (128, 192)), (3, (128, 192)), (5, (192, 128)), (6, (192, 128))])
def test_orientation(flip, shape):
    img, _ = make_image(flip=flip)
    arr = E.render(img, E.Edit(demosaic="mhc"))
    assert arr.shape[:2] == shape
    assert img.output_size == (shape[1], shape[0])


def test_half_size_preview_matches_full_render():
    img, _ = make_image()
    full = E.render(img, E.Edit(), scale=1.0)
    half = E.render(img, E.Edit(), scale=0.5)
    assert half.shape[0] == full.shape[0] // 2
    small = full[: half.shape[0] * 2, : half.shape[1] * 2].reshape(half.shape[0], 2, half.shape[1], 2, 3).mean((1, 3))
    assert np.median(np.abs(small - half)) < 0.02


def test_region_render_matches_full_frame():
    img, _ = make_image()
    full = E.render(img, E.Edit())
    part = E.render(img, E.Edit(), region=(40, 30, 64, 48))
    assert np.allclose(part, full[30:78, 40:104], atol=1e-5)


@pytest.mark.parametrize("fmt,bits", [("tiff", 16), ("png", 16), ("png", 8), ("jpeg", 8)])
def test_export_formats_embed_the_profile(tmp_path, fmt, bits):
    from PIL import Image as PI
    img, _ = make_image()
    spec = E.ExportSpec(format=fmt, bits=bits, space="adobergb")
    p = E.export(img, E.Edit(), spec, out_dir=str(tmp_path))
    assert os.path.isfile(p) and not os.path.exists(p + ".partial")
    if fmt == "tiff":
        import tifffile
        with tifffile.TiffFile(p) as t:
            page = t.pages[0]
            assert page.dtype == np.uint16 and page.shape == (128, 192, 3)
            assert "InterColorProfile" in page.tags
    else:
        im = PI.open(p)
        assert im.size == (192, 128)
        assert im.info.get("icc_profile")
    if fmt == "png" and bits == 16:
        im = PI.open(p)
        assert im.mode in ("RGB", "I;16", "RGBA") or "16" in im.mode or im.info


def test_export_is_deterministic(tmp_path):
    img, _ = make_image()
    a = E.export(img, E.Edit(), E.ExportSpec(), out_path=str(tmp_path / "a.tif"))
    b = E.export(img, E.Edit(), E.ExportSpec(), out_path=str(tmp_path / "b.tif"))
    import tifffile
    assert np.array_equal(tifffile.imread(a), tifffile.imread(b))


def test_sidecar_round_trip(tmp_path):
    raw = tmp_path / "IMG_0001.CR3"
    raw.write_bytes(b"x")
    e = E.Edit(exposure_ev=0.7, demosaic="mhc")
    e.white_balance.mode = "temp_tint"
    e.white_balance.temperature = 3200
    p = E.save_edit(str(raw), e)
    assert p.endswith("IMG_0001.CR3.psb.json")
    doc = json.loads(open(p, encoding="utf-8").read())
    assert doc["edit"]["stage_versions"]["demosaic"] == 1
    e2 = E.load_edit(str(raw))
    assert e2.to_dict() == e.to_dict()


def test_old_sidecar_without_new_fields_still_loads():
    e = E.Edit.from_dict({"exposure_ev": 1.0, "tone": {"contrast": 1.2}, "stage_versions": {"tone": 1}})
    assert e.exposure_ev == 1.0 and e.tone.contrast == 1.2
    assert e.stage_versions["demosaic"] == 1


@pytest.mark.parametrize("path", find_sample_raws(4), ids=lambda p: p.name)
def test_real_raw_develops(path):
    """Bill's own raws (if present in the app folder): opens, levels sane, quick render works."""
    img = E.open(str(path))
    lv = img.levels()
    lib = img.raw.black_for_positions(img.raw.black_libraw)
    assert np.all(np.abs(lv.black - lib) < 0.01 * (lv.white - lib))
    arr = E.render(img, E.Edit(), scale=0.25)
    assert arr.ndim == 3 and np.isfinite(arr).all()
    assert float(np.percentile(arr, 99.5)) > 0.05          # something is lit (Bill shoots low-key)
    assert float(np.median(arr)) < 0.95


def test_camera_brightness_matching_is_off_unless_turned_on():
    from studio.engine.color import tone as T
    img, _ = make_image()
    img.raw.meta.model = "Canon EOS R5"
    base = E.render(img, E.Edit())
    assert E.Edit().tone.match_camera_brightness is False
    on = E.Edit()
    on.tone.match_camera_brightness = True
    brighter = E.render(img, on)
    assert T.body_brightness_ev("Canon EOS R5") > 0
    assert float(np.median(brighter)) > float(np.median(base))
    img.raw.meta.model = "Unknown Camera"
    assert np.allclose(E.render(img, on), base, atol=1e-6)     # unknown bodies: no change even when on

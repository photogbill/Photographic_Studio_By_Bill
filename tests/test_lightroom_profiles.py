"""Lightroom's own camera profiles found on this PC (dcp.lightroom_*) and Tone > "Lightroom-like" (pipeline.
lightroom_tone), on synthetic profiles in a made-up Adobe folder."""
from __future__ import annotations

import os

import numpy as np
import pytest

from studio.engine.color import dcp

CAM = ("SONY", "ILCE-7M4")


def _profile(path, camera, name, tone=None, beo=None):
    cm = np.array([[0.7, -0.2, -0.05], [-0.4, 1.2, 0.2], [-0.05, 0.15, 0.6]])
    pr = dcp.Profile(name=name, camera=camera, copyright="test", illuminants=[17, 21], color_matrices=[cm, cm])
    if tone is not None:
        pr.tone = np.asarray(tone, float)
    pr.baseline_offset = beo
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dcp.write(pr, path)


@pytest.fixture
def lr_folder(tmp_path, monkeypatch):
    root = tmp_path / "CameraProfiles"
    _profile(str(root / "Adobe Standard" / "Sony ILCE-7M4 Adobe Standard.dcp"), "Sony ILCE-7M4", "Adobe Standard")
    curve = [[0, 0], [0.1, 0.2], [0.5, 0.8], [1, 1]]
    _profile(str(root / "Camera" / "Sony ILCE-7M4" / "Sony ILCE-7M4 Camera ST.dcp"), "Sony ILCE-7M4", "Camera ST",
             curve, -0.35)
    _profile(str(root / "Camera" / "Sony ILCE-7M4" / "Sony ILCE-7M4 Camera ST HDR.dcp"), "Sony ILCE-7M4",
             "Camera ST HDR", curve, -1.2)
    _profile(str(root / "Adobe Standard" / "Canon EOS R5 Adobe Standard.dcp"), "Canon EOS R5", "Adobe Standard")
    monkeypatch.setenv("STUDIO_LIGHTROOM_PROFILES", str(root))
    dcp._LR_INDEX.clear()
    try:
        dcp._index_file().unlink()
    except OSError:
        pass
    return root


def test_lightroom_profiles_are_found_for_the_photo_camera_only(lr_folder):
    got = dcp.lightroom_profiles(*CAM)
    assert [g["file"] for g in got] == ["lr:Adobe Standard", "lr:Camera ST", "lr:Camera ST HDR"]
    assert all(g["source"] == "lightroom" for g in got)
    assert [g["file"] for g in dcp.lightroom_profiles("Canon", "Canon EOS R5")] == ["lr:Adobe Standard"]
    assert dcp.lightroom_profiles("NIKON CORPORATION", "NIKON Z 8") == []
    assert "nikonz8" in dcp.camera_keys("NIKON CORPORATION", "NIKON Z 8")


def test_the_hidden_brightness_lift_is_read_from_the_camera_profiles(lr_folder):
    assert dcp.lightroom_baseline(*CAM) == pytest.approx(0.35)          # the HDR variant (-1.2) is left out
    assert dcp.lightroom_baseline("Canon", "Canon EOS R5") is None       # no camera-matching profiles: unknown


def test_an_lr_name_loads_with_its_curve_and_offset_and_is_missing_without_lightroom(lr_folder, monkeypatch):
    pr = dcp.load("lr:Camera ST", CAM)
    assert pr.tone.shape == (4, 2) and pr.baseline_offset == pytest.approx(-0.35)
    assert dcp.load("lr:Adobe Standard", CAM).tone is None
    with pytest.raises(FileNotFoundError):
        dcp.load("lr:Camera ST", ("Canon", "Canon EOS R5"))
    monkeypatch.setenv("STUDIO_LIGHTROOM_PROFILES", "")
    with pytest.raises(FileNotFoundError):
        dcp.load("lr:Camera ST", CAM)


def test_the_index_is_kept_and_reused_until_adobe_adds_profiles(lr_folder, monkeypatch):
    dcp.lightroom_profiles(*CAM)
    assert dcp._index_file().exists()
    dcp._LR_INDEX.clear()
    calls = []
    real = dcp._header
    monkeypatch.setattr(dcp, "_header", lambda p: calls.append(p) or real(p))
    dcp.lightroom_profiles(*CAM)
    assert calls == []                                                  # read back from the saved index
    newdir = lr_folder / "Camera" / "Sony ILCE-7RM5"
    _profile(str(newdir / "Sony ILCE-7RM5 Camera ST.dcp"), "Sony ILCE-7RM5", "Camera ST", None, -0.35)
    dcp._LR_INDEX.clear()
    assert dcp.lightroom_profiles("SONY", "ILCE-7RM5")                 # a new folder: scanned again
    assert calls


def test_adobe_default_curve_is_the_published_one():
    from studio.engine.color.acr_curve import ACR_DEFAULT, acr_curve
    assert ACR_DEFAULT.size == 1025 and np.all(np.diff(ACR_DEFAULT) >= 0)
    assert acr_curve(0.0) == 0.0 and acr_curve(1.0) == pytest.approx(1.0)
    assert float(acr_curve(0.18)) == pytest.approx(0.3881, abs=1e-3)


def _image(make, model):
    from conftest import synthetic_raw
    from studio.engine.pipeline.pipeline import Image
    y, x = np.mgrid[0:64, 0:96] / 96.0
    scene = np.stack([0.1 + 0.3 * x, 0.12 + 0.2 * y, 0.08 + 0.1 * x * y], -1)
    raw = synthetic_raw(np.clip(scene, 0, 1))
    raw.meta.make, raw.meta.model = make, model
    return Image(raw)


def test_lightroom_like_tone_lifts_by_the_camera_and_uses_the_profile_curve(lr_folder):
    from studio.engine.pipeline import pipeline as pl
    from studio.engine.pipeline.edit import Edit
    from studio.engine.color.acr_curve import acr_curve
    im = _image(*CAM)
    e = Edit()
    e.tone.mode = "lightroom"
    info = pl.lightroom_info(im, e)
    assert info["lift_known"] and info["lift"] == pytest.approx(0.35) and info["curve"] == "adobe default"
    base, mid = pl.lightroom_tone(im, e)
    assert float(base(np.array([0.1]))[0]) == pytest.approx(float(acr_curve(0.1 * 2 ** 0.35)), abs=1e-4)
    assert float(base(np.array([mid]))[0]) == pytest.approx(0.18, abs=2e-3)
    e.profile = "lr:Camera ST"                    # its own curve, and its offset cancels the lift (as in Lightroom)
    base2, _ = pl.lightroom_tone(im, e)
    assert float(base2(np.array([0.1]))[0]) == pytest.approx(0.2, abs=1e-4)
    out = pl.render(im, e, scale=0.5)
    assert np.isfinite(out).all() and out.max() <= 1.0 + 1e-5
    other = _image("Canon", "Canon EOS R5")
    assert not pl.lightroom_info(other, Edit())["lift_known"]


def test_the_panel_offers_lightroom_profiles_and_the_lightroom_like_tone(lr_folder):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    page.image = _image(*CAM)
    page._fill_profiles("")
    data = [page.prof_combo.itemData(i) for i in range(page.prof_combo.count())]
    assert "lr:Adobe Standard" in data and "lr:Camera ST" in data
    e = E.Edit()
    e.profile = "lr:Camera ST"
    e.tone.mode = "lightroom"
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.profile == "lr:Camera ST" and back.tone.mode == "lightroom"
    page.image = _image("Canon", "Canon EOS R5")
    page._fill_profiles("lr:Camera ST")                      # not on this PC for this camera: said, and kept
    i = page.prof_combo.findData("lr:Camera ST")
    assert i >= 0 and "not on this PC" in page.prof_combo.itemText(i)
    app.processEvents()


def test_extended_low_iso_is_pulled_as_the_camera_does():
    from studio.engine.color.tone import extended_low_iso_ev as pull
    assert pull("SONY", "ILCE-9M2", 50) == -1.0 and pull("SONY", "ILCE-7RM5", 64) == -1.0
    assert pull("SONY", "ILCE-9M2", 100) == 0.0 and pull("SONY", "ILCE-7SM3", 80) == 0.0
    assert pull("NIKON CORPORATION", "NIKON Z 8", 32) == pytest.approx(-1.0)
    assert pull("NIKON CORPORATION", "NIKON Z 8", 50) == pytest.approx(-0.356, abs=0.01)
    assert pull("Canon", "Canon EOS R5", 50) == pytest.approx(-1.0) and pull("Canon", "Canon EOS R5", 3200) == 0.0
    assert pull("Canon", "Canon EOS R5", None) == 0.0


def test_lightroom_like_darkens_a_sony_iso_50_photo_by_a_stop(lr_folder):
    from studio.engine.pipeline import pipeline as pl
    from studio.engine.pipeline.edit import Edit
    im = _image(*CAM)
    e = Edit()
    e.tone.mode = "lightroom"
    im.meta.iso = 100.0
    assert pl.lightroom_info(im, e)["lift"] == pytest.approx(0.35)
    im.meta.iso = 50.0
    info = pl.lightroom_info(im, e)
    assert info["iso_pull"] == -1.0 and info["lift"] == pytest.approx(-0.65)


def test_studio_standard_darkens_extended_low_iso_for_new_edits_only():
    from studio.engine.pipeline import pipeline as pl
    from studio.engine.pipeline.edit import Edit
    im = _image(*CAM)
    im.meta.iso = 50.0
    new, old = Edit(), Edit.from_dict({"stage_versions": {"tone": 1}})
    assert new.stage_versions["tone"] == 2 and old.stage_versions["tone"] == 1
    assert pl._mid_in(im, new) == pytest.approx(2.0 * pl._mid_in(im, old))      # one stop darker
    im.meta.iso = 100.0
    assert pl._mid_in(im, new) == pytest.approx(pl._mid_in(im, old))
    im.meta.iso = 50.0
    a = pl.render(im, new, scale=0.5).mean()
    b = pl.render(im, old, scale=0.5).mean()
    assert a < 0.8 * b

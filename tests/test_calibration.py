"""Phase 2: the calibration builder recovers known sensor numbers from synthetic checklist frames."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_raw
from studio.engine.calib import profile as P
from studio.engine.calib.builder import build_profile, classify

H, W = 400, 600


def frame(level_dn=0.0, gain=2.0, rn=3.0, black=512.0, iso=800.0, exposure=1 / 100, row_band=0.0, hot=(),
          seed=0, temp=30.0, vignette=False):
    """A synthetic raw frame: Poisson photons at 'gain' e-/DN, Gaussian read noise, optional row banding."""
    rng = np.random.default_rng(seed)
    img = synthetic_raw(np.zeros((H, W, 3), np.float32), black=black, white=15000.0, read_noise=0.0)
    full = np.full(img.mosaic.shape, black, np.float64)
    sig = np.full((H, W), level_dn)
    if vignette:
        y, x = np.mgrid[0:H, 0:W]
        sig = sig * (1 - 0.3 * (((x - W / 2) / W) ** 2 + ((y - H / 2) / H) ** 2))
    vis = sig + (rng.poisson(sig * gain) / gain - sig if level_dn > 0 else 0)
    vis = vis + rng.normal(0, rn, (H, W)) + rng.normal(0, row_band, (H, 1))
    full[img.top:, img.left:] += vis
    full[img.top:, :img.left] += rng.normal(0, rn, (H, img.left)) + 0.0
    for (y, x) in hot:
        full[img.top + y, img.left + x] += 3000
    img.mosaic = np.clip(np.rint(full), 0, 65535).astype(np.uint16)
    img.meta.make, img.meta.model, img.meta.iso = "Synthetic", "Test Camera", iso
    img.meta.exposure_s, img.meta.camera_temperature_c = exposure, temp
    img.path = f"synthetic_{iso:g}_{level_dn:g}_{seed}.cr2"
    return img


def test_classifier_sorts_darks_flats_and_rejects():
    assert classify(frame()).kind == "dark"
    assert classify(frame(level_dn=6000)).kind == "flat"
    assert classify(frame(level_dn=14400)).kind == "other"          # clipped
    busy = frame(level_dn=3000)
    v = busy.mosaic[busy.top:, busy.left:].astype(np.int32)
    v[:, : W // 2] += 4000                                         # not even light
    busy.mosaic[busy.top:, busy.left:] = np.clip(v, 0, 65535)
    assert classify(busy).kind == "other"


def test_full_profile_recovers_read_noise_gain_banding_and_hot_pixels():
    hot = [(50, 60), (200, 301), (333, 77)]
    frames = [frame(rn=3.0, row_band=1.5, hot=hot, seed=s) for s in range(2)]
    frames += [frame(level_dn=5000, gain=2.0, seed=10 + s, vignette=True) for s in range(2)]
    frames += [frame(level_dn=0, exposure=30.0, seed=20, temp=41.0)]
    prof, rep = build_profile(frames, save_local_data=False)
    assert prof.method == "full"
    e = prof.entry_for(800)
    assert np.allclose(e.read_noise_dn, 3.0, atol=0.25)
    assert np.allclose(e.black_dn, 512.0, atol=0.3)
    assert e.gain_e_per_dn == pytest.approx(2.0, rel=0.08)
    assert e.row_banding_dn == pytest.approx(1.5, abs=0.4)
    assert e.col_banding_dn < 0.5
    assert e.hot_pixels >= 3
    assert any(d["exposure_s"] == 30.0 for d in e.dark_current)
    assert rep.counts().get("dark") == 3 and rep.counts().get("flat") == 2


def test_quick_profile_from_dark_frames_only():
    prof, _ = build_profile([frame(iso=3200, rn=8.0, seed=s) for s in range(2)], save_local_data=False)
    assert prof.method == "quick"
    e = prof.entry_for(3200)
    assert e.gain_e_per_dn is None and np.allclose(e.read_noise_dn, 8.0, atol=0.5)


def test_profile_lookup_noise_model_and_round_trip(tmp_path):
    frames = [frame(seed=s) for s in range(2)] + [frame(level_dn=5000, gain=2.0, seed=5 + s) for s in range(2)]
    prof, _ = build_profile(frames, save_local_data=False)
    path = prof.save(str(tmp_path / "p.json"))
    back = P.CameraProfile.load(path)
    assert back.to_dict()["isos"] == prof.to_dict()["isos"]
    assert back.entry_for(1000) is not None and back.entry_for(3200) is None      # within 1/3 stop only
    rng = np.full((2, 2), 15000.0 - 512.0)
    nm = back.noise_model(800, rng)
    assert nm.source == "calibrated"
    assert nm.a[0, 0] == pytest.approx(1 / (back.entry_for(800).gain_e_per_dn * rng[0, 0]))
    assert np.sqrt(nm.b[0, 0]) * rng[0, 0] == pytest.approx(back.entry_for(800).read_noise_dn[0][0])


def test_saved_profile_contains_no_private_data(tmp_path):
    prof, _ = build_profile([frame(hot=[(10, 10)], seed=s) for s in range(2)], save_local_data=False)
    text = open(prof.save(str(tmp_path / "q.json")), encoding="utf-8").read().lower()
    for word in ("serial", "gps", "[10, 10]"):
        assert word not in text


def test_pipeline_prefers_this_bodys_profile():
    import os
    from studio.engine.pipeline.pipeline import Image
    frames = [frame(iso=6400, seed=s) for s in range(2)] + \
             [frame(iso=6400, level_dn=5000, gain=0.5, seed=7 + s) for s in range(2)]
    prof, _ = build_profile(frames, save_local_data=False)
    path = prof.save()                                   # the (test) calibration_local folder
    try:
        P._CACHE.clear()
        img = Image(frame(iso=6400, level_dn=800, seed=99))
        nm = img.noise_model()
        assert nm.source == "calibrated"
        rng = img.levels().range
        assert nm.a[0, 0] == pytest.approx(1 / (0.5 * rng[0, 0]), rel=0.1)
    finally:
        os.remove(path)
        P._CACHE.clear()

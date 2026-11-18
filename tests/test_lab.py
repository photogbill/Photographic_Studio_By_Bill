"""The Lab (BUILD_PLAN_NOISE F): clean photos, training pairs with real noise, a tiny training run, the trained
network running in Clean. The training test needs PyTorch and ONNX Runtime and is skipped without them."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import scene, synthetic_raw
from studio import lab
from studio.engine.calib import builder as B
from studio.engine.calib import deep as D
from studio.engine.calib import profile as P
from studio.engine.calib import synth as S
from studio.lab import data
from test_calibration import frame


@pytest.fixture
def camera_and_photos(tmp_path, monkeypatch):
    """A calibrated synthetic camera (darks, flats, noise bank) and a folder of 'clean photos'."""
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    P._CACHE.clear(), D._CACHE.clear()
    made = {}

    def add(folder, name, kind, **kw):
        p = tmp_path / folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode() * 60)
        made[str(p)] = (kind, kw)
        return str(p)

    def fake_open(path):
        kind, kw = made[path]
        if kind == "frame":
            img = frame(**kw)
        else:                                                   # a clean, detailed photo at base ISO
            img = synthetic_raw(scene(320, 448, seed=kw["seed"]) * 0.8, black=512.0, white=15000.0)
            img.meta.iso, img.meta.exposure_s, img.meta.datetime = kw.get("iso", 100.0), 1 / 200, kw["when"]
            img.meta.make, img.meta.model = "Synthetic", "Test Camera"
        img.path = path
        return img

    def fake_meta(path, _=None):
        return fake_open(path).meta

    import studio.engine.io.metadata as M
    import studio.engine.io.rawfile as R
    monkeypatch.setattr(R, "open_raw", fake_open)
    monkeypatch.setattr(M, "read_metadata", fake_meta)
    monkeypatch.setattr(data, "CROP", 96)
    monkeypatch.setattr(data, "CROPS_PER_PHOTO", 4)
    cal = []
    for iso, rn, g in ((800, 4.0, 0.45), (6400, 30.0, 0.057)):
        cal += [add("cal", f"d{iso}_{s}.cr2", "frame", iso=iso, rn=rn, seed=iso + s) for s in range(4)]
        cal += [add("cal", f"f{iso}_{s}.cr2", "frame", iso=iso, rn=rn, level_dn=5000, gain=g, seed=7000 + iso + s)
                for s in range(2)]
    prof = B.calibrate_drop(cal, workers=1).built[0][0]
    D.deep_calibrate(cal, workers=1)
    for k in range(10):
        add("clean", f"c{k}.cr2", "photo", seed=k, when=f"2026:0{1 + k % 5}:1{k} 1{k}:00:00")
    add("clean", "high_iso.cr2", "photo", seed=99, iso=3200.0, when="2026:06:01 10:00:00")
    yield prof, str(tmp_path / "clean")
    P._CACHE.clear(), D._CACHE.clear()


def test_the_software_says_how_many_photos_and_why():
    assert lab.PHOTOS_WANTED == 200
    for word in ("200", "variety", "texture", "lowest ISO", "never copied"):
        assert word in lab.PHOTOS_WHY
    assert lab.photos_verdict(250)[0] == "good"
    tier, text = lab.photos_verdict(90)
    assert tier == "thin" and "200" in text
    assert lab.photos_verdict(5)[0] == "few"


def test_survey_counts_what_is_usable(camera_and_photos):
    _, folder = camera_and_photos
    sv = data.survey(folder)
    assert sv["raws"] == 11 and sv["usable"] == 10              # the ISO 3200 photo is not clean truth
    assert sv["cameras"] == {"Test Camera": 10} and sv["days"] >= 5 and sv["sessions"] == 10


def test_training_pairs_carry_unit_noise_and_an_unbiased_target(camera_and_photos):
    prof, folder = camera_and_photos
    src = S.NoiseSource(prof, D.find_bank("Synthetic", "Test Camera"))
    crops = [c for c in (data.prepare_photo(f) for f in data.survey(folder)["files"]) if c]
    assert len(crops) == 10 and data.prepare_photo(folder + "/high_iso.cr2") is None
    cs = data.CleanSet(crops)
    assert len(cs.hold) == 1 and len(cs.train) == 9 and not set(cs.hold) & set(cs.train)
    rng = np.random.default_rng(0)
    x, t = data.batch(cs, src, rng, n=24, size=64)
    assert x.shape == t.shape == (24, 4, 32, 32) and x.dtype == np.float32
    d = x - t
    assert float(d.std()) == pytest.approx(1.0, abs=0.12)       # stabilised: the noise has strength one
    assert abs(float(d.mean())) < 0.06                          # and the target is not biased
    # the target, taken back through Clean's own exact inverse, is the clean photo: no brightness shift is taught
    from studio.engine.calib.noise import inverse_anscombe_exact
    m = src.model(6400.0)
    for level in (0.002, 0.02, 0.2, 0.7):
        t1 = data.expected_vst(np.full((4, 4), level, np.float32), float(m.a[0, 0]), float(m.b[0, 0]))
        assert inverse_anscombe_exact(t1, float(m.a[0, 0]), float(m.b[0, 0])).mean() == pytest.approx(level, rel=0.01)


def test_train_a_tiny_network_and_run_it_in_clean(camera_and_photos):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from studio.engine.raw import denoise as DN
    from studio.engine.raw import learned as LN
    from studio.lab import train as T
    prof, folder = camera_and_photos
    torch.manual_seed(0)
    path = T.train(folder, size="fast", hours=1.0, steps=60, batch_size=6, patch=64, eval_every=30,
                   force_few=True, quiet=True)
    card = LN.card(path)
    assert card["camera"] == "Test Camera" and card["clean_photos"] == 10 and "too few" in card["clean_verdict"]
    assert card["noise_left"] < 1.0                              # even 60 steps remove some noise on held-out
    LN._NETS.clear()
    net = LN.for_camera(("Synthetic", "Test Camera"))
    assert net is not None and LN.status("Synthetic", "Test Camera")["runtime"]
    planes = np.random.default_rng(1).standard_normal((4, 101, 150)).astype(np.float32) + 30.0
    out = net.run(planes)
    assert out.shape == planes.shape and np.isfinite(out).all()
    big = np.random.default_rng(2).standard_normal((4, 500, 620)).astype(np.float32) + 30.0     # several tiles
    ob = net.run(big)
    assert ob.shape == big.shape and (ob - 30.0).std() < (big - 30.0).std()
    # in Clean: the learned method runs, then the guard
    src = S.NoiseSource(prof, D.find_bank("Synthetic", "Test Camera"))
    rng = np.random.default_rng(3)
    clean = S.mosaic_of(S.dead_leaves(128, 128, seed=2) * 0.5)
    noisy = src.add(clean, 6400.0, rng)
    st = {}
    den = DN.denoise_mosaic(noisy, S.RGGB, src.model(6400.0),
                            DN.DenoiseParams(method="learned", camera=("Synthetic", "Test Camera"), guard=True),
                            stats=st)
    assert st["learned"] == net.name and "guard" in st and den.shape == noisy.shape and np.isfinite(den).all()
    # sixty steps do not make a denoiser: this one is poor. The guard must contain it, not trust it.
    raw_net = DN.denoise_mosaic(noisy, S.RGGB, src.model(6400.0),
                                DN.DenoiseParams(method="learned", camera=("Synthetic", "Test Camera")))
    assert np.mean((den - clean) ** 2) < np.mean((raw_net - clean) ** 2)
    st2 = {}
    DN.denoise_mosaic(noisy, S.RGGB, src.model(6400.0),
                      DN.DenoiseParams(method="learned", camera=("Synthetic", "No Such Camera")), stats=st2)
    assert "Hybrid used" in st2["learned"]                       # no model: says so, uses Hybrid


def test_training_refuses_too_few_photos_and_says_why(camera_and_photos, capsys):
    pytest.importorskip("torch")
    from studio.lab import train as T
    _, folder = camera_and_photos
    with pytest.raises(SystemExit) as e:
        T.train(folder, size="fast", steps=5, quiet=True)
    assert "add clean photos" in str(e.value)
    said = capsys.readouterr().out
    assert "Why around 200 varied photos" in said and "variety helps much more than number" in said


def test_cameras_page_teaches_clean(camera_and_photos):
    """The card says what stands between the camera and its network, how many clean photos and why, builds the
    noise bank, counts the chosen photos, and will not start training on too few."""
    import os
    import time
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import cameras as C
    prof, folder = camera_and_photos
    prof.save()
    app = QApplication.instance() or QApplication([])
    page = C.CamerasPage()
    page.settings.d.pop("lab_clean_folder", None)
    page.clean_info = None
    page.refresh()
    card = page.cards[0]
    assert "about 200, and varied" in card.why.text() and "variety helps much more than number" in card.why.text()
    said = [t for _, t in card.teach]
    assert "Full profile" in said[0] and "Noise bank" in said[1] and "none chosen yet" in said[2]
    assert page.train_model(prof, launch=lambda *a: 1 / 0) is False          # nothing chosen: not started
    assert "Choose the folder of clean photos first" in page.out.text() and "200" in page.out.text()

    def wait(cond):
        t0 = time.time()
        while not cond() and time.time() - t0 < 60:
            app.processEvents()
            time.sleep(0.02)
        for _ in range(10):
            app.processEvents()
        assert cond()
    page.survey_clean(folder)
    wait(lambda: page.clean_info is not None)
    assert page.clean_info["usable"] == 10 and "files" not in page.clean_info
    assert any("too few" in t for _, t in page.cards[0].teach)
    started = []
    assert page.train_model(prof, launch=lambda bat, args: started.append((bat, args))) is False
    assert not started and "too few" in page.out.text() and "variety" in page.out.text()
    page.clean_info["usable"] = 90                                           # thin, but enough to start
    assert page.train_model(prof, launch=lambda bat, args: started.append((bat, args))) is True
    assert started[0][0].endswith("Train Camera Model.bat") and started[0][1] == [folder, "--camera", "Test Camera"]
    assert "around 200 varied ones is the aim" in page.out.text()
    D._CACHE.clear()
    page.deep_calibration(prof, folder=os.path.join(os.path.dirname(folder), "cal"))
    wait(lambda: page.deep_last is not None and not page.busy)
    assert page.deep_last[0]["patches"] == 32 and "noise bank built" in page.out.text()
    page.grab()

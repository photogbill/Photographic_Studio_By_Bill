"""The drop-everything calibration (Bill, 2026-11-18): a big pile of lens-cap frames at different ISOs and camera
temperatures, measured one frame at a time, several cameras mixed, added to on another day."""
from __future__ import annotations

import math

import numpy as np
import pytest

from studio.engine.calib import builder as B
from studio.engine.calib import frames as F
from studio.engine.calib import profile as P
from test_calibration import frame


@pytest.fixture
def pile(tmp_path, monkeypatch):
    """Files on disk whose 'raw' content is a synthetic frame: open_raw is replaced, everything else is real."""
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    P._CACHE.clear()
    made, opened = {}, []

    def add(name, model="Test Camera", **kw):
        p = tmp_path / "drop" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode() * 50)                       # distinct content = distinct frame
        made[str(p)] = (model, kw)
        return str(p)

    def fake_open(path):
        opened.append(path)
        model, kw = made[path]
        img = frame(**kw)
        img.meta.model, img.path = model, path
        return img

    import studio.engine.io.rawfile as R
    monkeypatch.setattr(R, "open_raw", fake_open)
    yield add, opened, tmp_path
    P._CACHE.clear()


def test_read_noise_is_kept_apart_by_camera_temperature(pile):
    add, opened, _ = pile
    files = []
    for iso, rn_cold, rn_hot in ((400, 3.0, 4.0), (6400, 8.0, 11.0)):
        files += [add(f"c_{iso}_{s}.cr2", iso=iso, rn=rn_cold, temp=24.0, seed=s) for s in range(3)]
        files += [add(f"h_{iso}_{s}.cr2", iso=iso, rn=rn_hot, temp=40.0, seed=10 + s) for s in range(3)]
    res = B.calibrate_drop(files, workers=2)
    assert res.counts() == {"dark": 12} and len(res.built) == 1
    prof = res.built[0][0]
    e = prof.entry_for(400)
    by = {t["temp_c"]: np.mean(t["read_noise_dn"]) for t in e.by_temp}
    assert by[24.0] == pytest.approx(3.0, abs=0.2) and by[40.0] == pytest.approx(4.0, abs=0.25)
    rng = np.full((2, 2), 14000.0)
    cold = np.sqrt(prof.noise_model(6400, rng, temp_c=25.0).b[0, 0]) * 14000
    hot = np.sqrt(prof.noise_model(6400, rng, temp_c=41.0).b[0, 0]) * 14000
    assert cold == pytest.approx(8.0, abs=0.5) and hot == pytest.approx(11.0, abs=0.7)
    q = prof.quality
    assert q["temp_min_c"] == 24.0 and q["temp_max_c"] == 40.0
    assert q["coverage"]["cells"]["400"] == {"24": 3, "40": 3}
    assert not any("temperature" in a or "°C" in a for a in q["advice"])       # 16 degrees apart: enough


def test_long_exposures_give_the_heat_noise_and_its_doubling(pile):
    add, _, _ = pile
    files = [add(f"s_{s}.cr2", iso=800, rn=3.0, temp=24.0, seed=s) for s in range(2)]
    k24, dbl = 0.5, 6.0                                       # DN^2 per second at 24 C, doubling every 6 C
    for temp in (24.0, 32.0, 40.0):
        k = k24 * 2 ** ((temp - 24.0) / dbl)
        files += [add(f"l_{temp:g}_{s}.cr2", iso=800, rn=math.sqrt(9.0 + k * 30.0), exposure=30.0, temp=temp,
                      seed=100 + int(temp) + s) for s in range(2)]
    prof = B.calibrate_drop(files, workers=1).built[0][0]
    e = prof.entry_for(800)
    assert np.allclose(e.read_noise_dn, 3.0, atol=0.25)       # the long frames did not leak into read noise
    th = e.thermal
    assert th["doubling_c"] == pytest.approx(dbl, rel=0.2)
    assert th["var_dn2_per_s"] == pytest.approx(k24 * 2 ** ((th["at_temp_c"] - 24.0) / dbl), rel=0.2)
    rng = np.full((2, 2), 14000.0)
    var40 = prof.noise_model(800, rng, temp_c=40.0, exposure_s=30.0).b[0, 0] * 14000 ** 2
    assert var40 == pytest.approx(9.0 + k24 * 2 ** (16 / dbl) * 30.0, rel=0.25)
    assert prof.noise_model(800, rng, temp_c=40.0, exposure_s=1 / 100).b[0, 0] * 14000 ** 2 == pytest.approx(9, rel=.2)


def test_two_cameras_in_one_pile_and_what_is_left_out(pile):
    add, _, _ = pile
    files = [add(f"a{s}.cr2", model="Camera A", iso=100, seed=s) for s in range(2)]
    files += [add(f"b{s}.cr2", model="Camera B", iso=100, rn=5.0, seed=5 + s) for s in range(2)]
    files += [add("photo.cr2", model="Camera A", level_dn=14400)]                  # not a dark frame
    files += [add("flatonly.cr2", model="Camera C", level_dn=6000)]                # a camera with no darks
    res = B.calibrate_drop(files, workers=2)
    assert sorted(p.model for p, _, _ in res.built) == ["Camera A", "Camera B"]
    assert [f.name for f in res.left_out] == ["photo.cr2"] and res.left_out[0].reason
    b = next(p for p, _, _ in res.built if p.model == "Camera B")
    assert np.allclose(b.entry_for(100).read_noise_dn, 5.0, atol=0.4)
    assert P.find_profile("Synthetic", "Camera A") is not None                    # saved on this PC


def test_second_drop_adds_to_the_first_and_nothing_is_read_twice(pile):
    add, opened, tmp = pile
    first = [add(f"d{s}.cr2", iso=1600, seed=s) for s in range(2)]
    one = B.calibrate_drop(first, workers=1).built[0][0]
    assert [e.iso for e in one.isos] == [1600.0] and len(opened) == 2
    copy = tmp / "copy_of_d0.cr2"                              # the same file from another folder
    copy.write_bytes(open(first[0], "rb").read())
    second = [add(f"e{s}.cr2", iso=3200, seed=20 + s) for s in range(2)]
    res = B.calibrate_drop(first + [str(copy)] + second, workers=1)
    two = res.built[0][0]
    assert [e.iso for e in two.isos] == [1600.0, 3200.0]
    assert len(opened) == 4                                    # only the two new frames were opened
    assert two.quality["dark_frames"] == 4                     # the copy counted once
    assert sum(1 for f in res.scanned if f.cached) == 3
    assert F.forget_camera("Synthetic", "Test Camera") == 4 and F.load_store() == []


def test_cancel_keeps_what_was_measured(pile):
    add, opened, _ = pile
    files = [add(f"k{s}.cr2", iso=200, seed=s) for s in range(6)]
    seen = []
    res = B.calibrate_drop(files, progress=lambda n, t, st: seen.append(n), cancelled=lambda: len(seen) >= 2,
                           workers=1)
    assert res.cancelled and 2 <= len(res.scanned) < 6
    assert len(F.load_store()) == len(res.scanned)


def test_one_frame_holds_little(pile):
    """What is kept per frame is small: a pile of a thousand frames is tens of MB, not tens of GB."""
    add, _, tmp = pile
    st = F.scan([add("one.cr2", iso=100)], workers=1)[0]
    assert st.crop.shape == (256, 256) and st.crop.dtype == np.uint16
    import os
    assert os.path.getsize(os.path.join(F.store_dir(), st.key + ".npz")) < 200_000


def test_advice_says_what_is_missing(pile):
    add, _, _ = pile
    prof = B.calibrate_drop([add("only.cr2", iso=100, temp=30.0)], workers=1).built[0][0]
    text = " ".join(prof.quality["advice"])
    assert "Only one short dark frame at ISO 100" in text and "°C" in text and "long dark frames" in text
    assert "No flat frames" in text


def test_cameras_page_takes_a_dropped_folder(pile):
    """The page: drop a folder, every frame is measured in the background, a card per camera appears."""
    import os
    import time
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app.cameras import CamerasPage
    add, _, tmp = pile
    for iso in (100, 800, 6400):
        for temp, s in ((24.0, 0), (24.0, 1), (40.0, 2), (40.0, 3)):
            add(f"p_{iso}_{s}.cr2", iso=iso, temp=temp, seed=iso + s)
    add("holiday.cr2", level_dn=14400)
    app = QApplication.instance() or QApplication([])
    page = CamerasPage()
    assert page.cards == [] and "No camera profiles yet" in page.shared.text()
    said = []
    page.status.connect(said.append)
    page.add([str(tmp / "drop")])
    t0 = time.time()
    while (page.busy or page.last is None) and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
    assert not page.busy and page.last.counts() == {"dark": 12, "other": 1}
    assert len(page.cards) == 1 and page.cards[0].prof.model == "Test Camera"
    card = page.cards[0]
    assert card.grid.isos == ["100", "800", "6400"] and card.grid.temps == ["24", "40"]
    assert len(card.chart.lines) == 2                       # one line per camera temperature
    assert "12 dark frames" in page.out.text() and "holiday.cr2" in page.out.text()
    assert any("long dark frames" in a for a in card.advice)
    assert said and "Test Camera" in said[-1]
    page.grab()                                             # paints the grid and the chart without error
    page.start_over("Synthetic", "Test Camera", ask=False)
    assert page.cards == [] and F.load_store() == []


def test_two_frames_either_side_of_a_temperature_step_are_still_a_pair(pile):
    """Bill's first real pile: two frames per ISO at 26 and 27 C. They are a pair."""
    add, _, _ = pile
    files = [add("t26.cr2", iso=400, rn=3.0, temp=26.0, seed=1), add("t27.cr2", iso=400, rn=3.0, temp=27.0, seed=2)]
    assert B.temp_bin(26.0) != B.temp_bin(27.0)
    e = B.calibrate_drop(files, workers=1).built[0][0].entry_for(400)
    assert e.read_noise_paired and np.allclose(e.read_noise_dn, 3.0, atol=0.25)
    far = [add("c.cr2", iso=800, temp=20.0, seed=3), add("h.cr2", iso=800, temp=40.0, seed=4)]
    assert not B.calibrate_drop(far, workers=1).built[0][0].entry_for(800).read_noise_paired

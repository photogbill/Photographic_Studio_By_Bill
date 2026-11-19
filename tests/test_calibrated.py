"""BUILD_PLAN_NOISE block C: what the camera's calibration takes out of a photo before Clean (raw/calibrated.py),
and the numbers the builder and deep calibration now keep for it."""
from __future__ import annotations

import numpy as np
import pytest

from studio.engine.calib import builder as B
from studio.engine.calib import deep as D
from studio.engine.calib import profile as P
from studio.engine.raw import calibrated as CAL
from test_calibration import H, W, frame

HOT = [(50, 60), (200, 301), (333, 77)]
WARM_DARK = [(120, 130), (310, 90), (20, 250)]          # in the dark half of the test photo
WARM_BRIGHT = [(150, 380), (260, 450)]                  # in its bright half: hidden under photon noise
RN, GAIN, ISO = 4.0, 0.45, 1600.0


@pytest.fixture
def camera(tmp_path, monkeypatch):
    """A synthetic camera with faults that are the same in every frame: a row pattern, a column pattern, hot and
    warm photosites, and a picture-area black that sits 0.5 DN above the masked border's."""
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    P._CACHE.clear(), D._CACHE.clear(), CAL._CACHE.clear()
    g = np.random.default_rng(42)
    rows = g.normal(0, 2.0, H)
    cols = g.normal(0, 1.5, W)
    made = {}

    def faults(img, seed):
        m = img.mosaic.astype(np.float64)
        m[img.top:, :] += rows[:, None]                       # a row's offset is shared by the masked border
        m[img.top:, img.left:] += cols[None, :] + 0.5
        for (y, x) in HOT:
            m[img.top + y, img.left + x] += 3000
        for (y, x) in WARM_DARK + WARM_BRIGHT:
            m[img.top + y, img.left + x] += 60
        img.mosaic = np.clip(np.rint(m), 0, 65535).astype(np.uint16)
        return img

    def add(name, **kw):
        p = tmp_path / "cal" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode() * 50)
        made[str(p)] = kw
        return str(p)

    def fake_open(path):
        kw = made[path]
        img = faults(frame(**kw), kw.get("seed", 0))
        img.path = path
        return img

    import studio.engine.io.rawfile as R
    monkeypatch.setattr(R, "open_raw", fake_open)
    files = [add(f"d{s}.cr2", iso=ISO, rn=RN, row_band=1.0, seed=s) for s in range(8)]
    files += [add(f"f{s}.cr2", iso=ISO, rn=RN, level_dn=5000, gain=GAIN, seed=50 + s) for s in range(2)]
    prof = B.calibrate_drop(files, workers=1).built[0][0]
    rep, = D.deep_calibrate(files, workers=1)

    def photo(seed=7, row_band=1.0):
        """Left half deep shadow, right half bright, one real 'star' where no defect is mapped."""
        rng = np.random.default_rng(seed)
        img = frame(iso=ISO, rn=0.0, seed=seed)
        sig = np.full((H, W), 12.0)
        sig[:, W // 2:] = 8000.0
        vis = rng.poisson(sig * GAIN) / GAIN + rng.normal(0, RN, (H, W)) + rng.normal(0, row_band, (H, 1))
        full = img.mosaic.astype(np.float64)
        full[img.top:, img.left:] += vis
        full[img.top:, :img.left] += rng.normal(0, RN, (H, img.left))
        full[img.top + 80, img.left + 100] += 900             # the star
        img.mosaic = np.clip(np.rint(full), 0, 65535).astype(np.uint16)
        return faults(img, seed)
    yield prof, rep, photo, rows, cols
    P._CACHE.clear(), D._CACHE.clear(), CAL._CACHE.clear()


def test_builder_keeps_what_clean_needs(camera):
    prof, rep, _, rows, cols = camera
    e = prof.entry_for(ISO)
    assert np.mean(e.black_offset_dn) == pytest.approx(0.5, abs=0.2)            # picture black above the border's
    assert e.hot_pixels == 3 and e.warm_pixels == 5
    assert e.neighbour_corr is not None and abs(e.neighbour_corr) < 0.05 and not e.pushed
    fix = CAL.for_camera("Synthetic", "Test Camera")
    idx, dn = fix.defect_list(ISO, 1 / 100, 30.0)
    assert idx.size == 8 and np.sum(dn > 2500) == 3 and np.all((dn[dn < 2500] > 40) & (dn[dn < 2500] < 80))
    fx = rep["fixed"][f"{ISO:g}"]
    assert fx["fixed_rows_dn"] == pytest.approx(2.0, abs=0.35) and fx["rows_weight"] > 0.85
    assert fx["fixed_cols_dn"] == pytest.approx(1.5, abs=0.25) and fx["periodic"] == []
    r, c = D.find_bank("Synthetic", "Test Camera").fixed_lines(ISO)
    assert np.corrcoef(r, rows)[0, 1] > 0.95 and np.corrcoef(c, cols)[0, 1] > 0.97


def test_pushed_iso_and_comb_are_found(camera, tmp_path, monkeypatch):
    """A setting whose values are stretched in the camera uses one raw value in two, and its gain x ISO is off."""
    import studio.engine.io.rawfile as R
    real = R.open_raw
    made = {}

    def add(name, **kw):
        p = tmp_path / "push" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode() * 51)
        made[str(p)] = kw
        return str(p)

    def fake(path):
        if path not in made:
            return real(path)
        kw = dict(made[path])
        stretch = kw.pop("stretch", 1)
        img = frame(**kw)
        img.meta.model = "Push Camera"
        if stretch > 1:
            v = img.mosaic.astype(np.int64)
            img.mosaic = np.clip(512 + (v - 512) * stretch, 0, 65535).astype(np.uint16)
        img.path = path
        return img
    monkeypatch.setattr(R, "open_raw", fake)
    files = []
    for iso, rn in ((400, 3.0), (800, 6.0), (1600, 12.0), (3200, 24.0)):
        files += [add(f"d{iso}_{s}.cr2", iso=iso, rn=rn, seed=iso + s) for s in range(2)]
        files += [add(f"f{iso}_{s}.cr2", iso=iso, rn=rn, level_dn=3000, gain=360.0 / iso, seed=9 * iso + s)
                  for s in range(2)]
    files += [add(f"d6400_{s}.cr2", iso=6400, rn=24.0, stretch=2, seed=70 + s) for s in range(2)]
    files += [add(f"f6400_{s}.cr2", iso=6400, rn=24.0, level_dn=3000, gain=360.0 / 3200, stretch=2, seed=90 + s)
              for s in range(2)]
    prof = next(p for p, _, _ in B.calibrate_drop(files, workers=1).built if p.model == "Push Camera")
    by = {e.iso: e for e in prof.isos}
    assert not any(by[i].pushed for i in (400.0, 800.0, 1600.0, 3200.0))
    assert by[6400.0].pushed and by[6400.0].digital_gain == pytest.approx(2.0, abs=0.25)


def _norm(img):
    from studio.engine.pipeline.pipeline import Image
    from studio.engine.raw import linearize as lin
    image = Image(img)
    return image, lin.normalise(img, image.levels())


def test_calibration_is_subtracted_from_a_photo(camera):
    _, _, photo, rows, cols = camera
    image, mos = _norm(photo())
    span = float(np.mean(image.levels().range))
    before = mos.copy()
    rep = CAL.apply(mos, image, 0, 0, None)
    assert rep["used"] == ["black", "fixed pattern", "hot pixels"]
    # hot photosites: healed to their surroundings; warm ones healed where they show, left where they are hidden
    for (y, x) in HOT:
        assert before[y, x] * span > 2500 and abs(mos[y, x] * span - (12 if x < W // 2 else 8000)) < 600
    for (y, x) in WARM_DARK:
        assert before[y, x] * span > 40 and abs(mos[y, x] * span - 12) < 5 * RN
    for (y, x) in WARM_BRIGHT:
        assert mos[y, x] == pytest.approx(before[y, x] - cols[x] / span - rows[y] / span - 0.5 / span, abs=2e-4)
    assert rep["hot"] == 3 and rep["warm"] == 3 and rep["mapped"] == 8
    # the star is not on the map: nothing touches it but the subtractions
    assert mos[80, 100] * span > 800
    # the fixed banding is gone from the dark half; the black is on zero
    dark = mos[:, : W // 2] * span - 12.0
    b_dark = before[:, : W // 2] * span - 12.0
    assert abs(np.corrcoef(dark.mean(1), rows)[0, 1]) < 0.35 < np.corrcoef(b_dark.mean(1), rows)[0, 1]
    assert abs(np.corrcoef(dark.mean(0), cols[: W // 2])[0, 1]) < 0.3 < np.corrcoef(b_dark.mean(0), cols[: W // 2])[0, 1]
    level = lambda a: float(np.mean(np.clip(a, -60, 60)))               # noqa: E731  (defects left out)
    assert abs(level(dark)) < 0.2 and level(b_dark) > 0.3


def test_a_region_gets_what_the_whole_frame_gets(camera):
    _, _, photo, _, _ = camera
    image, mos = _norm(photo())
    whole = mos.copy()
    CAL.apply(whole, image, 0, 0, None)
    from studio.engine.raw import linearize as lin
    x0, y0, w, h = 40, 30, 300, 240
    part = lin.normalise(image.raw, image.levels(), (x0, y0, w, h))
    CAL.apply(part, image, x0, y0, None)
    assert np.allclose(part[2:-2, 2:-2], whole[y0 + 2:y0 + h - 2, x0 + 2:x0 + w - 2], atol=1e-6)


def test_in_the_pipeline_rows_are_not_taken_out_twice(camera):
    """With the per-photo row step on, the fixed rows are shared between it and the calibration: together they
    take the pattern out once."""
    from studio import engine as E
    from studio.engine.pipeline.pipeline import Image, linear_rgb
    _, _, photo, rows, _ = camera
    raw = photo(seed=11)
    out = {}
    for name, cal in (("off", False), ("on", True)):
        e = E.Edit()
        e.calibrated, e.banding = cal, True
        e.lens.auto_ca = e.lens.defringe = False
        image = Image(raw)
        rgb = linear_rgb(image, e, cache=False)
        g = rgb[..., 1][:, 8: W // 2 - 8]
        out[name] = (np.corrcoef(g.mean(1), rows[:g.shape[0]])[0, 1], image.calibration_report)
    assert abs(out["on"][0]) < abs(out["off"][0]) and abs(out["on"][0]) < 0.35
    assert out["on"][1]["used"] == ["black", "fixed pattern", "hot pixels"] and not out["off"][1]


def test_old_sidecars_and_cameras_without_a_profile(camera):
    from studio import engine as E
    assert E.Edit().calibrated and E.Edit.from_dict({}).calibrated
    old = E.Edit.from_dict({"exposure_ev": 0.5, "stage_versions": {"denoise": 3}})
    assert not old.calibrated, "a sidecar written before: renders exactly as it did"
    assert E.Edit.from_json(E.Edit().to_json()).calibrated
    off = E.Edit()
    off.calibrated = False
    assert off.raw_key() != E.Edit().raw_key()
    _, _, photo, _, _ = camera
    raw = photo()
    raw.meta.model = "Some Other Camera"
    image, mos = _norm(raw)
    keep = mos.copy()
    assert CAL.apply(mos, image, 0, 0, None) == {} and np.array_equal(mos, keep)


def test_auto_strength_needs_the_noise_measured_and_leans_with_the_slider(camera):
    """Clean's Auto chooses for a camera whose noise is measured, says plainly why not for one that is only
    estimated, and the Strength slider leans from the photo's own choice."""
    from studio import engine as E
    from studio.engine.pipeline.pipeline import Image, clean_auto, clean_params
    _, _, photo, _, _ = camera
    image = Image(photo(seed=21))
    e = E.Edit()
    e.denoise.enabled = True
    out = clean_auto(image, e)
    assert out and out["kind"] == "strength" and 0.3 <= out["luma"] <= 2.6 and image.auto_report["note"] == ""
    assert image.noise_model(True).confidence == "good"
    params, auto = clean_params(image, e)
    assert auto is out and params.luma == out["luma"] and params.chroma == out["chroma"]
    e.denoise.lean = 1.5
    assert clean_params(image, e)[0].luma == pytest.approx(1.5 * out["luma"])
    e.denoise.auto = False
    params, auto = clean_params(image, e)
    assert auto is None and params.luma == e.denoise.luma
    raw = photo(seed=22)
    raw.meta.model = "Some Other Camera"                       # no profile: its noise is only estimated
    other = Image(raw)
    e2 = E.Edit()
    e2.denoise.enabled = True
    assert clean_auto(other, e2) is None and "noise to be measured" in other.auto_report["note"]
    params, auto = clean_params(other, e2)
    assert auto is None and (params.luma, params.chroma) == (e2.denoise.luma, e2.denoise.chroma)


def test_the_camera_card_says_what_clean_takes_and_what_it_leaves_alone(camera):
    """Every item of the calibration that Clean uses is said in plain words, and so is what was looked at and
    deliberately not used (a master dark frame from too few frames would add noise)."""
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from studio.app import cameras as C
    from studio.app import theme as T
    prof, _, _, _, _ = camera
    uses = C.clean_uses(prof, D.find_bank(prof.make, prof.model))
    text = " ".join(t for _, t in uses)
    for want in ("Hot and warm photosites are healed", "Black level", "Banding: about", "No repeating (periodic)",
                 "Master dark frame: looked at and not used", "would ADD noise", "Auto strength in Develop"):
        assert want in text, want
    assert next(c for c, t in uses if "Master dark" in t) == T.DIM
    assert next(c for c, t in uses if "Banding: about" in t) == T.CYAN
    assert "Fixed banding: not measured yet" in " ".join(t for _, t in C.clean_uses(prof, None))

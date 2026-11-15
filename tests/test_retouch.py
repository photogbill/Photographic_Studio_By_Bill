"""Phase 9 (second part): heal, clone, red eye, dust spots and the dust map."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from conftest import synthetic_raw
from studio import engine as E
from studio.engine.local import dust as du
from studio.engine.local import retouch as rt
from studio.engine.pipeline import pipeline as P
from studio.engine.pipeline.pipeline import Image

LUM = np.array([0.288, 0.712, 0.0], np.float32)
H, W = 240, 360
YY, XX = np.mgrid[0:H, 0:W]


def _scene(seed=3, dust=((200, 120, 4.0, 0.35),), h=H, w=W):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.25 + 0.15 * xx / w
    tex = 1 + 0.08 * gaussian_filter(rng.standard_normal((h, w)), 1.2)
    sc = np.stack([base * tex * 1.1, base * tex, base * tex * 0.8], -1)
    for x, y, s, depth in dust:
        sc = sc * (1 - depth * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)))[..., None]
    return sc.astype(np.float32)


def _depth(a, x=200, y=120):
    """Brightness at the spot centre / brightness of the ring around it (1 = no spot)."""
    yv = a @ LUM
    win = (slice(y - 20, y + 20), slice(x - 20, x + 20))
    r = np.hypot(XX[win] - x, YY[win] - y)
    return yv[y - 4:y + 5, x - 4:x + 5].mean() / yv[win][r > 14].mean()


def _spot(kind="heal", x=200, y=120, r=12, src=(30, 0), **kw):
    return dict({"kind": kind, "pts": [[x / W, y / H]], "size": r / 360.0, "feather": 0.3, "opacity": 100,
                 "src": [src[0] / W, src[1] / H]}, **kw)


def test_no_spots_change_nothing():
    img = Image(synthetic_raw(_scene()))
    base = E.render(img, E.Edit())
    assert np.array_equal(base, E.render(img, E.Edit(retouch=[])))
    off = _spot()
    off["enabled"] = False
    assert np.array_equal(base, E.render(img, E.Edit(retouch=[off])))


def test_heal_removes_a_dust_spot_and_matches_the_surroundings():
    img = Image(synthetic_raw(_scene()))
    clean = Image(synthetic_raw(_scene(dust=())))
    before = _depth(E.render(img, E.Edit()))
    after = _depth(E.render(img, E.Edit(retouch=[_spot()])))
    ref = _depth(E.render(clean, E.Edit()))
    assert before < 0.85
    assert abs(after - ref) < 0.02
    # outside the spot nothing changes
    a, b = E.render(img, E.Edit()), E.render(img, E.Edit(retouch=[_spot()]))
    far = np.hypot(XX - 200, YY - 120) > 14
    assert np.abs(a - b)[far].max() < 1e-6


def test_heal_is_seamless_where_clone_is_not():
    """Source taken from a brighter place: heal matches the lighting at the edge, clone copies the brightness."""
    sc = _scene(dust=())
    sc[:, 240:] *= 1.6                                       # the source area is brighter
    img = Image(synthetic_raw(sc))
    healed = E.render(img, E.Edit(retouch=[_spot(x=180, y=120, src=(80, 0))])) @ LUM
    cloned = E.render(img, E.Edit(retouch=[_spot("clone", x=180, y=120, src=(80, 0))])) @ LUM
    ring = (np.hypot(XX - 180, YY - 120) > 13) & (np.hypot(XX - 180, YY - 120) < 18)
    inner = np.hypot(XX - 180, YY - 120) < 6
    assert abs(healed[inner].mean() / healed[ring].mean() - 1) < 0.05
    assert cloned[inner].mean() / cloned[ring].mean() > 1.2               # (display values: tone-compressed)


def test_region_and_half_size_renders_match_the_full_render():
    img = Image(synthetic_raw(_scene()))
    stroke = {"kind": "heal", "pts": [[0.3, 0.3], [0.4, 0.45]], "size": 0.02, "feather": 0.3, "src": [0.0, 0.2]}
    e = E.Edit(retouch=[_spot(), stroke, _spot("clone", x=90, y=60, r=8, src=(0, 40))])
    full = E.render(img, e)
    part = E.render(img, e, region=(150, 80, 100, 80))
    assert np.abs(part - full[80:160, 150:250]).max() < 1e-5
    half = E.render(img, e, scale=0.5)
    assert half.shape[:2] == (H // 2, W // 2)
    assert abs(_depth(np.repeat(np.repeat(half, 2, 0), 2, 1)) - _depth(full)) < 0.02


def test_spots_stay_on_their_subject_with_geometry_and_rotated_raws():
    img = Image(synthetic_raw(_scene(), flip=6))
    e = E.Edit(retouch=[_spot()])
    e.geometry.rotate = 4.0
    u, v = E.s_to_view(img, e, 200 / W, 120 / H)
    xs, ys = E.view_to_s(img, e, u, v)
    assert abs(float(xs) - 200 / W) < 1e-4 and abs(float(ys) - 120 / H) < 1e-4
    out = E.render(img, e) @ LUM
    plain = E.render(img, E.Edit(geometry=e.geometry)) @ LUM
    oh, ow = out.shape
    i, j = int(float(v) * oh), int(float(u) * ow)
    assert out[i - 3:i + 4, j - 3:j + 4].mean() > 1.15 * plain[i - 3:i + 4, j - 3:j + 4].mean()


def test_red_eye_pulls_red_down_only_where_red_dominates():
    sc = np.full((H, W, 3), 0.2, np.float32)
    pupil = np.hypot(XX - 120, YY - 100) < 6
    sc[pupil] = [0.5, 0.06, 0.06]
    img = Image(synthetic_raw(sc))
    out = E.render(img, E.Edit(retouch=[_spot("redeye", x=120, y=100, r=10, src=(0, 0))]))
    before = E.render(img, E.Edit())
    r0 = before[100, 120, 0] / before[100, 120, 1]
    r1 = out[100, 120, 0] / out[100, 120, 1]
    assert r0 > 3 and r1 < 1.5
    grey = np.hypot(XX - 120, YY - 100) > 8
    assert np.abs(out - before)[grey & (np.hypot(XX - 120, YY - 100) < 12)].max() < 0.02


def test_membrane_is_harmonic_and_matches_its_edges():
    inside = np.zeros((40, 40), bool)
    inside[10:30, 10:30] = True
    yy, xx = np.mgrid[0:40, 0:40]
    lin = (0.02 * xx + 0.01 * yy).astype(np.float32)[..., None]       # a plane is harmonic: reproduced exactly
    m = rt.membrane(inside, lin)
    assert np.abs(m[inside] - lin[inside]).max() < 1e-5


def test_source_is_kept_inside_the_frame():
    img = Image(synthetic_raw(_scene()))
    sp = _spot(x=10, y=10, r=6, src=(-40, -40))
    out = E.render(img, E.Edit(retouch=[sp]))
    assert np.isfinite(out).all()
    off = rt.clamp_source(sp, W, H)
    assert off[0] * W >= -1 and off[1] * H >= -1


def test_auto_source_avoids_other_dust():
    """Two dust spots side by side: the source for one must not be the other."""
    img = Image(synthetic_raw(_scene(dust=((200, 120, 4.0, 0.35), (230, 120, 4.0, 0.35)))))
    sp = _spot(src=(0, 0))
    off = E.auto_source(img, E.Edit(), sp)
    sx, sy = 200 + off[0] * W, 120 + off[1] * H
    assert np.hypot(sx - 230, sy - 120) > 14 and np.hypot(sx - 200, sy - 120) > 20
    healed = E.render(img, E.Edit(retouch=[dict(sp, src=off)]))
    assert _depth(healed) > 0.97


def test_find_dust_finds_the_spot_and_nothing_in_clean_texture():
    img = Image(synthetic_raw(_scene()))
    spots = E.find_dust(img, E.Edit())
    assert len(spots) == 1
    (x, y), = spots[0]["pts"]
    assert abs(x * W - 200) < 4 and abs(y * H - 120) < 4 and spots[0]["auto"] == "dust"
    assert _depth(E.render(img, E.Edit(retouch=spots))) > 0.97
    assert E.find_dust(Image(synthetic_raw(_scene(dust=()))), E.Edit()) == []
    assert E.find_dust(img, E.Edit(retouch=spots)) == []                 # already healed: not found again


def test_dust_map_learns_recurring_spots_only():
    """Same sensor dust in four different scenes, plus one dark blob that is part of one scene."""
    dust = ((200, 120, 4.0, 0.3), (80, 180, 5.0, 0.25))
    per = []
    for seed in range(4):
        extra = ((300, 40, 4.0, 0.4),) if seed == 0 else ()
        img = Image(synthetic_raw(_scene(seed=seed + 10, dust=dust + extra)))
        b, gx, gy = P._dust_base(img, E.Edit())
        bl = du.find_blobs(b @ du.LUMA, 2, gx, gy, r_min=3, r_max=32, min_contrast=0.035, k_texture=3.0)
        per.append([{"x": q["x"] / W, "y": q["y"] / H, "size": q["r"] / 360.0, "contrast": q["contrast"]} for q in bl])
    spots = du.cluster(per, W, H, min_seen=3)
    got = sorted((round(s["x"] * W), round(s["y"] * H)) for s in spots)
    assert len(got) == 2
    assert all(min(np.hypot(gx_ - x, gy_ - y) for x, y, _, _ in dust) < 5 for gx_, gy_ in got)
    m = du.new_map("Synthetic Test Camera", spots, 4, 11.0)
    # a new photo of another scene at the same aperture: both spots are found again and healed
    img = Image(synthetic_raw(_scene(seed=99, dust=dust)))
    b, gx, gy = P._dust_base(img, E.Edit())
    blobs = du.find_blobs(b @ du.LUMA, 2, gx, gy, r_min=3, r_max=32, min_contrast=0.02, k_texture=2.0, limit=600)
    found = du.match_map(m, blobs, W, H, 11.0)
    assert len(found) == 2


def test_learn_dust_map_end_to_end_saves_privately(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path))
    dust = ((200, 120, 4.0, 0.3),)
    imgs = {f"p{k}.cr2": Image(synthetic_raw(_scene(seed=20 + k, dust=dust))) for k in range(4)}
    monkeypatch.setattr(P, "open_image", lambda p: imgs[p])
    rep = E.learn_dust_map(list(imgs))
    assert rep["ok"] and rep["spots"] == 1 and rep["photos"] == 4
    assert str(tmp_path) in rep["path"]
    img = Image(synthetic_raw(_scene(seed=77, dust=dust)))
    spots = E.dust_from_map(img, E.Edit())
    assert len(spots) == 1 and spots[0]["auto"] == "map"
    assert _depth(E.render(img, E.Edit(retouch=spots))) > 0.97
    # the same scene twice counts once
    imgs2 = {f"q{k}.cr2": Image(synthetic_raw(_scene(seed=20, dust=dust))) for k in range(3)}
    monkeypatch.setattr(P, "open_image", lambda p: imgs2[p])
    rep2 = E.learn_dust_map(list(imgs2))
    assert not rep2["ok"] and len(rep2["skipped"]) == 2


def test_retouch_is_copied_with_settings_but_never_carried_by_looks():
    from studio.engine.presets import looks
    src = E.Edit(retouch=[_spot()])
    dst = looks.copy_groups(src, E.Edit(), ["retouch"])
    assert dst.retouch == src.retouch
    lk = looks.look_from_edit(src, "x", ["light"])
    assert "retouch" not in lk["settings"]
    assert looks.apply_look(E.Edit(), {"settings": {"retouch": [_spot()]}}).retouch is None
    assert looks.describe_change(E.Edit(), src).startswith("Retouch")

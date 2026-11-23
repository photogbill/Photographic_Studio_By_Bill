"""Body parts found for body sculpting (local/bodies.py) and the warps they drive (portrait 'outline', 'stretch'),
on synthetic figures whose measurements are known."""
from __future__ import annotations

import numpy as np
import pytest

from studio.engine.local import bodies as bd
from studio.engine.local import people as pp
from studio.engine.local import portrait as pt

SW, SH = 1000, 1400


def _figure(side_on: bool = False):
    """(33 points as S fractions, person mask on a 500 x 700 grid, full-res mask) of a standing figure, facing the
    camera (or side-on, facing left in the photo). Torso half width: 110 at the shoulders, 70 at the waist (y 650),
    100 at the hips; arms 30 wide each side of their line, thighs 45, calves 30."""
    P = np.zeros((33, 3))
    P[:, 2] = 0.95
    sx = 10 if side_on else 100
    hx = 8 if side_on else 60
    pts = {pp.NOSE: (470 if side_on else 500, 300), pp.L_EYE: (480, 290), pp.R_EYE: (520, 290),
           pp.L_EAR: (465, 300), pp.R_EAR: (535, 300), pp.MOUTH_L: (490, 330), pp.MOUTH_R: (510, 330),
           pp.L_SH: (500 + sx, 400), pp.R_SH: (500 - sx, 400), pp.L_EL: (670, 600), pp.R_EL: (330, 600),
           pp.L_WR: (672, 780), pp.R_WR: (328, 780), pp.L_IDX: (673, 820), pp.R_IDX: (327, 820),
           pp.L_HIP: (500 + hx, 800), pp.R_HIP: (500 - hx, 800), pp.L_KN: (550, 1050), pp.R_KN: (450, 1050),
           pp.L_AN: (550, 1300), pp.R_AN: (450, 1300)}
    if side_on:
        pts[pp.L_EAR], pts[pp.R_EAR] = (505, 300), (515, 300)
    for k, (x, y) in pts.items():
        P[k, :2] = (x / SW, y / SH)
    P[[1, 3, 4, 6, 17, 18, 21, 22, 29, 30], 2] = 0.2
    P[[25, 26, 27, 28, 31, 32], :2] = P[[25, 26, 27, 28, 27, 28], :2]
    P[[25, 26, 27, 28, 31, 32], 2] = 0.95
    ys, xs = np.mgrid[0:SH, 0:SW] + 0.5
    half = np.interp(ys, [380, 400, 650, 820, 880], [100, 110, 70, 100, 100])
    if side_on:
        half = np.interp(ys, [380, 400, 650, 820, 880], [60, 70, 70, 90, 90])
    m = ((np.abs(xs - 500) <= half) & (ys >= 380) & (ys <= 880)).astype(np.float32)
    m = np.maximum(m, (np.hypot(xs - 500, ys - 300) <= 45).astype(np.float32))
    m = np.maximum(m, ((np.abs(xs - 500) <= 28) & (ys >= 330) & (ys <= 400)).astype(np.float32))

    def tube(a, b, r):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        t = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / (d @ d), 0, 1)
        return (np.hypot(xs - a[0] - t * d[0], ys - a[1] - t * d[1]) <= r).astype(np.float32)
    if not side_on:
        for s in (1, -1):
            m = np.maximum(m, tube((500 + s * 170, 420), (500 + s * 170, 600), 30))
            m = np.maximum(m, tube((500 + s * 170, 600), (500 + s * 172, 780), 25))
    for s in (1, -1):
        m = np.maximum(m, tube((500 + s * hx, 800), (500 + s * 50, 1050), 45 if not side_on else 50))
        m = np.maximum(m, tube((500 + s * 50, 1050), (500 + s * 50, 1300), 30))
    grid = m.reshape(700, 2, 500, 2).mean(axis=(1, 3))
    if not side_on:                                   # the arms attach at the shoulder points in this figure
        P[pp.L_SH, :2] = (670 / SW, 410 / SH)
        P[pp.R_SH, :2] = (330 / SW, 410 / SH)
    return P.tolist(), grid, m


def _parts(side_on=False):
    P, grid, m = _figure(side_on)
    return bd.find_parts([P], grid, SW, SH), [P], grid, m


def _get(parts, name, side=""):
    for q in parts:
        if q["part"] == name and q.get("side", "") == side:
            return q
    raise AssertionError(f"{name} {side} not found in {[q['id'] for q in parts]}")


def test_parts_found_with_their_outlines_measured():
    parts, *_ = _parts()
    names = {q["part"] for q in parts}
    assert {"upper_arm", "forearm", "thigh", "calf", "neck", "shoulders", "waist", "hips", "chest"} <= names
    assert "tummy" not in names and "seat" not in names            # side-on only
    w = _get(parts, "waist")
    assert bd.how(w) == bd.MEASURED
    assert w["wl"] * SH == pytest.approx(71, abs=4) and w["wr"] * SH == pytest.approx(71, abs=4)
    mid = 0.5 * (w["a"][1] + w["b"][1]) * SH
    assert mid == pytest.approx(650, abs=25)                         # the narrowest place, not the hip line
    th = _get(parts, "thigh", "L")
    assert bd.how(th) == bd.MEASURED
    assert th["wl"] * SH == pytest.approx(45, abs=4) and th["wr"] * SH == pytest.approx(45, abs=4)
    assert _get(parts, "upper_arm", "R")["view"] == "front"


def test_side_on_finds_tummy_and_seat_and_moves_one_edge():
    parts, *_ = _parts(side_on=True)
    names = {q["part"] for q in parts}
    assert {"tummy", "seat", "chest"} <= names
    t = _get(parts, "tummy")
    assert sorted((t["fl"], t["fr"])) == [0.0, 1.0]
    s = _get(parts, "seat")
    assert (s["fl"], s["fr"]) == (t["fr"], t["fl"])                # the seat is on the back
    assert len(bd.lift_shapes(_get(parts, "chest"), [], SW, SH)) == 1


def _slimmed(part_name, slim=100.0, side="", mirror=False, side_on=False):
    parts, people, grid, m = _parts(side_on)
    q = _get(parts, part_name, side)
    item = bd.item_for(q, parts, people, SW, SH, mirror=mirror, slim=slim)
    return pt.warps({"sculpt": [item]}, SW, SH), q, m


def test_slim_narrows_the_waist_by_the_asked_share():
    ws, q, m = _slimmed("waist")
    assert ws and ws[0][0] == "outline"
    base = np.repeat(m[..., None], 3, -1)
    out = pt.apply_warps(base, 0.0, 0.0, 1, ws)
    row = int(round(0.5 * (q["a"][1] + q["b"][1]) * SH))
    before = float(base[row, :, 0].sum())
    after = float(out[row, :, 0].sum())
    # the waist row: 140 wide; 12% narrower at 100 (the arms beside it are out of reach)
    assert before - after == pytest.approx(0.12 * 2 * q["wl"] * SH, rel=0.15)


def test_joints_and_the_room_stay_put():
    ws, q, _ = _slimmed("thigh", side="L")
    X = np.array([550.0 + 10, 500 + 50, 200.0, 950.0, 560.0])
    Y = np.array([800.0, 1050.0, 925.0, 925.0, 925.0])     # hip, knee, far left, far right, mid-thigh
    dx, dy = pt._disp(ws[0], X, Y)
    assert abs(dx[0]) < 0.05 and abs(dy[0]) < 0.05 and abs(dx[1]) < 0.05 and abs(dy[1]) < 0.05
    assert np.all(np.abs(dx[2:4]) < 1e-9) and np.all(np.abs(dy[2:4]) < 1e-9)
    # a vertical line well clear of the body stays straight
    yl = np.linspace(500, 1300, 50)
    dxl, _ = pt._disp(ws[0], np.full_like(yl, 150.0), yl)
    assert np.abs(dxl).max() < 1e-9


def test_both_sides_does_the_other_limb_too():
    ws, q, _ = _slimmed("thigh", side="L", mirror=True)
    assert len(ws) == 2
    xs = [0.5 * (w[1] + w[3]) for w in ws]
    assert min(xs) < 500 < max(xs)


def test_every_body_warp_inverts_exactly():
    parts, people, grid, m = _parts()
    items = [bd.item_for(_get(parts, n, s), parts, people, SW, SH, slim=100) for n, s in
             (("waist", ""), ("thigh", "L"), ("upper_arm", "R"), ("neck", ""), ("hips", ""))]
    items.append(bd.legs_item(people, 0, SW, SH, amount=100))
    for it in items:
        ws = pt.warps({"sculpt": [it]}, SW, SH)
        assert ws
        X, Y = np.meshgrid(np.linspace(250, 750, 60), np.linspace(250, 1350, 90))
        dx, dy = pt._disp(ws[0], X, Y)
        SX, SY = pt.source_of(ws, X + dx, Y + dy, iters=8)
        assert np.abs(SX - X).max() < 0.05 and np.abs(SY - Y).max() < 0.05, it["name"]


def test_faces_and_hands_never_move():
    ws, q, _ = _slimmed("neck", slim=-100)
    dx, dy = pt._disp(ws[0], np.array([500.0, 530.0]), np.array([300.0, 300.0]))
    assert np.abs(dx).max() < 1e-6 and np.abs(dy).max() < 1e-6
    ws, q, _ = _slimmed("forearm", side="L")
    dx, dy = pt._disp(ws[0], np.array([672.0, 697.0]), np.array([800.0, 800.0]))
    assert np.abs(dx).max() < 1e-6 and np.abs(dy).max() < 1e-6


def test_longer_legs_stretch_below_the_hips_only():
    _, people, _, _ = _parts()
    it = bd.legs_item(people, 0, SW, SH, amount=100)
    ws = pt.warps({"sculpt": [it]}, SW, SH)
    w = ws[0]
    assert w[0] == "stretch"
    dx, dy = pt._disp(w, np.array([500.0, 500.0, 500.0, 500.0]), np.array([300.0, 799.0, 1300.0, 2200.0]))
    assert abs(dy[0]) < 1e-9 and abs(dy[1]) < 1.0
    assert dy[2] == pytest.approx(0.06 * 500, rel=0.02)          # the ankles go down 6% of the leg length
    assert abs(dy[3]) < 1e-9
    assert np.all(np.abs(dx) < 1e-9)
    assert bd.legs_item([np.zeros((33, 3)).tolist()], 0, SW, SH) is None


def test_lift_parts_place_two_shapes_facing_the_camera():
    parts, people, *_ = _parts()
    shp = bd.lift_shapes(_get(parts, "chest"), people, SW, SH)
    assert len(shp) == 2 and all(s["tool"] == "breast" and s["placed"] for s in shp)
    xs = sorted(s["c"][0] * SW for s in shp)
    assert xs[0] < 500 < xs[1]
    assert pt.warps({"sculpt": shp}, SW, SH)                       # the existing lift tool renders them


def test_pointing_finds_the_part_under_the_pointer():
    parts, *_ = _parts()
    q = bd.part_at(parts, 555 / SW, 925 / SH, SW, SH)
    assert q is not None and q["part"] == "thigh"
    assert bd.part_at(parts, 60 / SW, 100 / SH, SW, SH) is None


def test_a_dragged_part_measures_its_width_from_the_mask():
    _, _, grid, _ = _parts()
    q = bd.manual_part([555 / SW, 860 / SH], [550 / SW, 1000 / SH], grid, SW, SH)
    assert bd.how(q) == bd.MEASURED
    q2 = bd.manual_part([555 / SW, 860 / SH], [550 / SW, 1000 / SH], None, SW, SH)
    assert bd.how(q2) == bd.ESTIMATED
    it = bd.item_for(q, [], [], SW, SH, slim=50)
    assert pt.warps({"sculpt": [it]}, SW, SH)


def test_zero_and_disabled_do_nothing_and_the_history_names_the_part():
    parts, people, *_ = _parts()
    it = bd.item_for(_get(parts, "waist"), parts, people, SW, SH, slim=0)
    assert not pt.warps({"sculpt": [it]}, SW, SH)
    it["slim"] = 50
    it["enabled"] = False
    assert not pt.sculpt_active({"sculpt": [it]})
    it["enabled"] = True
    assert pt.sculpt_active({"sculpt": [it]})
    from studio.engine.presets.looks import _describe_portrait
    assert _describe_portrait({}, {"sculpt": [it]}) == "Sculpt: waist"


def test_panel_keeps_bodies_and_parts_and_the_viewer_names_the_part_under_the_pointer():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    from studio.engine.local import masks as mk
    app = QApplication.instance() or QApplication([])
    parts, people, grid, _ = _parts()
    item = bd.item_for(_get(parts, "waist"), parts, people, SW, SH, slim=40)
    store = {"w": 500, "h": 700, "data": mk.encode_bitmap(np.round(grid * 255).astype(np.uint8)), "people": people}
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.portrait = {"bodies": store, "sculpt": [item]}
    page.controls_from_edit(e)
    page.edit = e
    page._shsel = 0
    page._refresh_portrait_ui()
    assert page.bd_amount.isEnabled() and page.bd_amount.value() == 40
    assert "Waist" in page.bd_info.text()
    back = page.edit_from_controls()
    assert back.portrait["bodies"]["people"] == people and back.portrait["sculpt"][0]["kind"] == "part"
    v = page.viewer
    v.tool = "body"
    v.resize(800, 600)
    v.body_parts = [{"outline": [(0.2, 0.2), (0.4, 0.2), (0.4, 0.6), (0.2, 0.6)], "name": "Thigh", "how": "measured"},
                    {"outline": [(0.25, 0.3), (0.3, 0.3), (0.3, 0.4), (0.25, 0.4)], "name": "Knee", "how": "estimated"}]
    from PySide6.QtGui import QImage
    v.after = QImage(400, 300, QImage.Format.Format_RGB32)
    assert v._body_at((0.35, 0.5)) == 0 and v._body_at((0.27, 0.35)) == 1 and v._body_at((0.8, 0.8)) == -1
    v.body_hover, v._mouse = 1, (0.27, 0.35)
    v._pts = [(0.1, 0.1), (0.2, 0.3)]
    assert not v.grab().isNull()                       # paints the parts, the hovered name and a drag line
    got = []
    v.body_clicked.connect(got.append)
    v._pts = [(0.35, 0.5), (0.35, 0.5)]
    v._tool_release(None)
    assert got == [0]
    app.processEvents()

"""3D Scan (BUILD_PLAN_DP3D E, I.2): the camera solver, the mesh and the files, on synthetic views with known truth
(tests/scan_synth.py: a textured, bumpy log on textured ground, hand-held views with their own Dual Pixel depth).
Nothing is written outside outputs/."""
from __future__ import annotations

import functools
import json
import os

import numpy as np
import pytest

import scan_synth as S
from studio.engine.scan import geom as G
from studio.engine.scan import sift
from studio.engine.scan import solve as SV
from studio.engine.scan import twoview as TV

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "pytest_scan")


@functools.lru_cache(maxsize=None)
def _ring(n, seed, stray=False):
    svs = S.ring(n, seed=seed) + ([S.stray()] if stray else [])
    return svs, [S.to_view(s) for s in svs]


@functools.lru_cache(maxsize=None)
def _solved(n, seed, use_depth=True, stray=False):
    svs, vs = _ring(n, seed, stray)
    return svs, vs, SV.solve(vs, use_depth=use_depth, n_features=2000)


# ------------------------------------------------------------------------------------------ pieces
def test_features_survive_turning_and_scaling():
    _, vs = _ring(5, 4)
    from scipy.ndimage import affine_transform
    img = vs[0].image
    a = np.radians(40)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) / 1.25
    c = np.array(img.shape) / 2
    off = c - M @ c
    warped = affine_transform(img, M, offset=off, order=1, mode="nearest")
    fa, fb = sift.detect(img), sift.detect(warped)
    m = sift.match(fa, fb)
    pb = (fa.xy[m[:, 0]][:, ::-1] - off) @ np.linalg.inv(M).T
    good = np.linalg.norm(pb - fb.xy[m[:, 1]][:, ::-1], axis=1) < 2
    assert len(m) > 500 and good.mean() > 0.95


def test_two_view_geometry_is_exact_on_clean_points():
    rng = np.random.default_rng(0)
    R = G.rodrigues(np.array([0.1, 0.4, -0.05]))
    t = np.array([0.8, 0.1, 0.2])
    X = rng.uniform([-1, -1, 3], [1, 1, 6], (200, 3))
    x1 = X[:, :2] / X[:, 2:]
    Xc = X @ R.T + t
    x2 = Xc[:, :2] / Xc[:, 2:]
    x2[:30] += rng.normal(0, 0.05, (30, 2))                       # 15 % wrong matches
    R2, t2, inl = G.essential_pose_ransac(x1, x2, 0.002)
    assert G.angle_deg(R, R2) < 0.01 and inl[30:].all() and not inl[:30].any()
    Rp, tp, inl = G.pnp_ransac(X, x2, 0.002)
    assert G.angle_deg(R, Rp) < 0.01 and np.linalg.norm(tp - t) < 1e-6
    s, Ru, tu = G.umeyama(2.0 * X @ R.T + t, X)
    assert abs(s - 2.0) < 1e-9 and G.angle_deg(R, Ru) < 1e-6


def test_a_flat_texture_is_called_planar_and_posed_from_its_depth():
    pv = [S.to_view(s) for s in S.planar_pair()]
    fa, fb = sift.detect(pv[0].image), sift.detect(pv[1].image)
    pr = TV.pair(pv[0], pv[1], fa, fb, sift.match(fa, fb), 0, 1, True)
    assert pr is not None and pr.model == "homography" and pr.how == "depth"


# ------------------------------------------------------------------------------------------ the solver
def test_sixteen_hand_held_views_are_placed_to_half_a_degree():
    svs, vs, sol = _solved(16, 2)
    c = SV.compare(sol, [s.R_true for s in svs], [s.C_true for s in svs])
    assert c["placed"] == 16
    assert max(c["rot_deg"]) < 0.5
    # positions: under 1.5 % of the cameras' spread (1.7 m). Each stack's focal length is solved (focus breathing),
    # and a focal length a little long puts that camera a little further back along its view: the worst view here
    # comes out 2.5 % long and 2 cm back (docs/RESULTS.md, session 18); the median is under 0.5 %
    assert max(c["pos"]) < 0.015 and np.median(c["pos"]) < 0.005
    assert sol.summary()["reprojection_px"] < 0.5
    assert all(h in ("first", "both", "matches", "depth") for h in sol.how)


def test_five_views_72_degrees_apart_are_placed_only_with_the_depth():
    """Bill's shoot 2: a few wide views of one object. The depth places them all; plain structure from motion (the
    depth switched off) cannot. This is the reason the solver exists."""
    svs, vs, sol = _solved(5, 4, True, True)                      # with a stray view of something else
    c = SV.compare(sol, [s.R_true for s in svs[:5]] + [svs[5].R_true], [s.C_true for s in svs])
    assert sum(sol.placed[:5]) == 5 and max(c["rot_deg"]) < 1.0
    assert sol.affine                                             # too far round for plain matching
    assert not sol.placed[5] and "no detail" in sol.reason[5]     # the stray is named, with the reason
    _, _, plain = _solved(5, 4, False, False)
    c2 = SV.compare(plain, [s.R_true for s in svs[:5]], [s.C_true for s in svs[:5]])
    assert sum(plain.placed) <= 3 or max(c2["rot_deg"]) > 3 * max(c["rot_deg"])


def test_every_view_says_how_it_was_placed_and_its_depth_shape_is_recovered():
    svs, vs, sol = _solved(5, 4, True, True)
    for i in range(5):
        v, s = vs[i], svs[i]
        k_true = v.t_scale / (v.t_ref - s.b)                       # kappa of t = a / z + b
        assert abs(sol.kappa[i] - k_true) < 0.1 * k_true + 0.01
        assert SV.HOW_COLOUR[sol.how[i]] in ("cyan", "amber")
    q = [x for x in sol.quality[:5] if x]
    assert all(x["reprojection_px"] < 0.5 and x["depth_rms"] < 0.01 for x in q)


def test_focus_breathing_each_stack_finds_its_own_focal_length():
    """Macro: the lens moves out to focus, so the picture's focal length is up to twice the marked one, and each
    stack breathes a little differently. The files say 520 px; the truth is 625-675 px."""
    svs = S.ring(8, seed=6, f=650.0, breathing=0.04)
    vs = [S.to_view(s, f_marked=520.0, f_sigma=0.35) for s in svs]
    sol = SV.solve(vs, n_features=2000)
    c = SV.compare(sol, [s.R_true for s in svs], [s.C_true for s in svs])
    assert sum(sol.placed) == 8 and max(c["rot_deg"]) < 1.0
    f_true = np.array([s.f for s in svs])
    assert abs(np.mean(sol.f) / f_true.mean() - 1) < 0.03
    assert vs[0].f == 520.0                                       # the caller's views are left alone
    assert any("focal length" in n for n in sol.notes)


# ------------------------------------------------------------------------------------------ the model and its files
@functools.lru_cache(maxsize=None)
def _model(tag="m8"):
    from studio.engine.scan import model as MD
    svs, vs, sol = _solved(8, 6)
    out = os.path.join(OUT, tag)
    r = MD.build(vs, out, "LOG", MD.ModelParams(triangles=6000, lods=True, atlas=2048), MD.MAKES, solution=sol)
    return svs, vs, sol, r, out


def test_the_mesh_lies_on_the_true_surface_where_measured_and_the_unseen_underside_is_filled():
    from studio.engine.scan import model as MD
    svs, vs, sol, r, out = _model()
    Vw, T, measured = r["mesh_world"]
    # solution world -> truth by the cameras' centres
    idx = [i for i in range(len(vs)) if sol.placed[i]]
    s, R, t = G.umeyama(np.array([svs[i].C_true for i in idx]), np.array([sol.centre(i) for i in idx]))
    P = s * Vw @ R.T + t
    fc = P[T].mean(1)
    on_log = np.abs(fc[:, 0]) < S.HALF_L - 0.03
    th = np.arctan2(fc[:, 2] - S.R0, fc[:, 1])
    d = np.abs(np.hypot(fc[:, 1], fc[:, 2] - S.R0) - S.radius(th, fc[:, 0]))
    sel = measured & on_log
    assert sel.sum() > 1000
    assert np.median(d[sel]) < 0.01 * 2 * S.HALF_L                # under 1 % of the object's size
    # the underside (facing down) no camera saw: filled, not measured
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    down = fn[:, 2] / np.maximum(np.linalg.norm(fn, axis=1), 1e-12) < -0.8
    assert down.any() and (~measured[down]).mean() > 0.8
    q = r["json"]["quality"]
    assert 0.3 < q["measured_share"] < 0.95 and q["views_placed"] == 8


def test_the_model_files_keep_their_contract_and_read_back():
    from studio.engine.io import gltf as GL
    svs, vs, sol, r, out = _model()
    doc = r["json"]
    assert doc["studio_model"] == 1 and doc["written_by"].startswith("Photographic Studio By Bill")
    for k in ("glb", "mesh", "obj", "points", "cameras", "made", "json", "dataset"):
        assert k in r["paths"], k
    names = {os.path.basename(p) for p in r["paths"].values()}
    assert "LOG_model_Studio.glb" in names and "LOG_dataset_Studio" in names
    g, b = GL.read_glb(r["paths"]["glb"])
    assert GL.validate(g, b) == [] and [m["name"] for m in g["meshes"]] == ["LOG", "LOG_LOD1", "LOG_LOD2"]
    tri = [int(g["accessors"][m["primitives"][0]["indices"]]["count"]) // 3 for m in g["meshes"]]
    assert tri[0] <= 6600 and tri[0] > tri[1] > tri[2]
    assert g["materials"][0].get("normalTexture") is not None
    cams = json.load(open(r["paths"]["cameras"], encoding="utf-8"))
    assert sum(v["placed"] for v in cams["views"]) == 8 and all(v["colour"] in ("cyan", "amber") for v in cams["views"])
    assert doc["godot"]["normal_map"].startswith("OpenGL") and doc["units"] == "model units"
    assert doc["coach"]["lines"] and any("underside" in line for line in doc["coach"]["lines"])
    # a second run replaces its own files (no -2)
    from studio.engine.scan import model as MD
    r2 = MD.build(vs, out, "LOG", MD.ModelParams(triangles=3000, atlas=1024), ("glb", "cameras"), solution=sol)
    assert os.path.basename(r2["paths"]["glb"]) == "LOG_model_Studio.glb"


def test_the_blender_dataset_cameras_agree_both_ways_and_the_gate_refuses_a_poor_scan(tmp_path):
    from studio.engine.scan import dataset as DS
    svs, vs, sol, r, out = _model()
    ds = json.load(open(os.path.join(r["paths"]["dataset"], "dataset.json"), encoding="utf-8"))
    assert ds["format"] == "studio_blender_dataset" and ds["gate"]["passed"]
    cam = ds["cameras"][0]
    for k in ("image", "depth", "confidence", "agreement", "mask"):
        assert os.path.exists(os.path.join(r["paths"]["dataset"], cam["files"][k])), k
    d = np.load(os.path.join(r["paths"]["dataset"], cam["files"]["depth"]))
    assert d.dtype == np.float32 and np.isfinite(d).mean() > 0.05
    # one world point projected with the OpenCV numbers and with Blender's camera: the same pixel
    Rcv, tcv = np.array(cam["opencv"]["R"]), np.array(cam["opencv"]["t"])
    K = np.array(cam["opencv"]["K"])
    C = -Rcv.T @ tcv
    X = C + Rcv.T @ np.array([0.05, -0.03, 1.0])                   # a point in front of the camera
    x = K @ (Rcv @ X + tcv)
    px_cv = x[:2] / x[2]
    mw = np.array(cam["blender"]["matrix_world"])
    Xl = np.linalg.inv(mw) @ np.r_[X, 1.0]                         # Blender camera space: looks down -Z, +Y up
    W, H = cam["blender"]["resolution"]
    f = cam["blender"]["lens_mm"] / cam["blender"]["sensor_width_mm"] * W
    big = max(W, H)
    u = f * Xl[0] / -Xl[2] + (W - 1) / 2.0 - cam["blender"]["shift_x"] * big
    v = -f * Xl[1] / -Xl[2] + (H - 1) / 2.0 + cam["blender"]["shift_y"] * big
    assert np.allclose([u, v], px_cv, atol=1e-6)
    ok, why = DS.gate({"views_given": 6, "views_placed": 2, "reprojection_px": 2.4, "measured_share": 0.2,
                       "disagree_share": 0.4})
    assert not ok and len(why) == 5 and all(isinstance(w, str) and w for w in why)


def test_model_is_a_command_reading_focus_stack_results(tmp_path, capsys):
    """The command line on Focus Stack results (the picture and NAME.depth.npz beside it, as Focus Stack writes
    them): cameras and a glb; the dataset only through the gate."""
    from studio import cli
    from studio.engine.io.export import write_png_any
    svs, vs = _ring(5, 4)
    files = []
    for sv, v in zip(svs, vs):
        p = tmp_path / f"{sv.name}_Studio.png"
        write_png_any(str(p), v.colour)
        t = v.t
        lo, hi = np.percentile(t[v.conf > 0.5], [0.5, 99.5])
        np.savez_compressed(str(tmp_path / f"{sv.name}_Studio.depth.npz"),
                            depth=np.clip((t - lo) / (hi - lo), 0, 1).astype(np.float16),
                            confidence=v.conf.astype(np.float16), label=(v.conf > 0.5).astype(np.uint8) * 2,
                            label_strict=(v.mask & (v.conf > 0.5)).astype(np.uint8) * 2, range_steps=np.float32(hi - lo))
        files.append(str(p))
    capsys.readouterr()
    code = cli.main(["model"] + files + ["--make", "cameras,glb", "--triangles", "3000", "--atlas", "1024",
                                        "--out", str(tmp_path / "m"), "--json"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0, res["problems"]
    kinds = {o["kind"] for o in res["outputs"]}
    assert {"model_glb", "model_cameras", "model_json"} <= kinds
    assert res["data"]["model"]["views"]["placed"] >= 4
    code = cli.main(["model", files[0], "--json"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and "two views" in res["problems"][0]["text"]


# ------------------------------------------------------------------------------------------ one call per folder (Forge)
def test_intake_turns_one_folder_into_a_model_and_textures(tmp_path, capsys):
    """A folder as Bill fills it: five views round the log (Focus Stack results with their depth), one result of a
    wall elsewhere (another subject: a texture; a texture shot beside the object would rightly join it), one ordinary
    photo (an estimated texture). One call; one JSON listing a model, a measured texture and an estimated one."""
    from studio import cli
    from studio.engine.io.export import write_png_any
    svs, vs = _ring(5, 4)
    folder = tmp_path / "incoming"
    folder.mkdir()

    def result(sv, v, name):
        p = folder / f"{name}_Studio.png"
        write_png_any(str(p), v.colour)
        t = v.t
        lo, hi = np.percentile(t[v.conf > 0.5], [0.5, 99.5])
        np.savez_compressed(str(folder / f"{name}_Studio.depth.npz"),
                            depth=np.clip((t - lo) / (hi - lo), 0, 1).astype(np.float16),
                            confidence=v.conf.astype(np.float16), label=(v.conf > 0.5).astype(np.uint8) * 2,
                            label_strict=((v.mask if v.mask is not None else v.conf > 0.5) & (v.conf > 0.5))
                            .astype(np.uint8) * 2, range_steps=np.float32(hi - lo))
    for sv, v in zip(svs, vs):
        result(sv, v, sv.name)
    g = S.stray()
    gv = S.to_view(g)
    gv.mask = None
    result(g, gv, "WALL")
    write_png_any(str(folder / "BARK.png"), np.dstack([np.clip(S.tex("cap") * 255, 0, 255).astype(np.uint8)] * 3)[:300, :400])
    capsys.readouterr()
    code = cli.main(["intake", str(folder), "--out", str(tmp_path / "out"), "--size", "256", "--triangles", "3000",
                     "--json"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0, res["problems"]
    doc = json.load(open(res["outputs"][0]["output"], encoding="utf-8"))
    assert doc["studio_intake"] == 1
    kinds = sorted((a["kind"], a.get("measured")) for a in doc["assets"])
    assert ("model", None) in kinds
    model = next(a for a in doc["assets"] if a["kind"] == "model")
    assert len(model["views"]) >= 4 and os.path.exists(tmp_path / "out" / model["files"]["glb"])
    tex = {a["name"]: a for a in doc["assets"] if a["kind"] == "texture"}
    assert tex["BARK"]["measured"] is False and tex["BARK"]["colour"] == "amber"
    assert "WALL_Studio" in tex, (doc["left_out"], model["views"], doc["notes"])
    assert tex["WALL_Studio"]["measured"] is True and "WALL_Studio.png" not in model["views"]
    assert doc["summary"]["models"] == 1 and doc["summary"]["textures_estimated"] == 1
    assert any(a["kind"] == "relief" and a["name"] == "WALL_Studio" for a in doc["assets"])   # the lone view
    job = json.load(open(tmp_path / "out" / "intake_job.json", encoding="utf-8"))
    assert job["state"] == "done"


def test_a_known_length_sets_the_size_in_metres():
    from studio.engine.scan import model as MD
    svs, vs, sol = _solved(8, 6)
    r = MD.build(vs, os.path.join(OUT, "len"), "LOG", MD.ModelParams(triangles=2000, atlas=512, known_length_m=1.2),
                 ("glb",), solution=sol)
    V = r["mesh"][0]
    assert abs(np.max(np.ptp(V, axis=0)) - 1.2) < 1e-6
    assert r["json"]["units"] == "metres" and r["json"]["scale"]["how"].startswith("given")


# ------------------------------------------------------------------------------------------ front-only reliefs
def test_a_front_relief_follows_the_measured_depth_and_is_cut_at_depth_edges(tmp_path):
    """One view's measured depth as a relief mesh: nearer parts stand out (the mesh's depth follows the true
    inverse distance), no face bridges the jump from the log to the ground behind it, the glb is valid with a cut-out
    (alpha mask) texture, and the depth proportion is labelled estimated unless the solver measured it."""
    from studio.engine.io import gltf as GL
    from studio.engine.scan import relief as RL
    svs, vs = _ring(5, 4)
    v, sv = vs[0], svs[0]
    r = RL.build(v, str(tmp_path), "LOGVIEW", triangles=4000)
    doc = r["json"]
    assert doc["studio_relief"] == 1 and doc["cut_to_subject"] and doc["depth"]["colour"] == "amber"
    g, b = GL.read_glb(r["paths"]["glb"])
    assert GL.validate(g, b) == [] and g["materials"][0]["alphaMode"] == "MASK"
    Vl, Tl = r["mesh"]
    assert len(Tl) <= 4400
    # depth follows the truth: sample the true inverse distance under each vertex
    h, w = v.shape
    gh_aspect = h / float(w)
    px = (Vl[:, 0] + 0.5) * w - 0.5
    py = (-Vl[:, 1] / gh_aspect + 0.5) * h - 0.5
    from scipy.ndimage import map_coordinates
    inv = map_coordinates(np.where(np.isfinite(sv.z_true), 1.0 / np.where(np.isfinite(sv.z_true), sv.z_true, 1),
                                   0), [py, px], order=1)
    ok = inv > 0
    assert np.corrcoef(Vl[ok, 2], inv[ok])[0, 1] > 0.8
    # a measured depth shape (from the solver) gives the measured proportion, cyan
    r2 = RL.build(v, str(tmp_path), "LOGVIEW_K", triangles=2000, kappa=0.3)
    assert r2["json"]["depth"]["colour"] == "cyan"
    r3 = RL.build(v, str(tmp_path), "LOGVIEW", triangles=2000)          # the same name again: replaced, not -2
    assert os.path.basename(r3["paths"]["glb"]) == "LOGVIEW_relief_Studio.glb"

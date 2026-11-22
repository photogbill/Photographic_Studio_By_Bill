"""Dual Pixel 3D (BUILD_PLAN_DP3D): the foundations (A), on synthetic data with known truth. Nothing is written
outside outputs/.

A.1  which way is near, voted from receding planes (any orientation of the camera), and the camera's Dual Pixel
     profile that keeps the answer
A.2  the lens's recorded focus distance (Canon ShotInfo), or its absence
A.3  the fast offset search gives the first implementation's numbers; frames developed several at once give the
     same analysis as one at a time
A.4  a second analysis of the same frames comes from the cache, with the same depth
A.6  a stack where only some frames are Dual Pixel uses the ones that are
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

import depth_synth as DS
from studio.engine.dualpixel import depth as DEP
from studio.engine.dualpixel import dpprofile as DPP
from studio.engine.dualpixel import nearcheck as NC
from studio.engine.stack import align as A
from studio.engine.stack import depthfuse as DF
from studio.engine.stack import focus as FS


# ------------------------------------------------------------------------------------------ A.3 fast search
@pytest.mark.parametrize("h,w,size,step", [(53, 71, 16, 4), (50, 64, 48, 16), (37, 40, 15, 5), (8, 8, 48, 16)])
def test_grid_box_sums_are_uniform_filters(h, w, size, step):
    x = np.random.default_rng(3).standard_normal((h, w)).astype(np.float32)
    rows, cols = np.arange(0, h, step), np.arange(0, w, step)
    ref = uniform_filter(x.astype(np.float64), size, mode="nearest")[rows][:, cols]
    assert np.abs(DEP._box_mean(x, size, rows, cols) - ref).max() < 1e-9


def test_grid_upsampling_is_the_flow_fields_bilinear():
    g = np.random.default_rng(4).standard_normal((9, 13)).astype(np.float32)
    ref = A.Flow(g, g, g, 0.0, 0.0, 4.0).fields((37, 50))[0]
    assert np.abs(DEP._upsample_grid(g, 4.0, (37, 50)) - ref).max() < 1e-5


@pytest.fixture(scope="module")
def synth_halves():
    frames, halves, tr = DS.make_stack(n=5, h=384, w=576, handheld=False, uneven=False, seed=2)
    return halves


@pytest.mark.parametrize("tile,step", [(16, 4), (48, 16)])
def test_fast_offsets_match_the_first_implementation(synth_halves, tile, step):
    for la, lb in synth_halves[1:4]:
        d1, c1 = DEP.disparity(la, lb, tile, step)
        d0, c0 = DEP.disparity_reference(la, lb, tile, step)
        sure = c0 > 0.05
        assert np.abs(d1 - d0)[sure].max() < 0.02
        assert np.abs(c1 - c0).max() < 1e-3


# ------------------------------------------------------------------------------------------ A.1 near / far
def _receding(sign: int, flip: int, h=300, w=450, slope=6.0, noise=0.2, seed=0):
    """Offsets of a ground plane receding up the UPRIGHT frame (the bottom nearer), turned back into the sensor's
    frame for the given flip; ``sign`` = the sign of the offset in front of focus."""
    from studio.engine.pipeline.pipeline import apply_orientation
    rng = np.random.default_rng(seed)
    H, W = (w, h) if flip in (5, 6) else (h, w)
    y = np.linspace(-0.5, 0.5, H)[:, None] * np.ones((1, W))
    x = np.linspace(-0.5, 0.5, W)[None, :] * np.ones((H, 1))
    d_up = sign * slope * y + 0.8 * x + rng.normal(0, noise, (H, W))
    c_up = np.full((H, W), 0.8)
    c_up[: H // 6] = 0.0                              # sky: nothing measured
    # sensor frame: undo the orientation (apply_orientation(sensor) = upright)
    inv = {0: 0, 3: 3, 5: 6, 6: 5}[flip]
    d = np.asarray(apply_orientation(d_up.astype(np.float32), inv))
    c = np.asarray(apply_orientation(c_up.astype(np.float32), inv))
    assert np.allclose(apply_orientation(d, flip), d_up, atol=1e-5)
    return d, c


@pytest.mark.parametrize("flip", [0, 3, 5, 6])
@pytest.mark.parametrize("sign", [1, -1])
def test_a_receding_plane_votes_the_right_way_however_the_camera_was_held(sign, flip):
    d, c = _receding(sign, flip)
    fv = NC.frame_vote(d, c, flip)
    assert fv.vote == sign, fv


def test_a_flat_or_confused_frame_abstains():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1.0, (300, 450)).astype(np.float32)
    assert NC.frame_vote(d, np.full_like(d, 0.8), 0).vote == 0              # no steady change
    d2, _ = _receding(1, 0)
    assert NC.frame_vote(d2, np.zeros_like(d2), 0).vote == 0               # nothing measured
    # a hill in the middle (offsets up then down): not a receding plane
    y = np.linspace(-1, 1, 300)[:, None] * np.ones((1, 450))
    hill = (4.0 * (1 - 3 * y * y)).astype(np.float32)
    assert NC.frame_vote(hill, np.full_like(hill, 0.8), 0).vote == 0


def test_the_camera_is_locked_only_by_three_scenes_with_none_against():
    def stack(v):
        return NC.stack_vote([NC.FrameVote("a", v), NC.FrameVote("b", v), NC.FrameVote("c", 0)], "s")
    assert NC.decide([stack(-1), stack(-1), stack(-1), stack(0)])["locked"]
    assert NC.decide([stack(-1), stack(-1), stack(-1)])["sign"] == -1
    assert not NC.decide([stack(-1), stack(-1)])["locked"]                  # only two
    assert not NC.decide([stack(-1), stack(-1), stack(-1), stack(1)])["locked"]   # one against
    mixed = NC.stack_vote([NC.FrameVote("a", 1), NC.FrameVote("b", -1)], "m")
    assert mixed.vote == 0                                                  # a stack that disagrees abstains


def test_a_locked_answer_goes_to_the_cameras_profile_and_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    DPP.clear_cache()
    try:
        assert DEP.near_source("Canon EOS Test")[2] == "assumed"
        st = [NC.stack_vote([NC.FrameVote("f", 1)], f"s{i}") for i in range(3)]
        res = {"camera": "Canon EOS Test", "stacks": [s.to_dict() for s in st], "decision": NC.decide(st),
               "date": "2026-11-21"}
        path = NC.save_to_profile(res)
        assert path and os.path.isfile(path) and str(tmp_path) in path
        sign, known, tier, how = DEP.near_source("Canon EOS Test")
        assert (sign, known, tier) == (1, True, "measured") and "3" in how
        # an unsettled vote saves nothing
        st2 = [NC.stack_vote([NC.FrameVote("f", 1)], "s0")]
        assert NC.save_to_profile({"camera": "Canon EOS Other", "stacks": [s.to_dict() for s in st2],
                                   "decision": NC.decide(st2)}) is None
    finally:
        DPP.clear_cache()


def test_the_r5_is_known_from_bills_scenes():
    sign, known, tier, _ = DEP.near_source("Canon EOS R5")
    assert known and sign == -1 and tier in ("table", "measured", "shared")


# ------------------------------------------------------------------------------------------ A.2 focus distance
def test_focus_distance_is_read_when_the_lens_records_it_and_absent_when_not():
    from studio.engine.io.metadata import canon_focus_distance
    si = [0] * 19 + [123, 98, 64]
    assert canon_focus_distance(tuple(si)) == (1.23, 0.98)
    assert canon_focus_distance(tuple([0] * 23)) == (None, None)            # the Makro-Planar ZE on the R5
    up, lo = canon_focus_distance(tuple([0] * 19 + [65535, 500]))
    assert up == float("inf") and lo == 5.0


# ------------------------------------------------------------------------------------------ A.3-A.6 in a stack
def _install(monkeypatch, tag, halves_mask=None, **kw):
    frames, halves, tr = DS.make_stack(**kw)
    if halves_mask is not None:
        halves = [h if keep else None for h, keep in zip(halves, halves_mask)]
    paths = DS.install(monkeypatch, frames, halves, tag)
    return paths, tr


def test_frames_developed_several_at_once_give_the_same_analysis(monkeypatch):
    paths, tr = _install(monkeypatch, "par", n=6, h=384, w=576, seed=5)
    monkeypatch.setenv("STUDIO_STACK_WORKERS", "1")
    a1 = FS.analyse(paths, FS.FocusParams(order="capture"))
    monkeypatch.setenv("STUDIO_STACK_WORKERS", "3")
    a2 = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        assert a1.paths == a2.paths
        assert np.array_equal(np.asarray(a1.aligned), np.asarray(a2.aligned))
        assert np.array_equal(np.asarray(a1.dp_disp), np.asarray(a2.dp_disp))
        assert np.array_equal(a1.idx, a2.idx)
    finally:
        a1.close()
        a2.close()


def test_a_second_analysis_comes_from_the_cache_with_the_same_depth(monkeypatch, tmp_path):
    from studio.engine.stack import anacache as AC
    monkeypatch.setenv("STUDIO_ANALYSIS_CACHE", str(tmp_path / "anacache"))
    paths, tr = _install(monkeypatch, "cache", n=6, h=384, w=576, seed=6)
    a1 = FS.analyse(paths, FS.FocusParams(order="capture"), cache=True)
    r1 = DF.fuse(a1)
    a1.close()
    assert AC.summary()["analyses"] == 1
    a2 = FS.analyse(paths, FS.FocusParams(order="capture"), cache=True)
    try:
        assert a2.from_cache and any("cache" in n for n in a2.notes)
        r2 = DF.fuse(a2)
        assert np.allclose(r1.depth, r2.depth, atol=1e-6)
        assert r1.step_px == pytest.approx(r2.step_px)
        # a merge setting does not change the key; an analysis setting does
        assert AC.key_for(a2.paths, FS.FocusParams(order="capture", method="pyramid").analysis_key()) == \
            AC.key_for(a2.paths, FS.FocusParams(order="capture").analysis_key())
        assert AC.key_for(a2.paths, FS.FocusParams(order="capture", radius=2.0).analysis_key()) != \
            AC.key_for(a2.paths, FS.FocusParams(order="capture").analysis_key())
    finally:
        a2.close()
    assert AC.summary()["analyses"] == 1                                    # closing a cached analysis keeps it
    assert AC.trim(cap=0.0) == 1 and AC.summary()["analyses"] == 0


def test_a_mixed_stack_uses_the_dual_pixel_frames_it_has(monkeypatch):
    mask = [i % 2 == 0 for i in range(9)]                                    # frames 0, 2, 4, 6, 8 Dual Pixel
    paths, tr = _install(monkeypatch, "mixed", halves_mask=mask, n=9, h=512, w=768, seed=7)
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        assert an.dual_pixel
        assert sum(an.dp_frames) == 5
        assert any("Dual Pixel on 5 of 9" in n for n in an.notes)
        res = DF.fuse(an)
        assert res.method == "focus+dual_pixel"
        z, flat, moved, meas = DS.half_truth(tr, an)
        # the frames without halves give nothing (confidence 0), the fusion still measures two ways
        assert (res.label == DF.LABEL_BOTH).mean() > 0.2
        e = (res.depth - z)[meas & (res.label == DF.LABEL_BOTH)]
        e = e - np.median(e)
        assert float(np.sqrt(np.mean(e * e))) < 0.35
    finally:
        an.close()


# ------------------------------------------------------------------------------------------ the command line
def _run(args, capsys):
    import json
    from studio import cli
    capsys.readouterr()
    code = cli.main([str(a) for a in args] + ["--json"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, f"--json must print exactly one line, got {len(out)}"
    return code, json.loads(out[0])


def test_near_check_from_the_command_line_votes_and_saves(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    DPP.clear_cache()
    files = []
    for i in range(6):
        p = tmp_path / f"IMG_{i:04d}.CR3"
        p.write_bytes(b"")
        files.append(str(p))
    # three scenes, two frames each, a minute apart; the halves of every frame see a receding plane (near < 0)
    times = {f: f"2026:11:21 15:{10 + i // 2:02d}:{i % 2:02d}" for i, f in enumerate(files)}
    import types
    from studio.engine.stack import frames as F
    monkeypatch.setattr(F, "probe", lambda ps: [F.FrameInfo(path=p, name=os.path.basename(p), taken=times[p])
                                                for p in ps])
    monkeypatch.setattr(NC, "measure_frame", lambda p, gains=None: (*_receding(-1, 0, seed=hash(p) % 100), 0,
                                                                    2.0, None))
    from studio.engine.io import metadata as MD
    monkeypatch.setattr(MD, "read_metadata", lambda p, *a, **k: types.SimpleNamespace(model="Canon EOS Test"))
    try:
        code, res = _run(["dualpixel", *files, "--make", "near-check"], capsys)
        assert code == 0, res["problems"]
        o = res["outputs"][0]
        assert o["kind"] == "dp_near_check" and o["sign"] == -1 and o["locked"] and o["scenes_for"] == 3
        assert o["saved_to"] and str(tmp_path) in o["saved_to"]
        assert DEP.near_source("Canon EOS Test")[:3] == (-1, True, "measured")
    finally:
        DPP.clear_cache()


def test_stacks_lists_and_runs_every_stack_and_carries_on_after_a_stop(tmp_path, capsys, monkeypatch):
    import json
    from studio.engine.stack import batch as B
    frames, halves, tr = DS.make_stack(n=5, h=320, w=480, seed=8)
    paths = DS.install(monkeypatch, frames, halves, "stacks_a")
    single = tmp_path / "IMG_LONE.CR3"
    single.write_bytes(b"")
    burst = [str(tmp_path / f"IMG_B{i}.CR3") for i in range(3)]
    stacks = [B.StackInfo("A", "focus", paths, dual_pixel=5), B.StackInfo("IMG_LONE", "single", [str(single)]),
              B.StackInfo("B", "burst", burst)]
    monkeypatch.setattr(B, "find_stacks", lambda items, *a, **k: list(stacks))
    code, res = _run(["stacks", tmp_path], capsys)
    assert code == 0 and res["data"]["focus_stacks"] == 1 and len(res["outputs"]) == 3
    calls = []
    real = FS.analyse

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(FS, "analyse", counting)
    out = tmp_path / "out"
    code, res = _run(["stacks", tmp_path, "--run", "depth", "--out", out, "--set", "order=capture"], capsys)
    assert code == 0, res["problems"]
    names = sorted(os.path.basename(o["output"]) for o in res["outputs"])
    base = "3C9A9000_to_3C9A9004"
    assert names == sorted(f"{base}_{k}" for k in ("agreement_Studio.png", "confidence_Studio.png",
                                                    "height_Studio.png", "surface_Studio.json"))
    job = json.load(open(out / B.JOB_FILE, encoding="utf-8"))
    st = {it["stack"]: it["status"] for it in job["items"].values()}
    assert st == {"A": "done", "IMG_LONE": "skipped", "B": "skipped"} and job["state"] == "done"
    assert len(calls) == 1
    # a second run finds the job done: nothing is analysed again, the files are reported from the earlier run
    code, res = _run(["stacks", tmp_path, "--run", "depth", "--out", out], capsys)
    assert code == 0 and len(calls) == 1
    assert all(o.get("from_earlier_run") for o in res["outputs"]) and len(res["outputs"]) == 4
    # a stopped job (one item left pending) carries on with that item only
    k = next(k for k, it in job["items"].items() if it["stack"] == "A")
    job["items"][k]["status"] = "pending"
    job["state"] = "stopped"
    json.dump(job, open(out / B.JOB_FILE, "w", encoding="utf-8"))
    code, res = _run(["stacks", tmp_path, "--run", "depth", "--out", out, "--set", "order=capture"], capsys)
    assert code == 0 and len(calls) == 2
    code, res = _run(["stacks", tmp_path, "--run", "dpeth"], capsys)
    assert code == 2 and "depth" in res["problems"][0]["text"]


# ------------------------------------------------------------------------------------------ B: macro gains
def _band(mask, width):
    from scipy.ndimage import binary_dilation, binary_erosion
    return binary_dilation(mask, iterations=width) & ~binary_erosion(mask, iterations=width)


def test_measured_depth_merge_has_fewer_halos_at_a_bright_depth_edge(monkeypatch):
    paths, tr = _install(monkeypatch, "halo", n=9, h=512, w=768, seed=9, bright_step=True)
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        y0, y1, x0, x1 = an.box_half
        truth = A.bin_by(tr["tex"], 2)[y0:y1, x0:x1]
        z = A.bin_by(tr["t"], 2)[y0:y1, x0:x1]
        step = (np.abs(np.gradient(z, axis=0)) + np.abs(np.gradient(z, axis=1))) > 0.15
        band = _band(step, 6)
        away = ~_band(step, 20)
        away[:16], away[-16:], away[:, :16], away[:, -16:] = False, False, False, False
        old = FS.merge(an, FS.FocusParams(order="capture", method="auto", auto_measured=False))
        new = FS.merge(an, FS.FocusParams(order="capture", method="measured"))
        assert FS.resolve_method(an, FS.FocusParams()) == "measured"            # the default for Dual Pixel
        e_old = float(np.mean((old - truth)[band] ** 2))
        e_new = float(np.mean((new - truth)[band] ** 2))
        assert e_new < e_old, (e_new, e_old)

        def psnr(x):
            return -10 * np.log10(float(np.mean((x - truth)[away] ** 2)))
        assert psnr(new) >= psnr(old) - 0.3
    finally:
        an.close()


def test_a_part_that_moved_comes_from_one_frame_without_a_seam(monkeypatch):
    paths, tr = _install(monkeypatch, "ghost", n=9, h=512, w=768, seed=10, moved={4: 1.5, 5: 1.5, 6: 1.5})
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        res = DF.fuse(an)
        K, mk = FS.moving_frames(an, res)
        y0, y1, x0, x1 = an.box_half
        mv = A.bin_by(tr["moved"].astype(np.float32), 2)[y0:y1, x0:x1] > 0.5
        assert (K[mv] >= 0).mean() > 0.3                                      # the moved patch is found
        assert (K[~mv] >= 0).mean() < 0.02                                    # and little else
        on = FS.merge(an, FS.FocusParams(order="capture", method="depth", moving_single=True))
        off = FS.merge(an, FS.FocusParams(order="capture", method="depth", moving_single=False))
        core = mv & (mk > 0.99)
        assert core.sum() > 100
        src = np.stack([np.asarray(an.aligned[j], np.float32) for j in range(an.n)])
        pick = np.take_along_axis(src, np.clip(K, 0, an.n - 1)[None, ..., None].astype(np.int64), 0)[0]
        from scipy.ndimage import gaussian_filter

        def hf(x):
            g = A.luma(x)
            return g - gaussian_filter(g, 2.0)
        ghost_on = float(np.mean((hf(on) - hf(pick))[core] ** 2))
        ghost_off = float(np.mean((hf(off) - hf(pick))[core] ** 2))
        assert ghost_on < 0.05 * max(ghost_off, 1e-12)                        # one frame there, not a blend
        reg = mk > 0.5
        border = _band(reg, 3)
        rest = ~_band(reg, 12)
        rest[:16], rest[-16:], rest[:, :16], rest[:, -16:] = False, False, False, False
        gy, gx = np.gradient(A.luma(on))
        g2 = gy * gy + gx * gx
        gy0, gx0 = np.gradient(A.luma(off))
        g20 = gy0 * gy0 + gx0 * gx0
        # no seam: the gradient energy along the region's edge rises no more than 20% over the plain merge's
        assert float(np.mean(g2[border])) < 1.2 * float(np.mean(g20[border]))
    finally:
        an.close()


def test_the_gap_left_by_a_missing_frame_is_found_and_the_coach_says_where(monkeypatch):
    frames, halves, tr = DS.make_stack(n=9, h=512, w=768, seed=11, uneven=False)
    keep = [i for i in range(9) if i != 4]
    paths = DS.install(monkeypatch, [frames[i] for i in keep], [halves[i] for i in keep], "gap")
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        res = DF.fuse(an)
        assert res.dof_half is not None and 0.2 < res.dof_half < 1.0
        q = np.asarray(res.positions)
        jump = int(np.argmax(np.diff(q)))
        assert jump == 3 and np.diff(q)[jump] > 1.6                            # frames 4 and 5 (frame 5 is gone)
        assert res.gap is not None and res.gap.mean() > 0.01
        z = res.depth[res.gap]
        inside = (z > q[jump]) & (z < q[jump + 1])
        assert inside.mean() > 0.9                                             # the gap lies at the right depth
        assert any("frames 4 and 5" in c for c in res.coach)
        full = DS.make_stack  # noqa: F841  (the full stack has no such gap)
    finally:
        an.close()


def test_the_depth_travels_with_the_result_and_develop_selects_by_measured_distance(monkeypatch, tmp_path):
    from studio.engine import api
    from studio.engine.pipeline.edit import Edit
    from studio.engine.pipeline.pipeline import depth_shape, measured_depth
    paths, tr = _install(monkeypatch, "travel", n=6, h=384, w=576, seed=12)
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        out = FS.build(an, FS.FocusParams(order="capture"), str(tmp_path / "r_Focus_Studio.tif"))
    finally:
        an.close()
    side = FS.depth_sidecar_path(out)
    assert os.path.isfile(side)
    doc = api.load_doc(out)
    assert doc["stack"]["depth"]["file"] == os.path.basename(side)
    assert doc["stack"]["depth"]["source"] == "measured"
    img = api.open(out)
    md = measured_depth(img)
    assert md is not None and md["depth"].dtype == np.float32
    assert 0.0 <= float(md["depth"].min()) and float(md["depth"].max()) <= 1.0
    sh = depth_shape(img, Edit())
    assert sh["source"] == "measured" and sh["type"] == "depth"


# ------------------------------------------------------------------------------------------ C: surfaces for games
def _bumps(h=256, w=256):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.zeros((h, w), np.float32)
    z += 0.6 * np.exp(-((y - 80) ** 2 + (x - 80) ** 2) / (2 * 18.0 ** 2))          # a bump
    z -= 0.6 * np.exp(-((y - 170) ** 2 + (x - 170) ** 2) / (2 * 18.0 ** 2))         # a pit
    z[:, 200:204] -= 0.25                                                          # a narrow groove
    return z


def test_ao_cavity_and_curvature_follow_the_shape():
    from studio.engine.stack import maps as MP
    z = _bumps()
    ao = MP.ambient_occlusion(z, 20.0, radius=40)
    flat = float(np.median(ao[20:40, 150:190]))
    assert ao[170, 170] < flat - 0.1                                   # the bottom of the pit is occluded
    assert ao[80, 80] > flat - 0.02                                    # the top of the bump is open
    cav = MP.cavity(z, 20.0)
    assert cav[128, 201] < 0.6 < float(np.median(cav[20:40, 150:190]))  # the groove is a crevice
    c = MP.curvature(z, 20.0, 3.0)
    assert c[80, 80] > 0 > c[170, 170]
    t = MP.ambient_occlusion(np.tile(z, (1, 1)), 20.0, radius=40, wrap=True)
    assert t.shape == z.shape


def test_de_lighting_recovers_the_colour_under_a_known_light():
    from scipy.ndimage import gaussian_filter
    from studio.engine.stack import maps as MP
    from studio.engine.stack.surface import normals
    rng = np.random.default_rng(3)
    h = w = 384
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 0.5 * np.sin(x / 23.0) * np.cos(y / 31.0) + 0.4 * np.exp(-((y - 190) ** 2 + (x - 200) ** 2) / 3000.0)
    alb = 0.35 + 0.25 * gaussian_filter(rng.random((h, w)), 6)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)
    alb = alb + 0.05 * gaussian_filter(rng.standard_normal((h, w, 3)), (1, 1, 0))      # fine texture
    alb = np.clip(alb, 0.02, 1).astype(np.float32)
    slope = 40.0
    n = normals(z.astype(np.float32), slope, True)
    az, el = np.radians(130.0), np.radians(40.0)
    L = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32)
    shade = 0.25 + 0.9 * np.maximum(n @ L, 0.0)
    img = np.clip(alb * shade[..., None], 0, 1).astype(np.float32)
    got, fit, sh = MP.delight(img, z.astype(np.float32), slope, sigma=3.0)
    assert fit["ok"] and fit["r2"] > 0.5

    def err(a):                                      # up to one overall brightness
        k = float(np.sum(a * alb) / np.sum(a * a))
        return float(np.sqrt(np.mean((k * a - alb) ** 2)))
    assert err(got) < 0.5 * err(img), (err(got), err(img))
    d = np.degrees(np.arccos(np.clip(np.dot(np.asarray(fit["dir"]), L), -1, 1)))
    assert d < 25, d


def _bricks(h=250, w=370):
    img = np.full((h, w, 3), (0.55, 0.25, 0.18), np.float32)
    for r in range(7, h, 32):
        img[r:r + 4] = 0.8                                                # mortar rows
        off = 11 if (r // 32) % 2 == 0 else 43
        for c in range(off, w + 64, 64):
            img[r:r + 32, max(0, c):max(0, c + 4)] = 0.8                  # mortar joints
    img += 0.03 * np.random.default_rng(2).standard_normal(img.shape).astype(np.float32)
    return img


def test_the_seam_cut_tiling_repeats_without_a_join_and_without_ghosts():
    from studio.engine.stack import maps as MP
    from studio.engine.stack import surface as SURF
    img = _bricks()
    H, W = img.shape[:2]
    cw = MP.cut_weights_for(img)
    t = MP.tile_cut(img, cw)
    # the join where the texture repeats must not stand out from the picture's own neighbouring columns / rows
    cols = np.mean(np.abs(np.diff(t, axis=1)), axis=(0, 2))
    rows = np.mean(np.abs(np.diff(t, axis=0)), axis=(1, 2))
    wrap_x = float(np.mean(np.abs(t[:, 0] - t[:, -1])))
    wrap_y = float(np.mean(np.abs(t[0] - t[-1])))
    assert wrap_x < 2.0 * float(np.median(cols)) and wrap_y < 2.0 * float(np.median(rows)), (wrap_x, wrap_y)
    # ghosts: a pixel that is neither copy is a blend of two (a doubled mortar line); the cut has almost none
    copies = [img, np.roll(img, -(W // 2), 1), np.roll(img, -(H // 2), 0), np.roll(img, (-(H // 2), -(W // 2)), (0, 1))]
    near = np.min([np.max(np.abs(t - c), axis=2) for c in copies], axis=0)
    w = SURF.tile_weights(H, W)
    b = SURF.tile(img, w)
    near_b = np.min([np.max(np.abs(b - c), axis=2) for c in copies], axis=0)
    assert float((near > 0.05).mean()) < 0.3 * float((near_b > 0.05).mean())


def test_a_coin_gives_the_size_of_a_pixel():
    from scipy.ndimage import gaussian_filter
    from studio.engine.stack import maps as MP
    rng = np.random.default_rng(5)
    h, w = 900, 1300
    img = np.clip(0.35 + 0.15 * gaussian_filter(rng.standard_normal((h, w, 3)), (2, 2, 0)), 0, 1)
    y, x = np.mgrid[0:h, 0:w]
    r = 120.0
    disc = ((y - 420) ** 2 + (x - 700) ** 2) <= r * r
    img[disc] = (0.75, 0.74, 0.72)
    img = gaussian_filter(img, (1.0, 1.0, 0)).astype(np.float32)
    c = MP.find_coin(img, "us_quarter")
    assert c is not None
    assert abs(c["radius_px"] - r) / r < 0.02
    assert abs(c["mm_per_px"] - 24.26 / (2 * r)) / (24.26 / (2 * r)) < 0.02
    assert abs(c["centre"][0] - 700 / w) < 0.01 and abs(c["centre"][1] - 420 / h) < 0.01
    assert MP.scale_from_click(0.1, 0.5, 0.6, 0.5, 50.0, 1000, 800) == pytest.approx(0.1)
    m = MP.magnification_from_distance(50.0, 0.3)                # a 50 mm lens focused at 30 cm
    assert 0.15 < m < 0.35


def test_every_map_says_how_it_was_made_and_the_game_size_and_godot_notes(monkeypatch, tmp_path):
    import json
    from PIL import Image
    from studio.engine.stack import surface as SURF
    paths, tr = _install(monkeypatch, "maps", n=6, h=384, w=576, seed=13)
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        sp = SURF.SurfaceParams(maps="all", size=256, tileable=True, tile_method="cut")
        r = SURF.stack_surface(an, str(tmp_path), "m", sp, SURF.kinds_from(sp))
    finally:
        an.close()
    doc = r["json"]
    assert doc["studio_surface"] == 2
    for k in ("studio_surface", "written_by", "method", "source", "camera", "size", "files", "height", "lateral",
              "displacement", "normal", "dual_pixel", "coverage", "agreement", "tileable", "settings", "notes"):
        assert k in doc, k                                          # every key of version 1 is kept
    want = set(SURF.ALL_KINDS) - {"roughness"}
    assert set(doc["maps"]) == want and set(r["paths"]) == want | {"json"}
    for k, v in doc["maps"].items():
        assert v["how"] and v["colour"] in ("cyan", "amber", "violet", "none") and v["space"] in ("srgb", "linear")
    assert doc["maps"]["albedo"]["colour"] == "amber" and doc["maps"]["ao"]["colour"] == "cyan"
    for k in want:
        assert Image.open(r["paths"][k]).size == (256, 256)
    orm = np.asarray(Image.open(r["paths"]["orm"]))
    assert orm.shape == (256, 256, 3) and int(orm[..., 2].max()) == 0         # metallic 0
    assert doc["godot"]["heightmap_flip"] is False and doc["godot"]["normal_map"].startswith("OpenGL")
    assert doc["godot"]["textures"]["albedo"] == "srgb" and doc["godot"]["textures"]["height"] == "linear"
    assert doc["game_size"]["size"] == 256 and doc["tiling"]["method"] == "cut"
    assert "coverage" in doc and "gap" in doc["coverage"]
    assert json.load(open(r["paths"]["json"], encoding="utf-8"))["maps"]["detail"]["colour"] == "amber"


def test_a_subject_is_cut_out_by_its_measured_depth_not_its_colour():
    from studio.engine.stack import depthmask as DM
    h, w = 200, 300
    y, x = np.mgrid[0:h, 0:w]
    leaf = ((y - 100) / 60.0) ** 2 + ((x - 140) / 90.0) ** 2 <= 1.0
    depth = np.where(leaf, 0.8 + 0.05 * np.sin(x / 15.0), 0.2).astype(np.float32)
    label = np.where(leaf, 2, 0).astype(np.uint8)
    label[150:190, 220:290] = 1                                    # a scrap of measured litter far behind
    depth[150:190, 220:290] = 0.15
    label[95:105, 130:150] = 0                                     # a plain patch on the leaf: not measured
    rng = np.random.default_rng(1)
    guide = (0.4 + 0.05 * rng.standard_normal((h, w))).astype(np.float32)   # the same colour everywhere
    a, info = DM.subject(depth, label, guide)
    assert info["found"]
    iou = float(((a > 0.5) & leaf).sum()) / float(((a > 0.5) | leaf).sum())
    assert iou > 0.9                                               # the leaf, its plain patch included
    assert a[170, 250] < 0.1                                       # the litter is not the subject
    a2, info2 = DM.subject(depth, label, guide, click=(250 / w, 170 / h))
    assert a2[170, 250] > 0.5 and a2[100, 140] < 0.5                # clicking the litter picks the litter


# ------------------------------------------------------------------------------------------ F: glTF, OBJ, PLY
def test_a_glb_reads_back_with_every_required_field_and_its_textures(tmp_path):
    from studio.engine.io import gltf as G
    n = 9
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    pos = np.stack([u.ravel() - 0.5, 0.05 * np.sin(6 * u.ravel()), v.ravel() - 0.5], 1).astype(np.float32)
    tri = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            tri += [(a, a + n, a + 1), (a + 1, a + n, a + n + 1)]
    uvs = np.stack([u.ravel(), v.ravel()], 1).astype(np.float32)
    col = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
    nrm = np.full((32, 32, 3), (128, 128, 255), np.uint8)
    mat = G.Material("m", base_colour=col, normal=nrm, orm=np.zeros((32, 32, 3), np.uint8))
    lod1 = G.Mesh("thing_LOD1", pos, np.asarray(tri)[::2], uvs=uvs, material=0)
    p = G.write_glb(str(tmp_path / "t_Studio.glb"), [G.Mesh("thing", pos, np.asarray(tri), uvs=uvs, material=0),
                                                      lod1], [mat], extras={"studio": "test"})
    doc, binc = G.read_glb(p)
    assert G.validate(doc, binc) == []
    prim = doc["meshes"][0]["primitives"][0]
    assert set(prim["attributes"]) == {"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT"}   # a normal map: tangents
    P = G.accessor_array(doc, binc, prim["attributes"]["POSITION"])
    assert np.allclose(P, pos) and len(doc["images"]) == 3 and doc["meshes"][1]["name"] == "thing_LOD1"
    T = G.accessor_array(doc, binc, prim["attributes"]["TANGENT"])
    N = G.accessor_array(doc, binc, prim["attributes"]["NORMAL"])
    assert np.allclose(np.linalg.norm(T[:, :3], axis=1), 1, atol=1e-4)
    assert np.abs(np.sum(T[:, :3] * N, axis=1)).max() < 1e-3                 # tangents lie in the surface
    assert doc["asset"]["extras"] == {"studio": "test"} and doc["materials"][0]["occlusionTexture"]["index"] == 2
    o = G.write_obj(str(tmp_path / "t_Studio.obj"), G.Mesh("thing", pos, np.asarray(tri), uvs=uvs),
                    {"colour": "c.png"})
    V, F = G.read_obj(o[0])
    assert np.allclose(V, pos, atol=1e-5) and F.shape == (len(tri), 3) and os.path.isfile(o[1])
    q = G.write_ply(str(tmp_path / "t_Studio.ply"), pos, (pos * 0 + 200).astype(np.uint8),
                    np.full(len(pos), 2, np.uint8))
    rec = G.read_ply(q)
    assert len(rec) == len(pos) and np.allclose(rec["x"], pos[:, 0]) and set(rec["h"]) == {2}


def test_the_blender_script_needs_only_blender_and_the_standard_library():
    import ast
    import studio.blender
    p = os.path.join(os.path.dirname(studio.blender.__file__), "build_material.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"json", "math", "os", "sys", "bpy", "mathutils"}, mods


# ------------------------------------------------------------------------------------------ D: foliage
def _leaves(h=300, w=420):
    y, x = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(4)
    colour = np.clip(np.stack([0.35 + 0.1 * rng.random((h, w)), 0.3 + 0.1 * rng.random((h, w)),
                               0.25 + 0.05 * rng.random((h, w))], 2) * 255, 0, 255)
    leaf1 = ((y - 120) / 70.0) ** 2 + ((x - 120) / 50.0) ** 2 <= 1
    leaf2 = ((y - 170) / 60.0) ** 2 + ((x - 310) / 45.0) ** 2 <= 1
    colour[leaf1] = (60, 140, 50)
    colour[leaf2] = (70, 150, 40)
    depth = np.full((h, w), 0.1, np.float32)
    depth[leaf1] = 0.8 + 0.1 * np.cos((x[leaf1] - 120) / 30.0)                 # a cupped leaf
    depth[leaf2] = 0.6
    label = np.where(leaf1 | leaf2, 2, 0).astype(np.uint8)
    return depth, label, colour.astype(np.uint8), leaf1, leaf2


def test_two_leaves_make_two_cards_with_clean_edges_a_pivot_and_a_bent_mesh(tmp_path):
    import json
    from PIL import Image
    from studio.engine.io import gltf as G
    from studio.engine.stack import foliage as FO
    depth, label, colour, leaf1, leaf2 = _leaves()
    cards = FO.make_cards(depth, label, colour, FO.FoliageParams(), mm_px=0.05)
    assert len(cards) == 2
    c = cards[0]
    a = c.colour[..., 3]
    rgb = c.colour[..., :3].astype(np.float32)
    clear = a < 10
    assert clear.any() and float(rgb[clear].mean()) > 60                       # bled: no black margin
    assert abs(c.pivot[1] - 1.0) < 0.1                                          # the stem: the lowest point
    assert c.size_mm is not None and c.size_mm[0] > 4
    pos, uv, tri, col = FO.bent_mesh(c, 600)
    assert 150 <= len(tri) <= 900 and np.ptp(pos[:, 2]) > 0                     # bent, not flat
    assert col[np.argmin(np.hypot(pos[:, 0], pos[:, 1])), 0] < 0.2              # dark at the stem
    r = FO.write(str(tmp_path), "lf", cards, FO.FoliageParams(), FO.MAKES, {"model": "Canon EOS R5"},
                 {"kind": "test"})
    doc = json.load(open(r["json_path"], encoding="utf-8"))
    assert doc["studio_foliage"] == 1 and len(doc["leaves"]) == 2 and doc["atlas"]["size"] in (256, 512, 1024)
    sheet = np.asarray(Image.open(os.path.join(tmp_path, doc["atlas"]["files"]["colour"])))
    assert sheet.shape[2] == 4
    for rect in doc["atlas"]["cards"]:                                          # no overlap, inside the sheet
        u0, v0, u1, v1 = rect["uv"]
        assert 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1
    r0, r1 = doc["atlas"]["cards"][0]["uv"], doc["atlas"]["cards"][1]["uv"]
    assert r0[2] <= r1[0] or r1[2] <= r0[0] or r0[3] <= r1[1] or r1[3] <= r0[1]
    g = os.path.join(tmp_path, doc["leaves"][0]["glb"])
    gd, gb = G.read_glb(g)
    assert G.validate(gd, gb) == []
    m = gd["materials"][0]
    assert m["alphaMode"] == "MASK" and m["alphaCutoff"] == 0.5 and m["doubleSided"] is True
    assert "COLOR_0" in gd["meshes"][0]["primitives"][0]["attributes"]


def test_foliage_and_stacks_are_commands_and_tools(capsys):
    from studio import mcp_server as M
    names = {t["name"] for t in M.tools()}
    assert {"studio_foliage", "studio_stacks"} <= names
    code, res = _run(["foliage", "nothing_here.CR3", "--make", "leaves"], capsys)
    assert code == 2 and "cards" in res["problems"][0]["text"]


def test_blender_is_found_from_the_forge_read_only_and_the_command_is_exact(tmp_path, monkeypatch):
    import json
    from studio.engine.io import blender as BL
    forge = tmp_path / "RPG_Forge" / "forge" / "threed" / "blender"
    forge.mkdir(parents=True)
    exe = tmp_path / "blender.exe"
    exe.write_bytes(b"")
    (forge / "profiles.json").write_text(json.dumps({"blender_program": str(exe)}), encoding="utf-8")
    monkeypatch.setenv("STUDIO_FORGE_DIR", str(tmp_path / "RPG_Forge"))
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    assert BL.program() == (str(exe), "the RPG Forge's setting")
    cmd = BL.command(str(exe), "C:/x/a_surface_Studio.json", "C:/o", background=True)
    assert cmd[:4] == [str(exe), "--background", "--python", BL.SCRIPT] and cmd[-4:] == ["--surface",
                                                                                         "C:/x/a_surface_Studio.json",
                                                                                         "--out", "C:/o"]
    assert BL.command("b", "m_model_Studio.json")[-2:] == ["--model", "m_model_Studio.json"]
    assert os.path.isfile(BL.SCRIPT)


def test_without_a_coin_the_size_comes_from_the_lens_focus_distance_and_its_view(monkeypatch, tmp_path):
    from studio.engine.stack import surface as SURF
    paths, tr = _install(monkeypatch, "fdist", n=5, h=320, w=480, seed=14)
    an = FS.analyse(paths, FS.FocusParams(order="capture"))
    try:
        an.meta["focus_distance_m"] = 0.30                      # a lens that records where it is focused
        r = SURF.stack_surface(an, str(tmp_path), "fd", SURF.SurfaceParams(), SURF.DEPTH_KINDS)
    finally:
        an.close()
    doc = r["json"]
    assert doc["lateral"]["mm_per_pixel"] and "edge to edge" in doc["lateral"]["how_known"]
    assert doc["scale"]["colour"] == "amber" and len(doc["scale"]["view_mm"]) == 2


# ------------------------------------------------------------------------------------------ the Forge's texture folders
def test_one_ordinary_photo_gives_an_estimated_surface_labelled_amber_everywhere(tmp_path):
    """A texture shot without a stack or Dual Pixel: the height is estimated (dark taken as deep), follows the true
    shape where a crevice is darker, and every map made from it says amber, not measured."""
    import json
    from PIL import Image as PILImage
    from studio.engine.stack import surface as SURF
    from studio.engine.stack import maps as MP
    rng = np.random.default_rng(3)
    h, w = 256, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    true = 0.5 + 0.25 * np.sin(xx / 9.0) * np.cos(yy / 13.0) + 0.1 * uniform_filter(rng.random((h, w)), 9)
    # crevices darker (soft light: ambient occlusion makes low places dark), a little colour noise
    lum = np.clip(0.25 + 0.6 * true + 0.01 * rng.standard_normal((h, w)), 0, 1)
    img = np.dstack([lum * 0.9, lum * 0.8, lum * 0.6])
    est = MP.estimated_height(img)
    rho = np.corrcoef(est.ravel(), true.ravel())[0, 1]
    assert rho > 0.8
    p = tmp_path / "BARK_0001.png"
    PILImage.fromarray(np.rint(img * 255).astype(np.uint8)).save(p)
    r = SURF.photo_surface(str(p), str(tmp_path / "out"), "BARK_0001",
                           SURF.SurfaceParams(maps="all", tileable=True), SURF.kinds_from(SURF.SurfaceParams(maps="all")))
    doc = json.load(open(r["paths"]["json"], encoding="utf-8"))
    assert doc["method"] == "estimated" and doc["coverage"]["estimated"] == 1.0
    assert doc["coverage"]["measured_both"] == 0 and doc["coverage"]["measured_one"] == 0
    assert not any(k in doc["files"] for k in ("confidence", "agreement", "gaps"))
    for k in ("height", "normal", "ao", "cavity", "albedo", "orm"):
        assert doc["maps"][k]["colour"] == "amber", k
    assert "estimated" in doc["maps"]["height"]["how"] and doc["height"]["units"] == "relative"
    assert any("estimated" in n for n in doc["notes"])


def test_stacks_makes_estimated_surfaces_from_single_photos_and_a_bursts_sharpest_frame(tmp_path, monkeypatch):
    from studio.engine.stack import batch as B
    from studio.engine.stack import surface as SURF
    made = []

    def fake(path, out_dir, base, p, kinds, taken):
        made.append(os.path.basename(path))
        return {"paths": {"json": os.path.join(out_dir, base + "_surface_Studio.json")}, "json": {"notes": []}}
    monkeypatch.setattr(SURF, "photo_surface", fake)
    lone = B.StackInfo("IMG_LONE", "single", [str(tmp_path / "IMG_LONE.CR3")])
    burst = B.StackInfo("B", "burst", [str(tmp_path / f"IMG_B{i}.CR3") for i in range(3)],
                        best=str(tmp_path / "IMG_B1.CR3"))
    assert B._applies(lone, "surface") and B._applies(burst, "surface")
    assert not B._applies(lone, "depth") and not B._applies(lone, "foliage")
    doc = B.run([lone, burst], ["surface"], str(tmp_path / "out"), resume=False)
    assert sorted(made) == ["IMG_B1.CR3", "IMG_LONE.CR3"]
    assert all(it["status"] == "done" for it in doc["items"].values())

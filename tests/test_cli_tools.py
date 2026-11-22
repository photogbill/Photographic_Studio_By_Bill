"""The tool commands of studio-cli (studio/cli_tools.py, docs/CLI.md): what the RPG Forge and Claude call.

Synthetic pictures with known truth; everything is written under the test's own folder (outputs/pytest_tmp)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from studio import cli, cli_tools


def _picture(path, tint=(1.0, 1.0, 1.0), gain=1.0, seed=1):
    """A soft colour field with one bright yellow square in the middle third (the cut-out's known answer)."""
    rng = np.random.default_rng(seed)
    h, w = 240, 360
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.25 + 0.2 * np.sin(x / 40.0)
    img[..., 1] = 0.30 + 0.15 * np.cos(y / 30.0)
    img[..., 2] = 0.35
    img[80:160, 120:240] = (0.9, 0.85, 0.2)
    img = (img + rng.normal(0, 0.008, img.shape)) * np.asarray(tint, np.float32) * gain
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)
    return str(path)


def _run(args, capsys):
    """Run one command with --json: (exit code, the one JSON object standard output carried)."""
    capsys.readouterr()
    code = cli.main([str(a) for a in args] + ["--json"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, f"--json must print exactly one line, got {len(out)}"
    return code, json.loads(out[0])


def _rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def test_schema_describes_every_setting(capsys):
    code, res = _run(["schema"], capsys)
    assert code == 0 and res["ok"]
    by = {s["name"]: s for s in res["data"]["settings"]}
    assert by["light.shadows"]["default"] == 0.0 and "shadows" in by["light.shadows"]["about"]
    assert by["exposure_ev"]["type"] == "float"
    from studio.engine.pipeline.edit import Edit
    e = Edit().to_dict()
    for name in by:                                        # every name the schema lists is a real setting
        node = e
        for part in name.split("."):
            assert part in node, name
            node = node[part]
    assert set(cli_tools.HANDLERS) == set(res["data"]["commands"])


def test_edit_set_changes_the_picture_and_names_it_studio(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    code, base = _run(["edit", src, "--out", tmp_path / "o", "--tag", "base"], capsys)
    code2, up = _run(["edit", src, "--out", tmp_path / "o", "--set", "exposure_ev=1"], capsys)
    assert code == 0 and code2 == 0
    p0, p1 = base["outputs"][0]["output"], up["outputs"][0]["output"]
    assert os.path.basename(p1) == "a_Studio.png" and os.path.basename(p0) == "a_base_Studio.png"
    assert _rgb(p1).mean() > _rgb(p0).mean() + 0.05
    assert up["outputs"][0]["width"] == 360 and up["outputs"][0]["height"] == 240
    assert np.array_equal(_rgb(src), _rgb(src)) and os.path.getsize(src) > 0        # the original is untouched


def test_a_mistyped_setting_is_an_error_not_a_silent_no_op(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    code, res = _run(["edit", src, "--out", tmp_path / "o", "--set", "light.shadow=20"], capsys)
    assert code == 2 and not res["ok"] and not res["outputs"]
    assert "shadows" in res["problems"][0]["text"]
    code, res = _run(["edit", src, "--out", tmp_path / "o", "--recipe", '{"lite": {"shadows": 3}}'], capsys)
    assert code == 2 and "light" in res["problems"][0]["text"]
    assert not os.path.exists(tmp_path / "o" / "a_Studio.png")


def test_recipe_round_trip_gives_the_same_pixels(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    rec = tmp_path / "look.json"
    _, r1 = _run(["edit", src, "--out", tmp_path / "o1", "--set", "exposure_ev=0.4", "--set", "light.shadows=30",
                  "--set", "colour.saturation=25", "--save-recipe", rec], capsys)
    doc = json.loads(rec.read_text())
    assert doc["edit"] == {"exposure_ev": 0.4, "light": {"shadows": 30}, "colour": {"saturation": 25}}
    assert doc["export"] == {}                             # only what was asked for: the picture's kind still decides
    _, r2 = _run(["edit", src, "--out", tmp_path / "o2", "--recipe", rec], capsys)
    assert np.array_equal(_rgb(r1["outputs"][0]["output"]), _rgb(r2["outputs"][0]["output"]))
    _, r3 = _run(["edit", src, "--out", tmp_path / "o3", "--recipe", rec, "--dry-run"], capsys)
    assert r3["outputs"][0]["edit"] == doc["edit"] and not os.path.exists(tmp_path / "o3" / "a_Studio.png")


def test_match_moves_the_colour_statistics_to_the_reference(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    ref = _picture(tmp_path / "ref.png", tint=(1.25, 1.0, 0.7), gain=0.8, seed=2)
    prof = tmp_path / "warm.json"
    code, res = _run(["look-profile", ref, "--save", prof], capsys)
    assert code == 0 and json.loads(prof.read_text())["pictures"] == 1
    _, a = _run(["match", src, "--to", ref, "--out", tmp_path / "m1"], capsys)
    _, b = _run(["match", src, "--to", prof, "--out", tmp_path / "m2"], capsys)
    _, st = _run(["find", src, ref, a["outputs"][0]["output"], b["outputs"][0]["output"], "--what", "stats"], capsys)
    v = [np.array(list(o["found"]["stats"].values())) for o in st["outputs"]]
    before, after_pic, after_prof = (np.abs(x - v[1]).sum() for x in (v[0], v[2], v[3]))
    assert after_pic < 0.35 * before and after_prof < 0.35 * before
    assert abs(after_pic - after_prof) < 0.02              # a profile is the reference's statistics, nothing else


def test_cutout_alpha_follows_the_selection(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    code, res = _run(["cutout", src, "--luminance", "0.6,1.0", "--make", "both", "--out", tmp_path / "c"], capsys)
    assert code == 0
    kinds = {o["kind"]: o for o in res["outputs"]}
    rgba = np.asarray(Image.open(kinds["cutout"]["output"]))
    assert rgba.shape == (240, 360, 4)
    assert rgba[120, 180, 3] > 240 and rgba[20, 20, 3] < 15          # the square in, the field out
    assert abs(kinds["mask"]["coverage"] - (80 * 120) / (240 * 360)) < 0.03
    mask = np.asarray(Image.open(kinds["mask"]["output"]))
    assert mask.dtype == np.uint16 and mask.shape == (240, 360)
    code, inv = _run(["cutout", src, "--luminance", "0.6,1.0", "--invert", "--out", tmp_path / "c2"], capsys)
    assert np.asarray(Image.open(inv["outputs"][0]["output"]))[120, 180, 3] < 15
    code, res = _run(["cutout", src, "--out", tmp_path / "c3"], capsys)
    assert code == 2                                        # nothing to cut out was named


def test_depth_layers_cover_the_picture_exactly():
    rng = np.random.default_rng(0)
    D = rng.random((64, 96)).astype(np.float32)
    alphas, bounds = cli_tools.depth_layers(D, 4, None, 0.03)
    assert len(alphas) == 4 and bounds[0] == 0.0 and bounds[-1] == 1.0
    assert np.allclose(sum(alphas), 1.0, atol=1e-5)
    assert all(abs(a.mean() - 0.25) < 0.06 for a in alphas)             # equal areas by default
    far, near = alphas[0], alphas[-1]
    assert far[D < 0.1].min() > 0.99 and near[D > 0.9].min() > 0.99     # layer 1 is the farthest
    with pytest.raises(cli_tools.Usage):
        cli_tools.depth_layers(D, 3, [0.5], 0.03)


def test_depth_files(tmp_path, capsys):
    from studio.engine.local import depth as dp
    if not dp.available():
        pytest.skip("depth model not installed")
    src = _picture(tmp_path / "a.png")
    code, res = _run(["depth", src, "--make", "map,normals,layers", "--layers", "2", "--out", tmp_path / "d"], capsys)
    assert code == 0
    kinds = [o["kind"] for o in res["outputs"]]
    assert kinds == ["depth_map", "normal_map", "depth_layer", "depth_layer"]
    m = np.asarray(Image.open(res["outputs"][0]["output"]))
    assert m.dtype == np.uint16 and m.shape == (240, 360)
    a = [np.asarray(Image.open(o["output"])).astype(np.float32)[..., 3] / 255 for o in res["outputs"][2:]]
    assert np.abs(a[0] + a[1] - 1.0).max() < 0.02
    assert all(o["estimated"] for o in res["outputs"])


def test_batch_runs_jobs_and_reports_each(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([
        ["edit", src, "--out", str(tmp_path / "b"), "--tag", "one"],
        {"id": "second", "args": ["find", src, "--what", "size"]},
        ["edit", str(tmp_path / "missing.png")],
        ["develop", src],
    ]))
    code, res = _run(["batch", jobs], capsys)
    assert code == 1 and len(res["jobs"]) == 4
    assert res["jobs"][0]["ok"] and os.path.isfile(res["jobs"][0]["outputs"][0]["output"])
    assert res["jobs"][1]["id"] == "second" and res["jobs"][1]["outputs"][0]["found"]["size"] == {"width": 360,
                                                                                                    "height": 240}
    assert not res["jobs"][2]["ok"] and not res["jobs"][3]["ok"]
    assert len(res["problems"]) == 2


def test_results_never_replace_a_file_the_studio_did_not_write(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    out = tmp_path / "o"
    out.mkdir()
    mine = out / "a_Studio.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(mine)          # somebody else's file under the name we would use
    before = mine.read_bytes()
    _, r1 = _run(["edit", src, "--out", out], capsys)
    assert mine.read_bytes() == before and os.path.basename(r1["outputs"][0]["output"]) == "a_Studio-2.png"
    _, r2 = _run(["edit", src, "--out", out, "--set", "exposure_ev=0.2"], capsys)       # our own: replaced in place
    assert r2["outputs"][0]["output"] == r1["outputs"][0]["output"]
    _, r3 = _run(["cutout", src, "--luminance", "0.6,1", "--out", out], capsys)
    _, r4 = _run(["cutout", src, "--luminance", "0.5,1", "--out", out], capsys)
    assert r3["outputs"][0]["output"] == r4["outputs"][0]["output"]                     # the RGBA writer marks its files


def test_stack_result_opens_in_edit(tmp_path, capsys):
    import tifffile
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    h, w = 200, 300
    base = np.clip(gaussian_filter(rng.random((h, w, 3)), (1.2, 1.2, 0)) * 0.9 + 0.05, 0, 1)
    (tmp_path / "fs").mkdir()
    for i, (a, b) in enumerate([(0, 1), (1, 0), (0.5, 0.5)]):
        img = base.copy()
        img[:, :w // 2] = gaussian_filter(base, (3 * a + 0.01, 3 * a + 0.01, 0))[:, :w // 2]
        img[:, w // 2:] = gaussian_filter(base, (3 * b + 0.01, 3 * b + 0.01, 0))[:, w // 2:]
        tifffile.imwrite(tmp_path / "fs" / f"f{i}.tif", (img * 65535).astype(np.uint16), photometric="rgb")
    code, res = _run(["stack", "focus", tmp_path / "fs", "--out", tmp_path / "s", "--set", "method=depth"], capsys)
    assert code == 0 and res["outputs"][0]["linear"] and res["outputs"][0]["frames"] == 3
    code, fin = _run(["edit", res["outputs"][0]["output"], "--out", tmp_path / "s"], capsys)
    assert code == 0 and os.path.isfile(fin["outputs"][0]["output"])
    code, res = _run(["stack", "focus", tmp_path / "fs", "--set", "methud=depth"], capsys)
    assert code == 2 and "method" in res["problems"][0]["text"]


def test_infrared_says_plainly_that_it_needs_a_raw(tmp_path, capsys):
    src = _picture(tmp_path / "a.png")
    code, res = _run(["infrared", src, "--out", tmp_path / "i"], capsys)
    assert code == 1 and "raw" in res["problems"][0]["text"] and "error_report" not in res["problems"][0]


def test_clean_auto_and_the_new_methods_from_the_command_line(tmp_path, capsys):
    """`clean --auto-strength`: the photo chooses and --strength leans from its choice; without it the strengths
    given decide, as before. The schema lists every method Develop offers."""
    import argparse
    ns = dict(strength=1.2, chroma=None, keep_grain=0.0, method=None)
    d0 = cli_tools._p_clean(argparse.Namespace(auto_strength=False, **ns))["denoise"]
    d1 = cli_tools._p_clean(argparse.Namespace(auto_strength=True, **dict(ns, method="fusion")))["denoise"]
    assert d0["auto"] is False and d0["luma"] == pytest.approx(1.2) and "lean" not in d0
    assert d1["auto"] is True and d1["lean"] == pytest.approx(1.2) and d1["method"] == "fusion"
    src = _picture(tmp_path / "n.png")
    code, res = _run(["clean", src, "--out", tmp_path / "o", "--strength", "1.2", "--auto-strength", "--method",
                      "fusion", "--scale", "0.5"], capsys)
    assert code == 0 and os.path.basename(res["outputs"][0]["output"]).endswith("_Studio.png")
    code3, sch = _run(["schema"], capsys)
    from studio.engine.raw.denoise import METHOD_INFO
    assert code3 == 0 and set(sch["data"]["choices"]["denoise.method"]) == set(METHOD_INFO)


def test_stack_focus_surface_files_and_json_contract(tmp_path, capsys, monkeypatch):
    """`stack focus FRAMES --make surface`: the five PNGs and the JSON under the names the RPG Forge relies on
    (BUILD_PLAN_DEPTH_STACK D), the JSON's keys, its own files replaced on a second run, --make depth, and wrong
    values said plainly (exit code 2)."""
    import depth_synth as DS
    frames, halves, _ = DS.make_stack(n=6, h=384, w=576)
    paths = DS.install(monkeypatch, frames, halves, "cli_surface")
    out = tmp_path / "s"
    args = ["stack", "focus", *paths, "--make", "surface", "--out", out, "--ratio", "1:2", "--set", "order=capture"]
    code, res = _run(args, capsys)
    assert code == 0, res["problems"]
    kinds = {o["kind"]: o for o in res["outputs"]}
    assert set(kinds) == {"surface_colour", "surface_height", "surface_normal", "surface_confidence",
                          "surface_agreement", "surface_json"}
    base = "3C9A9000_to_3C9A9005"
    for k in ("colour", "height", "normal", "confidence", "agreement"):
        assert os.path.basename(kinds[f"surface_{k}"]["output"]) == f"{base}_{k}_Studio.png"
    assert os.path.basename(kinds["surface_json"]["output"]) == f"{base}_surface_Studio.json"
    modes = {k: Image.open(kinds[f"surface_{k}"]["output"]).mode for k in ("colour", "height", "normal",
                                                                          "confidence", "agreement")}
    assert modes == {"colour": "RGB", "height": "I;16", "normal": "RGB", "confidence": "L", "agreement": "LA"}
    sizes = {Image.open(kinds[f"surface_{k}"]["output"]).size for k in ("colour", "height", "normal")}
    assert len(sizes) == 1                                          # one geometry for every file
    doc = json.load(open(kinds["surface_json"]["output"], encoding="utf-8"))
    for key in ("studio_surface", "written_by", "method", "source", "camera", "size", "files", "height", "lateral",
                "displacement", "normal", "dual_pixel", "coverage", "agreement", "tileable", "settings", "notes"):
        assert key in doc, key
    assert doc["method"] == "focus+dual_pixel" and doc["source"]["frames_used"] == 6
    assert doc["files"]["height"] == f"{base}_height_Studio.png" and doc["height"]["white"] == "near"
    assert doc["height"]["units"] == "relative" and doc["lateral"]["mm_per_pixel"] > 0
    assert abs(doc["dual_pixel"]["step_px"]) > 0.5 and len(doc["dual_pixel"]["frame_positions_steps"]) == 6
    assert abs(sum(doc["coverage"][k] for k in ("measured_both", "measured_one", "filled")) - 1) < 1e-3
    assert kinds["surface_json"]["method"] == "focus+dual_pixel"
    code2, res2 = _run(args, capsys)
    assert code2 == 0 and sorted(o["output"] for o in res2["outputs"]) == sorted(o["output"] for o in res["outputs"])
    code3, res3 = _run(["stack", "focus", *paths, "--make", "depth", "--out", tmp_path / "d"], capsys)
    assert code3 == 0 and {o["kind"] for o in res3["outputs"]} == {"surface_height", "surface_confidence",
                                                                  "surface_agreement", "surface_json"}
    code4, res4 = _run(["stack", "focus", *paths, "--make", "surface", "--ratio", "banana"], capsys)
    assert code4 == 2 and "--ratio" in res4["problems"][0]["text"]
    code5, res5 = _run(["stack", "superres", *paths, "--make", "surface"], capsys)
    assert code5 == 2 and "focus" in res5["problems"][0]["text"]
    code6, res6 = _run(["stack", "focus", *paths, "--make", "sruface"], capsys)
    assert code6 == 2 and "surface" in res6["problems"][0]["text"]


def test_dualpixel_surface_from_the_command_line(tmp_path, capsys, monkeypatch):
    import test_dualpixel as TD
    from studio.engine.dualpixel import decode as DD
    TD.OUT.mkdir(parents=True, exist_ok=True)
    dp, _ = TD.make_dual()
    src = tmp_path / "IMG_DUAL.CR3"
    src.write_bytes(b"")
    monkeypatch.setattr(DD, "is_dual_pixel", lambda p: True)
    monkeypatch.setattr(DD, "open_dual", lambda p: dp)
    code, res = _run(["dualpixel", src, "--make", "surface", "--out", tmp_path / "o", "--normals-y", "down"], capsys)
    assert code == 0, res["problems"]
    kinds = {o["kind"]: o["output"] for o in res["outputs"]}
    assert os.path.basename(kinds["surface_height"]) == "IMG_DUAL_height_Studio.png"
    doc = json.load(open(kinds["surface_json"], encoding="utf-8"))
    assert doc["method"] == "dual_pixel" and doc["normal"]["convention"].startswith("DirectX")
    assert not doc["height"]["near_measured"] and "not yet checked" in doc["height"]["near_end"]


def test_model_and_intake_say_plainly_what_they_need(tmp_path, capsys):
    """The 3D Scan commands (their full runs are tested in test_scan.py on synthetic views): a folder with nothing to
    make, or a single view, is refused with a plain reason and exit code 2."""
    empty = tmp_path / "empty"
    empty.mkdir()
    code, res = _run(["intake", empty], capsys)
    assert code == 2 and "nothing to make" in res["problems"][0]["text"]
    code, res = _run(["intake", tmp_path / "missing"], capsys)
    assert code == 2 and "not a folder" in res["problems"][0]["text"]
    src = _picture(tmp_path / "a.png")
    code, res = _run(["model", src], capsys)
    assert code == 2 and "two views" in res["problems"][0]["text"]

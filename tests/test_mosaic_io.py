"""Flat Mosaic, in and out (BUILD_PLAN_MOSAIC.md M4 and Auto; plan section 6 tests 7 and 9): the files a mosaic
writes, its JSON, pieces, and telling a panorama from a mosaic. Files go to pytest's tmp_path (inside outputs/)."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mosaic_synth as S  # noqa: E402

from studio.engine.stack import mosaic as MO  # noqa: E402
from studio.engine.stack import mosaicbuild as MB  # noqa: E402
from studio.engine.stack import mosaicio as MI  # noqa: E402


@pytest.fixture(scope="module")
def built():
    g = S.Ground(x1=3.5, y1=1.4, seed=5, relief=S.bumps([(0.5, 0.15), (1.2, 0.35)], h=0.03, r=0.05))
    views, truth = S.strip(g, n_per_row=4, rows=2, length=1.5, seed=21, exposure=0.1)
    sol = MO.solve(views)
    params = MO.MosaicParams(resolution="custom", mm_per_px=2.0, tileable=False, pieces="0.1,0.1,0.3,0.4")
    r = MB.build(views, sol, params)
    return views, sol, params, r


def test_files_and_json(built, tmp_path):
    views, sol, params, r = built
    w = MI.write(r, sol, views, params, str(tmp_path), "road")
    p = w["paths"]
    for k in ("colour", "height", "normal", "confidence", "agreement", "json", "misfit", "coverage"):
        assert k in p and os.path.exists(p[k]), k
        if k != "json":
            assert p[k].endswith("_Studio.png")
    assert len(p["pieces"]) == 3 and all(os.path.exists(x) for x in p["pieces"])
    d = json.load(open(p["json"], encoding="utf-8"))
    assert d["method"] == "mosaic"
    assert d["lateral"]["mm_per_pixel"] == pytest.approx(2.0)
    # the measured repeat: the texture's width in metres at 2 mm per pixel
    W = d["size"]["width"]
    assert d["repeat_m"][0] == pytest.approx(W * 0.002, rel=1e-3)
    m = d["mosaic"]
    assert m["solve"]["method"] == "poses" and m["solve"]["placed"] == len(views)
    assert len(m["views"]) == len(views) and all(v["height_m"] for v in m["views"])
    assert m["files"]["coverage"].endswith("_coverage_Studio.png")
    assert m["coach"] and m["shooting_card"]
    # every map says how it was made, as the Forge's intake reads it
    for k in ("colour", "height", "normal"):
        assert d["maps"][k]["file"] and d["maps"][k]["colour"]
    if m["relief"] and m["relief"]["alpha"]:
        assert d["height"]["units"] == "mm"


def test_pieces_have_a_soft_edge(built, tmp_path):
    from PIL import Image
    views, sol, params, r = built
    w = MI.write(r, sol, views, params, str(tmp_path), "edge")
    col = [x for x in w["paths"]["pieces"] if x.endswith("_colour_Studio.png")][0]
    a = np.array(Image.open(col))
    assert a.shape[2] == 4
    assert a[0, 0, 3] == 0 and a[a.shape[0] // 2, a.shape[1] // 2, 3] == 255


def test_a_second_write_keeps_the_set_together(built, tmp_path):
    views, sol, params, r = built
    w1 = MI.write(r, sol, views, params, str(tmp_path), "same")
    open(os.path.join(tmp_path, "same_misfit_Studio.png.txt"), "w").close()
    taken = set()
    w2 = MI.write(r, sol, views, params, str(tmp_path), "same", taken)
    w3 = MI.write(r, sol, views, params, str(tmp_path), "same", taken)
    assert os.path.basename(w3["paths"]["colour"]) != os.path.basename(w2["paths"]["colour"])
    assert os.path.basename(w3["paths"]["misfit"]).startswith(os.path.basename(w3["paths"]["colour"])
                                                              .replace("_colour_Studio.png", ""))
    del w1


def test_bad_piece_is_refused(built, tmp_path):
    views, sol, params, r = built
    bad = MO.MosaicParams(**dict(params.to_dict(), pieces="0.5,0.5,0.4,0.9"))
    with pytest.raises(ValueError):
        MI.write(r, sol, views, bad, str(tmp_path), "bad")


def test_coach_names_low_overlap(built):
    views, sol, params, r = built
    tips = MI.coach(sol, views, r["plan"])
    assert any("Drift" in t or "drift" in t for t in tips)


def test_panorama_or_mosaic():
    import pano_synth as P
    tex = P.world()
    fr = []
    for k in range(4):
        R = P.rot(yaw=math.radians(-30 + 20 * k), pitch=math.radians(2 * k), roll=0.02 * k)
        img = P.render(tex, R, 600.0, 512, 384)
        fr.append(np.asarray(img if img.ndim == 2 else img.mean(-1), np.float32))
    assert MO.kind_of(fr)["kind"] == "panorama"
    assert MO.kind_of(fr, 450.0)["kind"] == "panorama"       # a wrong focal length does not fool it
    g = S.Ground(x1=3.5, y1=1.0, seed=8)
    views, _ = S.strip(g, n_per_row=4, rows=1, length=1.5, depth=False, focus=False)
    k = MO.kind_of([v.image for v in views])
    assert k["kind"] == "mosaic" and "moving" in k["text"]


def test_focus_stack_results_are_read_as_views(tmp_path):
    """Focus Stack results (a picture with NAME.depth.npz beside it) come in as views with their depth."""
    from PIL import Image
    g = S.Ground(x1=2.5, y1=1.0, seed=9, relief=S.bumps([(0.4, 0.1)], h=0.03, r=0.05))
    views, _ = S.strip(g, n_per_row=2, rows=1, length=0.5, depth=True)
    for v in views:
        p = os.path.join(tmp_path, f"{v.name}_Focus_Studio.png")
        Image.fromarray(v.colour).save(p)
        np.savez_compressed(os.path.splitext(p)[0] + ".depth.npz", depth=np.asarray(v.t, np.float16),
                            confidence=np.asarray(v.conf, np.float16), label=np.full(v.t.shape, 2, np.uint8))
    entries, left = MI.find_inputs([str(tmp_path)])
    assert [e["kind"] for e in entries] == ["result", "result"]
    got, problems = MI.read_views(entries)
    assert not problems and len(got) == 2
    assert all(MB.has_height(v) for v in got)
    assert got[0].meta["kind"] == "result"

"""The 3D Scan section's pages (BUILD_PLAN_DP3D H, I.4), offscreen: they build, show their empty state, find the stacks
in a folder and make surfaces for all of them on a synthetic Dual Pixel stack (served through monkeypatched frames
functions: there is no raw writer). Nothing is written outside outputs/."""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import ROOT  # noqa: E402
from test_stack_pages import _app, _wait  # noqa: E402

OUT = ROOT / "outputs" / "pytest_scan_pages"


def test_sets_page_finds_the_stacks_and_makes_every_surface(monkeypatch):
    import depth_synth as DS
    from studio.engine.stack import batch as B
    app, M = _app()
    frames, halves, _ = DS.make_stack(n=5, h=320, w=480, seed=21)
    paths = DS.install(monkeypatch, frames, halves, "scan_sets")
    folder = os.path.dirname(paths[0])
    lone = os.path.join(folder, "IMG_LONE.CR3")
    stacks = [B.StackInfo("A", "focus", paths, dual_pixel=5), B.StackInfo("IMG_LONE", "single", [lone])]
    monkeypatch.setattr(B, "find_stacks", lambda items, *a, **k: list(stacks))
    page = M.ScanPage(M.Settings())
    page.resize(1280, 720)
    page.show()
    app.processEvents()
    assert page.empty.isVisible() and not page.go_b.isEnabled()
    page.set_folder(folder)
    _wait(app, page)
    assert page.tree.topLevelItemCount() == 2 and page.go_b.isEnabled()
    assert page.selected() == [stacks[0]]                    # the ordinary single photo is not ticked
    page.out_dir = str(OUT / "out")
    page.all_maps.setChecked(False)
    page.do_leaves.setChecked(False)
    page.make_all()
    _wait(app, page)
    job = json.load(open(os.path.join(page.out_dir, B.JOB_FILE), encoding="utf-8"))
    assert job["state"] == "done"
    done = [it for it in job["items"].values() if it["status"] == "done"]
    assert len(done) == 1 and any(o.endswith("_surface_Studio.json") for o in done[0]["outputs"])
    assert "surface: done" in page.tree.topLevelItem(0).text(3)
    page.close()


def test_the_sidebar_has_a_3d_scan_section():
    app, M = _app()
    assert ("⬡", "3D Scan") in M.NAV and M.ROW["3D Scan"] == M.ROW["Card to Stacks"] + 1


def test_the_model_view_turns_the_model_and_says_how_good_it_is(monkeypatch):
    """Make a 3D model from the ticked stacks: the views are read, the model built, and the view below the list shows
    it turning, with the quality and the coach in plain words (the engine is replaced by a known result here; the
    engine itself is tested in test_scan.py)."""
    import numpy as np
    from studio.engine.scan import isosurface as IS
    from studio.engine.scan import model as MD
    from studio.engine.scan import views as VW
    from studio.engine.stack import batch as B
    app, M = _app()
    n = 24
    g = np.mgrid[0:n, 0:n, 0:n].astype(float)
    F = np.pad(np.sqrt(((g - 11.5) ** 2).sum(0)) - 8.0, 1, constant_values=1)
    V, Tr, _ = IS.surface_nets(F, (-1, -1, -1), 1.0)
    measured = V[Tr].mean(1)[:, 2] > 8
    result = {"mesh": (V, Tr, measured), "paths": {"json": str(OUT / "m.json")},
              "json": {"files": {"glb": "x.glb"}, "gate": {"passed": False, "reasons": ["only 2 view(s) could be "
                                                                                           "placed"]},
                       "quality": {"views_placed": 2, "views_given": 3, "reprojection_px": 0.4,
                                   "measured_share": 0.6, "filled_share": 0.4},
                       "coach": {"lines": ["The underside was not seen."]}}}
    calls = {}
    monkeypatch.setattr(VW, "from_stack", lambda paths, name, *a, **k: calls.setdefault("views", []).append(name))
    monkeypatch.setattr(MD, "build", lambda views, out_dir, base, *a, **k: result)
    stacks = [B.StackInfo(f"S{i}", "focus", [f"/x/a{i}.CR3", f"/x/b{i}.CR3"], dual_pixel=2) for i in range(3)]
    monkeypatch.setattr(B, "find_stacks", lambda items, *a, **k: list(stacks))
    page = M.ScanPage(M.Settings())
    page.resize(1280, 820)
    page.show()
    page.set_folder(str(OUT))
    _wait(app, page)
    assert page.model_b.isEnabled()
    page.untick_all()
    assert page.selected() == []
    for i in range(page.tree.topLevelItemCount()):
        page.tree.topLevelItem(i).setCheckState(0, __import__("PySide6.QtCore").QtCore.Qt.CheckState.Checked)
    page.make_model()
    _wait(app, page)
    assert len(calls["views"]) == 3 and page.model_view.isVisible()
    assert page.model_view.pic.pixmap() is not None and not page.model_view.pic.pixmap().isNull()
    assert "No Blender dataset" in page.model_view.text.text() and "underside" in page.model_view.text.text()
    page.model_view.mode.setCurrentIndex(1)
    page.model_view.turn.setValue(90)
    app.processEvents()
    page.close()

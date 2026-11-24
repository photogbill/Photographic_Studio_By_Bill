"""The screen (color/display.py): ProPhoto straight into the monitor's profile, sRGB only without one."""
from __future__ import annotations

import numpy as np

from studio.engine.color import display, icc
from studio.engine.io.export import ExportSpec, to_output


def _img():
    y, x = np.mgrid[0:40, 0:60] / 60.0
    return np.stack([0.05 + 0.9 * x, 0.1 + 0.6 * y, 0.02 + 0.5 * x * y], -1).astype(np.float32)


def test_a_wide_gamut_monitor_gets_the_photo_in_its_own_colours(tmp_path):
    p = tmp_path / "wide.icc"
    p.write_bytes(icc.profile_bytes("adobergb"))           # a monitor that covers Adobe RGB
    d = display.Display(str(p))
    assert d.managed and d.kind == "matrix" and "Adobe" in d.name
    got = d.to8(_img()).astype(int)
    want = to_output(_img(), ExportSpec(format="jpeg", space="adobergb", bits=8)).astype(int)
    assert np.abs(got - want).max() <= 2                   # the monitor's own encoding, not sRGB's
    srgb = to_output(_img(), ExportSpec(format="jpeg", space="srgb", bits=8)).astype(int)
    assert np.abs(got - srgb).max() > 10


def test_no_profile_is_said_and_shown_as_windows_shows_it(monkeypatch):
    monkeypatch.setenv("STUDIO_MONITOR_ICC", "")
    d = display.reload()
    assert not d.managed and "sRGB" in d.name
    a = d.to8(_img())
    assert np.array_equal(a, to_output(_img(), ExportSpec(format="jpeg", space="srgb", bits=8)))


def test_soft_proof_shows_the_print_and_marks_what_the_paper_cannot_hold(tmp_path):
    scr = tmp_path / "screen.icc"
    scr.write_bytes(icc.profile_bytes("adobergb"))
    prn = tmp_path / "paper.icc"
    prn.write_bytes(icc.profile_bytes("srgb"))             # a "paper" smaller than the photo's colours
    d = display.Display(str(scr))
    img = _img()
    img[:10, :10] = (0.0, 0.6, 0.05)                        # a green no sRGB paper can print
    plain = d.to8(img)
    try:
        display.set_proof(str(prn), "relative", gamut=False)
        proof = d.to8(img)
        assert proof.shape == plain.shape and np.abs(proof.astype(int) - plain.astype(int)).max() > 3
        display.set_proof(str(prn), "relative", gamut=True)
        marked = d.to8(img)
        assert (marked[:10, :10] == (255, 0, 255)).all(axis=-1).mean() > 0.9
        assert (marked[30:, 5:15] == (255, 0, 255)).all(axis=-1).mean() < 0.1
    finally:
        display.set_proof(None)
    assert np.array_equal(d.to8(img), plain)

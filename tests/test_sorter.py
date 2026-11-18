"""Card to finished stacks, first pass (studio/engine/stack/sorter.py) on a synthetic card with known truth.

The card: a hand-held focus stack (a textured scene whose depth runs left to right, the focus stepping through it,
the camera drifting a little), with four bad frames planted in it (one soft everywhere, one a duplicate depth that is
a little softer, one shaken sideways, one two stops brighter); then a burst of a different scene (focus fixed); a
single shot of a third scene; and, after a pause, a second focus stack. Frames are judged from their proxies, so the
test builds proxies directly (no files).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift, uniform_filter1d

from studio.engine.stack import sorter as S

H, W = 384, 576


def texture(seed):
    """Isotropic natural-looking detail at several scales, 0..1."""
    rng = np.random.default_rng(seed)
    t = sum(gaussian_filter(rng.normal(0, 1, (H, W)), s) * s for s in (0.8, 1.6, 3.2, 8.0))
    t = (t - t.min()) / (t.max() - t.min())
    for _ in range(25):                                     # some hard-edged blocks too
        y0, x0 = rng.integers(0, H - 40), rng.integers(0, W - 40)
        t[y0:y0 + rng.integers(10, 40), x0:x0 + rng.integers(10, 40)] = rng.uniform(0.1, 0.9)
    return t.astype(np.float32)


def at_focus(sharp, focus, n, dy=0.0, dx=0.0, extra=0.0):
    """The scene with the focus at depth ``focus`` (depth 0 at the left edge .. n-1 at the right)."""
    depth = np.linspace(0, n - 1, W)[None, :].repeat(H, 0)
    sig = np.clip(1.4 * np.abs(depth - focus) + extra, 0, 8)
    bank = [sharp] + [gaussian_filter(sharp, s) for s in np.arange(0.5, 8.01, 0.5)]
    k = np.clip((sig / 0.5).astype(int), 0, len(bank) - 2)
    f = sig / 0.5 - k
    out = np.choose(k, bank) * (1 - f) + np.choose(k + 1, bank) * f
    if dy or dx:
        out = nd_shift(out, (dy, dx), order=1, mode="nearest")
    return out.astype(np.float32)


def frame(name, grey, t, ev=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.clip(grey + rng.normal(0, 0.006, grey.shape), 0, 1).astype(np.float32)
    f = S.Frame(path=name, name=name, number=S._number(name), time_s=t, ev=ev, focal=90.0, iso=400)
    f.grey = g
    f.energy, f.aniso = S.sharpness_blocks(g)
    return f


def make_card():
    n = 9
    A, B, C, D = texture(1), texture(2), texture(3), texture(4)
    card, t, num = [], 1000.0, 100
    bad = {}

    def add(grey, ev=0.0, dt=0.5, tag=None):
        nonlocal t, num
        t += dt
        num += 1
        nm = f"IMG_{num:04d}.CR3"
        card.append(frame(nm, grey, t, ev, seed=num))
        if tag:
            bad[nm] = tag

    # stack 1: nine depths, hand-held drift of a few pixels, four planted bad frames
    for i in range(n):
        add(at_focus(A, i, n, dy=0.4 * i, dx=-0.6 * i))
        if i == 2:
            add(at_focus(A, n + 6, n, dy=1.0, dx=-1.5), tag="soft")             # focused far past the subject
        if i == 4:
            add(at_focus(A, 4, n, dy=1.6, dx=-2.4, extra=0.7), tag="adds")      # same depth again, a bit softer
        if i == 6:
            g = uniform_filter1d(at_focus(A, 6.5, n, dy=2.5, dx=-3.7), 11, axis=1)
            add(g, tag="shaken")                                                # shaken sideways
        if i == 7:
            add(np.clip(at_focus(A, 7, n, dy=2.9, dx=-4.3) * 1.0, 0, 1), ev=2.0, tag="exposure")
    # a burst of another scene: the focus does not move
    for i in range(5):
        add(at_focus(B, 3, n, dy=0.3 * i), dt=0.25 if i else 4.0)
    # one shot of a third scene, soon after
    add(at_focus(C, 4, n), dt=1.5)
    # after a pause, a second stack
    for i in range(6):
        add(at_focus(D, i * 1.6, n, dx=0.5 * i), dt=0.6 if i else 30.0)
    return card, bad


def group(card):
    groups = []
    for run in S.runs(card):
        kind = S.classify(run)
        g = S.Group(kind, run)
        if kind == "focus":
            S.judge(run)
        groups.append(g)
    return groups


def test_card_is_sorted_into_stacks_burst_and_single():
    card, _ = make_card()
    groups = group(card)
    assert [g.kind for g in groups] == ["focus", "burst", "single", "focus"], [g.text() for g in groups]
    assert [len(g.frames) for g in groups] == [13, 5, 1, 6]


def test_bad_frames_are_left_out_with_the_right_reason():
    card, bad = make_card()
    stack = group(card)[0]
    by = {f.name: f for f in stack.frames}
    assert "soft everywhere" in by[[k for k, v in bad.items() if v == "soft"][0]].reason
    adds = by[[k for k, v in bad.items() if v == "adds"][0]]
    assert adds.status == "out" and "adds nothing" in adds.reason
    assert "IMG_0106.CR3" in adds.reason                    # it names the sharper frame at its depth
    shaken = by[[k for k, v in bad.items() if v == "shaken"][0]]
    assert shaken.status in ("warn", "out") and "shaken" in shaken.reason
    assert "exposure +2.0 stops" in by[[k for k, v in bad.items() if v == "exposure"][0]].reason
    good = [f for f in stack.frames if f.name not in bad]
    assert all(f.status == "ok" for f in good), [(f.name, f.reason) for f in good if f.status != "ok"]


def test_strictness_changes_how_much_is_left_out():
    card, _ = make_card()
    run = S.runs(card)[0]
    gentle = sum(f.status == "out" for f in S.judge(run, "gentle"))
    strict = sum(f.status == "out" for f in S.judge(run, "strict"))
    assert gentle <= strict
    for k, v in S.STRICTNESS.items():                       # every choice says what it is best for and weak at
        assert v["best"] and v["weak"]


def test_report_counts_everything():
    card, _ = make_card()
    groups = group(card)
    text = S.report(groups)
    assert "2 focus stacks" in text and "1 burst" in text and "1 single" in text, text


def test_a_folder_of_files_sorted_end_to_end():
    """The same card written as JPEG files with capture times (EXIF), found in a folder tree and sorted from disk;
    a camera JPEG beside its raw and the Studio's own results are not taken as frames."""
    import datetime as dt
    import os
    from PIL import Image
    from conftest import ROOT
    card, bad = make_card()
    d = ROOT / "outputs" / "pytest_sorter" / "card" / "100CANON"
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        p.unlink()
    t0 = dt.datetime(2026, 11, 17, 12, 0, 0)
    for f in card:
        im = Image.fromarray((np.clip(f.grey, 0, 1) * 255).astype(np.uint8))
        ex = Image.Exif()
        when = (t0 + dt.timedelta(seconds=f.time_s - 1000.0)).strftime("%Y:%m:%d %H:%M:%S")
        ex[0x0132] = when
        ifd = ex.get_ifd(0x8769)
        ifd[0x9003] = when
        ifd[0x8827] = 1600 if bad.get(f.name) == "exposure" else 400
        ifd[0x829A] = (1, 100)
        ifd[0x829D] = (8, 1)
        ifd[0x920A] = (90, 1)
        im.save(d / f.name.replace(".CR3", ".JPG"), quality=95, exif=ex)
    (d / "IMG_0101_to_IMG_0113_Focus_Studio.tif").write_bytes(b"")
    paths = S.find_paths([d.parent])
    assert len(paths) == len(card) and all(p.endswith(".JPG") for p in paths)
    groups = S.sort_frames(paths)
    assert [g.kind for g in groups] == ["focus", "burst", "single", "focus"], [g.text() for g in groups]
    stack = groups[0]
    out = {os.path.basename(f.path).replace(".JPG", ".CR3") for f in stack.left_out}
    assert set(k for k, v in bad.items() if v != "shaken") <= out, (out, bad)
    assert groups[1].note == "for Best of burst"


def test_stack_group_leaves_out_a_frame_where_the_subject_moved(monkeypatch):
    """A synthetic focus bracket (tests/test_focus_stack.py) where, in one frame, the part it is sharpest in has
    swayed sideways: that frame is left out with "subject moved" after lining up, the stack is lined up again
    without it and written beside the frames as FIRST_to_LAST_Focus_Studio.tif. A frame that is only sharper or
    softer than its neighbours is not called moved."""
    import os
    from test_focus_stack import _names, install, make_bracket
    frames, tr = make_bracket(n=7, h=384, w=576, seed=3)
    h, w = frames[0].shape[:2]
    k = 3                                                   # sharp at depth 3: rows around the middle
    band = np.abs(tr["depth"] - k) < 0.8
    rows = np.nonzero(band.any(1))[0]
    y0, y1 = rows.min(), rows.max() + 1
    moved = frames[k].copy()
    moved[y0:y1, : w // 2 + 60] = np.roll(frames[k][y0:y1, : w // 2 + 60], 7, axis=1)
    frames[k] = moved
    names = _names(7)
    paths = install(monkeypatch, frames, names, "sorter_moved")
    g = S.Group("focus", [S.Frame(path=p, name=os.path.basename(p)) for p in paths])
    from studio.engine.stack.focus import FocusParams
    r = S.stack_group(g, params=FocusParams(local=False))
    by = {f.name: f for f in g.frames}
    assert by[names[k]].status == "out" and "subject moved" in by[names[k]].reason, by[names[k]].reason
    assert all(f.status != "out" for n, f in by.items() if n != names[k]), [(f.name, f.reason) for f in g.frames]
    assert os.path.basename(r["path"]) == "IMG_1000_to_IMG_1006_Focus_Studio.tif"
    assert os.path.exists(r["path"]) and r["used"] == 6


def test_focus_overlay_shows_what_missed_focus():
    """Show what's in focus: a frame's cyan sits on its own depth band; a stack with two depths missing has an
    amber gap there, the full stack has next to none."""
    n = 9
    A = texture(7)
    full = [frame(f"IMG_{200 + i:04d}.CR3", at_focus(A, i, n, dx=0.5 * i), 100.0 + 0.5 * i) for i in range(n)]
    rgb, inf, missed_full = S.focus_overlay(full)
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8 and inf > 0.3
    gap = [f for i, f in enumerate(full) if i not in (4, 5)]
    _, _, missed_gap = S.focus_overlay(gap)
    assert missed_gap > missed_full + 0.03, (missed_full, missed_gap)
    rgb2, inf2, _ = S.focus_overlay(full, 2)                # frame 2: sharp around depth 2 of 0..8 (left side)
    cyan = (rgb2[..., 2] > rgb2[..., 0] + 40)
    cols = np.nonzero(cyan.any(0))[0]
    centre = float(np.median(np.nonzero(cyan)[1]))
    assert inf2 > 0.03 and abs(centre / W - 2 / (n - 1)) < 0.12, (centre / W, cols.min(), cols.max())


def test_in_focus_mask_marks_the_sharp_band_of_one_photo():
    """Focus peaking for one photo (Review, F): the sharp band of a shallow-focus photo is marked, the far blurred
    side is not."""
    g = at_focus(texture(9), 2, 9)                          # sharp near depth 2 of 0..8: the left quarter
    m = S.in_focus_mask(g)
    left, right = m[:, : W // 3].mean(), m[:, 2 * W // 3:].mean()
    assert left > 0.05 and right < 0.2 * left, (left, right)

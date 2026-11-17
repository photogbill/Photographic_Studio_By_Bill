"""The Studio's own resize / warp / blur (engine/imgops.py), and the rule that OpenCV is not used (THIRD_PARTY.md):
the installers do not install it, so a tool that imports it works on the build machine and fails on every PC."""
import ast
import pathlib

import numpy as np
import pytest

from studio.engine import imgops as io

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_no_module_imports_opencv():
    bad = []
    for p in (ROOT / "studio").rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            if any((m or "").split(".")[0] == "cv2" for m in names):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"OpenCV is not installed by the Studio; use studio/engine/imgops.py instead: {bad}"


def test_every_import_is_installed_by_requirements():
    import sys
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    names = {"numpy": "numpy", "scipy": "scipy", "rawpy": "rawpy", "tifffile": "tifffile", "PIL": "pillow",
             "exifread": "exifread", "PySide6": "pyside6", "pytest": "pytest"}
    optional = {"lensfunpy"}                          # tried inside try/except; the Studio works without it
    std = set(sys.stdlib_module_names) | {"studio", "photographic_studio"}
    missing = set()
    for p in (ROOT / "studio").rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                mods = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module.split(".")[0]]
            else:
                continue
            for m in mods:
                if m in std or m in optional:
                    continue
                if m not in names or names[m] not in req:
                    missing.add(f"{m} ({p.relative_to(ROOT)})")
    assert not missing, sorted(missing)


def test_resize_linear_matches_the_pixel_centre_convention():
    a = np.arange(4, dtype=np.float32)[None, :].repeat(2, 0)     # 0 1 2 3
    up = io.resize_linear(a, 8, 2)[0]
    # centres at (i + 0.5) / 2 - 0.5: -0.25 (edge held) 0.25 0.75 1.25 1.75 2.25 2.75 3.25 (edge held)
    assert np.allclose(up, [0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0], atol=1e-6)
    assert np.allclose(io.resize_linear(up[None, :].repeat(2, 0), 4, 2)[0], [0.125, 1.0, 2.0, 2.875], atol=1e-6)
    rgb = np.random.default_rng(0).random((9, 7, 3)).astype(np.float32)
    assert io.resize_linear(rgb, 14, 18).shape == (18, 14, 3)


def test_resize_cubic_keeps_a_ramp_and_a_constant():
    ramp = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (8, 1))
    up = io.resize_cubic(ramp, 64, 8)
    assert np.abs(np.diff(up[0, 4:-4])).max() < 0.02 and np.all(np.diff(up[0, 4:-4]) > 0)
    assert np.allclose(io.resize_cubic(np.full((5, 6), 0.3, np.float32), 17, 11), 0.3, atol=1e-6)


def test_warp_affine_places_a_point_exactly():
    a = np.zeros((21, 21), np.float32)
    a[10, 10] = 1.0
    th, k = np.radians(30.0), 2.0
    A = k * np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    t = np.array([50.0, 40.0])
    out = io.warp_affine(a, np.hstack([A, t[:, None]]), 120, 100)
    ys, xs = np.mgrid[0:100, 0:120]
    cy, cx = (out * ys).sum() / out.sum(), (out * xs).sum() / out.sum()
    ex, ey = A @ np.array([10.0, 10.0]) + t
    assert abs(cx - ex) < 0.05 and abs(cy - ey) < 0.05
    assert io.warp_affine(a, np.array([[1.0, 0, 500], [0, 1.0, 500]]), 50, 50).max() == 0   # off the canvas


def test_gaussian_blur_keeps_the_sum_and_spreads_as_sigma():
    a = np.zeros((101, 101), np.float32)
    a[50, 50] = 1.0
    b = io.gaussian_blur(a, 3.0)
    ys = np.arange(101) - 50
    var = (b.sum(axis=1) * ys * ys).sum() / b.sum()
    assert abs(b.sum() - 1.0) < 1e-4 and abs(np.sqrt(var) - 3.0) < 0.05


def test_same_as_opencv_where_it_happens_to_be_installed():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    a = rng.random((37, 53, 3)).astype(np.float32)
    for w, h in ((106, 74), (20, 15), (1024, 1024)):
        assert np.abs(io.resize_linear(a, w, h) - cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)).max() < 1e-5
        assert np.abs(io.resize_cubic(a, w, h) - cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)).max() < 1e-5
    th, k = 0.3, 1.7
    M = np.hstack([k * np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]), [[40.3], [25.7]]])
    ref = cv2.warpAffine(a, M, (300, 220), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    assert np.abs(io.warp_affine(a, M, 300, 220) - ref).max() < 1e-4
    assert np.abs(io.gaussian_blur(a[..., 0], 2.3) - cv2.GaussianBlur(a[..., 0], (0, 0), 2.3)).max() < 1e-5

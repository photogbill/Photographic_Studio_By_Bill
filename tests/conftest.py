"""Shared test setup. Tests write only inside the app folder (outputs/pytest_tmp) - never to C:.

Synthetic data with known ground truth is generated here (section 17); no photo is stored in the repo.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STUDIO_LOG_DIR", str(ROOT / "outputs" / "pytest_logs"))
os.environ.setdefault("STUDIO_CALIB_DIR", str(ROOT / "outputs" / "pytest_calibration"))   # never the real one
os.environ.setdefault("STUDIO_SETTINGS_DIR", str(ROOT / "outputs" / "pytest_settings"))    # never the real one
os.environ.setdefault("STUDIO_ERROR_DIR", str(ROOT / "outputs" / "pytest_error_reports"))   # never the real one
os.environ.setdefault("STUDIO_ANALYSIS_CACHE", str(ROOT / "outputs" / "pytest_anacache"))   # never the real one

from studio import log as slog  # noqa: E402

slog.setup(console_level=None, filename="tests.log")


# ------------------------------------------------------------------------------------------ synthetic scenes
def scene(h=256, w=384, seed=0) -> np.ndarray:
    """A natural-looking linear-light test scene: detail lives mostly in luminance (edges, fine texture),
    colour varies smoothly, as in real photographs (colour-difference demosaics rely on this). Includes hard
    luminance edges, coloured blocks with soft colour borders, and a neutral grey ramp along the bottom.
    Values 0..~0.9, float32, HxWx3 (camera RGB after white balance)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    lum = 0.30 + 0.15 * np.sin(x / 23.0) * np.cos(y / 31.0)
    chroma = np.ones((h, w, 3), np.float32)
    for _ in range(14):
        y0, x0 = rng.integers(0, h - 40), rng.integers(0, w - 40)
        hh, ww = rng.integers(15, 40), rng.integers(15, 40)
        lum[y0:y0 + hh, x0:x0 + ww] = rng.uniform(0.06, 0.7)          # sharp luminance edge
        chroma[y0:y0 + hh, x0:x0 + ww] = rng.uniform(0.6, 1.4, 3)     # colour change
    for c in range(3):
        chroma[..., c] = gaussian_filter(chroma[..., c], 2.5)          # colour borders are soft
    lum = lum + 0.04 * np.sin(x / 2.9) * np.cos(y / 3.3)              # fine luminance texture
    img = lum[..., None] * chroma
    img[h - 24:] = np.linspace(0.02, 0.85, w, dtype=np.float32)[None, :, None]   # neutral grey ramp
    return np.clip(img, 0.0, 0.95).astype(np.float32)


def mosaic_of(img: np.ndarray, pattern=((0, 1), (1, 2))) -> np.ndarray:
    p = np.asarray(pattern)
    h, w, _ = img.shape
    m = np.empty((h, w), np.float32)
    for i in (0, 1):
        for j in (0, 1):
            m[i::2, j::2] = img[i::2, j::2, p[i, j]]
    return m


def psnr(a, b, border=8) -> float:
    a = np.clip(a[border:-border, border:-border], 0, 1)
    b = b[border:-border, border:-border]
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


# ------------------------------------------------------------------------------------------ synthetic raw
CAM_XYZ_5D4 = np.array([[0.6446, -0.0366, -0.0864], [-0.4436, 1.2204, 0.2513], [-0.0952, 0.2496, 0.6348]])


def synthetic_raw(scene_wb: np.ndarray, black=512.0, white=15000.0, wb=(2.0, 1.0, 1.5), left=32, top=16,
                  read_noise=0.0, seed=1, flip=0, clip_white=False):
    """Build a RawImage directly (no file): the scene is camera RGB after white balance, 1.0 = clip.
    ``clip_white`` saturates photosites at the white level, as a real sensor does (highlight tests)."""
    from studio.engine.io.metadata import Metadata
    from studio.engine.io.rawfile import RawImage
    rng = np.random.default_rng(seed)
    h, w, _ = scene_wb.shape
    raw_rgb = scene_wb / np.asarray(wb, np.float32)            # undo white balance
    mos = mosaic_of(raw_rgb)                                   # RGGB at the visible origin
    full = np.full((h + top, w + left), black, np.float32)
    full[top:, left:] = black + mos * (white - black)
    if read_noise:
        full += rng.normal(0, read_noise, full.shape)
    full = np.clip(np.rint(full), 0, white if clip_white else 65535).astype(np.uint16)
    cfa_idx = np.array([[0, 1], [3, 2]])
    cfa_rgb = np.array([[0, 1], [1, 2]])
    md = Metadata(make="Synthetic", model="Test Camera", iso=100.0, iso_source="test")
    return RawImage(path="synthetic.cr2", meta=md, mosaic=full, top=top, left=left, height=h, width=w,
                    crop=(0, 0, w, h), cfa_idx=cfa_idx, cfa_rgb=cfa_rgb, color_desc="RGBG",
                    black_libraw=np.array([black] * 4), white_level=white, linear_max=np.zeros(4),
                    wb_as_shot=np.asarray(wb, float), wb_daylight=np.asarray(wb, float), cam_xyz=CAM_XYZ_5D4,
                    flip=flip)


# ------------------------------------------------------------------------------------------ synthetic TIFF metadata
def tiff_bytes(entries: dict, endian="<", sub_ifds: dict | None = None) -> bytes:
    """Build a little TIFF: IFD0 from ``entries`` {tag: (type, values)} plus optional sub-IFDs
    {pointer_tag: entries}. Supports types 2 (ASCII), 3 (SHORT), 4 (LONG), 5 (RATIONAL), 7 (UNDEFINED)."""
    sub_ifds = sub_ifds or {}

    def enc(typ, vals):
        if typ == 2:
            return vals.encode() + b"\0", len(vals) + 1
        if typ == 7:
            return vals, len(vals)
        if typ == 5:
            return b"".join(struct.pack(endian + "II", int(n), int(d)) for n, d in vals), len(vals)
        fmt = {3: "H", 4: "I"}[typ]
        return struct.pack(endian + fmt * len(vals), *vals), len(vals)

    out = bytearray(b"II*\0" if endian == "<" else b"MM\0*")
    out += struct.pack(endian + "I", 8)

    def write_ifd(ents, pos):
        """Lay out an IFD at ``pos``; returns bytes and records where pointer tags must be patched."""
        n = len(ents)
        data_start = pos + 2 + 12 * n + 4
        body = bytearray(struct.pack(endian + "H", n))
        data = bytearray()
        patches = {}
        for tag in sorted(ents):
            typ, vals = ents[tag]
            raw, count = enc(typ, vals)
            entry_pos = pos + len(body)
            if len(raw) <= 4:
                body += struct.pack(endian + "HHI", tag, typ, count) + raw.ljust(4, b"\0")
            else:
                off = data_start + len(data)
                body += struct.pack(endian + "HHII", tag, typ, count, off)
                data += raw + (b"\0" if len(raw) % 2 else b"")
            patches[tag] = entry_pos + 8
        body += struct.pack(endian + "I", 0)
        return body + data, patches

    ifd0 = dict(entries)
    for ptr in sub_ifds:
        ifd0[ptr] = (4, [0])
    blob, patches = write_ifd(ifd0, 8)
    out += blob
    for ptr, ents in sub_ifds.items():
        pos = len(out)
        sblob, _ = write_ifd(ents, pos)
        out += sblob
        struct.pack_into(endian + "I", out, patches[ptr], pos)
    return bytes(out)


def canon_shotinfo(base_raw: int, auto_raw: int = 0, temp_raw: int = 150) -> list:
    v = [0] * 34
    v[0] = 68
    v[1] = auto_raw & 0xFFFF
    v[2] = base_raw & 0xFFFF
    v[12] = temp_raw
    return v


@pytest.fixture
def outdir(tmp_path):
    return tmp_path


def find_sample_raws(limit=6) -> list:
    """Raw files Bill keeps in the app folder (never committed). Tests using them skip when there are none."""
    exts = (".cr2", ".cr3")
    env = os.environ.get("STUDIO_TEST_RAWS")
    folders = [Path(env)] if env else [ROOT, ROOT / "high iso raw"]
    files = []
    for f in folders:
        if f.is_dir():
            files += sorted(p for p in f.iterdir() if p.suffix.lower() in exts)
    return files[:limit]


_EXIT = {"status": 0}


def pytest_sessionfinish(session, exitstatus):
    """Wait for the Studio's background jobs (renders, reports) before the interpreter exits."""
    import sys
    m = sys.modules.get("studio.app.main")
    if m is not None:
        m.finish_jobs(timeout_ms=60000)             # tests wait for everything (the app itself does not)
    _EXIT["status"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """End the test process the way the Studio itself ends: without Python tearing Qt down. With the hundreds of
    widgets the tests make, that final teardown destroyed Qt objects in an order that segfaulted after every test
    had passed (exit code 139 with "441 passed"); the pass / fail status is kept."""
    if "PySide6.QtWidgets" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_EXIT["status"])

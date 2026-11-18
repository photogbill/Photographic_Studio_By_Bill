"""Calibration frames, measured one at a time (section 7; Bill, 2026-11-18: "it should be able to handle a huge
drop of raw photos at different ISO and temp. The key thing is the lens cap on.").

The first builder opened every frame at once, which a few hundred 45 MP raws cannot survive. Here each frame is
opened, measured and let go: what is kept is a few numbers, a small crop from the centre of the sensor (to tell
noise from fixed pattern when two frames are compared) and the positions of its hot pixels. That is all the
builder needs, so:

  * a drop of any size uses the memory of one frame per worker;
  * what was measured is kept on this PC (calibration_local/frames), so frames can be added on another day and
    nothing is read twice. A frame is known by its content, not its path: the same file from the card and from
    a copy counts once;
  * frames of several cameras can be in one pile. Each camera gets its own profile.

Nothing here is shared: the store holds crops of the user's own frames (rule 0.3.2).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog, paths
from ..raw.linearize import clipped_stats

log = slog.get("engine.calib")

VERSION = 1                 # of what is measured: raise it and stored frames are measured again
CROP = 256                  # side of the kept centre crop (photosites)
HOT_SIGMA = 10.0            # a hot pixel stands this many read-noise sigmas above black
HOT_CAP = 250_000           # more "hot pixels" than this in one frame is not a list worth keeping
LEAK_SIGMAS = 1.5           # a short dark frame whose broad areas differ by more than this is suspect


@dataclass
class FrameStats:
    path: str
    name: str = ""
    key: str = ""                       # content key (store file name)
    make: str = ""
    model: str = ""
    iso: float | None = None
    exposure_s: float | None = None
    temp_c: float | None = None
    datetime: str = ""
    kind: str = "other"                 # dark | flat | other | unreadable
    reason: str = ""
    mean_dn: float = 0.0                # centre level above black (classification)
    width: int = 0
    height: int = 0
    clip_dn: float = 0.0
    black: list | None = None           # 2x2, darks
    sigma: list | None = None           # 2x2 spatial noise of this one frame (row pattern removed), darks
    row_band: float | None = None
    col_band: float | None = None
    centre_dn: float | None = None      # clipped mean of the centre, raw DN (dark current)
    hot_count: int = 0
    uneven: float = 0.0                 # spread of broad areas, in sigmas (a light leak shows here)
    notes: list = field(default_factory=list)
    cached: bool = False                # read from the store, not measured now
    crop: np.ndarray | None = None      # uint16 centre crop, CFA phase of the visible area kept
    hot: np.ndarray | None = None       # flat indices (y * width + x) of hot pixels; None = not kept

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("crop"), d.pop("hot"), d.pop("cached")
        return d

    @property
    def camera(self) -> tuple:
        return (self.make, self.model)


# ------------------------------------------------------------------------------------------------ measuring
def _centre(a: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = a.shape
    ch, cw = int(h * frac) // 2 * 2, int(w * frac) // 2 * 2
    y0, x0 = ((h - ch) // 2) // 2 * 2, ((w - cw) // 2) // 2 * 2
    return a[y0:y0 + ch, x0:x0 + cw]


def _per_position(a: np.ndarray, fn) -> np.ndarray:
    return np.array([[fn(a[i::2, j::2]) for j in (0, 1)] for i in (0, 1)], float)


def _crop(vis: np.ndarray) -> np.ndarray:
    h, w = vis.shape
    cs = min(CROP, h // 2 * 2, w // 2 * 2)
    y0, x0 = ((h - cs) // 2) // 2 * 2, ((w - cs) // 2) // 2 * 2
    return np.ascontiguousarray(vis[y0:y0 + cs, x0:x0 + cs]).astype(np.uint16)


def _plain_reason(reason: str) -> str:
    """Why a frame was left out, for someone who shot with the cap on and wonders what went wrong."""
    if reason.startswith("not even enough"):
        return "a normal photo: not a dark frame (lens cap on) and not an evenly lit flat"
    if reason.startswith("too dark for a flat"):
        return ("nearly dark, but light is reaching the sensor: check the lens cap and cover the eyepiece, or "
                "this is just a very dark photo")
    if reason.startswith("over-exposed"):
        return "over-exposed (clipped): not usable as a flat"
    return reason or "not a calibration frame"


def measure(img, key: str = "") -> FrameStats:
    """Everything the builder needs from one opened frame. The frame can be let go afterwards."""
    from .builder import classify
    info = classify(img)
    m = img.meta
    vis = img.visible
    st = FrameStats(path=img.path, name=os.path.basename(img.path), key=key, make=m.make or "", model=m.model or "",
                    iso=m.iso, exposure_s=m.exposure_s, temp_c=m.camera_temperature_c, datetime=m.datetime or "",
                    kind=info.kind, reason=info.reason, mean_dn=float(info.mean_dn), width=int(vis.shape[1]),
                    height=int(vis.shape[0]), clip_dn=float(img.clip_level()))
    if st.kind not in ("dark", "flat"):
        st.reason = _plain_reason(st.reason)
        return st
    st.crop = _crop(vis)
    if st.kind == "flat":
        return st
    v = _centre(vis, 0.9).astype(np.float32)
    black = _per_position(v, lambda p: clipped_stats(p[::2, ::2])[0])
    for i in (0, 1):
        for j in (0, 1):
            v[i::2, j::2] -= np.float32(black[i, j])
    s0 = float(clipped_stats(v[::3, ::3])[1]) or 1.0
    keep = np.abs(v) < 6 * s0
    n_r, n_c = np.maximum(keep.sum(1), 1), np.maximum(keep.sum(0), 1)
    rows = np.where(keep, v, 0).sum(1) / n_r
    cols = np.where(keep, v, 0).sum(0) / n_c
    d = v - rows[:, None].astype(np.float32)                     # the row pattern is reported on its own
    sig = _per_position(d, lambda p: clipped_stats(p[::2, ::2])[1])
    s_all = float(np.sqrt(np.mean(sig ** 2))) or s0
    st.black = black.round(3).tolist()
    st.sigma = sig.round(3).tolist()
    st.row_band = round(math.sqrt(max(float(rows.var()) - s_all ** 2 / v.shape[1], 0.0)), 3)
    st.col_band = round(math.sqrt(max(float(cols.var()) - s_all ** 2 / v.shape[0], 0.0)), 3)
    c = _centre(vis, 0.5)
    st.centre_dn = round(float(clipped_stats(c[::3, ::3])[0]), 3)
    # broad unevenness: a dark frame is the same everywhere; a light leak (eyepiece, a loose cap) is not
    bh, bw = v.shape[0] // 8, v.shape[1] // 8
    if bh >= 8 and bw >= 8:
        vv = np.where(keep, v, 0)[:bh * 8, :bw * 8].reshape(8, bh, 8, bw)
        kk = keep[:bh * 8, :bw * 8].reshape(8, bh, 8, bw)
        blocks = vv.sum(axis=(1, 3)) / np.maximum(kk.sum(axis=(1, 3)), 1)
        st.uneven = round(float(np.percentile(blocks, 95) - np.percentile(blocks, 5)) / max(s_all, 1e-6), 3)
        if st.uneven > LEAK_SIGMAS and (st.exposure_s or 0) < 1.0 and st.uneven * s_all > 2.0:
            st.notes.append("one part of the frame is brighter than the rest: light may be getting in (cover the "
                            "eyepiece, check the cap)")
    del v, d, keep
    # hot pixels over the whole visible area
    parts, w = [], vis.shape[1]
    for i in (0, 1):
        for j in (0, 1):
            thr = black[i, j] + HOT_SIGMA * max(float(sig[i, j]), 0.5)
            yy, xx = np.nonzero(vis[i::2, j::2] > thr)
            parts.append((yy.astype(np.int64) * 2 + i) * w + (xx.astype(np.int64) * 2 + j))
    hot = np.sort(np.concatenate(parts))
    st.hot_count = int(hot.size)
    if hot.size <= HOT_CAP:
        st.hot = hot.astype(np.uint32)
    else:
        st.notes.append(f"{hot.size:,} photosites far above black: too many to be hot pixels (is the cap on?)")
    return st


# ------------------------------------------------------------------------------------------------ the store
def store_dir() -> str:
    p = paths.get("CALIBRATION_LOCAL") / "frames"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def content_key(path: str) -> str:
    """The same file is the same frame wherever it lies: its size and its first 256 KB."""
    h = hashlib.blake2b(digest_size=12)
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(262144))
    return h.hexdigest()


def _store_path(key: str) -> str:
    return os.path.join(store_dir(), key + ".npz")


def save(st: FrameStats) -> None:
    if not st.key:
        return
    arrays = {"meta": np.array(json.dumps({"version": VERSION, **st.to_json()}))}
    if st.crop is not None:
        arrays["crop"] = st.crop
    if st.hot is not None:
        arrays["hot"] = st.hot
    tmp = _store_path(st.key) + ".partial"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, _store_path(st.key))


def load(key: str) -> FrameStats | None:
    p = _store_path(key)
    if not os.path.isfile(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.pop("version", None) != VERSION:
                return None
            st = FrameStats(**{k: v for k, v in meta.items() if k in FrameStats.__dataclass_fields__})
            st.crop = z["crop"] if "crop" in z.files else None
            st.hot = z["hot"] if "hot" in z.files else None
        st.cached = True
        return st
    except Exception:                                   # a store file cut short: measure the frame again
        slog.log_exception(f"calibration store file unreadable: {p}", log)
        return None


def load_store(camera: tuple | None = None) -> list:
    """Every frame measured so far on this PC (of one camera, or of all)."""
    out = []
    for f in sorted(os.listdir(store_dir())):
        if f.endswith(".npz"):
            st = load(f[:-4])
            if st is not None and (camera is None or st.camera == tuple(camera)):
                out.append(st)
    return out


def forget_camera(make: str, model: str) -> int:
    """Start a camera over: its measured frames leave the store (the raw files are not touched)."""
    n = 0
    for st in load_store((make, model)):
        try:
            os.remove(_store_path(st.key))
            n += 1
        except OSError:
            slog.log_exception("could not remove a calibration store file", log)
    return n


# ------------------------------------------------------------------------------------------------ a whole drop
def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def scan_file(path: str) -> FrameStats:
    """One file: from the store if it was measured before, else opened, measured, stored and let go."""
    from ..io.rawfile import UnsupportedRaw, open_raw
    name = os.path.basename(path)
    try:
        key = content_key(path)
    except OSError as e:
        return FrameStats(path=path, name=name, kind="unreadable", reason=f"could not be read ({e})")
    st = load(key)
    if st is not None:
        st.path, st.name = path, name
        return st
    try:
        img = open_raw(path)
    except (UnsupportedRaw, FileNotFoundError, OSError) as e:
        return FrameStats(path=path, name=name, key=key, kind="unreadable", reason=str(e))
    st = measure(img, key)
    del img
    if st.kind in ("dark", "flat"):
        save(st)
    return st


def scan(files: list, progress=None, cancelled=None, workers: int | None = None) -> list:
    """Measure a pile of files. ``progress(done, total, stats)`` after each; ``cancelled()`` true stops taking new
    files (what was measured is kept in the store). Returns the stats in the order of ``files``."""
    files = list(files)
    out: list = [None] * len(files)
    lock, done = threading.Lock(), [0]
    workers = workers or default_workers()

    def one(i):
        if cancelled is not None and cancelled():
            return i, None
        try:
            return i, scan_file(files[i])
        except Exception as e:                           # a fault of ours on one frame must not end the pile
            slog.log_exception(f"calibration: measuring {files[i]} failed", log)
            slog.save_error_report(f"Calibration could not measure {os.path.basename(files[i])}")
            return i, FrameStats(path=files[i], name=os.path.basename(files[i]), kind="unreadable",
                                 reason=f"the Studio could not measure it ({e}); an error report was saved")

    with slog.stage("calib.scan", log, files=len(files), workers=workers):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, i) for i in range(len(files))]
            for fu in as_completed(futs):
                i, st = fu.result()
                out[i] = st
                with lock:
                    done[0] += 1
                    n = done[0]
                if progress is not None and st is not None:
                    progress(n, len(files), st)
    return [s for s in out if s is not None]

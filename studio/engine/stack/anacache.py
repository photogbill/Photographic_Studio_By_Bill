"""A cache of Focus Stack analyses (BUILD_PLAN_DP3D A.4).

Pass A (half-size frames lined up, sharpness, Dual Pixel offsets) is the slow part of a stack and does not change
with the merge, the surface settings, foliage or models made later from the same stack. So it is kept, one folder
per stack, in ``analysis_cache/`` beside the app (never on C:\\, never in the repository), keyed by the frames
(path, size, modification time) and the settings that change the analysis (FocusParams.analysis_key()).

A folder holds the big arrays as .npy files opened as memory maps (aligned, energy, dp_disp, dp_conf) and the
small fields (transforms, flows, gains, box, maps, notes) as small.npz + meta.json. meta.json is written last:
a folder without it was interrupted and is ignored (and removed when the cache is trimmed). The cache has a size
cap (Settings: "Analysis cache", default 60 GB); the least recently used folders go first. No Qt here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import numpy as np

from ... import log as slog, paths

log = slog.get("engine.stack.anacache")

VERSION = 1
DEFAULT_CAP_GB = 60.0
BIG = ("aligned", "energy", "dp_disp", "dp_conf")


def root() -> str:
    env = os.environ.get("STUDIO_ANALYSIS_CACHE")
    p = env or str(paths.DATA_ROOT / "analysis_cache")
    os.makedirs(p, exist_ok=True)
    return p


def cap_gb() -> float:
    """The size cap from the Studio's settings (app.json "analysis_cache_gb"), else 60 GB."""
    try:
        d = json.loads((paths.get("SETTINGS") / "app.json").read_text(encoding="utf-8"))
        v = float(d.get("analysis_cache_gb", DEFAULT_CAP_GB))
        return max(0.0, v)
    except (OSError, ValueError, TypeError):
        return DEFAULT_CAP_GB


def key_for(frame_paths: list, analysis_key: tuple) -> str:
    """The cache key: every frame's absolute path, size and modification time, and the analysis settings."""
    h = hashlib.sha256()
    h.update(f"v{VERSION}".encode())
    for p in frame_paths:
        ap = os.path.normcase(os.path.abspath(p))
        try:
            st = os.stat(p)
            h.update(f"{ap}|{st.st_size}|{int(st.st_mtime)}\n".encode("utf-8", "replace"))
        except OSError:
            h.update(f"{ap}|missing\n".encode("utf-8", "replace"))
    h.update(repr(tuple(analysis_key)).encode())
    return h.hexdigest()[:24]


class CacheScratch:
    """Scratch space for one analysis that lives in the cache: the same calls as frames.Scratch (array, drop,
    close), but close() only lets the files go (the folder stays: that is the point). Temporary arrays that are
    not part of the cached analysis (the developed frames before lining up) are made in the folder too and
    dropped as usual."""

    def __init__(self, folder: str):
        self.dir = folder
        os.makedirs(folder, exist_ok=True)
        self._maps: list = []

    def array(self, name: str, shape, dtype=np.float32) -> np.ndarray:
        p = os.path.join(self.dir, name + ".npy")
        a = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=tuple(shape))
        self._maps.append(a)
        return a

    def open(self, name: str) -> np.ndarray:
        a = np.load(os.path.join(self.dir, name + ".npy"), mmap_mode="r")
        self._maps.append(a)
        return a

    def drop(self, a: np.ndarray):
        try:
            fn = a.filename
            a._mmap.close()                           # noqa: SLF001
            self._maps = [x for x in self._maps if x is not a]
            os.remove(fn)
        except Exception:                             # noqa: BLE001  (a stray file is trimmed later)
            pass

    def close(self):
        for a in self._maps:
            try:
                if getattr(a, "mode", "r") != "r":
                    a.flush()
                a._mmap.close()                       # noqa: SLF001
            except Exception:                         # noqa: BLE001
                pass
        self._maps = []

    def discard(self):
        """An analysis that failed or was cancelled: its half-written folder goes."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def new_entry(key: str) -> CacheScratch:
    """A fresh folder for an analysis being made, at its final name (Windows cannot rename a folder whose files are
    open as memory maps, and the analysis keeps using them). It counts only once meta.json is written; an old,
    unfinished folder of the same key is cleared first."""
    d = entry_dir(key)
    if os.path.isdir(d) and not os.path.isfile(os.path.join(d, "meta.json")):
        shutil.rmtree(d, ignore_errors=True)
    return CacheScratch(d)


def entry_dir(key: str) -> str:
    return os.path.join(root(), key)


def has(key: str) -> bool:
    return os.path.isfile(os.path.join(entry_dir(key), "meta.json"))


def finish(sc: CacheScratch, key: str, meta: dict, small: dict) -> str:
    """Write the small fields, flush the big arrays (they stay open: the analysis goes on using them) and write
    meta.json last, which makes the entry count. Then keep the cache under its cap. Returns the folder."""
    for a in sc._maps:                               # noqa: SLF001
        try:
            if getattr(a, "mode", "r") != "r":
                a.flush()
        except Exception:                            # noqa: BLE001
            pass
    np.savez(os.path.join(sc.dir, "small.npz"), **small)
    meta = dict(meta, version=VERSION, key=key, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = os.path.join(sc.dir, "meta.json.partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, os.path.join(sc.dir, "meta.json"))
    trim(keep=sc.dir)
    return sc.dir


def load(key: str):
    """(CacheScratch over the folder, meta, small) or None."""
    d = entry_dir(key)
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("version") != VERSION:
            return None
        small = dict(np.load(os.path.join(d, "small.npz"), allow_pickle=False))
        os.utime(os.path.join(d, "meta.json"))       # used now: the last to be trimmed
        return CacheScratch(d), meta, small
    except (OSError, ValueError, KeyError) as e:
        log.info("analysis cache %s unreadable (%s): analysed again", key, e)
        return None


def size_bytes(folder: str) -> int:
    tot = 0
    for dp, _, fns in os.walk(folder):
        for f in fns:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def trim(cap: float | None = None, keep: str | None = None) -> int:
    """Keep the cache under its cap: interrupted folders go first, then the least recently used (``keep``: a folder
    in use now, never removed). Returns the number of folders removed."""
    cap_b = (cap_gb() if cap is None else float(cap)) * 1e9
    r = root()
    entries = []
    removed = 0
    for name in os.listdir(r):
        d = os.path.join(r, name)
        if not os.path.isdir(d) or (keep and os.path.normcase(os.path.abspath(d)) ==
                                    os.path.normcase(os.path.abspath(keep))):
            continue
        meta = os.path.join(d, "meta.json")
        if not os.path.isfile(meta):
            # an interrupted analysis; one being written right now is younger than a day
            try:
                if time.time() - os.path.getmtime(d) > 86400:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
            continue
        entries.append((os.path.getmtime(meta), d, size_bytes(d)))
    total = sum(e[2] for e in entries)
    for _, d, sz in sorted(entries):
        if total <= cap_b:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= sz
        removed += 1
    if removed:
        log.info("analysis cache trimmed: %d folder(s) removed", removed)
    return removed


def clear() -> int:
    """Remove every cached analysis (Settings > Clear the analysis cache). Returns the number removed."""
    r = root()
    n = 0
    for name in os.listdir(r):
        d = os.path.join(r, name)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def summary() -> dict:
    r = root()
    entries = [os.path.join(r, n) for n in os.listdir(r) if os.path.isfile(os.path.join(r, n, "meta.json"))]
    return {"folder": r, "analyses": len(entries), "gb": round(sum(size_bytes(d) for d in entries) / 1e9, 2),
            "cap_gb": cap_gb()}

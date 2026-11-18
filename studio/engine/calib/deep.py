"""Deep calibration (BUILD_PLAN_NOISE block B): what the dark frames still hold after the profile is built.

The profile keeps a few numbers per ISO. The frames hold the noise itself. This second pass over the raw files
keeps, privately in calibration_local/<camera>/:

  * a **noise bank**: patches of real dark frames per ISO and exposure class (short / mid / long). The truth
    harness and the training of a camera's own network draw real noise from it: true read noise, banding, fixed
    pattern, warm pixels and the tails no formula has (B.2);
  * the **fixed row and column pattern** per ISO: the part of the banding that is the same in every photo, which
    can be subtracted exactly (B.1's small, certain part; the per-photosite master dark has to earn its place
    first, plan 0.3);
  * the **shape** of the noise per ISO: how heavy its tails are and how much neighbours agree (B.4).

Nothing here is shared (rule 0.3.2): the bank is made of the user's own frames.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ... import log as slog
from . import frames as F
from .profile import local_data_dir

log = slog.get("engine.calib.deep")

PATCH = 256                 # side of one bank patch (photosites; CFA phase of the visible area kept)
PER_FRAME = 4               # patches taken from one frame, from different places
MAX_PATCHES = 48            # per ISO and exposure class (about 6 MB)
MIN_FIXED = 4               # frames needed before a fixed row / column pattern is trusted
BANK, BANK_META, FIXED = "noise_bank.npz", "noise_bank.json", "fixed_pattern.npz"


def exposure_class(seconds) -> str:
    s = float(seconds or 0)
    return "short" if s < 0.1 else "mid" if s < 5.0 else "long"


def bank_key(iso: float, cls: str = "short") -> str:
    return f"{float(iso):g}|{cls}"


def _places(key: str, h: int, w: int, n: int) -> list:
    """Where the patches of one frame are taken: different for every frame, the same every time for that frame."""
    seed = int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16)
    rng = np.random.default_rng(seed)
    my, mx = max(int(h * 0.05) // 2 * 2, 0), max(int(w * 0.05) // 2 * 2, 0)
    out = []
    for _ in range(n):
        y = int(rng.integers(my // 2, max((h - PATCH - my) // 2, my // 2 + 1))) * 2
        x = int(rng.integers(mx // 2, max((w - PATCH - mx) // 2, mx // 2 + 1))) * 2
        out.append((min(y, max(h - PATCH, 0)) // 2 * 2, min(x, max(w - PATCH, 0)) // 2 * 2))
    return out


def _lines(vis: np.ndarray, black: np.ndarray, sigma: float) -> tuple:
    """Row and column means of a dark frame above black, outliers (hot pixels) left out."""
    v = vis.astype(np.float32)
    for i in (0, 1):
        for j in (0, 1):
            v[i::2, j::2] -= np.float32(black[i][j])
    keep = np.abs(v) < 6.0 * max(sigma, 0.5)
    v = np.where(keep, v, 0.0)
    return (v.sum(1) / np.maximum(keep.sum(1), 1)).astype(np.float32), \
           (v.sum(0) / np.maximum(keep.sum(0), 1)).astype(np.float32)


def _shape(patches: np.ndarray, black: np.ndarray) -> dict:
    """Tails and neighbour agreement of the noise in one CFA plane of the patches."""
    p = patches[:, 0::2, 0::2].astype(np.float64) - float(black[0][0])
    p = p - np.median(p, axis=2, keepdims=True)
    s = float(p.std()) or 1.0
    q = p[np.abs(p) < 8 * s]
    v = float(q.var()) or 1.0
    return dict(excess_kurtosis=round(float((q ** 4).mean() / v ** 2 - 3.0), 3),
                neighbour_h=round(float((p[:, :, :-1] * p[:, :, 1:]).mean() / v), 4),
                neighbour_v=round(float((p[:, :-1] * p[:, 1:]).mean() / v), 4))


def deep_calibrate(files: list, progress=None, cancelled=None, workers: int | None = None) -> list:
    """Read the dark frames among ``files`` once more and build each camera's noise bank and fixed pattern.
    ``progress(done, total, name)``. Returns one report dict per camera."""
    from ..io.rawfile import open_raw
    stats = [s for s in F.scan(files, cancelled=cancelled, workers=workers) if s.kind == "dark" and s.iso]
    lock = threading.Lock()
    patches, meta = defaultdict(list), defaultdict(lambda: defaultdict(list))
    rows, cols = defaultdict(list), defaultdict(list)
    done = [0]

    def one(st):
        if cancelled is not None and cancelled():
            return
        cam, key = st.camera, bank_key(st.iso, exposure_class(st.exposure_s))
        with lock:
            full = len(patches[(cam, key)]) >= MAX_PATCHES
            lines_full = len(rows[(cam, st.iso)]) >= 16 or exposure_class(st.exposure_s) != "short"
        if full and lines_full:
            return
        img = open_raw(st.path)
        vis = img.visible
        got = [np.ascontiguousarray(vis[y:y + PATCH, x:x + PATCH]).astype(np.uint16)
               for y, x in _places(st.key or st.name, vis.shape[0], vis.shape[1], PER_FRAME)]
        got = [g for g in got if g.shape == got[0].shape]
        ln = None if lines_full else _lines(vis, st.black, float(np.mean(st.sigma)))
        del img, vis
        with lock:
            room = MAX_PATCHES - len(patches[(cam, key)])
            patches[(cam, key)] += got[:max(room, 0)]
            m = meta[(cam, key)]
            m["black"].append(st.black), m["clip"].append(st.clip_dn), m["sigma"].append(st.sigma)
            m["temp"].append(st.temp_c), m["exposure"].append(st.exposure_s), m["frames"].append(st.name)
            if ln is not None:
                rows[(cam, st.iso)].append(ln[0]), cols[(cam, st.iso)].append(ln[1])

    with slog.stage("calib.deep", log, darks=len(stats)):
        with ThreadPoolExecutor(max_workers=workers or F.default_workers()) as ex:
            futs = {ex.submit(one, s): s for s in stats}
            for fu in as_completed(futs):
                fu.result()
                done[0] += 1
                if progress is not None:
                    progress(done[0], len(stats), futs[fu].name)
    reports = []
    for cam in sorted({c for c, _ in patches}):
        d = local_data_dir(*cam)
        arrays, info = {}, {}
        for (c, key), lst in sorted(patches.items()):
            if c != cam or not lst:
                continue
            arr = np.stack(lst)
            m = meta[(c, key)]
            black = np.median(np.asarray(m["black"], float), axis=0)
            temps = [t for t in m["temp"] if t is not None]
            arrays[key] = arr
            info[key] = dict(iso=float(key.split("|")[0]), cls=key.split("|")[1], patches=int(arr.shape[0]),
                             frames=len(set(m["frames"])), black=np.round(black, 3).tolist(),
                             clip_dn=float(np.median(m["clip"])),
                             sigma_dn=round(float(np.sqrt(np.mean(np.asarray(m["sigma"], float) ** 2))), 3),
                             exposure_s=float(np.median([e or 0 for e in m["exposure"]])),
                             temp_min_c=min(temps) if temps else None, temp_max_c=max(temps) if temps else None,
                             **_shape(arr, black))
        fixed, fixed_info = {}, {}
        for (c, iso), lst in sorted(rows.items()):
            if c != cam or len(lst) < MIN_FIXED or len({r.shape for r in lst}) != 1:
                continue
            r, cl = np.mean(lst, axis=0), np.mean(cols[(c, iso)], axis=0)
            r, cl = r - np.median(r), cl - np.median(cl)
            n = len(lst)
            one_frame = float(np.mean([np.std(x - np.median(x)) for x in lst]))
            # the mean of n frames still holds 1/n of the random row noise: take that out of the estimate
            fixed_rows = float(np.sqrt(max(float(r.var()) - max(one_frame ** 2 - float(r.var()), 0.0) / max(n - 1, 1),
                                           0.0)))
            fixed[f"rows_{iso:g}"], fixed[f"cols_{iso:g}"] = r.astype(np.float32), cl.astype(np.float32)
            fixed_info[f"{iso:g}"] = dict(frames=n, fixed_rows_dn=round(fixed_rows, 4),
                                          one_frame_rows_dn=round(one_frame, 4),
                                          fixed_cols_dn=round(float(cl.std()), 4))
        folder = os.path.dirname(os.path.abspath(stats[0].path)) if stats else ""
        with open(os.path.join(d, BANK + ".partial"), "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(os.path.join(d, BANK + ".partial"), os.path.join(d, BANK))
        with open(os.path.join(d, FIXED + ".partial"), "wb") as fh:
            np.savez_compressed(fh, **fixed)
        os.replace(os.path.join(d, FIXED + ".partial"), os.path.join(d, FIXED))
        rep = dict(make=cam[0], model=cam[1], groups=info, fixed=fixed_info, raws_folder=folder,
                   patches=int(sum(v["patches"] for v in info.values())))
        with open(os.path.join(d, BANK_META), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        _CACHE.pop(cam, None)
        log.info("deep calibration: %s %s, %d patches in %d groups, fixed pattern at %d ISOs", cam[0], cam[1],
                 rep["patches"], len(info), len(fixed_info))
        reports.append(rep)
    return reports


# ------------------------------------------------------------------------------------------------ reading it
class NoiseBank:
    """A camera's stored real noise. ``patches(iso, cls)`` -> (uint16 (n, P, P), info) or None."""

    def __init__(self, make: str, model: str):
        d = local_data_dir(make, model)
        self.make, self.model = make, model
        self.meta = json.load(open(os.path.join(d, BANK_META), encoding="utf-8"))
        self._z = np.load(os.path.join(d, BANK), allow_pickle=False)
        self._fixed = np.load(os.path.join(d, FIXED), allow_pickle=False) if os.path.isfile(
            os.path.join(d, FIXED)) else None
        self._mem: dict = {}

    @property
    def groups(self) -> dict:
        return self.meta.get("groups", {})

    def isos(self, cls: str = "short") -> list:
        return sorted(g["iso"] for g in self.groups.values() if g["cls"] == cls)

    def patches(self, iso: float, cls: str = "short"):
        key = bank_key(iso, cls)
        if key not in self.groups:
            return None
        if key not in self._mem:
            self._mem[key] = self._z[key]
        return self._mem[key], self.groups[key]

    def fixed_lines(self, iso: float):
        """(rows, cols) fixed pattern above black in DN for this ISO, or None."""
        if self._fixed is None or f"rows_{float(iso):g}" not in self._fixed.files:
            return None
        return self._fixed[f"rows_{float(iso):g}"], self._fixed[f"cols_{float(iso):g}"]


_CACHE: dict = {}


def find_bank(make: str, model: str) -> NoiseBank | None:
    key = (make, model)
    if key not in _CACHE:
        try:
            _CACHE[key] = NoiseBank(make, model)
        except (OSError, ValueError, KeyError):
            _CACHE[key] = None
    return _CACHE[key]

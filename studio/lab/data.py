"""Training data for a camera's own network: clean photos + that camera's real noise (BUILD_PLAN_NOISE F.2).

No PyTorch here: everything is NumPy, so the Studio can survey a folder and the tests can run without it.

Clean truth is a base-ISO raw reduced 2x in its colour planes (synth.reduce_planes): its own noise halved, still a
true mosaic. From each photo a handful of crops are kept (the most detailed ones and a few at random, so skies and
walls are in there too) as 16-bit floats in calibration_local/lab/clean. Training then draws crops, picks an ISO,
scales the exposure, and lets synth.NoiseSource put the camera's noise on.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .. import log as slog, paths
from ..engine.calib import synth as S
from ..engine.calib.noise import _exact_table, anscombe

log = slog.get("lab.data")
CROP = 512                  # side of a stored clean crop (photosites of the reduced mosaic)
CROPS_PER_PHOTO = 12
CLEAN_MAX_ISO = 400.0
HOLD_OUT_EVERY = 8          # every 8th photo is never trained on: the honest score comes from those


def clean_dir() -> str:
    d = paths.get("CALIBRATION_LOCAL") / "lab" / "clean"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def list_raws(folder: str) -> list:
    from ..engine.io.rawfile import RAW_EXTENSIONS
    out = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith((".", "$")) and d not in ("calibration_local", "cache")]
        out += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(RAW_EXTENSIONS)]
    return out


def survey(folder: str, progress=None, cancelled=None) -> dict:
    """Count what a folder holds for training, from the files' own data (nothing is decoded): how many raws, how
    many at ISO 400 or lower (usable as clean truth), which cameras, and how spread out they are in time (a
    rough sign of variety: 200 frames of one burst are one photo)."""
    from ..engine.io.metadata import read_metadata
    files = list_raws(folder)
    usable, cams, days, hours = [], {}, set(), set()
    for k, f in enumerate(files):
        if cancelled is not None and cancelled():
            break
        try:
            m = read_metadata(f, None)
        except Exception:                                     # noqa: BLE001  (an unreadable file is not a photo)
            continue
        if m.iso and m.iso <= CLEAN_MAX_ISO and (m.exposure_s or 0) < 30:
            usable.append(f)
            cams[m.model or "?"] = cams.get(m.model or "?", 0) + 1
            if m.datetime:
                days.add(m.datetime[:10]), hours.add(m.datetime[:13])
        if progress is not None and (k % 25 == 0 or k == len(files) - 1):
            progress(k + 1, len(files))
    return dict(folder=folder, raws=len(files), usable=len(usable), files=usable, cameras=cams,
                days=len(days), sessions=len(hours))


def _to_rggb(mos: np.ndarray, cfa_rgb: np.ndarray) -> np.ndarray:
    """Shift so that red is at the origin: every camera's truth then has the same pattern."""
    pos = [(i, j) for i in (0, 1) for j in (0, 1) if int(cfa_rgb[i, j]) == 0][0]
    m = mos[pos[0]:, pos[1]:]
    return m[:(m.shape[0] // 2) * 2, :(m.shape[1] // 2) * 2]


def prepare_photo(path: str) -> str | None:
    """One clean photo -> its stored crops (skipped when already there). Returns the crop file, or None when
    the photo cannot be used (unreadable, not base ISO, blown out)."""
    from ..engine.io.rawfile import UnsupportedRaw, open_raw
    key = hashlib.blake2b((os.path.abspath(path) + str(os.path.getsize(path))).encode(), digest_size=10).hexdigest()
    out = os.path.join(clean_dir(), key + ".npz")
    if os.path.isfile(out):
        return out
    try:
        img = open_raw(path)
    except (UnsupportedRaw, OSError):
        return None
    if not img.meta.iso or img.meta.iso > CLEAN_MAX_ISO:
        return None
    black = img.black_for_positions(img.black_libraw)
    span = img.clip_level() - float(np.mean(black))
    vis = img.visible.astype(np.float32)
    for i in (0, 1):
        for j in (0, 1):
            vis[i::2, j::2] -= np.float32(black[i, j])
    mos = S.reduce_planes(_to_rggb(vis / np.float32(span), img.cfa_rgb), 2)
    del vis, img
    h, w = mos.shape
    if h < CROP or w < CROP:
        return None
    # candidates on a grid; score = fine detail in luminance (skies score low, foliage high)
    cand = [(y, x) for y in range(0, h - CROP + 1, CROP // 2) for x in range(0, w - CROP + 1, CROP // 2)]
    score = []
    for y, x in cand:
        c = mos[y:y + CROP:2, x:x + CROP:2]
        score.append(float(np.abs(np.diff(c, axis=0)).mean() + np.abs(np.diff(c, axis=1)).mean())
                     / (float(c.mean()) + 0.01) if float(c.max()) < 0.98 else -1.0)
    order = [k for k in np.argsort(score)[::-1] if score[k] >= 0]
    if not order:
        return None
    rng = np.random.default_rng(int(key[:8], 16))
    n_top = CROPS_PER_PHOTO * 2 // 3
    pick = list(order[:n_top]) + list(rng.permutation(order[n_top:])[:CROPS_PER_PHOTO - n_top])
    crops = np.stack([mos[cand[k][0]:cand[k][0] + CROP, cand[k][1]:cand[k][1] + CROP] for k in pick])
    with open(out + ".partial", "wb") as fh:
        np.savez_compressed(fh, crops=crops.astype(np.float16), source=np.array(os.path.basename(path)))
    os.replace(out + ".partial", out)
    return out


class CleanSet:
    """The stored clean crops, split by photo into training and held-out."""

    def __init__(self, crop_files: list):
        files = sorted(crop_files)
        self.hold = [f for k, f in enumerate(files) if k % HOLD_OUT_EVERY == HOLD_OUT_EVERY - 1]
        self.train = [f for f in files if f not in set(self.hold)] or files
        self._mem: dict = {}

    def __len__(self) -> int:
        return len(self.train) + len(self.hold)

    def crops(self, f: str) -> np.ndarray:
        if f not in self._mem:
            if len(self._mem) > 400:
                self._mem.pop(next(iter(self._mem)))
            self._mem[f] = np.load(f)["crops"]
        return self._mem[f]

    def draw(self, rng: np.random.Generator, size: int, held_out: bool = False) -> np.ndarray:
        """One clean mosaic patch (size x size, RGGB, normalised), with a random exposure and orientation."""
        files = self.hold if held_out and self.hold else self.train
        c = self.crops(files[int(rng.integers(len(files)))])
        p = c[int(rng.integers(c.shape[0]))].astype(np.float32)
        size = min(size, (min(p.shape) // 8) * 8)
        y = int(rng.integers(0, (p.shape[0] - size) // 2 + 1)) * 2
        x = int(rng.integers(0, (p.shape[1] - size) // 2 + 1)) * 2
        p = p[y:y + size, x:x + size]
        if rng.random() < 0.5:
            p = p.T                                           # RGGB transposed is still RGGB
        gain = float(np.exp(rng.uniform(np.log(0.03), np.log(1.6))))     # deep shadow to bright
        return np.clip(p * gain, 0.0, 1.0).astype(np.float32)


def pack(mos: np.ndarray) -> np.ndarray:
    """(h, w) RGGB mosaic -> (4, h/2, w/2): R, G1, G2, B."""
    return np.stack([mos[0::2, 0::2], mos[0::2, 1::2], mos[1::2, 0::2], mos[1::2, 1::2]])


def expected_vst(clean: np.ndarray, a: float, b: float) -> np.ndarray:
    """What the stabilised value of a clean signal is on average (the target the network is taught toward, so
    that Clean's exact inverse brings it back unbiased)."""
    lam, ef = _exact_table(b / (a * a))
    mu = np.maximum(clean, 0.0) / a
    hi = 2.0 * np.sqrt(mu + 0.125 + b / (a * a))              # asymptotic form beyond the table
    return np.where(mu > lam[-1], hi, np.interp(mu, lam, ef)).astype(np.float32)


def make_pair(clean: np.ndarray, source: S.NoiseSource, iso: float, rng: np.random.Generator) -> tuple:
    """(noisy planes, target planes), both (4, h/2, w/2) in unit-noise units, for one clean mosaic patch."""
    model = source.model(iso)
    noisy = source.add(clean, iso, rng)
    a, b = float(model.a[0, 0]), model.b
    x = np.stack([anscombe(pl, a, float(b[i, j])) for pl, (i, j) in zip(pack(noisy), ((0, 0), (0, 1), (1, 0), (1, 1)))])
    t = np.stack([expected_vst(pl, a, float(b[i, j])) for pl, (i, j) in zip(pack(clean), ((0, 0), (0, 1), (1, 0), (1, 1)))])
    return x.astype(np.float32), t


def batch(clean_set: CleanSet, source: S.NoiseSource, rng: np.random.Generator, n: int = 16, size: int = 256,
          held_out: bool = False, isos: list | None = None) -> tuple:
    isos = isos or source.isos()
    xs, ts = [], []
    for _ in range(n):
        x, t = make_pair(clean_set.draw(rng, size, held_out), source, float(isos[int(rng.integers(len(isos)))]), rng)
        xs.append(x), ts.append(t)
    return np.stack(xs), np.stack(ts)


def texture_kept(out: np.ndarray, noisy: np.ndarray, target: np.ndarray) -> float:
    """Share of the truth's fine detail that survived (1 = all): the result's detail projected on the truth's.
    Noise does not line up with the truth, so leaving noise in cannot raise this."""
    def fine(v):
        s = (v + np.roll(v, 1, -1) + np.roll(v, -1, -1) + np.roll(v, 1, -2) + np.roll(v, -1, -2)) / 5.0
        return v - s
    d, t = fine(out), fine(target)
    return float((d * t).sum() / max(float((t * t).sum()), 1e-9))


def write_card(path_onnx: str, **info) -> None:
    with open(os.path.splitext(path_onnx)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)

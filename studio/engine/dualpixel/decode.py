"""Dual Pixel RAW: the two frames, the two halves, and each half's shading (BUILD_PLAN_TABS 3.2).

Canon's Dual Pixel RAW (R5, 5D Mark IV, EOS R) stores two raw frames: frame 1 is the normal picture (both
photodiodes of every pixel, "A+B"), frame 2 is one photodiode only ("half 1": light that came through one side of
the lens). The other half is the difference. Each half looks through one side of the lens, so the two halves see
the scene from two slightly different points: a little parallax (depth), and opposite left-right shading (each half
is darker toward one side, with a colour cast: Bill measured B/A from 2.7 down to 0.31 across the frame at f/11).

Reading the second frame (checked 2026-11-16 on Bill's R5 eagle file, docs/RESEARCH.md):
  - CR3 (R5, EOS R): a Dual Pixel CR3 carries five tracks instead of four; the fifth is the second raw frame, in
    the same CRAW format as the main one. LibRaw 0.22 hands back the main frame again for ``shot_select=1``, so
    the Studio reads the fifth track itself: a temporary copy of the file whose main-raw track tables (sample
    description, size, offset) point at the fifth track, decoded by LibRaw as usual, then deleted. The result is
    bit-identical to DPRSplit's second DNG. Finding a Dual Pixel CR3 needs only the file's box structure (no
    decode), so the Library can mark them cheaply.
  - CR2 (5D Mark IV): LibRaw's ``shot_select=1`` (one raw decode to know).
  - A pair of DNGs from DPRSplit (frame 1 = A+B, frame 2 = one half) works too (``open_dual_pair``).
A normal raw has no second frame: said in plain words with how to turn Dual Pixel RAW on. Which side of the lens
"half 1" looks through is not assumed (see depth.py).

Arrays are the camera crop (even origin, CFA kept), black subtracted, 1.0 = the sensor's clip, float32. No Qt.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from ..io.rawfile import UnsupportedRaw

log = slog.get("engine.dualpixel")

HOW_TO = ("Dual Pixel RAW is a camera setting: on the R5, Shooting menu (red) > Dual Pixel RAW > Enable; on the "
          "5D Mark IV, Shooting menu > Dual Pixel RAW > Enable. Files are about twice the size and bursts are "
          "slower while it is on.")
NOT_DUAL = "This photo was not shot in Dual Pixel RAW."
CLIP = 0.995                        # a photosite at or above this (normalised) is clipped


@dataclass
class DualPixel:
    image: object                   # engine Image of frame 1 (A+B): the rest of the Studio keeps working on it
    ab: np.ndarray                  # A+B, normalised, camera crop
    a: np.ndarray                   # half 1, same scale as ab (about ab / 2), from frame 2 with its own black level
    b: np.ndarray                   # half 2 = ab - a, floored at 0
    clipped_ab: np.ndarray          # bool
    clipped_a: np.ndarray           # bool
    pattern: np.ndarray             # 2x2 of 0/1/2 at the crop origin
    wb: np.ndarray                  # as-shot multipliers (G = 1)
    cam_xyz: np.ndarray
    flip: int
    crop: tuple                     # (x, y, w, h) of the camera crop in the visible area
    level_ratio: float              # mean A+B over mean half 1 (about 2)
    source: str = "dual pixel raw"  # dual pixel raw | dprsplit pair | synthetic
    path: str = ""
    notes: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        return self.ab.shape

    def half_image(self, which: int = 1, gains=None):
        """An engine Image of one half, scaled to the brightness of the full picture (x2, shading corrected when
        ``gains`` is given), developed exactly like the photo by the normal pipeline. Values above the photo's clip
        are kept (that is where the extra highlight stop lives)."""
        half = self.a if which == 1 else self.b
        g = None
        if gains is not None:
            g = (gains[0] if which == 1 else gains[1]).full(self.shape)
        return _half_as_image(self, half, g)


def _half_as_image(dp: DualPixel, half: np.ndarray, gain: np.ndarray | None):
    """Build a RawImage whose visible camera crop holds 2 x half (in raw numbers of frame 1), the rest of the
    sensor (masked borders) copied from frame 1, so levels, banding and every later stage behave the same."""
    from ..pipeline.pipeline import Image
    src = dp.image
    raw = src.raw
    lv = src.levels(True)
    x, y, w, h = dp.crop
    full = raw.mosaic.copy()
    vis = full[raw.top:raw.top + raw.height, raw.left:raw.left + raw.width]
    v = 2.0 * half if gain is None else 2.0 * half * gain
    rng = lv.range
    blk = lv.black
    out = np.empty(v.shape, np.float32)
    for i in (0, 1):
        for j in (0, 1):
            out[i::2, j::2] = blk[i, j] + v[i::2, j::2] * np.float32(rng[i, j])
    vis[y:y + h, x:x + w] = np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    hr = dataclasses.replace(raw, mosaic=full)
    img = Image(hr)
    img._levels = dict(src._levels) if src._levels else {}
    img._ca = src.lateral_ca()                     # the same lens: one lateral CA measurement for all three
    return img


def _plane_means(x: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(x)
    return float(x[m].mean()) if m.any() else 0.0


def make_dual(image, half_mosaic_full: np.ndarray, black_half=None, source: str = "dual pixel raw",
              path: str = "") -> DualPixel:
    """A DualPixel from the A+B engine Image and frame 2's full sensor mosaic (uint16, same layout as frame 1).
    ``black_half``: frame 2's black level per CFA position of the visible area (2x2), default frame 1's."""
    from ..raw import linearize as lin
    raw = image.raw
    lv = image.levels(True)
    cx, cy, cw, ch = raw.crop
    cw, ch = cw // 2 * 2, ch // 2 * 2
    ab = lin.normalise(raw, lv, (cx, cy, cw, ch))
    vis = half_mosaic_full[raw.top:raw.top + raw.height, raw.left:raw.left + raw.width]
    hv = vis[cy:cy + ch, cx:cx + cw].astype(np.float32)
    bh = lv.black if black_half is None else np.asarray(black_half, np.float32).reshape(2, 2)
    a = np.empty(hv.shape, np.float32)
    for i in (0, 1):
        for j in (0, 1):
            a[i::2, j::2] = (hv[i::2, j::2] - bh[i, j]) / np.float32(lv.range[i, j])
    clipped_ab = ab >= CLIP
    white_h = float(raw.white_level)
    clipped_a = hv >= 0.995 * white_h
    b = np.maximum(ab - a, 0.0)
    ok = (~clipped_ab) & (ab > 0.02) & (ab < 0.9)
    ratio = _plane_means(ab, ok) / max(_plane_means(a, ok), 1e-9)
    dp = DualPixel(image=image, ab=ab, a=a, b=b, clipped_ab=clipped_ab, clipped_a=clipped_a,
                   pattern=np.asarray(raw.cfa_rgb).copy(), wb=np.asarray(raw.wb_as_shot, float).copy(),
                   cam_xyz=np.asarray(raw.cam_xyz, float).copy(), flip=int(raw.flip), crop=(cx, cy, cw, ch),
                   level_ratio=float(ratio), source=source, path=path or raw.path)
    if not 1.6 < ratio < 2.6:
        dp.notes.append(f"The second frame is {ratio:.2f}x darker than the photo (about 2 is usual): the halves "
                        f"may not be what the Studio expects.")
    log.info("dual pixel %s: level ratio %.3f, %.2f%% of A+B clipped, %.2f%% of half 1 clipped",
             os.path.basename(dp.path), ratio, 100 * clipped_ab.mean(), 100 * clipped_a.mean())
    return dp


# ------------------------------------------------------------------------------------------ CR3 second frame
def _boxes(buf, off: int, end: int) -> list:
    import struct
    out = []
    while off + 8 <= end:
        sz, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if sz == 1:
            sz = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        if sz < 8:
            break
        out.append((typ.decode("latin-1"), off, sz, hdr))
        off += sz
    return out


def cr3_raw_tracks(path: str) -> list:
    """The full-size raw tracks of a CR3 as [(stsd, stsz, co64, sample_size)], each a (offset, size) of the box in
    the file; previews (small CRAW tracks) and metadata are left out. One entry for a normal CR3, two for Dual
    Pixel RAW. Reads the first megabyte only. [] for anything else."""
    import struct
    try:
        with open(path, "rb") as fh:
            buf = fh.read(1 << 20)
    except OSError:
        return []
    if buf[4:8] != b"ftyp" or b"crx " not in buf[8:24]:
        return []
    top = _boxes(buf, 0, len(buf))
    moov = [b for b in top if b[0] == "moov"]
    if not moov:
        return []
    _, mo, ms, mh = moov[0]
    out = []
    for name, to, ts, th in _boxes(buf, mo + mh, min(len(buf), mo + ms)):
        if name != "trak":
            continue
        cur = (to, ts, th)
        ok = True
        for want in ("mdia", "minf", "stbl"):
            kids = [k for k in _boxes(buf, cur[0] + cur[2], cur[0] + cur[1]) if k[0] == want]
            if not kids:
                ok = False
                break
            cur = (kids[0][1], kids[0][2], kids[0][3])
        if not ok:
            continue
        kids = {k[0]: k for k in _boxes(buf, cur[0] + cur[2], cur[0] + cur[1])}
        if not all(k in kids for k in ("stsd", "stsz")) or ("co64" not in kids and "stco" not in kids):
            continue
        sd = kids["stsd"]
        if buf[sd[1] + sd[3] + 12:sd[1] + sd[3] + 16] != b"CRAW":
            continue
        sz = kids["stsz"]
        size0, count = struct.unpack(">II", buf[sz[1] + sz[3] + 4:sz[1] + sz[3] + 12])
        sample = size0 or (struct.unpack(">I", buf[sz[1] + sz[3] + 12:sz[1] + sz[3] + 16])[0] if count else 0)
        co = kids.get("co64") or kids["stco"]
        out.append(((sd[1], sd[2]), (sz[1], sz[2]), (co[1], co[2]), int(sample)))
    if not out:
        return []
    big = max(t[3] for t in out)
    return [t for t in out if t[3] > 0.25 * big]


def _cr3_second_frame(path: str, data: bytes | None = None):
    """(full sensor mosaic uint16, black per LibRaw channel) of a Dual Pixel CR3's second frame. The file is read
    into memory once, the main-raw track tables are pointed at the second raw track in that copy, and LibRaw
    decodes it from memory (BUILD_PLAN_DP3D A.3.2: no 70-120 MB temporary file). A rawpy that cannot read from
    memory falls back to the temporary copy (said in the log)."""
    import io
    import rawpy
    tr = cr3_raw_tracks(path)
    if len(tr) < 2:
        raise UnsupportedRaw(f"{os.path.basename(path)}: {NOT_DUAL} {HOW_TO}")
    main, second = tr[0], tr[1]
    if any(main[k][1] != second[k][1] for k in range(3)):
        raise UnsupportedRaw(f"{os.path.basename(path)}: a Dual Pixel file laid out in a way the Studio does not "
                             f"read yet. Make the two frames with DPRSplit and use 'Use two DPRSplit files'.")
    if data is None:
        with open(path, "rb") as fh:
            data = fh.read()
    buf = bytearray(data)
    for k in range(3):
        blob = bytes(buf[second[k][0]:second[k][0] + second[k][1]])
        buf[main[k][0]:main[k][0] + main[k][1]] = blob
    try:
        with rawpy.imread(io.BytesIO(bytes(buf))) as r:
            m = np.array(r.raw_image, dtype=np.uint16, copy=True)
            black = np.array(r.black_level_per_channel, dtype=np.float32)
        return m, black
    except (TypeError, AttributeError) as e:         # an old rawpy that only takes a path
        log.info("rawpy cannot read from memory (%s): the second frame is read through a temporary copy", e)
        return _cr3_second_frame_via_copy(path, main, second)


def _cr3_second_frame_via_copy(path: str, main, second):
    """The first way of reading the second frame (session 12): a temporary copy whose main-raw track tables point
    at the second raw track, decoded by LibRaw, then deleted."""
    import shutil
    import uuid
    import rawpy
    from ... import paths
    d = paths.temp_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = os.path.join(str(d), f"dp_{uuid.uuid4().hex[:10]}.CR3")
    try:
        shutil.copyfile(path, tmp)
        with open(path, "rb") as src, open(tmp, "r+b") as dst:
            for k in range(3):
                src.seek(second[k][0])
                blob = src.read(second[k][1])
                dst.seek(main[k][0])
                dst.write(blob)
        with rawpy.imread(tmp) as r:
            m = np.array(r.raw_image, dtype=np.uint16, copy=True)
            black = np.array(r.black_level_per_channel, dtype=np.float32)
        return m, black
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def is_dual_pixel(path: str) -> bool:
    """True when the file holds a second raw frame of the same size. A CR3 is told from its box structure alone
    (cheap: the Library can ask); a CR2 needs one raw decode (about a second), so ask that only in the Lab."""
    import rawpy
    if str(path).lower().endswith(".cr3"):
        return len(cr3_raw_tracks(path)) >= 2
    try:
        with rawpy.imread(path) as r0:
            s0 = (r0.sizes.raw_height, r0.sizes.raw_width)
        with rawpy.imread(path, shot_select=1) as r1:
            if (r1.sizes.raw_height, r1.sizes.raw_width) != s0:
                return False
            r1.unpack()                              # a normal CR3 opens here and only fails to decode
            return True
    except Exception:                                # noqa: BLE001  (no second frame, not a raw: not Dual Pixel)
        return False


def open_dual(path: str, image=None) -> DualPixel:
    """Both frames of a Dual Pixel RAW file. Raises UnsupportedRaw (NOT_DUAL + how to turn it on) for a normal raw,
    or the usual UnsupportedRaw for a file the decoder cannot read. ``image``: the photo already opened with
    api.open (a stack develops it first: the A+B frame is then decoded once, BUILD_PLAN_DP3D A.3.1)."""
    import rawpy
    from .. import api
    name = os.path.basename(path)
    if image is None:
        image = api.open(path)                       # raises UnsupportedRaw / FileNotFoundError itself
    if name.lower().endswith(".cr3"):
        m1, black = _cr3_second_frame(path)
        if m1.shape != image.raw.mosaic.shape:
            raise UnsupportedRaw(f"{name}: the second frame is not the size of the first.")
        return make_dual(image, m1, image.raw.black_for_positions(black), "dual pixel raw", path)
    try:
        with rawpy.imread(path, shot_select=1) as r1:
            if (r1.sizes.raw_height, r1.sizes.raw_width) != image.raw.mosaic.shape:
                raise UnsupportedRaw(f"{name}: {NOT_DUAL} {HOW_TO}")
            m1 = np.array(r1.raw_image, dtype=np.uint16, copy=True)
            black = np.array(r1.black_level_per_channel, dtype=np.float32)
    except UnsupportedRaw:
        raise
    except Exception as e:                           # noqa: BLE001  (LibRaw: no such image)
        log.info("%s: no second frame (%s)", name, e)
        raise UnsupportedRaw(f"{name}: {NOT_DUAL} {HOW_TO}") from e
    bh = image.raw.black_for_positions(black)
    return make_dual(image, m1, bh, "dual pixel raw", path)


def open_dual_pair(path_ab: str, path_half: str) -> DualPixel:
    """Two DNGs from DPRSplit: the A+B frame and one half. Checked by size and by brightness (the half is about
    half as bright; given the other way round, they are swapped)."""
    from .. import api
    ia, ih = api.open(path_ab), api.open(path_half)
    if ia.raw.mosaic.shape != ih.raw.mosaic.shape:
        raise UnsupportedRaw("These two files are not the same size: they are not the two frames of one photo.")
    dp = make_dual(ia, ih.raw.mosaic, ih.raw.black_for_positions(ih.raw.black_libraw), "dprsplit pair", path_ab)
    if dp.level_ratio < 0.75:                        # given the half first
        dp = make_dual(ih, ia.raw.mosaic, ia.raw.black_for_positions(ia.raw.black_libraw), "dprsplit pair",
                       path_half)
    if not 1.6 < dp.level_ratio < 2.6:
        raise UnsupportedRaw(f"These two files do not look like a Dual Pixel pair (the second is "
                             f"{dp.level_ratio:.2f}x darker, about 2 was expected).")
    return dp


# ------------------------------------------------------------------------------------------ shading
@dataclass
class GainField:
    """A smooth per-photosite gain for one half: exp of a quartic polynomial of position, per CFA position."""
    coef: dict                      # (i, j) -> 10 coefficients
    shape: tuple                    # (h, w) of the mosaic it was measured on
    source: str = "estimated"       # estimated (amber) | measured (cyan, from the lens wizard, later)

    def plane(self, i: int, j: int, shape=None) -> np.ndarray:
        h, w = (shape or self.shape)[:2]
        ph, pw = (h - i + 1) // 2, (w - j + 1) // 2
        y = (i + 2 * np.arange(ph, dtype=np.float64))
        x = (j + 2 * np.arange(pw, dtype=np.float64))
        u = (x - (w - 1) / 2.0) / w
        v = (y - (h - 1) / 2.0) / h
        # the quartic is a sum of u^p v^q terms: evaluated as (powers of v) @ C @ (powers of u), the same numbers
        # as on the full grid, about 30x faster (45 MP: 39 s -> about 1 s for both halves)
        C = np.zeros((5, 5))
        for (p, q), k in zip(_TERMS, self.coef[(i, j)]):
            C[q, p] = k
        Up = np.stack([u ** p for p in range(5)])
        Vq = np.stack([v ** q for q in range(5)])
        return np.exp(Vq.T @ C @ Up).astype(np.float32)

    def full(self, shape=None) -> np.ndarray:
        h, w = (shape or self.shape)[:2]
        out = np.empty((h, w), np.float32)
        for i in (0, 1):
            for j in (0, 1):
                out[i::2, j::2] = self.plane(i, j, (h, w))
        return out


# 15 terms: quartic in u, v. The plan said cubic; a cubic misses the edges of the shading Bill measured (a 9:1
# left-right swing between the halves) by about 3%, a quartic fits it within 1%.
_TERMS = [(p, q) for p in range(5) for q in range(5) if p + q <= 4]


def _poly(U, V):
    return np.stack([U ** p * V ** q for p, q in _TERMS], axis=-1)


def _poly_eval(U, V, c):
    out = np.zeros(U.shape, np.float64)
    for (p, q), k in zip(_TERMS, c):
        out += k * U ** p * V ** q
    return out


def _lowpass_valid(x: np.ndarray, ok: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter
    w = ok.astype(np.float32)
    num = gaussian_filter(np.where(ok, x, 0).astype(np.float32), sigma, mode="nearest")
    den = gaussian_filter(w, sigma, mode="nearest")
    return num / np.maximum(den, 1e-6), den


def _fit_log_ratio(r: np.ndarray, ok: np.ndarray, ys, xs, h, w):
    """Robust quartic fit of log ratio over the binned grid (rounds dropping the worst 10%)."""
    v = (ys - (h - 1) / 2.0) / h
    u = (xs - (w - 1) / 2.0) / w
    U, V = np.meshgrid(u, v)
    m = ok & np.isfinite(r) & (r > 0)
    if m.sum() < 20:
        return np.zeros(len(_TERMS))
    X = _poly(U[m], V[m])
    y = np.log(r[m])
    keep = np.ones(len(y), bool)
    c = np.zeros(len(_TERMS))
    for _ in range(3):
        c, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        res = np.abs(X @ c - y)
        thr = np.quantile(res[keep], 0.9)
        nk = res <= thr
        if nk.sum() < 20:
            break
        keep = nk
    return c


def shading(dp: DualPixel, sigma_frac: float = 1.0 / 12, binning: int = 8) -> tuple[GainField, GainField]:
    """Per-half flat field estimated from this photo: for each CFA position, half / (A+B / 2) in 8 x 8 blocks (the
    picture cancels in the ratio; clipped and near-black photosites left out), lightly smoothed, fitted with a
    quartic in position (robust: the worst 10% dropped, which removes depth edges where the halves disagree). The
    gains bring each half to A+B / 2 on average. ``sigma_frac`` (of the frame width) caps the smoothing. A
    measured flat field from the lens wizard replaces this later (same return type)."""
    h, w = dp.shape
    coef_a, coef_b = {}, {}
    k = binning
    for i in (0, 1):
        for j in (0, 1):
            ab = dp.ab[i::2, j::2]
            a = dp.a[i::2, j::2]
            b = dp.b[i::2, j::2]
            ok = (~dp.clipped_ab[i::2, j::2]) & (~dp.clipped_a[i::2, j::2]) & (ab > 0.01) & (ab < 0.9)
            ph, pw = ab.shape[0] // k * k, ab.shape[1] // k * k

            def bsum(x):
                return x[:ph, :pw].reshape(ph // k, k, pw // k, k).sum(axis=(1, 3))
            n = bsum(ok.astype(np.float32))
            sab = bsum(np.where(ok, ab, 0))
            sa = bsum(np.where(ok, a, 0))
            sb = bsum(np.where(ok, b, 0))
            sig = min(1.0, max(0.5, (pw // k) * sigma_frac / 8.0))
            valid = n > 0.75 * k * k
            abl, _ = _lowpass_valid(sab / np.maximum(n, 1), valid, sig)
            al, _ = _lowpass_valid(sa / np.maximum(n, 1), valid, sig)
            bl, _ = _lowpass_valid(sb / np.maximum(n, 1), valid, sig)
            ys = i + 2 * (np.arange(ph // k) * k + (k - 1) / 2.0)
            xs = j + 2 * (np.arange(pw // k) * k + (k - 1) / 2.0)
            ca = _fit_log_ratio(al / np.maximum(abl / 2.0, 1e-9), valid, ys, xs, h, w)
            cb = _fit_log_ratio(bl / np.maximum(abl / 2.0, 1e-9), valid, ys, xs, h, w)
            coef_a[(i, j)] = (-ca).tolist()           # gain = 1 / ratio
            coef_b[(i, j)] = (-cb).tolist()
    return GainField(coef_a, (h, w)), GainField(coef_b, (h, w))

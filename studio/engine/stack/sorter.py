"""Card to finished stacks (BUILD_PLAN_PLATE_GLOW_BURST_MACRO.md, block F): sort a card or folder of photos into focus
stacks, bursts and single shots, and leave out the frames of a stack that would hurt it, each with a plain reason.

Bill hand-holds his macro work and focuses by hand, so a card holds runs of frames of one subject with the focus moving
through it, some soft, some shaken, some taken while the subject swayed. This module finds those runs and judges every
frame before the Focus Stack engine (``stack/focus.py``) lines them up and merges them.

Speed: a frame is judged from the full-size JPEG every Canon raw carries, decoded at 1/8 size (PIL draft mode), so a
card of hundreds of raws is sorted in seconds per hundred. What needs the raw itself (lining up, the moved check, the
merge) happens later, per stack, in ``stack/focus.py``.

What the first pass judges per frame (``judge``), all relative to the frame's own stack:
* sharpness map: noise-floored Laplacian energy in blocks, normalised per block by the stack's best (so a frame's
  "share" is where it is the sharpest frame of the stack);
* soft everywhere: in no part of the frame does it come close to the sharpest frame there (and it is almost never the
  sharpest itself): a smooth subject in front of a busy background counts like any other part;
* adds nothing: it is the sharpest frame almost nowhere and a neighbour is as sharp wherever it is;
* shaken: its gradients point one way (structure-tensor anisotropy) much more than its neighbours';
* camera moved too far: its shift from the stack's middle frame is a large part of the frame;
* exposure jump: its exposure (ISO x time / f-number^2) differs from the stack's median by more than one stop.
"""
from __future__ import annotations

import io
import logging
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, uniform_filter

from ... import log as slog

log = logging.getLogger(__name__)

PROXY = 1024                 # long side of the image a frame is judged on
BLOCKS = (24, 36)            # sharpness blocks (rows, cols)
GAP_S = 2.0                  # a longer pause between frames ends a run
SAME_SCENE = 0.12            # phase-correlation peak between neighbours below this: a different scene
SHIFT_MAX = 0.25             # a neighbour shifted more than this share of the frame: a different framing
SOFT_ALL = 0.25              # nowhere within this share of the best frame's sharpness, and rarely the sharpest: soft
SOFT = 0.45                  # ... within this share, and rarely the sharpest: adds nothing (a softer repeat)
RARE = 0.02                  # "rarely the sharpest": in less than this share of the parts with detail
TEX_FLOOR = 0.04             # blocks with detail: the best frame's sharpness above this share of the stack's top
NOTHING = 0.004              # sharpest frame for less than this share of the textured blocks: adds nothing
SHAKE = 1.6                  # anisotropy this many times the neighbours' median: shaken
LURCH = 0.22                 # shift from the stack's middle frame over this share of the frame: camera moved too far
STOPS = 1.0                  # exposure more than this many stops from the stack's median: exposure jump
MIN_STACK = 3
BURST_SWING = 0.5            # median (best - median) / best over the textured blocks below this: the focus did not move

STRICTNESS = {
    "gentle": {"name": "Gentle", "scale": 0.6,
               "best": "keeping every frame that might add something; stacks shot carefully on a rail or tripod",
               "weak": "a shaken or soft frame can stay in and lower the result where it is used"},
    "normal": {"name": "Normal", "scale": 1.0,
               "best": "hand-held stacks: leaves out the frames that clearly hurt and keeps the rest",
               "weak": "a borderline frame may go either way; check the reasons and put one back with a click"},
    "strict": {"name": "Strict", "scale": 1.5,
               "best": "busy hand-held runs with many frames per depth: keeps only the clearly good ones",
               "weak": "can leave a thin gap in depth where the only frame there was a little soft"},
}
KIND_TEXT = {"focus": "Focus stack", "burst": "Burst (not a focus stack)", "single": "Single shot"}
WHAT = ("Point it at a card or a folder: it finds every focus stack, leaves out the frames that would hurt each one "
        "(and says why), and stacks them all. Bursts and single shots are listed and left alone.")
HOW = ("Choose folder..., drop a folder here, or in the Library right-click a selection: Find focus stacks. After an "
       "import from a card, the import's summary offers it too.")
HONEST = ("A frame left out is only left out of the stack: nothing is deleted or moved, and one click puts it back. "
          "Grouping is inferred from time, scene and how the sharp area moves (amber when unsure). A gap in depth "
          "that no frame got sharp stays soft in the result. A subject that moved in every frame cannot be made "
          "sharp everywhere. A hand-held sweep also shifts the viewpoint a little, so near depth edges the line-up "
          "can only partly follow it, and frames that drifted are cropped to the area every frame covers.")


@dataclass
class Frame:
    path: str
    name: str
    number: int = -1             # the camera's file number (3C9A2345 -> 2345); orders frames within a second
    time_s: float | None = None  # capture time, seconds (whole seconds from the raw's metadata)
    ev: float | None = None      # log2(ISO x exposure / f^2): one stop = 1.0
    focal: float = 0.0
    iso: float = 0.0
    af: str = ""                         # the camera's focus mode (one-shot | servo | ai-focus | manual | "")
    grey: np.ndarray | None = None       # proxy luminance (perceptual), long side PROXY
    energy: np.ndarray | None = None     # (BLOCKS) noise-floored sharpness per block
    aniso: float = 0.0                   # gradient anisotropy over its sharpest blocks
    status: str = "ok"                   # ok (cyan) | warn (amber) | out (red)
    reason: str = ""
    share: float = 0.0                   # share of the stack's textured blocks where it is the sharpest frame
    shift: tuple = (0.0, 0.0)            # (dy, dx) from the stack's middle frame, proxy pixels
    problem: str = ""                    # could not be read


@dataclass
class Group:
    kind: str                    # focus | burst | single
    frames: list = field(default_factory=list)
    note: str = ""

    @property
    def used(self) -> list:
        return [f for f in self.frames if f.status != "out"]

    @property
    def left_out(self) -> list:
        return [f for f in self.frames if f.status == "out"]

    def text(self) -> str:
        n, k = len(self.frames), len(self.left_out)
        s = f"{KIND_TEXT[self.kind]}: {self.frames[0].name} to {self.frames[-1].name} ({n} frame{'s' * (n != 1)})"
        if self.kind == "focus" and k:
            s += f", {k} left out"
        return s + (f". {self.note}" if self.note else "")


# ------------------------------------------------------------------------------------------------ reading a frame
def _number(name: str) -> int:
    m = re.search(r"(\d{3,})(?=\D*$)", os.path.splitext(name)[0])
    return int(m.group(1)) if m else -1


def _seconds(text: str) -> float | None:
    m = re.match(r"(\d{4})[:\-](\d\d)[:\-](\d\d)[ T](\d\d):(\d\d):(\d\d)(?:\.(\d+))?", text or "")
    if not m:
        return None
    import datetime as _dt
    y, mo, d, h, mi, s = (int(v) for v in m.groups()[:6])
    frac = float("0." + m.group(7)) if m.group(7) else 0.0
    try:
        return _dt.datetime(y, mo, d, h, mi, s).timestamp() + frac
    except ValueError:
        return None


def proxy_grey(path: str) -> np.ndarray:
    """The frame's luminance with its long side about PROXY pixels: from the raw's embedded JPEG when there is one
    (fast), else from a half-size development."""
    from PIL import Image as PI
    im = None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        im = PI.open(path)
    else:
        try:
            import rawpy
            with rawpy.imread(path) as r:
                th = r.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                im = PI.open(io.BytesIO(th.data))
            else:
                im = PI.fromarray(th.data)
        except Exception:                                   # noqa: BLE001 - no embedded preview: develop instead
            im = None
    if im is not None:
        if hasattr(im, "draft"):
            im.draft("L", (PROXY, PROXY))
        im = im.convert("L")
        if max(im.size) > PROXY * 1.5:
            k = PROXY / float(max(im.size))
            im = im.resize((max(1, int(im.size[0] * k)), max(1, int(im.size[1] * k))), PI.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
        try:                                                 # the embedded JPEG is upright already for most bodies;
            from .frames import probe                        # the raw's flip is applied when the JPEG is not
            fi = probe([path])[0]
            if fi.flip and a.shape[0] < a.shape[1] and fi.height > fi.width:
                a = np.rot90(a, -1 if fi.flip == 6 else 1)
        except Exception:                                    # noqa: BLE001
            pass
        return a
    from .frames import scene_rgb, stack_edit, open_frame
    img = open_frame(path)
    rgb = scene_rgb(path, stack_edit(img), half=True)
    lum = rgb.mean(-1)
    lum = np.clip(lum / max(float(np.percentile(lum, 99.5)), 1e-6), 0, 1) ** (1 / 2.2)
    k = PROXY / float(max(lum.shape))
    from scipy.ndimage import zoom
    return zoom(lum, k, order=1).astype(np.float32)


def sharpness_blocks(grey: np.ndarray) -> tuple[np.ndarray, float]:
    """Noise-floored sharpness per block (BLOCKS) and the gradient anisotropy over the frame's sharpest blocks."""
    g = gaussian_filter(grey, 0.7)
    lap = laplace(g)
    e = uniform_filter(lap * lap, 5)
    noise = float(np.median(e)) * 0.5                       # flat areas: what noise and JPEG texture give
    e = np.maximum(e - noise, 0.0)
    H, W = e.shape
    by, bx = BLOCKS
    hh, ww = H // by * by, W // bx * bx
    blk = e[:hh, :ww].reshape(by, hh // by, bx, ww // bx).mean(axis=(1, 3))
    gy, gx = np.gradient(g)
    jxx = uniform_filter(gx * gx, 9)
    jyy = uniform_filter(gy * gy, 9)
    jxy = uniform_filter(gx * gy, 9)
    lam_d = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = lam_d / np.maximum(jxx + jyy, 1e-9)              # 0 = no direction .. 1 = all one way
    sharp = e > np.percentile(e, 97)
    aniso = float(np.median(coh[sharp])) if sharp.any() else 0.0
    return blk.astype(np.float32), aniso


def read_frames(paths, progress=None, cancel=None) -> list:
    from .frames import probe
    infos = probe(paths)
    out = []
    for k, fi in enumerate(infos):
        if cancel is not None and cancel.is_set():
            from .focus import Cancelled
            raise Cancelled()
        f = Frame(path=fi.path, name=fi.name, number=_number(fi.name), time_s=_seconds(fi.taken),
                  focal=float(fi.focal_mm or 0.0), iso=float(fi.iso or 0.0), af=fi.focus_mode)
        if fi.iso and fi.exposure_s and fi.fnumber:
            f.ev = math.log2(float(fi.iso) * float(fi.exposure_s) / float(fi.fnumber) ** 2)
        if not fi.ok:
            f.problem = fi.problem or "could not be read"
        else:
            try:
                f.grey = proxy_grey(fi.path)
                f.energy, f.aniso = sharpness_blocks(f.grey)
            except Exception as e:                           # noqa: BLE001 - a fault of the Studio: report it
                slog.save_error_report(f"Sorting photos: reading {fi.name}", e)
                f.problem = f"could not be judged ({e})"
        out.append(f)
        if progress is not None:
            progress((k + 1) / max(1, len(infos)), f"Reading {fi.name}")
    out.sort(key=lambda f: ((f.time_s or 0.0), f.number, f.name))
    return out


# ------------------------------------------------------------------------------------------------- grouping
def _hp(g: np.ndarray, size: int = 256) -> np.ndarray:
    from scipy.ndimage import zoom
    k = size / float(max(g.shape))
    s = zoom(g, k, order=1)
    return (s - gaussian_filter(s, 6.0)).astype(np.float32)


def _pair(a: Frame, b: Frame) -> tuple[float, float, float]:
    """Phase correlation of two frames' proxies: (peak, dy, dx) with dy, dx as shares of the frame."""
    from .align import phase_shift
    A, B = _hp(a.grey), _hp(b.grey)
    if A.shape != B.shape:
        return 0.0, 1.0, 1.0
    dy, dx, peak = phase_shift(A, B, iters=1)
    return float(peak), float(dy) / A.shape[0], float(dx) / A.shape[1]


def runs(frames: list, gap_s: float = GAP_S) -> list:
    """Split the card into runs: consecutive frames of one scene, close in time, same focal length."""
    out, cur = [], []
    for f in frames:
        if f.grey is None:
            if cur:
                out.append(cur)
            out.append([f])
            cur = []
            continue
        if cur:
            p = cur[-1]
            dt = (f.time_s - p.time_s) if (f.time_s is not None and p.time_s is not None) else None
            far = dt is not None and dt > gap_s
            other_lens = abs(f.focal - p.focal) > 0.5
            peak, dy, dx = _pair(p, f)
            moved = abs(dy) > SHIFT_MAX or abs(dx) > SHIFT_MAX
            # a frame shot within a second of the last one belongs to the run even when it is too blurred to
            # compare (the first or last frame of a hand-held sweep often is): judge() decides whether it helps
            quick = dt is not None and dt <= 1.0 and not other_lens
            if far or other_lens or (not quick and (peak < SAME_SCENE or moved)):
                out.append(cur)
                cur = []
        cur.append(f)
    if cur:
        out.append(cur)
    return out


def lined_up(run: list):
    """The run's sharpness blocks with every frame moved onto the middle frame (hand-held frames drift by a large
    part of the frame, and comparing blocks that hold different things says nothing). Shifts are chained neighbour to
    neighbour (neighbours look most alike), falling back to the middle frame when a neighbour pair does not agree.
    Returns (E (n, by, bx), valid blocks (by, bx): inside every frame, shifts [(dy, dx)] in shares of the frame)."""
    n = len(run)
    m = n // 2
    pos = [np.zeros(2)]
    for k in range(1, n):
        peak, dy, dx = _pair(run[k - 1], run[k])
        pos.append(pos[-1] + (np.array([dy, dx]) if peak >= SAME_SCENE * 0.5 else np.nan))
    pos = np.array(pos)
    for k in range(n):                                       # a step that failed: measure against the middle frame
        if np.isnan(pos[k]).any():
            _, dy, dx = _pair(run[m], run[k])
            pos[k] = (pos[m] if not np.isnan(pos[m]).any() else 0.0) + np.array([dy, dx])
    rel = pos - pos[m]
    H, W = run[m].grey.shape
    by, bx = BLOCKS
    E, valid = [], np.ones(BLOCKS, bool)
    yy = (np.arange(by) + 0.5) * (H / by)
    xx = (np.arange(bx) + 0.5) * (W / bx)
    for k, f in enumerate(run):
        dy, dx = rel[k][0] * H, rel[k][1] * W
        g = f.grey
        if g.shape != (H, W):
            from scipy.ndimage import zoom
            g = zoom(g, (H / g.shape[0], W / g.shape[1]), order=1)[:H, :W]
        g = _shift_whole(g, dy, dx)              # whole pixels: interpolating would soften the moved frames
        e, _ = sharpness_blocks(g)
        E.append(e)
        # a block is valid when its centre (with a margin of a block) holds this frame's own pixels
        okx = (xx - dx > W / bx) & (xx - dx < W - W / bx)
        oky = (yy - dy > H / by) & (yy - dy < H - H / by)
        valid &= oky[:, None] & okx[None, :]
    return np.stack(E), valid, [tuple(r) for r in rel]


def _shift_whole(g: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Move an image by whole pixels (edges repeated): content at p goes to p + (dy, dx)."""
    H, W = g.shape
    iy, ix = int(round(dy)), int(round(dx))
    if not (iy or ix):
        return g
    g = np.pad(g, ((max(iy, 0), max(-iy, 0)), (max(ix, 0), max(-ix, 0))), mode="edge")
    return g[max(-iy, 0):max(-iy, 0) + H, max(-ix, 0):max(-ix, 0) + W]


def _pixel_sharpness(grey: np.ndarray) -> np.ndarray:
    g = gaussian_filter(grey, 0.7)
    lap = laplace(g)
    e = uniform_filter(lap * lap, 5)
    return np.maximum(e - float(np.median(e)) * 0.5, 0.0)


def in_focus_mask(grey: np.ndarray, rel: float = 0.25) -> np.ndarray:
    """Focus peaking for one photo: where its fine detail is at least ``rel`` of the sharpest detail in the photo
    (and above what noise gives). Judged against the photo itself, so it shows which part is sharpest, not whether
    that part is sharp enough: a photo soft everywhere still shows its least soft part."""
    e = gaussian_filter(_pixel_sharpness(np.asarray(grey, np.float32)), 1.0)
    top = float(np.percentile(e, 99.7))
    return e > max(rel * top, 1e-9)


def focus_overlay(run: list, k: int | None = None, cyan: str = "#22d3ee", amber: str = "#f59e0b") -> tuple:
    """What is in focus, to see what missed (Bill 2026-11-17). On the run's proxies, lined up on the middle frame:
    k given: that frame in grey with CYAN where it is in focus (as sharp as the stack's best there, within 60%);
    k None: a quick sharpest-frame-per-place merge in grey with CYAN where some used frame is in focus and AMBER
    where there is detail at a coarser size but no frame got it sharp (missed focus, or background blur the
    photographer wanted). Only the area every frame covers counts. Returns (rgb uint8 (H, W, 3), share in focus,
    share missed)."""
    from scipy.ndimage import zoom
    n = len(run)
    m = n // 2
    _, _, rel = lined_up(run)
    H, W = run[m].grey.shape
    used = [j for j, f in enumerate(run) if f.status != "out"] or list(range(n))
    G, E = [], []
    common = np.ones((H, W), bool)
    for j in range(n):
        g = run[j].grey
        if g.shape != (H, W):
            g = zoom(g, (H / g.shape[0], W / g.shape[1]), order=1)[:H, :W]
        dy, dx = rel[j][0] * H, rel[j][1] * W
        iy, ix = int(round(dy)), int(round(dx))
        v = np.zeros((H, W), bool)                          # where this frame has its own pixels (8 px margin)
        v[max(iy, 0) + 8:H + min(iy, 0) - 8, max(ix, 0) + 8:W + min(ix, 0) - 8] = True
        if j in used:
            common &= v
        g = _shift_whole(g, dy, dx)
        G.append(g)
        E.append(gaussian_filter(_pixel_sharpness(g), 2.0))
    E = np.stack(E) * common
    emax = E[used].max(0)
    floor = 0.08 * float(np.percentile(emax[common], 99.5) if common.any() else emax.max()) + 1e-12
    sharp_any = (emax > floor) & common
    if k is not None:
        base = G[k]
    else:                                                   # a quick merge: each place from its sharpest frame
        pick = np.stack([gaussian_filter(E[j], 4.0) for j in used]).argmax(0)
        base = np.choose(pick, [G[j] for j in used])
    rgb = np.repeat((np.clip(base, 0, 1) ** 0.9 * 0.75)[..., None], 3, -1)
    rgb[~common] *= 0.35                                    # outside the area every frame covers: dimmed

    def hexrgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    c_cyan, c_amber = hexrgb(cyan), hexrgb(amber)
    if k is not None:
        inf = (E[k] >= 0.6 * emax) & sharp_any
        rgb[inf] = 0.5 * rgb[inf] + 0.5 * c_cyan
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), float(inf.mean()), 0.0
    cmax = np.max([uniform_filter(laplace(gaussian_filter(G[j], 3.0)) ** 2, 9) for j in used], 0) * common
    from scipy.ndimage import binary_dilation
    near_sharp = binary_dilation(sharp_any, iterations=6)  # the soft skirt of a sharp edge is not a miss
    missed = ~near_sharp & common & (cmax > 0.1 * float(np.percentile(cmax[common], 99.5) if common.any() else 0))
    rgb[sharp_any] = 0.5 * rgb[sharp_any] + 0.5 * c_cyan
    rgb[missed] = 0.45 * rgb[missed] + 0.55 * c_amber
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), float(sharp_any.mean()), float(missed.mean())


def classify(run: list) -> str:
    """focus: the sharp region moves through the run; burst: it stays put, or the camera's AI Servo AF was
    following a subject (the sharp area moves with the subject, not through depth: a dog running at the camera);
    single: one frame."""
    if len(run) < MIN_STACK or any(f.energy is None for f in run):
        return "single" if len(run) == 1 else "burst"
    if all(f.af == "servo" for f in run):
        return "burst"
    E, valid, _ = lined_up(run)                              # (n, by, bx)
    best = E.max(0)
    tex = (best > np.percentile(best[valid], 40)) & valid if valid.sum() >= 20 else best > np.percentile(best, 40)
    if tex.sum() < 10:
        return "burst"
    # in a burst every frame is about as sharp as the best in every block; in a focus stack most frames are far
    # softer than the best in most blocks (real cards: bursts 0.1-0.25, stacks 0.8-0.97)
    swing = float(np.median(((best - np.median(E, 0)) / np.maximum(best, 1e-12))[tex]))
    if swing < BURST_SWING:
        return "burst"
    arg = E.argmax(0)[tex]
    n = len(run)
    counts = np.bincount(arg, minlength=n) / float(tex.sum())
    spread = int((counts > 0.02).sum())
    # how far, in frames, the sharpest frame moves across the scene (a stack spans many; a burst stays near one)
    lo, hi = np.percentile(arg, [10, 90])
    return "focus" if spread >= 3 and (hi - lo) >= max(2, 0.2 * n) else "burst"


# -------------------------------------------------------------------------------------------------- judging
def judge(frames: list, strictness: str = "normal") -> list:
    """Status and reason for every frame of one focus stack (module text). Returns the frames."""
    s = STRICTNESS.get(strictness, STRICTNESS["normal"])["scale"]
    n = len(frames)
    for f in frames:
        f.status, f.reason, f.share = "ok", "", 0.0
    E, valid, rel = lined_up(frames)
    best = E.max(0)
    if valid.sum() < 20:
        valid = np.ones_like(valid)
    # detail anywhere above a small share of the stack's best: a smooth subject (a leaf in front of grass) counts
    # too, or a frame sharp only on it would look soft next to frames sharp on the busy background
    tex = (best > TEX_FLOOR * float(np.percentile(best[valid], 99))) & valid
    if tex.sum() < 10:
        tex = valid
    arg = E.argmax(0)
    counts = np.bincount(arg[tex], minlength=n) / float(max(1, tex.sum()))
    # how close each frame comes to the best frame, block by block, in the tenth of the blocks where it comes closest
    near = np.array([float(np.percentile((E[k] / np.maximum(best, 1e-12))[tex], 90)) for k in range(n)])
    evs = [f.ev for f in frames if f.ev is not None]
    ev_med = float(np.median(evs)) if evs else None
    an = np.array([f.aniso for f in frames])
    for k, f in enumerate(frames):
        f.share = float(counts[k])
        lo, hi = max(0, k - 3), min(n, k + 4)
        neigh = [an[j] for j in range(lo, hi) if j != k]
        an_ref = float(np.median(neigh)) if neigh else float(np.median(an))
        f.shift = rel[k]
        reasons_out, reasons_warn = [], []
        rare = counts[k] < RARE * s                           # (a soft frame that is still the sharpest
        if near[k] < SOFT_ALL * s and rare:                    # somewhere is kept: it is the best there is there)
            reasons_out.append("soft everywhere")
        if ev_med is not None and f.ev is not None and abs(f.ev - ev_med) > STOPS / s:
            reasons_out.append(f"exposure {f.ev - ev_med:+.1f} stops from the rest")
        if math.hypot(*f.shift) > LURCH / s:
            reasons_out.append("camera moved too far (it would crop the whole stack)")
        if an_ref > 0 and f.aniso > SHAKE / s * an_ref and f.aniso > 0.35:
            (reasons_out if f.aniso > 1.3 * SHAKE / s * an_ref else reasons_warn).append("shaken (blur one way)")
        if not reasons_out and (counts[k] < NOTHING * s or (near[k] < SOFT * s and rare)):
            mine = tex & (E[k] >= np.percentile(E[k][tex], 90)) if tex.any() else tex
            if mine.any():
                w = np.bincount(arg[mine], minlength=n)
                w[k] = 0
                j = int(np.argmax(w))
                reasons_out.append(f"adds nothing ({frames[j].name} is sharper at the same depth)")
            else:
                reasons_out.append("adds nothing")
        if reasons_out:
            f.status, f.reason = "out", "; ".join(reasons_out)
        elif reasons_warn:
            f.status, f.reason = "warn", "; ".join(reasons_warn)
    log.info("judged %d frames: %d out, %d warn", n, sum(f.status == "out" for f in frames),
             sum(f.status == "warn" for f in frames))
    return frames


def sort_frames(paths, strictness: str = "normal", gap_s: float = GAP_S, progress=None, cancel=None) -> list:
    """Read, group and judge: the whole first pass. Returns Groups in card order."""
    with slog.stage("sorter.sort", log, n=len(paths)):
        frames = read_frames(paths, progress, cancel)
        groups = []
        for r in runs(frames, gap_s):
            kind = classify(r)
            g = Group(kind, r)
            if kind == "focus":
                judge(r, strictness)
                if len(g.used) < MIN_STACK:
                    g.note = "too few good frames left to stack"
            elif kind == "burst" and len(r) >= MIN_STACK:
                g.note = ("AI Servo AF: the camera was following a subject; for Best of burst"
                          if all(f.af == "servo" for f in r) else "for Best of burst")
            if any(f.problem for f in r):
                g.note = (g.note + " " if g.note else "") + "; ".join(f"{f.name}: {f.problem}" for f in r if f.problem)
            groups.append(g)
    return groups


def report(groups: list) -> str:
    stacks = [g for g in groups if g.kind == "focus"]
    used = sum(len(g.used) for g in stacks)
    out = sum(len(g.left_out) for g in stacks)
    reasons = {}
    for g in stacks:
        for f in g.left_out:
            key = f.reason.split(";")[0].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
    n = sum(len(g.frames) for g in groups)
    bursts = sum(1 for g in groups if g.kind == "burst")
    singles = sum(1 for g in groups if g.kind == "single")
    why = ", ".join(f"{v} {k}" for k, v in sorted(reasons.items(), key=lambda t: -t[1]))
    s = (f"{len(stacks)} focus stack{'s' * (len(stacks) != 1)} found in {n} photos; {used} frames used, {out} left out"
         + (f" ({why})" if why else "") + f"; {bursts} burst{'s' * (bursts != 1)}, {singles} single shot"
         + ("s" * (singles != 1)) + ".")
    return s


def left_out_map(paths, strictness: str = "normal", gap_s: float = GAP_S, progress=None, cancel=None) -> dict:
    """For an import: {path: reason} of the frames that would be left out of a focus stack (Bill 2026-11-17: they
    are still copied, into the day's "Left out" folder). Bursts and single shots are never in it."""
    out = {}
    for g in sort_frames(paths, strictness, gap_s, progress, cancel):
        if g.kind == "focus":
            for f in g.left_out:
                out[f.path] = f.reason
    return out


def find_paths(items) -> list:
    """Files and folders (searched down through their sub-folders) to the photos the Studio can stack."""
    from .frames import supported
    out = []
    for it in items:
        it = os.fspath(it)
        if os.path.isdir(it):
            for root, dirs, files in os.walk(it):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                out += [os.path.join(root, n) for n in sorted(files)
                        if supported(os.path.join(root, n)) and "_Studio" not in n]
        elif os.path.isfile(it) and supported(it):
            out.append(it)
    from .frames import _is_raw
    raw_stems = {os.path.normcase(os.path.splitext(os.path.abspath(p))[0]) for p in out if _is_raw(p)}
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k in seen:
            continue
        if not _is_raw(p) and os.path.splitext(k)[0] in raw_stems:      # the camera's JPEG beside its raw
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


# ------------------------------------------------------------------------------------------------- stacking
MOVED_OUT = 0.25             # share of a frame's own sharp part that disagrees with its depth neighbours: left out
MOVED_WARN = 0.12            # ... used, with a warning
MOVED_MIN_AREA = 5e-4        # a sharp part smaller than this share of the frame is too small to judge
MOVED_DIFF = 0.5             # blurred difference over local contrast (and noise) above this: disagrees


def moved_shares(an) -> list:
    """For every frame of a Focus Stack analysis (focus order): the share of the place where it is the sharpest
    frame that disagrees with the frames on BOTH sides of it in depth once lined up, and that place's share of the
    frame. Disagreeing: lightly blurred copies differ by more than MOVED_DIFF times the local contrast (plus the
    noise of the pair, measured where there is no detail). A subject that swayed, or a frame taken much later in a
    back-and-forth sweep after the subject moved, disagrees; a frame that is only sharper or softer than its
    neighbours mostly does not. The engine's own moved measure is taken before local alignment and also counts
    hand-held parallax, so it is not used to leave frames out."""
    from scipy.ndimage import binary_dilation
    n = an.n
    if n < 2:
        return [(0.0, 0.0)] * n
    idx = np.asarray(an.idx)
    pres = an.presence()
    flat = pres < 0.2
    cache: dict = {}

    def low(k):
        if k not in cache:
            cache[k] = gaussian_filter(np.asarray(an.aligned[k], np.float32).mean(-1), 1.0)
        return cache[k]
    out = []
    for k in range(n):
        for old in [j for j in cache if j < k - 1]:
            cache.pop(old)
        R = (idx == k) & (pres > 0.5)
        area = float(R.mean())
        if area < MOVED_MIN_AREA:
            out.append((0.0, area))
            continue
        Rd = binary_dilation(R, iterations=3)

        def disagree(a, b):
            diff = np.abs(a - b)
            m = 0.5 * (a + b)
            sd = np.sqrt(np.maximum(uniform_filter(m * m, 9) - uniform_filter(m, 9) ** 2, 0.0))
            noise = 1.4826 * float(np.median(diff[flat])) if flat.sum() > 1000 else 0.0
            return diff / (sd + 2.5 * noise + 0.02 * m + 1e-4)
        nb = [j for j in (k - 1, k + 1) if 0 <= j < n]
        ds = [disagree(low(k), low(j)) for j in nb]
        share = float(np.mean(np.minimum.reduce(ds)[Rd] > MOVED_DIFF))
        if share > MOVED_WARN:
            # a suspect: blur this frame (sharp here) until it looks most like each neighbour (softer here), so a
            # big focus step between few frames is not taken for movement; what is left disagrees in shape
            ds = []
            for j in nb:
                best = None
                for sig in (0.0, 0.7, 1.4, 2.5, 4.0):
                    dd = disagree(gaussian_filter(low(k), sig) if sig else low(k), low(j))
                    sc = float(np.mean(dd[Rd] > MOVED_DIFF))
                    if best is None or sc < best[0]:
                        best = (sc, dd)
                ds.append(best[1])
            share = float(np.mean(np.minimum.reduce(ds)[Rd] > MOVED_DIFF))
        out.append((share, area))
    return out


def stack_group(g: Group, out_dir: str | None = None, params=None, progress=None, cancel=None) -> dict:
    """Line up, check and merge one focus stack with the Focus Stack engine. Frames the engine finds moved (the
    subject swayed, an insect walked) beyond MOVED are left out and the stack is lined up again once. Returns
    {"path", "used", "left_out", "notes"}; writes FIRST_to_LAST_Stack_Studio.tif beside the frames (or in out_dir)."""
    from . import focus as FO
    from .frames import default_out_dir, result_name
    from ..io.export import free_target
    params = params or FO.FocusParams()
    for attempt in range(2):
        paths = [f.path for f in g.used]
        if len(paths) < MIN_STACK:
            raise ValueError("too few good frames left to stack")
        an = FO.analyse(paths, params, progress=progress, cancel=cancel)
        by = {f.path: f for f in g.frames}
        try:
            mv = moved_shares(an)
        except Exception as e:                               # noqa: BLE001 - a fault: report, stack without it
            slog.save_error_report("Card to Stacks: the moved check", e)
            mv = [(0.0, 0.0)] * an.n
        out = [(p, m) for p, (m, _) in zip(an.paths, mv) if m > MOVED_OUT and p in by]
        if attempt == 0 and out and len(paths) - len(out) >= MIN_STACK:
            for p, m in out:
                by[p].status = "out"
                by[p].reason = (f"subject moved (about {100 * m:.0f}% of the part it is sharpest in disagrees with "
                                "the frames either side of it in depth)")
            an.close()
            continue
        for p, (m, _) in zip(an.paths, mv):
            f = by.get(p)
            if f is not None and f.status == "ok" and m > MOVED_WARN:
                f.status, f.reason = "warn", f"may have moved (about {100 * m:.0f}% of its sharp part disagrees)"
        for r in an.reports:
            f = by.get(r.path)
            if f is not None and not r.used and f.status != "out":
                f.status, f.reason = "out", r.reason or "the engine could not use it"
        d = out_dir or default_out_dir(paths[0])[0]
        os.makedirs(d, exist_ok=True)
        name = result_name([f.path for f in g.frames], "Focus")
        out = free_target(os.path.join(d, name))
        try:
            path = FO.build(an, params, out, progress=progress, cancel=cancel)
            notes = list(an.notes)
        finally:
            an.close()
        return {"path": path, "used": len(g.used), "left_out": len(g.left_out), "notes": notes}
    raise RuntimeError("unreachable")

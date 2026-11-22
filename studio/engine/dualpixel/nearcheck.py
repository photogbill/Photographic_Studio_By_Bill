"""Which way is near, read from scenes whose near end is obvious (BUILD_PLAN_DP3D A.1).

The two halves of a Dual Pixel photo cannot say on their own whether a place lies in front of or behind the plane
of focus (swap the halves and swap the sides: the same pictures). So the sign is read once per camera from scenes
whose near end everyone can see: ground receding at an angle (a path, a lawn, leaf litter, a table top), where the
bottom of the upright frame is nearer than the top.

Why it works: for a flat surface seen by a pinhole camera, inverse distance is an affine function of the image
position, and a Dual Pixel offset is an affine function of inverse distance (zero on the plane of focus). So over a
receding plane the offset is a plane in (row, column) too, and its slope down the upright frame has the sign of the
offset of places nearer than focus, whatever the focus distance was. A robust plane fit on the confident offsets
gives that slope; a frame whose fit is weak, or whose offsets do not change steadily down the frame, abstains.

Every frame votes; a stack votes when at least two thirds of its voting frames agree; the camera's sign is locked
only when at least three stacks vote the same way and none votes against (Bill's rule from his RF work: two or more
independent measures, and trust the agreement). Offsets live in the sensor's frame; they are turned upright first
(a portrait frame's "bottom" is a sensor column). No Qt here.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.dualpixel.nearcheck")

CONF_MIN = 0.3            # offsets trusted for the fit
MAX_SHIFT = 8.0           # search range (half-size px): twice a single photo's, so the ends of a deep stack vote too
LIMIT_PX = 0.75 * 2 * MAX_SHIFT   # |offset| (full-size px) past this: pinned at the end of the search, not an offset
MIN_SHARE = 0.03          # a frame needs confident offsets on at least this share of the picture
MIN_T = 8.0               # the slope must stand this many standard errors clear of zero
MIN_RHO = 0.6             # the row medians must change steadily down the frame (|Spearman| of the bin medians)
MIN_SWING_PX = 0.4        # the fitted offset must change by at least this much from top to bottom (full-size px)
STACK_AGREE = 2.0 / 3.0   # share of a stack's voting frames that must agree
LOCK_STACKS = 3           # stacks that must agree (with none against) to lock the camera's sign


@dataclass
class FrameVote:
    name: str
    vote: int = 0                 # +1 / -1: sign of the offset in front of focus; 0 = abstains
    slope_px: float = 0.0         # fitted offset change from the top of the upright frame to the bottom, full px
    t: float = 0.0                # slope / its standard error
    rho: float = 0.0              # Spearman of the row medians with the row
    share: float = 0.0            # share of the picture with confident offsets
    why: str = ""                 # plain words when it abstains

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class StackVote:
    name: str
    frames: list = field(default_factory=list)       # FrameVote
    vote: int = 0
    why: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "vote": self.vote, "why": self.why,
                "frames": [f.to_dict() for f in self.frames]}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def frame_vote(disp: np.ndarray, conf: np.ndarray, flip: int, name: str = "") -> FrameVote:
    """One frame's vote from its offset map and confidence (half size, sensor frame) and the photo's flip."""
    from ..pipeline.pipeline import apply_orientation
    d = np.asarray(apply_orientation(np.asarray(disp, np.float32), flip), np.float32)
    c = np.asarray(apply_orientation(np.asarray(conf, np.float32), flip), np.float32)
    H, W = d.shape
    fv = FrameVote(name=name)
    ok = (c > CONF_MIN) & (np.abs(d) < LIMIT_PX)
    fv.share = float(ok.mean())
    if fv.share < MIN_SHARE:
        fv.why = f"too little measured ({100 * fv.share:.1f}% of the frame)"
        return fv
    ys, xs = np.nonzero(ok)
    if ys.size > 200_000:                            # thin out evenly: the fit needs thousands, not millions
        k = int(np.ceil(ys.size / 200_000))
        ys, xs = ys[::k], xs[::k]
    v = d[ys, xs].astype(np.float64)
    w = c[ys, xs].astype(np.float64)
    y = ys / max(H - 1.0, 1.0)                       # 0 top .. 1 bottom (nearer on a receding ground)
    x = xs / max(W - 1.0, 1.0) - 0.5
    X = np.stack([np.ones_like(y), y - 0.5, x], 1)
    beta = np.zeros(3)
    ww = w.copy()
    for _ in range(8):                               # iteratively reweighted least squares (Huber)
        Xw = X * ww[:, None]
        beta = np.linalg.lstsq(Xw.T @ X, Xw.T @ v, rcond=None)[0]
        r = v - X @ beta
        s = float(1.4826 * np.median(np.abs(r))) + 1e-6
        hub = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12))
        ww = w * hub
    r = v - X @ beta
    s = float(1.4826 * np.median(np.abs(r))) + 1e-6
    # standard error of the slope from the scatter, counting neighbours as one (the offsets are smooth over a
    # window of about 16 half-size pixels: that many places are not independent)
    eff = max(10.0, ys.size / 64.0)
    vy = float(np.var(y))
    se = s / np.sqrt(max(eff * vy, 1e-12))
    fv.slope_px = float(beta[1])
    fv.t = float(abs(beta[1]) / se)
    # the offsets must change steadily down the frame, not just on average (ten bands of rows)
    edges = np.linspace(0, 1, 11)
    mids, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b if b < 1 else y <= b)
        if m.sum() >= 50:
            mids.append(0.5 * (a + b))
            meds.append(float(np.median(v[m] - beta[2] * x[m])))
    fv.rho = _spearman(np.asarray(mids), np.asarray(meds)) if len(mids) >= 4 else 0.0
    if fv.t < MIN_T:
        fv.why = f"the offsets do not change clearly down the frame (t {fv.t:.1f})"
    elif abs(fv.slope_px) < MIN_SWING_PX:
        fv.why = f"the offsets change too little down the frame ({abs(fv.slope_px):.2f} px)"
    elif len(mids) < 4 or abs(fv.rho) < MIN_RHO or np.sign(fv.rho) != np.sign(fv.slope_px):
        fv.why = f"the offsets do not change steadily down the frame (rho {fv.rho:.2f})"
    else:
        fv.vote = int(np.sign(fv.slope_px))
    return fv


def stack_vote(frames: list, name: str = "") -> StackVote:
    sv = StackVote(name=name, frames=list(frames))
    votes = [f.vote for f in frames if f.vote]
    if not votes:
        sv.why = "no frame could vote"
        return sv
    plus = sum(1 for v in votes if v > 0)
    minus = len(votes) - plus
    top = max(plus, minus)
    if top / len(votes) >= STACK_AGREE:
        sv.vote = 1 if plus > minus else -1
        sv.why = f"{top} of {len(votes)} voting frames agree"
    else:
        sv.why = f"its frames disagree ({plus} for +, {minus} for -)"
    return sv


def decide(stacks: list) -> dict:
    """{"sign": +1 / -1 / None, "locked": bool, "for": n, "against": n, "abstain": n, "sentence"}."""
    plus = sum(1 for s in stacks if s.vote > 0)
    minus = sum(1 for s in stacks if s.vote < 0)
    abst = sum(1 for s in stacks if s.vote == 0)
    sign = 1 if plus > minus else -1 if minus > plus else None
    agree, against = (plus, minus) if sign == 1 else (minus, plus)
    locked = sign is not None and agree >= LOCK_STACKS and against == 0
    if locked:
        sent = (f"Near is read the same way in {agree} independent scenes and against in none: locked "
                f"(offset {'positive' if sign > 0 else 'negative'} in front of focus).")
    elif sign is None:
        sent = "The scenes do not settle which way is near yet: add frames of ground receding at an angle."
    else:
        sent = (f"{agree} scene(s) say near is {'positive' if sign > 0 else 'negative'}, {against} against: not "
                f"locked yet (needs {LOCK_STACKS} with none against). Add more receding scenes.")
    return {"sign": sign, "locked": bool(locked), "for": int(agree if sign else 0),
            "against": int(against if sign else 0), "abstain": int(abst), "sentence": sent}


def measure_frame(path: str, gains=None, max_shift: float = MAX_SHIFT):
    """(disp, conf, flip, level_ratio, gains) of one Dual Pixel photo (single-photo window: 48 / 16; the search
    reaches +-16 full-size px: the near and far ends of a stack are far out of focus, and a receding plane's
    offsets stay a plane at any defocus)."""
    from . import decode as DD, depth as DEP
    dp = DD.open_dual(path)
    try:
        g = gains if gains is not None else DD.shading(dp)
        la, lb = DEP.halves_luma(dp, g)
        d, c = DEP.disparity(la, lb, 48, 16, max_shift)
        return d, c, int(dp.flip), float(dp.level_ratio), g
    finally:
        try:
            dp.image.drop_cache()
        except Exception:                            # noqa: BLE001  (only a cache)
            pass


def run(groups: dict, progress=None, cancel=None) -> dict:
    """The whole vote: ``groups`` {stack name: [paths]} (one photo alone is a stack of one). Returns {"stacks":
    [StackVote dicts], "decision": decide(...), "camera": model}. Frames that are not Dual Pixel are named and
    skipped."""
    from . import decode as DD
    from ..io.metadata import read_metadata
    stacks, model = [], ""
    total = sum(len(v) for v in groups.values()) or 1
    done = 0
    for name, paths_ in groups.items():
        fvs = []
        for p in paths_:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            if progress is not None:
                try:
                    progress(done / total, f"Reading the halves of {os.path.basename(p)}")
                except Exception:                    # noqa: BLE001  (a display must not stop the job)
                    pass
            done += 1
            try:
                if not model:
                    model = read_metadata(p, head_bytes=512 * 1024).model
                d, c, flip, _, _ = measure_frame(p)
                fv = frame_vote(d, c, flip, os.path.basename(p))
            except DD.UnsupportedRaw as e:
                fv = FrameVote(name=os.path.basename(p), why=str(e).split(":")[-1].strip()[:120])
            except Exception as e:                   # noqa: BLE001  (one bad frame abstains; the rest vote)
                slog.log_exception(f"near check {p}", log)
                slog.save_error_report(f"Which way is near: reading {os.path.basename(p)}")
                fv = FrameVote(name=os.path.basename(p), why=f"could not be read ({type(e).__name__})")
            log.info("near vote %s: %+d slope %.3f t %.1f rho %.2f share %.3f %s", fv.name, fv.vote, fv.slope_px,
                     fv.t, fv.rho, fv.share, fv.why)
            fvs.append(fv)
        stacks.append(stack_vote(fvs, name))
    dec = decide(stacks)
    return {"camera": model, "stacks": [s.to_dict() for s in stacks], "decision": dec,
            "date": _dt.date.today().isoformat()}


def save_to_profile(result: dict, model: str | None = None) -> str | None:
    """Save a locked answer to this PC's Dual Pixel profile for the camera (cyan). Returns the path, or None when
    the answer is not locked (nothing is saved from an unsettled vote)."""
    from . import dpprofile
    dec = result.get("decision") or {}
    if not dec.get("locked"):
        return None
    model = model or result.get("camera") or ""
    prof = dpprofile.local_or_new(model)
    prof.near_sign = int(dec["sign"])
    prof.near_how = (f"measured on your own frames {result.get('date', '')}: {dec['for']} receding scene(s) agree, "
                     f"none against")
    prof.near_date = result.get("date", "")
    prof.near_votes = [{"stack": s["name"], "vote": s["vote"], "why": s["why"],
                        "frames": len(s["frames"])} for s in result.get("stacks", [])]
    return prof.save()


def groups_by_time(paths: list, gap_s: float = 4.0) -> dict:
    """Paths split into stacks by a gap of more than ``gap_s`` seconds between capture times (the files' own
    clocks), named by their first and last file."""
    from ..stack import frames as F
    infos = F.probe(paths)
    import datetime as dt

    def t(fi):
        try:
            return dt.datetime.strptime(fi.taken[:19], "%Y:%m:%d %H:%M:%S").timestamp()
        except (ValueError, TypeError):
            return None
    out, cur, last = {}, [], None
    for fi in infos:
        if not fi.ok:
            continue
        tt = t(fi)
        if cur and (tt is None or last is None or tt - last > gap_s):
            out[_gname(cur)] = [f.path for f in cur]
            cur = []
        cur.append(fi)
        last = tt
    if cur:
        out[_gname(cur)] = [f.path for f in cur]
    return out


def _gname(fis) -> str:
    a = os.path.splitext(fis[0].name)[0]
    b = os.path.splitext(fis[-1].name)[0]
    return a if a == b else f"{a} to {b}"

"""Surfaces for games and 3-D: colour, height, normal, confidence and agreement, plus a JSON file saying how they
were made (BUILD_PLAN_DEPTH_STACK C, D).

From a measured depth (depthfuse.DepthResult) and the picture it belongs to:
    NAME_colour_Studio.png       the picture (a focus stack: its all-in-focus merge), 8-bit sRGB
    NAME_height_Studio.png       16-bit grey, white = near, spread over the measured range (filled-in places can
                                 fall outside it and are held at the ends)
    NAME_normal_Studio.png       from the height; OpenGL (green up: Godot, Blender) or DirectX (green down)
    NAME_confidence_Studio.png   8-bit grey, 255 = sure; 0 where the height was filled in, not measured
    NAME_agreement_Studio.png    grey + alpha: 255 = the measures agree, 128 = they stray half a step (the red line),
                                 0 = a step or more; transparent where there was only one measure to look at
    NAME_surface_Studio.json     the scale (relative or mm, and how it was known), the focus step and its spread,
                                 the frames used, what share is measured or filled, the method, and the files
``--make depth`` writes the height, confidence and agreement and the JSON (no colour, no normal: the fast part).

Height in real units only when it can be known: millimetres when the focus moved a known distance per frame
(--step-mm, a rail); the size of a pixel on the subject from the magnification (--ratio 1:2) and the camera's pixel
size. Otherwise the height is relative, and the JSON says so plainly.

Tileable (optional): the picture offset by half and the seams healed with one set of weights for every file, so
colour, height and the rest stay in register; the normal map is worked out again from the tiled height. No Qt here.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from ... import __version__, log as slog
from . import depthfuse as DF

log = slog.get("engine.stack.surface")

KINDS = ("colour", "height", "normal", "confidence", "agreement")
DEPTH_KINDS = ("height", "confidence", "agreement")

# pixel pitch (mm) by camera, for the size of a pixel on the subject (--ratio); width of the sensor / pixels across
PIXEL_MM = {
    "canon eos r5": 36.0 / 8192, "canon eos r5 mark ii": 36.0 / 8192, "canon eos 5d mark iv": 36.0 / 6720,
    "canon eos 6d": 35.8 / 5472, "canon eos 6d mark ii": 35.9 / 6240, "canon eos r": 36.0 / 6720,
    "canon eos r6": 35.9 / 5472, "canon eos r6 mark ii": 35.9 / 6000, "canon eos 5ds r": 36.0 / 8688,
    "canon eos 5ds": 36.0 / 8688, "canon eos r8": 35.9 / 6000, "canon eos rp": 35.9 / 6240,
}


@dataclass
class SurfaceParams:
    ratio: str = ""                 # magnification as shot, e.g. 1:2 (0.5) or 1:1: gives the size of one pixel on
                                    # the subject in mm (with the camera's pixel size); empty = not known
    step_mm: float = 0.0            # how far the plane of focus moved between frames, mm (a focusing rail): with it
                                    # the height is in mm; 0 = not known (the height is relative)
    near: str = "auto"              # which end of the stack is nearest: auto | first | last
    normals_y: str = "up"           # up (OpenGL: Godot, Blender) | down (DirectX)
    normal_strength: float = 1.0    # steepness of the normal map when the height is relative (1 = the steep tenth
                                    # of the surface tilts about 45 degrees); ignored when the height is in mm
    tileable: bool = False          # offset by half and heal the seams of every file the same way (a texture that
                                    # repeats without a visible join)
    swap: bool = False              # one Dual Pixel photo: near and far the other way round

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SurfaceParams":
        return SurfaceParams(**{k: v for k, v in (d or {}).items() if k in SurfaceParams.__dataclass_fields__})


def parse_ratio(text) -> float | None:
    """'1:2' -> 0.5, '2:1' -> 2.0, '0.5' -> 0.5; None for empty. ValueError for anything else."""
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().replace(" ", "")
    if ":" in t:
        a, b = t.split(":", 1)
        v = float(a) / float(b)
    else:
        v = float(t)
    if not 0.001 <= v <= 20:
        raise ValueError(f"a magnification of {text} is not believable (1:2 is 0.5, 1:1 is 1)")
    return v


def pixel_mm(model: str) -> float | None:
    return PIXEL_MM.get(" ".join(str(model or "").split()).lower())


# ------------------------------------------------------------------------------------------ which way is near
def decide_near(res: DF.DepthResult, model: str, p: SurfaceParams) -> tuple[bool, str, bool, list]:
    """(near_high, how, measured, notes): near_high True when larger depth values are nearer. A stack's depth grows
    with the frames' focus order; one Dual Pixel photo's depth is -offset."""
    from ..dualpixel.depth import near_sign
    ns, ns_measured = near_sign(model)
    notes = []
    if res.method == "dual_pixel":
        near_high = (ns < 0) != bool(p.swap)
        if ns_measured:
            how = "the Dual Pixel halves, read the way this camera was checked on a known scene"
        else:
            how = ("the Studio's usual reading of the Dual Pixel halves (not yet checked on this camera: if the "
                   "surface comes out inside out, swap near and far)")
        if p.swap:
            how = "set by you (near and far swapped)"
        return near_high, how, ns_measured and not p.swap, notes
    if p.near in ("first", "last"):
        return p.near == "last", f"set by you: the {p.near} frame is the nearest", False, notes
    if res.step_px is not None:
        first_near_dp = (np.sign(res.step_px) == ns)
        if ns_measured:
            return (not first_near_dp, "measured: the Dual Pixel halves (this camera's reading was checked on a "
                    "known scene)", True, notes)
        if not first_near_dp:
            notes.append("The first frame was taken as the nearest; the Dual Pixel halves, read the Studio's "
                         "usual way (not yet checked on this camera), say the last frame is. If the surface comes "
                         "out inside out, flip near and far.")
    return (False, "assumed: the first frame in focus order is the nearest (start a stack at the nearest point; "
            "flip near and far if the surface comes out inside out)", False, notes)


# ------------------------------------------------------------------------------------------ maps at full size
def _up(m: np.ndarray, shape) -> np.ndarray:
    from .focus import _up2
    if m.shape[:2] == tuple(shape[:2]):
        return np.asarray(m, np.float32)
    if (m.shape[0] * 2, m.shape[1] * 2) == tuple(shape[:2]):
        return _up2(np.asarray(m, np.float32), shape)
    from scipy.ndimage import zoom
    return zoom(np.asarray(m, np.float32), (shape[0] / m.shape[0], shape[1] / m.shape[1]), order=1,
                mode="nearest", grid_mode=True)[:shape[0], :shape[1]]


def full_maps(res: DF.DepthResult, shape, flip: int, near_high: bool) -> dict:
    """Upright, full-size float maps: height (near high, depth units), confidence, agreement (NaN = not compared),
    label."""
    from ..pipeline.pipeline import apply_orientation
    h = res.depth if near_high else -res.depth
    cmp_ = np.isfinite(res.agreement)
    out = {"height": _up(h, shape), "confidence": np.clip(_up(res.confidence, shape), 0, 1),
           "label": _up(res.label.astype(np.float32), shape)}
    ag = _up(np.nan_to_num(res.agreement), shape)
    ok = _up(cmp_.astype(np.float32), shape) > 0.5
    out["agreement"] = np.where(ok, ag, np.nan).astype(np.float32)
    return {k: np.ascontiguousarray(apply_orientation(v, flip)) for k, v in out.items()}


def fit_to(m: np.ndarray, shape) -> np.ndarray:
    """A map brought to the picture's size: a raw's picture can be a pixel or two larger than its even-sized
    camera crop, so up to 4 pixels are added (edge repeated) or cut; anything more is resized."""
    H, W = shape[:2]
    h, w = m.shape[:2]
    if abs(H - h) <= 4 and abs(W - w) <= 4:
        m = m[:min(h, H), :min(w, W)]
        if m.shape[0] < H or m.shape[1] < W:
            m = np.pad(m, ((0, H - m.shape[0]), (0, W - m.shape[1])), mode="edge")
        return np.ascontiguousarray(m)
    from scipy.ndimage import zoom
    log.info("surface: map %s resized to the picture %s", (h, w), (H, W))
    fin = np.isfinite(m)
    out = zoom(np.nan_to_num(m).astype(np.float32), (H / h, W / w), order=1, grid_mode=True, mode="nearest")[:H, :W]
    if not fin.all():
        ok = zoom(fin.astype(np.float32), (H / h, W / w), order=1, grid_mode=True, mode="nearest")[:H, :W] > 0.5
        out = np.where(ok, out, np.nan)
    return out.astype(np.float32)


def height_range(height: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """(lo, hi) of the height over the measured places (0.5 and 99.5 percentiles)."""
    m = label >= 0.5
    v = height[m] if m.sum() > 100 else height.ravel()
    if v.size > 2_000_000:
        v = v[:: int(np.ceil(v.size / 2_000_000))]
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return float(lo), float(hi)


def normals(height: np.ndarray, k: float, y_up: bool, wrap: bool = False, rows: int = 512) -> np.ndarray:
    """Unit normals (h, w, 3) float32 of a height map (near = high) with slope scale ``k`` (height units per pixel
    turned into a slope). OpenGL (y_up): green = up; DirectX: green = down. ``wrap``: the map repeats (tileable).
    Worked in strips of rows (a 45 MP map would otherwise need gigabytes)."""
    hh = np.asarray(height, np.float32)
    H, W = hh.shape
    out = np.empty((H, W, 3), np.float32)
    kk = np.float32(k)
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        a, b = max(0, y0 - 1), min(H, y1 + 1)
        blk = hh[a:b]
        if wrap:
            up = hh[(np.arange(y0, y1) - 1) % H]
            dn = hh[(np.arange(y0, y1) + 1) % H]
            gy = 0.5 * (dn - up)
            mid = hh[y0:y1]
            gx = 0.5 * (np.roll(mid, -1, 1) - np.roll(mid, 1, 1))
        else:
            g = np.gradient(blk, axis=0)
            gy = g[y0 - a:y0 - a + (y1 - y0)]
            gx = np.gradient(hh[y0:y1], axis=1)
        nx, ny = -gx * kk, (gy if y_up else -gy) * kk        # image rows run down
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
        out[y0:y1, :, 0] = nx * inv
        out[y0:y1, :, 1] = ny * inv
        out[y0:y1, :, 2] = inv
    return out


def auto_slope(height01: np.ndarray, strength: float) -> float:
    """The slope scale that tilts the steep tenth of a relative height map by about 45 degrees (x strength)."""
    gy, gx = np.gradient(np.asarray(height01, np.float64)[::2, ::2])
    p90 = float(np.percentile(np.hypot(gx, gy), 90)) / 2.0         # every other pixel: per pixel at full size
    return float(strength) / max(p90, 1e-9)


# ------------------------------------------------------------------------------------------ tileable
def _smooth(d, width):
    t = np.clip(np.asarray(d, np.float32) / np.float32(max(width, 1e-6)), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def tile_weights(H: int, W: int, band: float = 0.12, line: float = 0.08) -> dict:
    """Weights for four copies of the picture: as it is (A, kept in the middle), offset by half both ways (B: the
    corners), offset by half down (C: the middle of the top and bottom edges) and across (D: the middle of the left
    and right edges). The edges are made only from copies that run on across them, so the result repeats without a
    join; each copy is kept away from its own seam."""
    L = float(min(H, W))
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    cy, cx = H - H // 2 - 0.5, W - W // 2 - 0.5               # where the offset copies have their seams
    edge = np.minimum(np.minimum(y, H - 1 - y), np.minimum(x, W - 1 - x))
    a = _smooth(edge, band * L)
    sx = _smooth(np.abs(x - cx), line * L) * np.ones((H, 1), np.float32)
    sy = _smooth(np.abs(y - cy), line * L) * np.ones((1, W), np.float32)
    wB, wC, wD = sx * sy, (1 - sx) * sy, sx * (1 - sy)
    tot = np.maximum(wB + wC + wD, 1e-6)
    r = 1.0 - a
    return {"A": a, "B": r * wB / tot, "C": r * wC / tot, "D": r * wD / tot}


def tile(arr: np.ndarray, w: dict) -> np.ndarray:
    """One map made tileable with the weights from tile_weights (the same weights for every map); a colour picture
    one channel at a time."""
    if np.ndim(arr) == 3:
        return np.stack([tile(arr[..., c], w) for c in range(arr.shape[2])], 2)
    a = np.asarray(arr, np.float32)
    H, W = a.shape[:2]
    B = np.roll(a, (-(H // 2), -(W // 2)), (0, 1))
    C = np.roll(a, -(H // 2), 0)
    D = np.roll(a, -(W // 2), 1)
    return (w["A"] * a + w["B"] * B + w["C"] * C + w["D"] * D).astype(np.float32)


# ------------------------------------------------------------------------------------------ files
def names(base: str, kinds=KINDS) -> dict:
    d = {k: f"{base}_{k}_Studio.png" for k in kinds}
    d["json"] = f"{base}_surface_Studio.json"
    return d


def targets(out_dir: str, base: str, kinds=KINDS, taken: set | None = None) -> dict:
    """The paths, all with the same ending: when any one of them cannot be used (a file the Studio did not write,
    or one already written in this run), every file moves to NAME-2 ... together, so the set stays a set."""
    from ..io.export import is_own_export

    def key(p):
        return os.path.normcase(os.path.abspath(p))

    def usable(p):
        if taken is not None and key(p) in taken:
            return False
        if not os.path.lexists(p):
            return True
        return os.path.isfile(p) and is_own_export(p)

    n = 1
    while True:
        b = base if n == 1 else f"{base}-{n}"
        paths = {k: os.path.join(out_dir, v) for k, v in names(b, kinds).items()}
        if all(usable(p) for p in paths.values()):
            break
        n += 1
        if n > 9999:
            raise OSError(f"no free file names for {base}")
    if taken is not None:
        taken.update(key(p) for p in paths.values())
    return paths


def write(out_dir: str, base: str, res: DF.DepthResult, shape, flip: int, meta: dict, source: dict,
          p: SurfaceParams, colour: np.ndarray | None = None, kinds=KINDS, taken: set | None = None) -> dict:
    """Write the surface files and the JSON. ``colour``: uint8 sRGB (H, W, 3), upright, full size (None for --make
    depth). ``shape``: the full-size (h, w) of the depth's frame before turning upright. Returns {"paths": {kind:
    path}, "json": the JSON as written}."""
    from ..color import icc
    from ..io.export import write_png_any
    os.makedirs(out_dir, exist_ok=True)
    kinds = tuple(k for k in kinds if k in KINDS)
    if "colour" in kinds and colour is None:
        raise ValueError("the colour file needs the picture")
    paths = targets(out_dir, base, kinds, taken)
    model = str(meta.get("model") or "")
    near_high, near_how, near_measured, notes = decide_near(res, model, p)
    notes = list(res.notes) + notes
    m = full_maps(res, shape, flip, near_high)
    H, W = m["height"].shape
    if colour is not None and colour.shape[:2] != (H, W):
        m = {k: fit_to(v, colour.shape[:2]) for k, v in m.items()}
        H, W = colour.shape[:2]
    lo, hi = height_range(m["height"], m["label"])
    h01 = np.clip((m["height"] - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    conf = m["confidence"]
    ag = m["agreement"]
    agree01 = np.where(np.isfinite(ag), 1.0 - np.clip(np.nan_to_num(ag) / (2 * DF.DISAGREE_STEPS), 0, 1), 0.0)
    alpha = np.isfinite(ag).astype(np.float32)
    col = None if colour is None else np.asarray(colour, np.float32)
    if p.tileable:
        w = tile_weights(H, W)
        h01 = np.clip(tile(h01, w), 0, 1)
        conf = np.clip(tile(conf, w), 0, 1)
        agree01 = np.clip(tile(agree01, w), 0, 1)
        alpha = np.clip(tile(alpha, w), 0, 1)
        if col is not None:
            col = tile(col, w)

    # scale
    mag = parse_ratio(p.ratio)
    pmm = pixel_mm(model)
    mm_px = (pmm / mag) if (mag and pmm) else None
    steps = hi - lo
    unit = {"steps": "focus steps (the typical step between neighbouring frames)",
            "frames": "frames (focus steps, taken as even: without Dual Pixel an uneven hand-held step cannot be seen)",
            "dual pixel px": "pixels of Dual Pixel offset"}[res.units]
    h_mm = steps * float(p.step_mm) if (p.step_mm and res.units in ("steps", "frames")) else None
    lateral_how = (f"magnification {p.ratio} and the {model}'s {1000 * pmm:.2f} micrometre pixels" if mm_px else
                   ("magnification given, but this camera's pixel size is not in the Studio's table" if mag else
                    "not known (give --ratio, e.g. 1:2)"))
    if h_mm is not None:
        height_how = f"you gave the focus step: {p.step_mm:g} mm per step"
    elif res.units == "dual pixel px":
        height_how = "relative: one Dual Pixel photo gives offsets, not distances"
    else:
        height_how = ("relative: the distance the focus moved between frames is not known (give --step-mm for a "
                      "rail; the camera's recorded focus distance is not read yet)")

    # normal
    files = {}
    if "normal" in kinds:
        if h_mm is not None and mm_px:
            k = (h_mm / mm_px)                      # height 0..1 = h_mm; one pixel = mm_px: slope per unit of h01
            nscale = "physical: height and pixel size both in mm"
        else:
            k = auto_slope(h01, p.normal_strength)
            nscale = (f"relative: the steep tenth of the surface tilts about 45 degrees (strength "
                      f"{p.normal_strength:g})")
        nrm = normals(h01, k, p.normals_y == "up", wrap=p.tileable)
        nrm *= 127.5
        nrm += 127.5
        np.rint(nrm, out=nrm)
        write_png_any(paths["normal"], nrm.astype(np.uint8), None, meta)
        del nrm
        files["normal"] = paths["normal"]
    if "colour" in kinds:
        write_png_any(paths["colour"], np.clip(np.rint(col), 0, 255).astype(np.uint8), icc.profile_bytes("srgb"),
                      meta)
        files["colour"] = paths["colour"]
    if "height" in kinds:
        write_png_any(paths["height"], np.rint(h01 * 65535.0).astype(np.uint16), None, meta)
        files["height"] = paths["height"]
    if "confidence" in kinds:
        write_png_any(paths["confidence"], np.rint(conf * 255.0).astype(np.uint8), None, meta)
        files["confidence"] = paths["confidence"]
    if "agreement" in kinds:
        la = np.stack([np.rint(agree01 * 255.0), np.rint(alpha * 255.0)], -1).astype(np.uint8)
        write_png_any(paths["agreement"], la, None, meta)
        files["agreement"] = paths["agreement"]

    shares = res.shares()
    info = DF.METHOD_INFO[res.method]
    doc = {
        "studio_surface": 1,
        "written_by": f"Photographic Studio By Bill {__version__}",
        "method": res.method,
        "method_text": info["name"] + ": " + info["does"],
        "best_for": info["best"], "weak_at": info["weak"],
        "source": source,
        "camera": {k: meta.get(k) for k in ("make", "model", "lens", "fnumber", "focal_mm", "iso")},
        "size": {"width": int(W), "height": int(H)},
        "files": {k: os.path.basename(v) for k, v in files.items()},
        "height": {
            "white": "near", "bits": 16, "units": "mm" if h_mm is not None else "relative",
            "how_known": height_how,
            "depth_units": unit,
            "range": {"in_depth_units": round(steps, 4), "mm": round(h_mm, 4) if h_mm is not None else None},
            "black_means": "the farthest measured place (filled-in places can go past it and are held at 0)",
            "white_means": "the nearest measured place",
            "near_end": near_how, "near_measured": bool(near_measured),
        },
        "lateral": {"mm_per_pixel": round(mm_px, 6) if mm_px else None, "how_known": lateral_how,
                    "width_mm": round(W * mm_px, 3) if mm_px else None,
                    "height_mm": round(H * mm_px, 3) if mm_px else None},
        "displacement": {"height_range_over_width": round(h_mm / (W * mm_px), 6) if (h_mm and mm_px) else None,
                         "about": "the height range as a share of the texture's width (for a displacement "
                                  "setting); null when either is not known"},
        "normal": ({"convention": "OpenGL (green up)" if p.normals_y == "up" else "DirectX (green down)",
                    "scale": nscale} if "normal" in kinds else None),
        "dual_pixel": ({"step_px": round(res.step_px, 4), "step_spread": round(res.step_spread or 0.0, 4),
                        "about_step": "offset change between neighbouring frames, full-size pixels (s); spread: "
                                      "how unevenly the focus stepped, as a share of s",
                        "frame_positions_steps": res.positions,
                        "pairs": _pairs_json(res.pair_steps),
                        "noise_steps": {"one_frame": round(res.sigma_dp, 4) if res.sigma_dp else None,
                                        "focus": round(res.sigma_focus, 4) if res.sigma_focus else None}}
                       if res.step_px is not None else None),
        "coverage": shares,
        "coverage_about": "measured_both: focus and Dual Pixel (cyan); measured_one: one of them (amber); filled: "
                          "neither, filled in from the neighbours (violet, confidence 0); disagree: the measures "
                          "stray more than half a step (red)",
        "agreement": dict(res.agreement_stats(), red_line_steps=DF.DISAGREE_STEPS,
                          png="grey 255 = agree, 128 = half a step apart (the red line), 0 = a step or more; "
                              "transparent = only one measure there"),
        "confidence_png": "255 = sure; 0 = filled in, not measured",
        "tileable": bool(p.tileable),
        "settings": p.to_dict(),
        "notes": notes,
    }
    if "json" in paths:
        tmp = paths["json"] + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_plain(doc), fh, indent=2)
        os.replace(tmp, paths["json"])
        files["json"] = paths["json"]
    log.info("surface %s: %s, %s, %d x %d, measured both %.0f%%, filled %.0f%%", base, res.method,
             doc["height"]["units"], W, H, 100 * shares["measured_both"], 100 * shares["filled"])
    return {"paths": files, "json": doc}


def _pairs_json(pairs: list) -> list:
    return [{"frames": [int(q["pair"]) + 1, int(q["pair"]) + 2],
             "step_px": None if q["step_px"] is None else round(float(q["step_px"]), 4),
             "samples": int(q["samples"])} for q in pairs]


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ------------------------------------------------------------------------------------------ colour
def srgb8(display_pp: np.ndarray) -> np.ndarray:
    """Display-referred linear ProPhoto (render()) -> uint8 sRGB."""
    from ..io.export import ExportSpec, to_output
    return to_output(np.ascontiguousarray(display_pp, np.float32), ExportSpec(format="png", space="srgb", bits=8))


def colour_of_scene(scene_upright: np.ndarray, meta: dict, scratch_dir: str) -> np.ndarray:
    """A scene-linear result (a stack's merge, upright) developed with the Studio's default look, as uint8 sRGB:
    written as a linear intermediate in the job's scratch folder and rendered like any photo."""
    from .. import api
    from ..io.intermediate import write_intermediate
    from ..pipeline.edit import Edit
    tmp = os.path.join(scratch_dir, "surface_colour.tif")
    write_intermediate(tmp, np.ascontiguousarray(scene_upright, np.float32), meta,
                       {"kind": "focus", "method": "surface colour", "sources": [], "scale": 1.0})
    img = api.open(tmp)
    try:
        return srgb8(api.render(img, Edit()))
    finally:
        img.drop_cache()
        del img


# ------------------------------------------------------------------------------------------ the two sources
def stack_surface(an, out_dir: str, base: str, p: SurfaceParams, kinds=KINDS, progress=None, cancel=None,
                  taken: set | None = None, method: str = "auto") -> dict:
    """The surface of a Focus Stack analysis: the fused depth (Dual Pixel when the analysis kept it), and for the
    colour the all-in-focus merge at full size (``method``, default auto), developed with the default look."""
    from . import focus as FS
    from . import frames as F
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import apply_orientation
    FS._tick(progress, 0.02, "Measuring depth")
    res = DF.fuse(an)
    colour = None
    if "colour" in kinds:
        fp = FS.FocusParams.from_dict(an.params.to_dict())
        fp.method = method if method in FS.METHODS else "auto"
        with F.Scratch("surface") as sc:
            out = FS.merge(an, fp, True, lambda f, m: FS._tick(progress, 0.05 + 0.8 * f, m), cancel, sc,
                           Edit.from_dict(an.edit))
            FS._check(cancel)
            FS._tick(progress, 0.88, "Developing the colour")
            up = np.ascontiguousarray(apply_orientation(out, an.flip))
            del out
            colour = colour_of_scene(up, an.meta, sc.dir)
            del up
    FS._check(cancel)
    FS._tick(progress, 0.95, "Writing the surface")
    y0, y1, x0, x1 = an.box_full
    names_ = [os.path.basename(q) for q in an.paths]
    source = {"kind": "focus stack", "frames": names_, "frames_used": an.n,
              "reference_frame": names_[an.ref] if names_ else "", "dual_pixel": bool(an.dual_pixel),
              "left_out": [r.name for r in an.reports if r.status == "dropped"]}
    r = write(out_dir, base, res, (y1 - y0, x1 - x0), an.flip, an.meta, source, p, colour, kinds, taken)
    FS._tick(progress, 1.0, "Done")
    return r


def dual_surface(dp, out_dir: str, base: str, p: SurfaceParams, kinds=KINDS, taken: set | None = None,
                 maps: dict | None = None) -> dict:
    """The surface of one Dual Pixel photo (decode.DualPixel): its own offsets, the photo developed with the
    default look as the colour. ``maps``: {"disp", "conf", "la", "lb"} already measured (half size)."""
    from ..dualpixel import decode as DD, depth as DEP
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import render
    if maps is None:
        gains = DD.shading(dp)
        la, lb = DEP.halves_luma(dp, gains)
        disp, conf = DEP.disparity(la, lb)
    else:
        la, lb, disp, conf = maps["la"], maps["lb"], maps["disp"], maps["conf"]
    res = DF.from_dual_pixel(disp, conf, la + lb)
    md = dp.image.meta
    meta = {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s, "fnumber": md.fnumber,
            "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime}
    colour = srgb8(render(dp.image, Edit())) if "colour" in kinds else None
    source = {"kind": "one dual pixel photo", "frames": [os.path.basename(dp.path)], "frames_used": 1,
              "reference_frame": os.path.basename(dp.path), "dual_pixel": True, "left_out": []}
    return write(out_dir, base, res, dp.shape, dp.flip, meta, source, p, colour, kinds, taken)


# ------------------------------------------------------------------------------------------ the Depth view
# the Studio's palette (app/theme.py), plan 23.1: measured two ways, one way, filled in; red = disagree
VIEW_RGB = {DF.LABEL_BOTH: (0x22, 0xD3, 0xEE), DF.LABEL_ONE: (0xF5, 0x9E, 0x0B), DF.LABEL_FILLED: (0xC0, 0x84, 0xFC)}
VIEW_RED = (0xF8, 0x71, 0x71)


def depth_view(res: DF.DepthResult, near_high: bool, flip: int, overlay: bool = True) -> np.ndarray:
    """uint8 RGB, upright, the size of the depth (half size): brightness = height (bright = near), colour = how
    it was measured (cyan both ways, amber one, violet filled in); ``overlay``: red where the measures disagree."""
    from ..pipeline.pipeline import apply_orientation
    h = res.depth if near_high else -res.depth
    lo, hi = height_range(h, res.label)
    t = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
    pal = np.zeros((3, 3), np.float32)
    for k, c in VIEW_RGB.items():
        pal[k] = c
    rgb = pal[np.clip(res.label, 0, 2)] * (0.16 + 0.84 * t)[..., None]
    if overlay:
        dis = np.nan_to_num(res.agreement) > DF.DISAGREE_STEPS
        rgb[dis] = 0.35 * rgb[dis] + 0.65 * np.asarray(VIEW_RED, np.float32)
    return np.ascontiguousarray(apply_orientation(np.clip(rgb, 0, 255).astype(np.uint8), flip))


def summary(res: DF.DepthResult) -> str:
    """One or two plain sentences on what was measured, for the page."""
    sh = res.shares()
    pct = lambda v: f"{100 * v:.0f}%" if v >= 0.01 or v == 0 else "<1%"      # noqa: E731
    if res.method == "focus+dual_pixel":
        s = (f"Measured two ways {pct(sh['measured_both'])}, one way {pct(sh['measured_one'])}, filled in "
             f"{pct(sh['filled'])}; the measures disagree on {pct(sh['disagree'])}.")
        if res.step_px is not None:
            s += (f" One focus step = {abs(res.step_px):.2f} px of Dual Pixel offset; the steps vary by "
                  f"{100 * (res.step_spread or 0):.0f}%.")
        return s
    if res.method == "focus":
        return (f"Depth from focus alone (these frames are not Dual Pixel RAW): measured {pct(sh['measured_one'])}, "
                f"filled in {pct(sh['filled'])}. Uneven focus steps cannot be seen without Dual Pixel.")
    return f"One Dual Pixel photo: measured {pct(sh['measured_one'])}, filled in {pct(sh['filled'])}."

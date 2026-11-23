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
# BUILD_PLAN_DP3D C: maps on request (--maps); every one says in the JSON how it was made (maps.HOW)
EXTRA_KINDS = ("albedo", "detail", "ao", "cavity", "curvature", "orm", "roughness", "gaps")
ALL_KINDS = KINDS + EXTRA_KINDS
SURFACE_VERSION = 2

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
    tile_flatten: bool = True       # tileable: first take out the large-scale light falloff (vignetting) of the colour
                                    # and the overall tilt of the height, so the copies meet without a step
    tile_method: str = "cut"        # tileable: cut (a minimum-error cut through each seam band: structured
                                    # textures such as brick) | blend (the copies blended: even, random textures)
    size: int = 0                   # game size: 0 = as shot; 1024 / 2048 / 4096 = the measured area's best square,
                                    # resized (power of two)
    scale: str = ""                 # how big it is: "" | "coin" / "coin:us_quarter" (found in the picture) |
                                    # "click:U1,V1,U2,V2,MM" (two points 0..1 a known distance apart)
    ao_radius: float = 0.0          # ambient occlusion reach, pixels (0 = 1/32 of the long edge)
    maps: str = ""                  # which files, comma list (--maps): "" = colour, height, normal, confidence,
                                    # agreement; "all" = every map; or any of colour, albedo, height, detail, normal,
                                    # ao, cavity, curvature, orm, roughness, confidence, agreement, gaps

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SurfaceParams":
        return SurfaceParams(**{k: v for k, v in (d or {}).items() if k in SurfaceParams.__dataclass_fields__})


def kinds_from(p: "SurfaceParams", default=KINDS) -> tuple:
    """The files a surface writes: --maps, else ``default``. "all" = every map except roughness (an estimate of
    something nothing here measures: asked for by name only). ValueError names a wrong one."""
    t = str(p.maps or "").strip().lower()
    if not t:
        return tuple(default)
    if t == "all":
        return tuple(k for k in ALL_KINDS if k != "roughness")
    ks = [x.strip() for x in t.replace(" ", "").split(",") if x.strip()]
    bad = [k for k in ks if k not in ALL_KINDS]
    if bad:
        raise ValueError(f"no such map: {', '.join(bad)} (choose from {', '.join(ALL_KINDS)} or all)")
    return tuple(k for k in ALL_KINDS if k in ks)


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
    if res.method == "estimated":
        return True, "estimated: dark taken as deep (one ordinary photo; nothing measured)", False, notes
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


def _lateral(p: SurfaceParams, model: str, meta: dict, col, W: int, H: int, notes: list) -> tuple:
    """(mm per pixel or None, how, scale info, magnification, pixel mm): from the magnification, a coin or two
    clicked points, or the lens's focus distance."""
    from . import maps as MP
    mag = parse_ratio(p.ratio)
    pmm = pixel_mm(model)
    mm_px = (pmm / mag) if (mag and pmm) else None
    lateral_how = (f"magnification {p.ratio} and the {model}'s {1000 * pmm:.2f} micrometre pixels" if mm_px else
                   ("magnification given, but this camera's pixel size is not in the Studio's table" if mag else
                    "not known (give --ratio, e.g. 1:2)"))
    scale_info = {"lateral_from": "ratio" if mm_px else None}
    if p.scale:
        sc = _scale_from_setting(p.scale, col, W, H)
        if sc is not None:
            mm_px = sc["mm_per_px"]
            lateral_how = sc["how"]
            scale_info = dict(sc, lateral_from=sc["kind"])
        else:
            notes.append("No coin was found in the picture: the size stays as it was (relative unless --ratio).")
    if mm_px is None and meta.get("focus_distance_m") and meta.get("focal_mm") and pmm:
        # Bill (2026-11-21): scale from how far away the lens says it is focused and how much the lens sees edge to
        # edge at that distance (no coin): the view across the frame = sensor width / magnification
        mg = MP.magnification_from_distance(float(meta["focal_mm"]), float(meta["focus_distance_m"]))
        if mg:
            mm_px = pmm / mg
            view = MP.view_edge_to_edge(pmm, mg, W, H)
            lateral_how = (f"the focus distance the lens recorded ({float(meta['focus_distance_m']):.2f} m) and its "
                           f"{float(meta['focal_mm']):g} mm view at that distance: about {view[0]:.0f} x {view[1]:.0f} "
                           f"mm edge to edge (the lens's distance is coarse: amber)")
            scale_info = {"lateral_from": "focus distance and the lens's view (recorded by the lens: coarse)",
                          "magnification": mg, "view_mm": [round(view[0], 1), round(view[1], 1)],
                          "colour": "amber"}
    return mm_px, lateral_how, scale_info, mag, pmm


def write(out_dir: str, base: str, res: DF.DepthResult, shape, flip: int, meta: dict, source: dict,
          p: SurfaceParams, colour: np.ndarray | None = None, kinds=KINDS, taken: set | None = None,
          near_known: str | None = None, lateral: dict | None = None, height_unit_mm: float | None = None,
          height_how: str | None = None, extra: dict | None = None) -> dict:
    """Write the surface files and the JSON. ``colour``: uint8 sRGB (H, W, 3), upright, full size (None for --make
    depth). ``shape``: the full-size (h, w) of the depth's frame before turning upright. Returns {"paths": {kind:
    path}, "json": the JSON as written}.

    For a result that knows its own size (the Flat Mosaic): ``lateral`` {"mm_per_px", "how", "info"} replaces the
    ratio / coin / focus-distance scale; ``height_unit_mm`` is millimetres per depth unit (the height in mm) with
    ``height_how`` saying how it was known; ``extra`` is merged into the JSON."""
    from ..color import icc
    from ..io.export import write_png_any
    from . import maps as MP
    os.makedirs(out_dir, exist_ok=True)
    estimated = res.method == "estimated"
    if estimated:                                    # nothing measured: no confidence, agreement or gap maps
        kinds = tuple(k for k in kinds if k not in ("confidence", "agreement", "gaps"))
    kinds = tuple(k for k in ALL_KINDS if k in kinds)
    need_colour = any(k in kinds for k in ("colour", "albedo", "detail", "roughness")) or \
        ("orm" in kinds and "roughness" in kinds)
    if need_colour and colour is None:
        raise ValueError("the colour, albedo, detail and roughness files need the picture")
    paths = targets(out_dir, base, kinds, taken)
    model = str(meta.get("model") or "")
    if near_known:                                   # the depth already says near = high (a Focus Stack result's file)
        near_high, near_how, near_measured, notes = True, near_known, True, []
    else:
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
    lab = m["label"]
    agree01 = np.where(np.isfinite(ag), 1.0 - np.clip(np.nan_to_num(ag) / (2 * DF.DISAGREE_STEPS), 0, 1), 0.0)
    alpha = np.isfinite(ag).astype(np.float32)
    col = None if colour is None else np.asarray(colour, np.float32)
    gaps01 = None
    if "gaps" in kinds:
        if res.gap is not None:
            from .focus import _up2  # noqa: F401  (the same resampling as the other maps)
            from ..pipeline.pipeline import apply_orientation
            g = _up(res.gap.astype(np.float32), shape)
            gaps01 = np.ascontiguousarray(apply_orientation(g, flip)).astype(np.float32)
            if gaps01.shape != (H, W):
                gaps01 = fit_to(gaps01, (H, W))
        else:
            gaps01 = np.zeros((H, W), np.float32)

    # scale (before any crop: the coin is looked for in the whole picture)
    if lateral is not None:
        mm_px = lateral.get("mm_per_px")
        lateral_how = lateral.get("how", "")
        scale_info = dict(lateral.get("info") or {}, lateral_from=lateral.get("from", "the result itself"))
    else:
        mm_px, lateral_how, scale_info, _, _ = _lateral(p, model, meta, col, W, H, notes)
    # game size: the measured area's best square, every map cut and resized the same way
    crop = None
    if p.size:
        y0, x0, side = MP.best_square(lab)
        crop = (y0, x0, side)

        def cut(a):
            return None if a is None else MP.resize_to(a[y0:y0 + side, x0:x0 + side], int(p.size))
        h01, conf, agree01, alpha = cut(h01), cut(conf), cut(agree01), cut(alpha)
        lab = cut(lab)
        col = cut(col)
        gaps01 = cut(gaps01)
        if mm_px:
            mm_px = mm_px * side / float(p.size)
        H = W = int(p.size)
    steps = hi - lo
    unit = {"steps": "focus steps (the typical step between neighbouring frames)",
            "frames": "frames (focus steps, taken as even: without Dual Pixel an uneven hand-held step cannot be seen)",
            "dual pixel px": "pixels of Dual Pixel offset",
            "estimated": "relative (estimated from the picture, not measured)"}.get(res.units, res.units)
    h_mm = steps * float(p.step_mm) if (p.step_mm and res.units in ("steps", "frames")) else None
    if height_unit_mm:                               # the result knows its height in mm (the Flat Mosaic)
        h_mm = steps * float(height_unit_mm)
    if height_how:
        pass                                         # the result says how its height is known
    elif h_mm is not None:
        height_how = f"you gave the focus step: {p.step_mm:g} mm per step"
    elif res.units == "dual pixel px":
        height_how = "relative: one Dual Pixel photo gives offsets, not distances"
    elif estimated:
        height_how = ("estimated, relative: one ordinary photo measures no depth (a focus stack or a Dual Pixel RAW "
                      "would)")
    else:
        height_how = ("relative: the distance the focus moved between frames is not known (give --step-mm for a "
                      "rail, or calibrate the Dual Pixel scale once with a coin in the frame)")

    # tileable: first the large scale out (it can never repeat: a vignetted corner or a tilted surface steps at
    # every seam), then one geometry for every map
    tiled = None
    flat_doc = None
    if p.tileable and p.tile_flatten:
        flat_doc = {"scale": "1/8 of the edge", "height": "overall tilt and broad bumps taken out; finer relief kept"}
        h01 = MP.flatten_height(h01)
        if col is not None:
            col, stops = MP.flatten_colour(col)
            flat_doc["colour"] = (f"large-scale brightness evened out (vignetting, light falloff): up to "
                                  f"{stops:.2f} stops corrected")
        notes.append("Tileable: the large-scale light falloff of the colour and the overall tilt of the height were "
                     "taken out before the seams were healed, so the texture repeats without steps (the height is "
                     "relief only; the slope of the ground as shot is not in it).")
    if p.tileable:
        if p.tile_method == "blend":
            w = tile_weights(H, W)
            tiled = ("blend", w)
            tl = lambda a: tile(a, w)                    # noqa: E731
        else:
            guide = col if col is not None else h01
            cw = MP.cut_weights_for(guide / (255.0 if col is not None else 1.0))
            tiled = ("cut", cw)
            tl = lambda a: MP.tile_cut(a, cw)            # noqa: E731
        h01 = np.clip(tl(h01), 0, 1)
        conf = np.clip(tl(conf), 0, 1)
        agree01 = np.clip(tl(agree01), 0, 1)
        alpha = np.clip(tl(alpha), 0, 1)
        lab = tl(lab)
        if col is not None:
            col = tl(col)
        if gaps01 is not None:
            gaps01 = np.clip(tl(gaps01), 0, 1)

    # the slope scale shared by the normal map and the maps computed from the height
    if h_mm is not None and mm_px:
        k = (h_mm / mm_px)
        nscale = "physical: height and pixel size both in mm"
    else:
        k = auto_slope(h01, p.normal_strength)
        nscale = (f"relative: the steep tenth of the surface tilts about 45 degrees (strength "
                  f"{p.normal_strength:g})")
    wrap = bool(p.tileable)
    files = {}
    maps_doc = {}

    def png(kind, arr, icc_b=None):
        write_png_any(paths[kind], arr, icc_b, meta)
        files[kind] = paths[kind]
        how, colour_, space = MP.HOW.get(kind, ("measured" if kind in ("height", "confidence", "agreement")
                                                 else "the picture", "cyan" if kind != "colour" else "none",
                                                 "srgb" if kind == "colour" else "linear"))
        maps_doc[kind] = {"file": os.path.basename(paths[kind]), "how": how, "colour": colour_, "space": space}

    if "normal" in kinds:
        nrm = normals(h01, k, p.normals_y == "up", wrap=wrap)
        nrm *= 127.5
        nrm += 127.5
        np.rint(nrm, out=nrm)
        png("normal", nrm.astype(np.uint8))
        maps_doc["normal"].update(how="computed from the measured height", colour="cyan")
        del nrm
    if "colour" in kinds:
        png("colour", np.clip(np.rint(col), 0, 255).astype(np.uint8), icc.profile_bytes("srgb"))
    if "height" in kinds:
        png("height", np.rint(h01 * 65535.0).astype(np.uint16))
        maps_doc["height"].update(how="measured (" + DF.METHOD_INFO[res.method]["name"] + "); filled places held "
                                  "between their measured neighbours (see confidence)")
    if "confidence" in kinds:
        png("confidence", np.rint(conf * 255.0).astype(np.uint8))
    if "agreement" in kinds:
        la = np.stack([np.rint(agree01 * 255.0), np.rint(alpha * 255.0)], -1).astype(np.uint8)
        png("agreement", la)
    light = None
    ao = None
    if any(x in kinds for x in ("ao", "orm")):
        ao = MP.ambient_occlusion(h01, k, p.ao_radius or None, wrap=wrap)
        if "ao" in kinds:
            png("ao", np.rint(ao * 255.0).astype(np.uint8))
    if "cavity" in kinds:
        png("cavity", np.rint(MP.cavity(h01, k, wrap=wrap) * 255.0).astype(np.uint8))
    if "curvature" in kinds:
        png("curvature", np.rint(MP.curvature_png(MP.curvature(h01, k, wrap=wrap)) * 65535.0).astype(np.uint16))
    alb = shading = rough = None
    if any(x in kinds for x in ("albedo", "detail", "roughness")):
        lin = _srgb_to_linear(np.clip(col, 0, 255) / 255.0)
        alb, light, shading = MP.delight(lin, h01, k, trust=lab > 0.5, wrap=wrap)
        if "albedo" in kinds:
            png("albedo", np.clip(np.rint(_linear_to_srgb(alb) * 255.0), 0, 255).astype(np.uint8),
                icc.profile_bytes("srgb"))
        if "detail" in kinds:
            png("detail", np.rint(MP.detail_relief(lin, shading, wrap=wrap) * 65535.0).astype(np.uint16))
        if "roughness" in kinds:
            rough = MP.roughness_estimate(alb, wrap=wrap)
            png("roughness", np.rint(rough * 255.0).astype(np.uint8))
    if "orm" in kinds:
        orm = MP.pack_orm(ao, rough, 0.9)
        png("orm", np.rint(orm * 255.0).astype(np.uint8))
        if rough is None:
            maps_doc["orm"]["how"] = ("packed: occlusion (R) computed from the measured height, roughness (G) a "
                                      "constant 0.9 (not measured; ask for --maps roughness for an estimate), "
                                      "metallic (B) 0")
    if "gaps" in kinds:
        png("gaps", np.rint(np.clip(gaps01, 0, 1) * 255.0).astype(np.uint8))

    if estimated:                                    # every map made from the height is a guess: amber
        for kind, d in maps_doc.items():
            if kind == "colour":
                continue
            d["colour"] = "amber"
            if kind in ("height", "normal", "ao", "cavity", "curvature"):
                d["how"] = ("estimated from one ordinary photo (dark taken as deep), not measured" if kind == "height"
                            else f"computed from the estimated height (one ordinary photo), not measured")
        notes.append("One ordinary photo: the height and every map made from it are estimated (amber), not "
                     "measured. A focus stack, or one Dual Pixel RAW, measures them.")
    shares = res.shares()
    info = DF.METHOD_INFO[res.method]
    doc = {
        "studio_surface": SURFACE_VERSION,
        "written_by": f"Photographic Studio By Bill {__version__}",
        "method": res.method,
        "method_text": info["name"] + ": " + info["does"],
        "best_for": info["best"], "weak_at": info["weak"],
        "source": source,
        "camera": {k_: meta.get(k_) for k_ in ("make", "model", "lens", "fnumber", "focal_mm", "iso")},
        "size": {"width": int(W), "height": int(H)},
        "files": {k_: os.path.basename(v) for k_, v in files.items()},
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
                          "stray more than half a step (red); gap: measured places that no frame had in focus",
        "agreement": dict(res.agreement_stats(), red_line_steps=DF.DISAGREE_STEPS,
                          png="grey 255 = agree, 128 = half a step apart (the red line), 0 = a step or more; "
                              "transparent = only one measure there"),
        "confidence_png": "255 = sure; 0 = filled in, not measured",
        "tileable": bool(p.tileable),
        "settings": p.to_dict(),
        "notes": notes,
        # version 2 (BUILD_PLAN_DP3D C): every map says how it was made; scale, focus coach, Godot import notes
        "maps": maps_doc,
        "scale": scale_info,
        "tiling": ({"method": tiled[0], "about": "cut: a minimum-error cut through each seam band, the same for every "
                    "map (structured textures); blend: the copies blended (even, random textures)",
                    "flattened": flat_doc}
                   if tiled else None),
        "game_size": ({"size": int(p.size), "crop": {"y": crop[0], "x": crop[1], "side": crop[2]},
                       "about": "the measured area's best square, resized"} if crop else None),
        "focus": {"dof_half_steps": round(res.dof_half, 4) if res.dof_half else None, "coach": list(res.coach)},
        "light": ({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in light.items()}
                  if light else None),
        "godot": godot_notes(kinds, p, mm_px, h_mm),
        # the size one copy of the texture covers on the ground, so the game sets its tiling from a measurement
        "repeat_m": ([round(W * mm_px / 1000.0, 4), round(H * mm_px / 1000.0, 4)] if mm_px else None),
    }
    if extra:
        doc.update(extra)
    if "json" in paths:
        tmp = paths["json"] + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_plain(doc), fh, indent=2)
        os.replace(tmp, paths["json"])
        files["json"] = paths["json"]
    log.info("surface %s: %s, %s, %d x %d, measured both %.0f%%, filled %.0f%%, maps %s", base, res.method,
             doc["height"]["units"], W, H, 100 * shares["measured_both"], 100 * shares["filled"], ",".join(kinds))
    return {"paths": files, "json": doc}


def needs_colour(kinds) -> bool:
    return any(k in kinds for k in ("colour", "albedo", "detail", "roughness"))


def godot_notes(kinds, p: SurfaceParams, mm_px, h_mm) -> dict:
    """What a Godot 4 import needs, so the RPG Forge sets it without guessing (C.7, F.4). Godot's
    StandardMaterial3D reads a height map as brighter = higher unless heightmap_flip_texture is on (that setting is
    for depth maps, where brighter is lower); the Studio's height is white = near = higher, so no flip. Normal maps
    are OpenGL (green up) in Godot."""
    tex = {}
    for k in kinds:
        tex[k] = "srgb" if k in ("colour", "albedo") else "linear"
    out = {"normal_map": "OpenGL (no flip)" if p.normals_y == "up" else "DirectX: flip green on import",
           "heightmap_flip": False,
           "heightmap_about": "white = near = higher; Godot's heightmap_flip_texture stays off (it is for depth maps). "
                              "From the Godot 4 class reference; not yet checked by rendering in the game.",
           "textures": tex,
           "albedo": "use NAME_albedo for albedo_texture when the light should come from the game, else NAME_colour",
           "orm": "ORMMaterial3D: R occlusion, G roughness, B metallic" if "orm" in kinds else None,
           "repeat": bool(p.tileable)}
    if mm_px:
        out["texture_size_m"] = None
    return out


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _scale_from_setting(text: str, col, W: int, H: int) -> dict | None:
    """The size of a pixel from --scale: coin[:NAME] (found in the picture) or click:U1,V1,U2,V2,MM."""
    from . import maps as MP
    t = str(text).strip().lower()
    if t.startswith("click:"):
        try:
            u1, v1, u2, v2, mm = (float(x) for x in t[6:].split(","))
        except ValueError:
            raise ValueError("--scale click:U1,V1,U2,V2,MM (two points 0..1 and the distance between them in mm)")
        mmpx = MP.scale_from_click(u1, v1, u2, v2, mm, W, H)
        return {"kind": "click", "mm_per_px": mmpx, "how": f"two points you clicked, {mm:g} mm apart"}
    if t.startswith("coin"):
        if col is None:
            raise ValueError("--scale coin needs the colour file (the coin is found in the picture)")
        name = t.split(":", 1)[1] if ":" in t else "us_quarter"
        c = MP.find_coin(np.asarray(col, np.float32) / 255.0, name)
        if c is None:
            return None
        return {"kind": "coin", "mm_per_px": c["mm_per_px"], "coin": c["coin"], "coin_mm": c["coin_mm"],
                "centre": c["centre"], "radius_px": c["radius_px"], "score": c["score"],
                "how": f"a {c['coin'].replace('_', ' ')} ({c['coin_mm']} mm) found in the picture"}
    raise ValueError("--scale: coin, coin:NAME or click:U1,V1,U2,V2,MM")


def _pairs_json(pairs: list) -> list:
    return [{"frames": [int(q["pair"]) + 1, int(q.get("to", q["pair"] + 1)) + 1],
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
def stack_colour(an, method: str = "auto", progress=None, cancel=None) -> np.ndarray:
    """The all-in-focus merge of a Focus Stack analysis at full size, upright, developed with the default look, as
    uint8 sRGB; kept in the analysis cache for the merge setting (surfaces, foliage and models share it)."""
    from . import focus as FS
    from . import frames as F
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import apply_orientation
    fp = FS.FocusParams.from_dict(an.params.to_dict())
    fp.method = method if method in FS.METHODS else "auto"
    colour = _cached_colour(an, fp)
    if colour is not None:
        return colour
    with F.Scratch("surface") as sc:
        out = FS.merge(an, fp, True, lambda f, m: FS._tick(progress, 0.05 + 0.8 * f, m), cancel, sc,
                       Edit.from_dict(an.edit))
        FS._check(cancel)
        FS._tick(progress, 0.88, "Developing the colour")
        up = np.ascontiguousarray(apply_orientation(out, an.flip))
        del out
        colour = colour_of_scene(up, an.meta, sc.dir)
        del up
    _keep_colour(an, fp, colour)
    return colour


def stack_surface(an, out_dir: str, base: str, p: SurfaceParams, kinds=KINDS, progress=None, cancel=None,
                  taken: set | None = None, method: str = "auto") -> dict:
    """The surface of a Focus Stack analysis: the fused depth (Dual Pixel when the analysis kept it), and for the
    colour the all-in-focus merge at full size (``method``, default auto), developed with the default look."""
    from . import focus as FS
    FS._tick(progress, 0.02, "Measuring depth")
    res = DF.fuse(an)
    colour = stack_colour(an, method, progress, cancel) if needs_colour(kinds) else None
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


def _colour_cache_path(an, fp) -> str | None:
    """Where an analysis kept in the analysis cache (A.4) keeps its developed full-size merge for a merge setting:
    surfaces, foliage and models from one stack need the same colour, and the full-size merge is the slow part."""
    from .anacache import CacheScratch
    sc = getattr(an, "scratch", None)
    if not isinstance(sc, CacheScratch):
        return None
    import hashlib
    d = {k: v for k, v in fp.to_dict().items() if k not in ("retouch",)}
    d["retouch"] = fp.retouch
    key = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return os.path.join(sc.dir, f"colour_{key}.npy")


def _cached_colour(an, fp):
    p = _colour_cache_path(an, fp)
    if p and os.path.isfile(p):
        try:
            c = np.load(p)
            log.info("surface colour from the analysis cache (%s)", os.path.basename(p))
            return c
        except (OSError, ValueError) as e:
            log.info("cached colour unreadable (%s): merged again", e)
    return None


def _keep_colour(an, fp, colour):
    p = _colour_cache_path(an, fp)
    if not p or colour is None:
        return
    try:
        tmp = p + ".partial.npy"
        np.save(tmp, np.ascontiguousarray(colour, np.uint8))
        os.replace(tmp, p)
    except OSError as e:                             # a full disk: only the speed-up is lost
        log.info("surface colour not kept in the cache (%s)", e)


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
    colour = srgb8(render(dp.image, Edit())) if needs_colour(kinds) else None
    source = {"kind": "one dual pixel photo", "frames": [os.path.basename(dp.path)], "frames_used": 1,
              "reference_frame": os.path.basename(dp.path), "dual_pixel": True, "left_out": []}
    return write(out_dir, base, res, dp.shape, dp.flip, meta, source, p, colour, kinds, taken)


def estimated_result(colour8: np.ndarray, wrap: bool = False) -> DF.DepthResult:
    """A DepthResult for one ordinary photo (``colour8``: uint8 sRGB, upright): the height ESTIMATED at half size
    (maps.estimated_height), labelled estimated everywhere."""
    from . import maps as MP
    c = np.asarray(colour8, np.float32)[::2, ::2] / 255.0
    est = MP.estimated_height(_srgb_to_linear(c), wrap=wrap)
    shp = est.shape
    return DF.DepthResult(depth=est, confidence=np.ones(shp, np.float32),
                          label=np.full(shp, DF.LABEL_ONE, np.uint8), agreement=np.full(shp, np.nan, np.float32),
                          focus_depth=None, focus_quality=None, dp_depth=None, method="estimated",
                          units="estimated")


def photo_surface(path: str, out_dir: str, base: str, p: SurfaceParams, kinds=KINDS, taken: set | None = None) -> dict:
    """The surface of one ordinary photo (raw, TIFF, PNG or JPEG): the photo developed with the default look as the
    colour, the height ESTIMATED from it (every map amber; the JSON says so). For the Forge's texture folders shot
    without a stack or Dual Pixel."""
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import open_image, render
    img = open_image(path)
    try:
        colour = srgb8(render(img, Edit()))
        md = img.meta
        meta = {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s,
                "fnumber": md.fnumber, "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime,
                "focus_distance_m": getattr(md, "focus_distance_m", None)}
    finally:
        try:
            img.drop_cache()
        except Exception:                            # noqa: BLE001
            pass
    res = estimated_result(colour)
    source = {"kind": "one ordinary photo (estimated)", "frames": [os.path.basename(path)], "frames_used": 1,
              "reference_frame": os.path.basename(path), "dual_pixel": False, "left_out": []}
    return write(out_dir, base, res, colour.shape[:2], 0, meta, source, p, colour, kinds, taken)


# ------------------------------------------------------------------------------------------ the Depth view
# the Studio's palette (app/theme.py), plan 23.1: measured two ways, one way, filled in; red = disagree
VIEW_RGB = {DF.LABEL_BOTH: (0x22, 0xD3, 0xEE), DF.LABEL_ONE: (0xF5, 0x9E, 0x0B), DF.LABEL_FILLED: (0xC0, 0x84, 0xFC)}
VIEW_RED = (0xF8, 0x71, 0x71)


def depth_view(res: DF.DepthResult, near_high: bool, flip: int, overlay: bool = True,
               gaps: bool = False) -> np.ndarray:
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
    if gaps and res.gap is not None and res.gap.any():
        # places no frame had in focus (B.3): amber stripes, so they read apart from "measured one way"
        yy, xx = np.indices(res.gap.shape)
        stripe = res.gap & (((yy + xx) // 4) % 2 == 0)
        rgb[stripe] = 0.3 * rgb[stripe] + 0.7 * np.asarray(VIEW_RGB[DF.LABEL_ONE], np.float32)
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

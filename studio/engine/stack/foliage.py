"""Foliage for the game: leaf cards, atlases and glb files from measured depth (BUILD_PLAN_DP3D D).

Bill: "leaves, etc, into a godot 4 video game via blender". A leaf shot wide open against a soft background (a focus
stack or one Dual Pixel photo) is cut out by its measured depth (depthmask.py: the bokeh behind measures nothing),
and each separate leaf in the frame becomes a card:

  colour     the all-in-focus merge, its colour bled into the transparent margin (so mipmaps do not draw a dark halo
             round the leaf), straight (not premultiplied) alpha, the edge anti-aliased
  albedo     the colour with the light divided out (estimated, amber: maps.delight)
  normal     from the leaf's measured relief (its cup and fold), OpenGL
  thickness  how much light comes through (amber): a constant until the Studio can see a frame lit from behind
  mesh       the leaf's 3-D bend as a low grid over its alpha (200-800 triangles), as well as the flat card
  pivot      the stem end (the lowest point of the leaf), and a sway colour per vertex (black at the stem, white at
             the tip) so a wind shader can move the tip and hold the stem

Cards go out as PNGs + NAME_foliage_Studio.json, packed into one atlas sheet (power of two, shelf packing, every map
in the same layout), and as glb (one per card, and one for the atlas: alpha scissor at 0.5, double-sided). No Qt.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import __version__, log as slog

log = slog.get("engine.stack.foliage")

MAKES = ("cards", "atlas", "glb")
FOLIAGE_VERSION = 1


@dataclass
class FoliageParams:
    pick: int = 0                   # keep at most this many leaves (the largest first); 0 = every leaf found
    min_share: float = 0.004        # a leaf smaller than this share of the picture is not a card
    max_atlas: int = 4096           # the atlas grows to at most this (cards shrink evenly to fit); --size overrides
    triangles: int = 600            # the bent mesh's size (200..800 is usual for games)
    atlas_size: int = 0             # the atlas sheet (power of two); 0 = the smallest that holds the cards at size
    padding: int = 8                # pixels between cards in the atlas
    thickness: float = 0.5          # how translucent (0 opaque .. 1 thin); a constant (amber): not measured yet
    ratio: str = ""                 # magnification, for the size in mm (as the surface's --ratio)
    scale: str = ""                 # coin / click:... (as the surface's --scale)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Card:
    index: int
    box: tuple                      # (y0, y1, x0, x1) in the full-size upright picture
    colour: np.ndarray              # uint8 (h, w, 4) straight alpha, colour bled into the margin
    albedo: np.ndarray              # uint8 (h, w, 4)
    normal: np.ndarray              # uint8 (h, w, 3)
    height: np.ndarray              # float32 0..1 (near = 1) over the card
    alpha: np.ndarray               # float32 0..1
    pivot: tuple                    # (u, v) 0..1 within the card: the stem end
    size_mm: tuple | None           # (width, height) when the scale is known
    share: float
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------------------------------ finding leaves
def leaf_regions(depth01: np.ndarray, label: np.ndarray, min_share: float) -> list:
    """Each separate measured region big enough to be a leaf (D.2), largest first: [(label id, size)]."""
    from . import depthmask as DM
    lab, n = DM.regions(depth01, label)
    if n == 0:
        return [], lab
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = [(i, int(sizes[i])) for i in range(1, n + 1) if sizes[i] >= min_share * lab.size]
    return sorted(keep, key=lambda t: -t[1]), lab


def region_alpha(lab: np.ndarray, pick: int, depth01: np.ndarray, label: np.ndarray) -> np.ndarray:
    """One region grown over measured neighbours at its depth, closed and with its holes filled (as depthmask)."""
    from scipy.ndimage import binary_closing, binary_dilation
    from .depthmask import JUMP
    m = lab == pick
    med = float(np.median(depth01[m]))
    near_same = (label > 0) & (np.abs(depth01 - med) < 2.5 * JUMP)
    grow = m.copy()
    for _ in range(3):
        grow = binary_dilation(grow, iterations=4) & (near_same | m)
    from .depthmask import fill_small_holes
    return fill_small_holes(binary_closing(grow, iterations=2)).astype(np.float32)


def bleed(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Colour pushed out from the leaf into its transparent margin (push-pull from the opaque places), so texture
    filtering and mipmaps blend leaf colour, never black, at the edge."""
    from .depthfuse import push_pull
    w = (alpha > 0.5).astype(np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[2]):
        filled = push_pull(np.where(w > 0, rgb[..., c], 0.0), w)
        out[..., c] = np.where(w > 0, rgb[..., c], filled)
    return out


def pivot_of(alpha: np.ndarray) -> tuple:
    """The stem end: the lowest point of the leaf (largest row with alpha), its middle column. (u, v) 0..1."""
    m = alpha > 0.5
    if not m.any():
        return 0.5, 1.0
    rows = np.nonzero(m.any(axis=1))[0]
    r = int(rows.max())
    cols = np.nonzero(m[r])[0]
    return float((cols.mean() + 0.5) / alpha.shape[1]), float((r + 0.5) / alpha.shape[0])


def make_cards(depth01: np.ndarray, label: np.ndarray, colour_u8: np.ndarray, p: FoliageParams,
               mm_px: float | None = None, slope: float | None = None) -> list:
    """Cards from a measured depth (upright, near = 1, any size: brought to the colour's size) and the colour
    (uint8 sRGB, upright, full size)."""
    from scipy.ndimage import zoom
    from . import maps as MP
    from .surface import _linear_to_srgb, _srgb_to_linear, auto_slope, normals
    H, W = colour_u8.shape[:2]
    regs, lab = leaf_regions(depth01, label, p.min_share)
    if p.pick:
        regs = regs[:int(p.pick)]
    fy, fx = H / depth01.shape[0], W / depth01.shape[1]
    cards = []
    for k, (rid, size) in enumerate(regs):
        a_small = region_alpha(lab, rid, depth01, label)
        ys, xs = np.nonzero(a_small > 0.5)
        if ys.size == 0:
            continue
        pad = 3
        y0, y1 = max(0, ys.min() - pad), min(a_small.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(a_small.shape[1], xs.max() + pad + 1)
        Y0, Y1, X0, X1 = int(y0 * fy), min(H, int(math.ceil(y1 * fy))), int(x0 * fx), min(W, int(math.ceil(x1 * fx)))
        h, w = Y1 - Y0, X1 - X0
        if h < 8 or w < 8:
            continue

        def up(m):
            z = zoom(m[y0:y1, x0:x1].astype(np.float32), (h / (y1 - y0), w / (x1 - x0)), order=1,
                     grid_mode=True, mode="nearest")
            return np.pad(z, ((0, max(0, h - z.shape[0])), (0, max(0, w - z.shape[1]))), mode="edge")[:h, :w]
        col = colour_u8[Y0:Y1, X0:X1].astype(np.float32)
        a = up(a_small)
        from ..detail.finish import guided
        g = (0.25 * col[..., 0] + 0.5 * col[..., 1] + 0.25 * col[..., 2]) / 255.0
        a = np.clip(guided(g.astype(np.float32), a.astype(np.float32), max(2, int(round(fy * 2))), 0.004), 0, 1)
        hgt = up(np.where(label > 0, depth01, np.nan_to_num(depth01)))
        lin = _srgb_to_linear(col / 255.0)
        sl = slope if slope is not None else auto_slope(hgt, 1.0)
        alb, fit, sh = MP.delight(lin, hgt, sl, trust=a > 0.5)
        bled = bleed(col, a)
        bled_alb = bleed(_linear_to_srgb(alb) * 255.0, a)
        nrm = normals(hgt, sl, True)
        A8 = np.rint(a * 255.0).astype(np.uint8)[..., None]
        cards.append(Card(
            index=k, box=(Y0, Y1, X0, X1),
            colour=np.concatenate([np.clip(np.rint(bled), 0, 255).astype(np.uint8), A8], 2),
            albedo=np.concatenate([np.clip(np.rint(bled_alb), 0, 255).astype(np.uint8), A8], 2),
            normal=np.rint(nrm * 127.5 + 127.5).astype(np.uint8), height=hgt.astype(np.float32),
            alpha=a.astype(np.float32), pivot=pivot_of(a),
            size_mm=(round(w * mm_px, 2), round(h * mm_px, 2)) if mm_px else None,
            share=round(float(size) / lab.size, 4),
            notes=[] if fit.get("ok") else ["the light could not be fitted on this leaf: its albedo is its colour"]))
    return cards


# ------------------------------------------------------------------------------------------ meshes
def bent_mesh(card: Card, triangles: int = 600, depth_scale: float | None = None):
    """(positions (n, 3), uvs (n, 2), triangles (m, 3), sway colours (n, 4)): a grid over the leaf's alpha with its
    measured height, pivot at the origin, the card's width = 1 model unit (or metres when the size is known), Y up,
    the leaf facing +Z. ``depth_scale``: height range in the same units (default: 10% of the width)."""
    h, w = card.alpha.shape
    # a grid whose kept triangles come near the target
    fill = max(float((card.alpha > 0.5).mean()), 0.05)
    cells = max(4.0, triangles / (2.0 * fill))
    gx = max(3, int(round(math.sqrt(cells * w / h))) + 1)
    gy = max(3, int(round(cells / max(gx - 1, 1))) + 1)
    us, vs = np.linspace(0, 1, gx), np.linspace(0, 1, gy)
    U, V = np.meshgrid(us, vs)
    from scipy.ndimage import map_coordinates
    pa = map_coordinates(card.alpha, [V.ravel() * (h - 1), U.ravel() * (w - 1)], order=1).reshape(U.shape)
    ph = map_coordinates(card.height, [V.ravel() * (h - 1), U.ravel() * (w - 1)], order=1).reshape(U.shape)
    keep = pa > 0.5
    width = card.size_mm[0] / 1000.0 if card.size_mm else 1.0
    height_u = width * h / w
    ds = depth_scale if depth_scale is not None else 0.1 * width
    pu, pv = card.pivot
    X = (U - pu) * width
    Y = (pv - V) * height_u
    Z = (ph - float(np.median(ph[keep])) if keep.any() else ph) * ds
    idx = -np.ones(U.shape, np.int64)
    idx[keep] = np.arange(int(keep.sum()))
    pos = np.stack([X[keep], Y[keep], Z[keep]], 1).astype(np.float32)
    uv = np.stack([U[keep], V[keep]], 1).astype(np.float32)
    tris = []
    for j in range(gy - 1):
        for i in range(gx - 1):
            a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i], idx[j + 1, i + 1]
            if a >= 0 and c >= 0 and b >= 0:
                tris.append((a, c, b))
            if b >= 0 and c >= 0 and d >= 0:
                tris.append((b, c, d))
    tri = np.asarray(tris, np.int64).reshape(-1, 3)
    dist = np.hypot(pos[:, 0], pos[:, 1])
    t = dist / max(float(dist.max()), 1e-9)
    col = np.stack([t, t, t, np.ones_like(t)], 1).astype(np.float32)
    return pos, uv, tri, col


def quad(card: Card):
    """The flat card: two triangles over the card's rectangle, pivot at the origin."""
    width = card.size_mm[0] / 1000.0 if card.size_mm else 1.0
    h, w = card.alpha.shape
    height_u = width * h / w
    pu, pv = card.pivot
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    pos = np.array([[(u - pu) * width, (pv - v) * height_u, 0.0] for u, v in uv], np.float32)
    tri = np.array([[0, 2, 1], [1, 2, 3]], np.int64)
    t = np.hypot(pos[:, 0], pos[:, 1])
    t = t / max(float(t.max()), 1e-9)
    col = np.stack([t, t, t, np.ones_like(t)], 1).astype(np.float32)
    return pos, uv, tri, col


# ------------------------------------------------------------------------------------------ atlas (D.3)
def shelf_pack(sizes: list, sheet: int, pad: int) -> list | None:
    """Rectangles (w, h) packed in rows (tallest first) into a sheet x sheet square: [(x, y)] in the input order,
    or None when they do not fit."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos = [None] * len(sizes)
    x = y = pad
    row_h = 0
    for i in order:
        w, h = sizes[i]
        if w + 2 * pad > sheet or h + 2 * pad > sheet:
            return None
        if x + w + pad > sheet:
            x = pad
            y += row_h + pad
            row_h = 0
        if y + h + pad > sheet:
            return None
        pos[i] = (x, y)
        x += w + pad
        row_h = max(row_h, h)
    return pos


def atlas(cards: list, size: int = 0, pad: int = 8, cap: int = 4096) -> tuple:
    """(sheets {colour, albedo, normal}, layout [{x, y, w, h, scale}], sheet size): every card in one power-of-two
    sheet, the same layout for every map; cards are scaled down evenly when ``size`` is too small for them."""
    from PIL import Image
    sizes = [(c.alpha.shape[1], c.alpha.shape[0]) for c in cards]
    sheet = size or 256
    scale = 1.0
    while True:
        sz = [(max(1, int(w * scale)), max(1, int(h * scale))) for w, h in sizes]
        pos = shelf_pack(sz, sheet, pad)
        if pos is not None:
            break
        if not size and sheet < cap:
            sheet *= 2
        else:
            scale *= 0.9
            if scale < 0.02:
                raise ValueError("the leaves do not fit in the atlas")
    sheets = {"colour": np.zeros((sheet, sheet, 4), np.uint8), "albedo": np.zeros((sheet, sheet, 4), np.uint8),
              "normal": np.full((sheet, sheet, 3), (128, 128, 255), np.uint8)}
    layout = []
    for c, (x, y), (w, h) in zip(cards, pos, sz):
        for k, arr in (("colour", c.colour), ("albedo", c.albedo), ("normal", c.normal)):
            im = Image.fromarray(arr)
            if im.size != (w, h):
                im = im.resize((w, h), Image.Resampling.BOX if w < arr.shape[1] else Image.Resampling.BICUBIC)
            sheets[k][y:y + h, x:x + w] = np.asarray(im)
        layout.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "scale": round(scale, 4)})
    return sheets, layout, sheet


# ------------------------------------------------------------------------------------------ writing
def names(base: str) -> dict:
    return {"json": f"{base}_foliage_Studio.json", "atlas_colour": f"{base}_foliage_atlas_colour_Studio.png",
            "atlas_albedo": f"{base}_foliage_atlas_albedo_Studio.png",
            "atlas_normal": f"{base}_foliage_atlas_normal_Studio.png", "atlas_glb": f"{base}_foliage_atlas_Studio.glb"}


def card_names(base: str, i: int) -> dict:
    b = f"{base}_leaf{i + 1}"
    return {"colour": f"{b}_colour_Studio.png", "albedo": f"{b}_albedo_Studio.png",
            "normal": f"{b}_normal_Studio.png", "glb": f"{b}_Studio.glb"}


def write(out_dir: str, base: str, cards: list, p: FoliageParams, makes, meta: dict, source: dict,
          scale_how: str = "", taken: set | None = None) -> dict:
    """Write the cards, the atlas and the glb files asked for, and NAME_foliage_Studio.json. Returns {"paths":
    [...], "json": doc}."""
    from ..io import gltf as G
    from ..io.export import free_target, write_png_any
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def tgt(name):
        q = free_target(os.path.join(out_dir, name), None, taken)
        written.append(q)
        return q
    entries = []
    for c in cards:
        cn = card_names(base, c.index)
        e = {"leaf": c.index + 1, "share_of_picture": c.share, "size_mm": c.size_mm,
             "pivot_uv": [round(c.pivot[0], 4), round(c.pivot[1], 4)],
             "pixels": [int(c.alpha.shape[1]), int(c.alpha.shape[0])], "notes": c.notes}
        if "cards" in makes:
            files = {}
            for k in ("colour", "albedo", "normal"):
                q = tgt(cn[k])
                write_png_any(q, getattr(c, k), None, meta)
                files[k] = os.path.basename(q)
            e["files"] = files
        if "glb" in makes:
            pos, uv, tri, col = bent_mesh(c, p.triangles)
            qp, quv, qtri, qcol = quad(c)
            mat = G.Material(name=f"leaf{c.index + 1}", base_colour=c.colour, normal=c.normal, alpha_mode="MASK",
                             alpha_cutoff=0.5, double_sided=True, roughness=0.8,
                             extras={"how": "colour measured (the merge); normal from the measured relief; "
                                            "alpha from the measured depth"})
            meshes = [G.Mesh(f"leaf{c.index + 1}_bent", pos, tri, uvs=uv, colours=col, material=0,
                             extras={"triangles": int(tri.shape[0])}),
                      G.Mesh(f"leaf{c.index + 1}_card", qp, qtri, uvs=quv, colours=qcol, material=0)]
            q = tgt(cn["glb"])
            G.write_glb(q, meshes, [mat], extras=_extras(c, p))
            e["glb"] = os.path.basename(q)
            e["mesh_triangles"] = int(tri.shape[0])
        entries.append(e)
    doc_atlas = None
    if "atlas" in makes and cards:
        sheets, layout, sheet = atlas(cards, p.atlas_size, p.padding, p.max_atlas)
        nm = names(base)
        files = {}
        for k in ("colour", "albedo", "normal"):
            q = tgt(nm[f"atlas_{k}"])
            write_png_any(q, sheets[k], None, meta)
            files[k] = os.path.basename(q)
        rects = []
        for c, L in zip(cards, layout):
            rects.append({"leaf": c.index + 1, "uv": [round(L["x"] / sheet, 6), round(L["y"] / sheet, 6),
                                                      round((L["x"] + L["w"]) / sheet, 6),
                                                      round((L["y"] + L["h"]) / sheet, 6)],
                          "pixels": [L["x"], L["y"], L["w"], L["h"]], "size_mm": c.size_mm,
                          "pivot_uv": [round((L["x"] + c.pivot[0] * L["w"]) / sheet, 6),
                                       round((L["y"] + c.pivot[1] * L["h"]) / sheet, 6)]})
        doc_atlas = {"size": sheet, "files": files, "padding": p.padding, "cards": rects,
                     "packing": "shelves (tallest first), the same layout for every map"}
        if "glb" in makes:
            meshes = []
            for c, L in zip(cards, layout):
                qp, quv, qtri, qcol = quad(c)
                u0, v0 = L["x"] / sheet, L["y"] / sheet
                quv2 = np.stack([u0 + quv[:, 0] * L["w"] / sheet, v0 + quv[:, 1] * L["h"] / sheet], 1)
                meshes.append(G.Mesh(f"leaf{c.index + 1}", qp, qtri, uvs=quv2.astype(np.float32), colours=qcol,
                                     material=0))
            mat = G.Material(name="foliage_atlas", base_colour=sheets["colour"], normal=sheets["normal"],
                             alpha_mode="MASK", alpha_cutoff=0.5, double_sided=True, roughness=0.8)
            q = tgt(nm["atlas_glb"])
            G.write_glb(q, meshes, [mat], extras={"studio": "foliage atlas", "leaves": len(cards)})
            doc_atlas["glb"] = os.path.basename(q)
    doc = {
        "studio_foliage": FOLIAGE_VERSION, "written_by": f"Photographic Studio By Bill {__version__}",
        "source": source, "camera": {k: meta.get(k) for k in ("make", "model", "lens", "fnumber", "focal_mm")},
        "leaves": entries, "atlas": doc_atlas, "settings": p.to_dict(),
        "alpha": "straight (not premultiplied); colour bled into the transparent margin",
        "how": {"colour": "measured: the all-in-focus merge", "alpha": "measured depth (C.9), snapped to the colour "
                "edges", "normal": "computed from the measured relief", "albedo": "estimated (amber): the light "
                "fitted to the measured shape and divided out", "thickness": f"a constant {p.thickness:g} (amber): "
                "not measured (a frame lit from behind would measure it)"},
        "scale": scale_how or "relative: the card's width is 1 model unit (give --ratio or put a coin in the frame "
                              "for millimetres)",
        "godot": {"material": "alpha scissor (threshold 0.5), cull disabled (double-sided)",
                  "backlight": f"backlight / transmission from the thickness ({p.thickness:g})",
                  "wind": "COLOR_0: black at the stem, white at the tip: sway by the vertex colour",
                  "normal_map": "OpenGL (no flip)", "albedo_srgb": True},
    }
    q = free_target(os.path.join(out_dir, names(base)["json"]), None, taken)
    with open(q, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    written.append(q)
    return {"paths": written, "json": doc, "json_path": q}


def _extras(c: Card, p: FoliageParams) -> dict:
    return {"studio": "leaf card", "leaf": c.index + 1, "size_mm": c.size_mm, "pivot_uv": list(c.pivot),
            "thickness": p.thickness, "thickness_how": "constant (not measured)",
            "wind": "COLOR_0 black at the stem, white at the tip"}


# ------------------------------------------------------------------------------------------ from stacks
def from_analysis(an, p: FoliageParams, progress=None, cancel=None) -> tuple:
    """(cards, meta, source, scale how) from a Focus Stack analysis."""
    from . import depthfuse as DF
    from . import surface as SURF
    from ..pipeline.pipeline import apply_orientation
    res = DF.fuse(an)
    near_high, *_ = SURF.decide_near(res, str(an.meta.get("model") or ""), SURF.SurfaceParams())
    h = res.depth if near_high else -res.depth
    lo, hi = SURF.height_range(h, res.label)
    from .depthmask import strict_label
    d01 = np.ascontiguousarray(apply_orientation(np.clip((h - lo) / (hi - lo), 0, 1), an.flip)).astype(np.float32)
    label = np.ascontiguousarray(apply_orientation(strict_label(res), an.flip))
    colour = SURF.stack_colour(an, "auto", progress, cancel)
    mm_px, how = _scale(p, an.meta, colour)
    cards = make_cards(d01, label, colour, p, mm_px)
    src = {"kind": "focus stack", "frames": [os.path.basename(q) for q in an.paths], "dual_pixel": bool(an.dual_pixel)}
    return cards, an.meta, src, how


def from_dual_photo(path: str, p: FoliageParams) -> tuple:
    from . import depthmask as DM
    from . import surface as SURF
    from ..dualpixel import decode as DD
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import render
    got = DM.measured_depth_for(path)
    if got is None:
        raise ValueError(f"{os.path.basename(path)} is not a Dual Pixel RAW: leaves need measured depth")
    d01, label, _ = got
    dp = DD.open_dual(path)
    try:
        colour = SURF.srgb8(render(dp.image, Edit()))
        md = dp.image.meta
        meta = {"make": md.make, "model": md.model, "lens": md.lens, "fnumber": md.fnumber, "focal_mm": md.focal_mm}
    finally:
        try:
            dp.image.drop_cache()
        except Exception:                            # noqa: BLE001
            pass
    mm_px, how = _scale(p, meta, colour)
    return make_cards(d01, label, colour, p, mm_px), meta, {"kind": "one dual pixel photo",
                                                           "frames": [os.path.basename(path)]}, how


def _scale(p: FoliageParams, meta: dict, colour) -> tuple:
    from . import surface as SURF
    mm_px, how = None, ""
    mag = SURF.parse_ratio(p.ratio) if p.ratio else None
    pmm = SURF.pixel_mm(str(meta.get("model") or ""))
    if mag and pmm:
        mm_px, how = pmm / mag, f"magnification {p.ratio}"
    if p.scale:
        sc = SURF._scale_from_setting(p.scale, np.asarray(colour, np.float32), colour.shape[1], colour.shape[0])
        if sc:
            mm_px, how = sc["mm_per_px"], sc["how"]
    return mm_px, how


def batch_job(si, out_dir, sp, fp, taken, cancel, workers):
    """stacks --run foliage: one stack (or one Dual Pixel photo) through the foliage cards, atlas and glb."""
    from . import focus as FS
    p = FoliageParams(ratio=getattr(sp, "ratio", "") or "", scale=getattr(sp, "scale", "") or "")
    if len(si.paths) == 1:
        cards, meta, src, how = from_dual_photo(si.paths[0], p)
    else:
        an = FS.analyse(si.paths, FS.FocusParams.from_dict((fp or FS.FocusParams()).to_dict()), None, cancel,
                        cache=True, workers=workers)
        try:
            cards, meta, src, how = from_analysis(an, p, None, cancel)
        finally:
            an.close()
    if not cards:
        raise ValueError("no leaf was found by measured depth")
    r = write(out_dir, si.base, cards, p, MAKES, meta, src, how, taken)
    return r["paths"], [f"{len(cards)} leaf card(s)"]

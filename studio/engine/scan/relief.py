"""A front-only relief from one view (Bill, 2026-11-21: "how about the ones for the front only relief?").

When views cannot be joined into a whole model (too few, too far apart: the angel set), each stack still measured its
own depth. A relief is that depth as a mesh: the front of the subject as one view saw it (a 2.5-D shape, not a full
model), textured with the view's all-in-focus picture, cut to the subject by its measured-depth mask when one was
found (else the whole frame), as a glb for Blender and Godot: hero props seen from the front, parallax cards, or a
base to bake height and normal maps from.

* The mesh: a grid over the picture (``GRID`` cells on the long side), lifted by the measured depth (near = high),
  with no face across a depth edge (where neighbouring depths jump, the surface is cut instead of stretched), the
  holes Dual Pixel could not measure filled from their surroundings (violet in the "made" map); simplified to the
  triangle target by quadric clustering (``decimate``), the full grid's detail baked into a normal map.
* **Depth proportion**: one view measures depth only up to its distance (a near object's few millimetres and a far
  one's metres give the same Dual Pixel offsets), so how deep the relief is against its width is a setting: by
  default the subject is made half as deep as it is wide (amber, "estimated"), or ``depth_ratio`` as the
  photographer wants (in Blender or Godot it can also be scaled along one axis). A view whose depth shape the camera
  solver found (it was placed with others) uses that instead (cyan).
* Files: ``NAME_relief_Studio.glb``, ``NAME_relief_made_Studio.png`` (cyan measured, violet filled), and
  ``NAME_relief_Studio.json`` ("studio_relief": 1).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates, uniform_filter

RELIEF_VERSION = 1
GRID = 384
DEFAULT_RATIO = 0.5
EDGE_JUMP = 0.08                # a jump of more than this share of the depth range between neighbours cuts the surface


def names(base: str) -> dict:
    return {"glb": f"{base}_relief_Studio.glb", "made": f"{base}_relief_made_Studio.png",
            "json": f"{base}_relief_Studio.json"}


def _targets(out_dir, base, taken):
    from ..io.export import is_own_export
    n = 1
    while True:
        b = base if n == 1 else f"{base}-{n}"
        paths = {k: os.path.join(out_dir, v) for k, v in names(b).items()}
        keys = {os.path.normcase(os.path.abspath(p)) for p in paths.values()}
        free = all((not os.path.exists(p)) or is_own_export(p) for p in paths.values())
        if free and not (taken and keys & taken):
            if taken is not None:
                taken.update(keys)
            return paths
        n += 1


def _fill(d, known):
    """Unknown depths taken from the nearest known ones, then softened at the joins."""
    if known.all() or not known.any():
        return d
    idx = distance_transform_edt(~known, return_distances=False, return_indices=True)
    f = d[tuple(idx)]
    soft = uniform_filter(f, 5)
    return np.where(known, d, soft)


def build(view, out_dir: str, base: str, depth_ratio: float | None = None, triangles: int = 20000,
          max_texture: int = 4096, kappa: float | None = None, taken: set | None = None) -> dict:
    """The relief of one view (scan.views.View). ``kappa``: the view's depth shape when the solver found it."""
    from ... import __version__
    from . import decimate as DC
    from ..io import gltf as GL
    from ..io.export import write_png_any
    h, w = view.shape
    tau = np.asarray(view.tau, np.float32)
    conf = np.asarray(view.conf, np.float32)
    known = (conf > 0.3) & np.isfinite(tau)
    region = view.mask.copy() if view.mask is not None else np.ones((h, w), bool)
    cut = view.mask is not None
    if (known & region).sum() < 100:
        raise ValueError("too little measured depth in this view for a relief")
    tau_f = _fill(np.where(known, tau, 0.0), known)
    # the grid
    s = GRID / float(max(h, w))
    gh, gw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    ys = (np.arange(gh) + 0.5) / gh * h - 0.5
    xs = (np.arange(gw) + 0.5) / gw * w - 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    td = map_coordinates(tau_f, [gy, gx], order=1, mode="nearest")
    kd = map_coordinates(known.astype(np.float32), [gy, gx], order=1, mode="nearest") > 0.5
    rd = map_coordinates(region.astype(np.float32), [gy, gx], order=0, mode="nearest") > 0.5
    if cut:
        rd = binary_erosion(rd, iterations=1) | (rd & kd)
    vals = td[rd]
    lo, hi = np.percentile(vals, [1, 99])
    hgt = np.clip((td - lo) / max(float(hi - lo), 1e-6), 0, 1)
    # depth proportion
    ys_r, xs_r = np.nonzero(rd)
    width = (xs_r.max() - xs_r.min() + 1) / float(gw) if len(xs_r) else 1.0
    if kappa is not None and kappa > 0:
        # inverse distance 1 + kappa tau: depth (in units of the view's distance) of the range lo..hi
        z_lo, z_hi = 1.0 / max(1 + kappa * lo, 1e-3), 1.0 / max(1 + kappa * hi, 1e-3)
        fov_w = w / view.f                                   # the picture's width at distance 1
        ratio = float(abs(z_lo - z_hi) / max(width * fov_w, 1e-6))
        ratio_how, ratio_colour = "from the camera solver's depth shape for this view (measured)", "cyan"
    elif depth_ratio:
        ratio = float(depth_ratio)
        ratio_how, ratio_colour = f"set by you: {ratio:g} x the subject's width", "amber"
    else:
        ratio = DEFAULT_RATIO
        ratio_how = ("estimated: one view measures depth only up to its distance, so the subject is made half as "
                     "deep as it is wide; scale it along its depth in Blender or Godot, or give Relief depth")
        ratio_colour = "amber"
    aspect = gh / float(gw)
    X = (np.arange(gw) + 0.5) / gw - 0.5
    Y = ((np.arange(gh) + 0.5) / gh - 0.5) * aspect
    gxx, gyy = np.meshgrid(X, Y)
    Z = hgt * ratio * width
    V = np.stack([gxx, -gyy, Z], -1).reshape(-1, 3)        # x right, y up, z towards the viewer (glTF)
    UV = np.stack([(gx + 0.5) / w, (gy + 0.5) / h], -1).reshape(-1, 2)
    # faces: two triangles per cell inside the region, none across a depth edge
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    inside = rd[:-1, :-1] & rd[:-1, 1:] & rd[1:, :-1] & rd[1:, 1:]
    hc = np.stack([hgt[:-1, :-1], hgt[:-1, 1:], hgt[1:, :-1], hgt[1:, 1:]])
    smooth = (hc.max(0) - hc.min(0)) < EDGE_JUMP
    ok = inside & smooth
    T = np.concatenate([np.stack([a[ok], c[ok], b[ok]], 1), np.stack([b[ok], c[ok], d[ok]], 1)])
    if not len(T):
        raise ValueError("no surface left after cutting at the depth edges")
    from .isosurface import remove_unused
    meas_v = kd.reshape(-1).astype(np.float32)
    Vf, Tf, UVf, measf = remove_unused(V, T, UV, meas_v)
    Vl, Tl, UVl = DC.simplify(Vf, Tf, int(triangles), np.zeros(len(Vf), int), UVf)
    # texture: the view's colour cropped to what the mesh covers, alpha where it covers
    col = view.colour if view.colour is not None else np.dstack([np.clip(view.image * 255, 0, 255)] * 3).astype(np.uint8)
    H, W = col.shape[:2]
    u0, v0 = np.floor(UVf.min(0) * [W, H]).astype(int)
    u1, v1 = np.ceil(UVf.max(0) * [W, H]).astype(int)
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, W), min(v1, H)
    crop = col[v0:v1, u0:u1]
    sc = min(1.0, max_texture / float(max(crop.shape[:2])))
    if sc < 1.0:
        from PIL import Image as PILImage
        crop = np.asarray(PILImage.fromarray(crop).resize((max(1, int(crop.shape[1] * sc)),
                                                          max(1, int(crop.shape[0] * sc))), PILImage.LANCZOS))
    th, tw = crop.shape[:2]

    def to_tex(uv):
        return np.stack([(uv[:, 0] * W - u0) / max(u1 - u0, 1), (uv[:, 1] * H - v0) / max(v1 - v0, 1)], 1)
    uvf_t, uvl_t = to_tex(UVf), to_tex(UVl)
    from . import raster as RA
    cover, bary, _ = RA.rasterize(uvf_t * [tw, th] - 0.5, np.zeros(len(uvf_t)), Tf, th, tw)
    alpha = (cover >= 0)
    rgba = np.dstack([crop, (alpha * 255).astype(np.uint8)])
    made = np.zeros((th, tw, 3), np.uint8)
    if alpha.any():
        mv = (measf[Tf[cover[alpha]]] * bary[alpha]).sum(1) >= 0.5
        mm = np.zeros((th, tw), bool)
        mm[alpha] = mv
        made[mm] = (0x22, 0xD3, 0xEE)
        made[alpha & ~mm] = (0xC0, 0x84, 0xFC)
    nrm = DC.bake_normals(Vf, Tf, uvf_t, Vl, Tl, uvl_t, int(max(th, tw)))[:th, :tw] if th == tw else \
        _bake_rect(DC, Vf, Tf, uvf_t, Vl, Tl, uvl_t, th, tw)
    os.makedirs(out_dir, exist_ok=True)
    paths = _targets(out_dir, base, taken)
    mat = GL.Material(name=f"{base}_relief", base_colour=rgba, normal=nrm, roughness=0.85, alpha_mode="MASK",
                      double_sided=True)
    mesh = GL.Mesh(name=f"{base}_relief", positions=Vl.astype(np.float32), indices=Tl,
                   normals=GL.vertex_normals(Vl.astype(np.float32), Tl), uvs=uvl_t.astype(np.float32), material=0)
    GL.write_glb(paths["glb"], [mesh], [mat], extras={"studio_relief": RELIEF_VERSION})
    write_png_any(paths["made"], made)
    measured_share = float(alpha[alpha].size and (made[alpha][:, 0] == 0x22).mean())
    doc = {"studio_relief": RELIEF_VERSION, "written_by": f"Photographic Studio By Bill {__version__}",
           "files": {k: os.path.basename(v) for k, v in paths.items()},
           "source": {"view": view.name, "path": view.path},
           "what": "a front-only relief: the subject as this one view saw it (not a full model)",
           "cut_to_subject": bool(cut),
           "depth": {"ratio": round(ratio, 4), "how": ratio_how, "colour": ratio_colour},
           "measured_share": round(measured_share, 4), "triangles": int(len(Tl)), "triangles_full": int(len(Tf)),
           "units": "relative: 1 = the picture's width", "axes": "glTF: x right, y up, z towards the viewer",
           "made_png": "cyan: measured; violet: filled (Dual Pixel measured nothing there)",
           "godot": {"import": "the .glb directly; alpha scissor (MASK) keeps the cut-out edge",
                     "normal_map": "OpenGL (no flip)"}}
    tmp = paths["json"] + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, paths["json"])
    return {"paths": paths, "json": doc, "mesh": (Vl, Tl)}


def _bake_rect(DC, Vf, Tf, uvf, Vl, Tl, uvl, th, tw):
    """Normal map for a non-square texture: baked square at the larger side, then resized to (th, tw)."""
    S = int(max(th, tw))
    sq = DC.bake_normals(Vf, Tf, uvf, Vl, Tl, uvl, S)
    from PIL import Image as PILImage
    return np.asarray(PILImage.fromarray(sq).resize((tw, th), PILImage.BILINEAR))

"""The whole 3D Scan job: views in, a textured model and its files out (BUILD_PLAN_DP3D E, F, G.3).

views -> camera solver (``solve``) -> dense depth two ways per view (``dense``) -> fused volume closed by the
silhouettes (``fusion``) -> surface (``isosurface``) -> texture atlas (``texture``) -> game mesh and LODs with baked
normal maps (``decimate``) -> files:

    NAME_model_Studio.glb            the game mesh (LOD0, and LOD1 / LOD2 with --lods), colour and normal maps embedded
    NAME_model_full_Studio.glb       the full-resolution mesh (--make mesh)
    NAME_model_Studio.obj (+ .mtl)   for any tool (--make obj), with NAME_model_colour/normal_Studio.png
    NAME_model_made_Studio.png       how each texel was made: cyan measured, violet filled
    NAME_points_Studio.ply           the solver's points (--make points)
    NAME_cameras_Studio.json         every view: placed or not (and why), how, pose, focal length, quality
    NAME_dataset_Studio/             the Blender dataset (``dataset``), only when the quality gate passes
    NAME_model_Studio.json           the contract the RPG Forge reads ("studio_model": 1)

The model frame is glTF's: Y up (the cameras' average up), origin at the bottom centre of the object, metres when
the scale is known (from the lenses' recorded focus distance: amber), else model units (1 = the first view's
distance to its subject).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.scan.model")

MODEL_VERSION = 1
MAKES = ("cameras", "points", "mesh", "glb", "obj", "dataset")
DEFAULT_MAKES = ("cameras", "points", "glb", "dataset")


@dataclass
class ModelParams:
    triangles: int = 30000           # the game mesh (LOD0); LOD1 a quarter, LOD2 a fifteenth
    lods: bool = False
    ground_cut: bool = False
    voxels: int = 160                # the fused volume's resolution (its cube root of voxels)
    atlas: int = 4096                # largest texture side
    affine: str = "auto"             # auto | on | off: matching views far round (affine simulation)
    use_depth: bool = True           # False: plain structure from motion (for comparison only)
    gate: bool = True                # the Blender dataset only when the quality gate passes
    features: int = 4000
    known_length_m: float = 0.0      # the object's longest side, when the photographer knows it (else relative units)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def names(base: str) -> dict:
    return {"json": f"{base}_model_Studio.json", "glb": f"{base}_model_Studio.glb",
            "mesh": f"{base}_model_full_Studio.glb", "obj": f"{base}_model_Studio.obj",
            "colour": f"{base}_model_colour_Studio.png", "normal": f"{base}_model_normal_Studio.png",
            "made": f"{base}_model_made_Studio.png", "points": f"{base}_points_Studio.ply",
            "cameras": f"{base}_cameras_Studio.json", "dataset": f"{base}_dataset_Studio"}


def targets(out_dir: str, base: str, taken: set | None = None) -> dict:
    """The set of paths, moved together to NAME-2 ... when any one cannot be used (a file or folder the Studio did
    not write, or one already written in this run)."""
    from ..io.export import is_own_export

    def key(p):
        return os.path.normcase(os.path.abspath(p))

    def usable(p):
        if taken is not None and key(p) in taken:
            return False
        if not os.path.lexists(p):
            return True
        if os.path.isdir(p):
            return os.path.isfile(os.path.join(p, "dataset.json")) and \
                is_own_export(os.path.join(p, "dataset.json"))
        return os.path.isfile(p) and is_own_export(p)
    n = 1
    while True:
        b = base if n == 1 else f"{base}-{n}"
        paths = {k: os.path.join(out_dir, v) for k, v in names(b).items()}
        if all(usable(p) for p in paths.values()):
            break
        n += 1
        if n > 9999:
            raise OSError(f"no free file names for {base}")
    if taken is not None:
        taken.update(key(p) for p in paths.values())
    return paths


def up_vector(sol) -> np.ndarray:
    """The cameras' average up (each camera's -y axis in the world): hand-held cameras are held roughly level."""
    ups = [-sol.R[i][1] for i in range(len(sol.names)) if sol.placed[i]]
    u = np.mean(ups, 0)
    return u / max(np.linalg.norm(u), 1e-12)


def scale_from_views(sol, views) -> tuple[float | None, str]:
    """(metres per model unit, how) from the lenses' recorded focus distances: a view's middle depth (tau = 0) is
    1 / m model units away; the lens says it was focused at d metres. Amber: the lens's distance is coarse."""
    vals = []
    for i, v in enumerate(views):
        d = (v.meta or {}).get("focus_distance_m")
        if sol.placed[i] and d and sol.m[i]:
            vals.append(float(d) * float(sol.m[i]))
    if not vals:
        return None, "not known: the model is in relative units (1 = the first view's distance to its subject)"
    return float(np.median(vals)), (f"the focus distance the lenses recorded ({len(vals)} view(s)); coarse, so the "
                                    "size is amber")


def _ground_cut(V, T, up, margin_share=0.01):
    """Remove what lies below a plane fitted to the lowest part (RANSAC on the lowest tenth along up)."""
    h = V @ up
    low = V[h <= np.percentile(h, 10)]
    if len(low) < 30:
        return V, T, False
    rng = np.random.default_rng(0)
    best = (0, None)
    ext = float(np.ptp(h)) or 1.0
    for _ in range(200):
        p = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up) < 0.8:
            continue
        d = np.abs((low - p[0]) @ n)
        cnt = int((d < 0.01 * ext).sum())
        if cnt > best[0]:
            best = (cnt, (n if n @ up > 0 else -n, p[0]))
    if best[1] is None or best[0] < 0.3 * len(low):
        return V, T, False
    n, p0 = best[1]
    keep_v = (V - p0) @ n > margin_share * ext
    keep_t = keep_v[T].all(1)
    from .isosurface import remove_unused
    V2, T2 = remove_unused(V, T[keep_t])
    return V2, T2, True


def build(views, out_dir: str, base: str, p: ModelParams | None = None, makes=DEFAULT_MAKES, progress=None,
          cancel=None, taken: set | None = None, solution=None, group: int | None = None) -> dict:
    """Run the whole job and write the files asked for. Returns {"paths": {kind: path}, "json": doc, "solution",
    "gate": (passed, reasons)}."""
    from . import coach as CO
    from . import dataset as DS
    from . import decimate as DC
    from . import dense as DN
    from . import fusion as FU
    from . import solve as SV
    from . import texture as TX
    from ..io import gltf as GL
    from ..io.export import write_png_any
    from ..stack.foliage import bleed
    from ... import __version__
    p = p or ModelParams()
    makes = tuple(m for m in MAKES if m in makes)

    def prog(f, msg):
        if progress:
            progress(f, msg)

    def check():
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
    sol = solution or SV.solve(views, use_depth=p.use_depth, affine=p.affine,
                               progress=(lambda f, m: prog(0.45 * f, m)), cancel=cancel, n_features=p.features)
    check()
    groups = {}
    for i, g in enumerate(sol.group):
        if g >= 0 and sol.placed[i]:
            groups.setdefault(g, []).append(i)
    if not groups or max(len(v) for v in groups.values()) < 2:
        raise ValueError("no model: " + SV.why_nothing_placed(sol))
    gid = group if (group is not None and group in groups) else max(groups, key=lambda g: len(groups[g]))
    members = set(groups[gid])
    notes = list(sol.notes)
    if len(groups) > 1:
        notes.append(f"The views make {len(groups)} separate objects: this model is the largest "
                     f"({len(members)} views); run the others on their own.")
    sub = SV.Solution(names=sol.names, placed=[sol.placed[i] and i in members for i in range(len(sol.names))],
                      R=sol.R, t=sol.t, f=sol.f, m=sol.m, kappa=sol.kappa, how=sol.how, reason=sol.reason,
                      group=sol.group, k1=sol.k1, points=sol.points, point_src=sol.point_src,
                      obs_view=sol.obs_view, obs_point=sol.obs_point, obs_xy=sol.obs_xy, quality=sol.quality,
                      pairs=sol.pairs, use_depth=sol.use_depth, affine=sol.affine)
    # dense depth two ways, per placed view
    dense = {}
    order = sorted(members)
    for k, i in enumerate(order):
        check()
        prog(0.45 + 0.15 * k / len(order), f"measuring the depth of {views[i].name} two ways")
        dense[i] = DN.view_depth(sub, views, i)
    vol = FU.integrate(sub, views, dense, max_voxels=int(p.voxels) ** 3,
                       progress=lambda f, m: prog(0.6 + 0.1 * f, m))
    check()
    prog(0.7, "building the surface")
    V, T, meas = FU.mesh(vol)
    up = up_vector(sub)
    cut = False
    if p.ground_cut:
        V, T, cut = _ground_cut(V, T, up)
        if cut:
            meas = None
    if meas is None or len(meas) != len(V):
        # measured share per vertex again after a cut: from the volume's weight at each vertex
        idx = np.clip(np.rint((V - vol.origin) / vol.voxel).astype(int), 0, np.array(vol.weight.shape) - 1)
        meas = (vol.weight[idx[:, 0], idx[:, 1], idx[:, 2]] > 1e-3).astype(np.float32)
    check()
    prog(0.75, "texturing")
    tex = TX.build(V, T, meas, sub, views, max_size=int(p.atlas))
    S = tex.atlas.shape[0]
    # scale and the model frame: Y up, origin at the bottom centre
    mpu, scale_how = scale_from_views(sub, views)
    k = mpu if mpu else 1.0
    units = "metres" if mpu else "model units"
    yv = up
    xv = np.cross(yv, [0, 0, 1.0]) if abs(yv[2]) < 0.9 else np.cross(yv, [1.0, 0, 0])
    xv /= np.linalg.norm(xv)
    zv = np.cross(xv, yv)
    Rm = np.stack([xv, yv, zv])                               # world -> model axes
    h = tex.V @ yv
    origin = tex.V.mean(0) - yv * (tex.V.mean(0) @ yv - h.min())
    if p.known_length_m and p.known_length_m > 0:     # the photographer's own measure of the longest side
        ext = float(np.max(np.ptp((tex.V - origin) @ Rm.T, axis=0)))
        if ext > 0:
            mpu = float(p.known_length_m) / ext
            k = mpu
            units = "metres"
            scale_how = (f"given: the object's longest side is {p.known_length_m:g} m (as you measured it); the "
                         "rest of the model follows from the photos")

    def to_model(X):
        return (np.asarray(X) - origin) @ Rm.T * k
    prog(0.8, "game sizes")
    V0, T0, uv0 = DC.simplify(tex.V, tex.T, int(p.triangles), tex.vchart, tex.uv)
    nrm0 = DC.bake_normals(tex.V, tex.T, tex.uv, V0, T0, uv0, S)
    covered = np.any(nrm0 != np.array([128, 128, 255], np.uint8), axis=2) | np.any(tex.made > 0, axis=2)
    nrm0 = np.clip(np.rint(bleed(nrm0.astype(np.float32), covered.astype(np.float32))), 0, 255).astype(np.uint8)
    lods = [(V0, T0, uv0, nrm0)]
    if p.lods:
        for div in (4, 15):
            Vl, Tl, uvl = DC.simplify(tex.V, tex.T, max(200, int(p.triangles) // div), tex.vchart, tex.uv)
            nl = DC.bake_normals(tex.V, tex.T, tex.uv, Vl, Tl, uvl, S)
            nl = np.clip(np.rint(bleed(nl.astype(np.float32), covered.astype(np.float32))), 0, 255).astype(np.uint8)
            lods.append((Vl, Tl, uvl, nl))
    check()
    # quality and the gate
    q_views = [x for i, x in enumerate(sub.quality) if x and i in members]
    face_area = np.linalg.norm(np.cross(tex.V[tex.T[:, 1]] - tex.V[tex.T[:, 0]],
                                        tex.V[tex.T[:, 2]] - tex.V[tex.T[:, 0]]), axis=1)
    measured_share = float(face_area[tex.measured_face].sum() / max(face_area.sum(), 1e-12))
    dsh = [dv.shares() for dv in dense.values()]
    tot_meas = sum(s["measured_both"] + s["measured_one"] + s["disagree"] for s in dsh) or 1e-12
    quality = {
        "views_given": len(views), "views_placed": len(members),
        "reprojection_px": round(float(np.mean([x["reprojection_px"] for x in q_views if x["reprojection_px"]
                                                is not None])), 3) if q_views else None,
        "depth_agreement": round(float(np.mean([x["depth_rms"] for x in q_views if x["depth_rms"] is not None])), 4)
        if any(x["depth_rms"] is not None for x in q_views) else None,
        "measured_share": round(measured_share, 4), "filled_share": round(1 - measured_share, 4),
        "disagree_share": round(sum(s["disagree"] for s in dsh) / tot_meas, 4),
        "triangles_full": int(len(tex.T)), "triangles_game": int(len(T0)), "charts": int(tex.charts),
        "voxel": round(float(vol.voxel * k), 6)}
    gate_res = DS.gate(quality)
    coach = CO.advise(sub, tex.V, tex.T, tex.measured_face, up, mpu is not None)
    # files
    os.makedirs(out_dir, exist_ok=True)
    paths = targets(out_dir, base, taken)
    written = {}
    mat = GL.Material(name=f"{base}_scan", base_colour=tex.atlas, normal=lods[0][3], roughness=0.85)
    extras = {"studio_model": MODEL_VERSION, "units": units, "measured_share": quality["measured_share"]}

    def glmesh(Vx, Tx, uvx, name):
        Vm = to_model(Vx).astype(np.float32)
        N = GL.vertex_normals(Vm, Tx)
        return GL.Mesh(name=name, positions=Vm, indices=Tx, normals=N, uvs=uvx.astype(np.float32), material=0)
    if "glb" in makes or "dataset" in makes:
        meshes = [glmesh(V0, T0, uv0, base)]
        mats = [mat]
        for li, (Vl, Tl, uvl, nl) in enumerate(lods[1:], 1):
            mats.append(GL.Material(name=f"{base}_scan_LOD{li}", base_colour=tex.atlas, normal=nl, roughness=0.85))
            m_ = glmesh(Vl, Tl, uvl, f"{base}_LOD{li}")
            m_.material = li
            meshes.append(m_)
        GL.write_glb(paths["glb"], meshes, mats, extras=extras)
        written["glb"] = paths["glb"]
    if "mesh" in makes:
        GL.write_glb(paths["mesh"], [glmesh(tex.V, tex.T, tex.uv, f"{base}_full")],
                     [GL.Material(name=f"{base}_scan_full", base_colour=tex.atlas, roughness=0.85)], extras=extras)
        written["mesh"] = paths["mesh"]
    write_png_any(paths["made"], tex.made)
    written["made"] = paths["made"]
    if "obj" in makes:
        write_png_any(paths["colour"], tex.atlas)
        write_png_any(paths["normal"], lods[0][3])
        written["colour"], written["normal"] = paths["colour"], paths["normal"]
        for q in GL.write_obj(paths["obj"], glmesh(V0, T0, uv0, base),
                              {"colour": os.path.basename(paths["colour"]),
                               "normal": os.path.basename(paths["normal"])}):
            written.setdefault("obj", q)
    pts_m = to_model(sub.points) if len(sub.points) else np.zeros((0, 3))
    pcol = _point_colours(sub, views)
    phow = np.where(sub.point_src == 1, 2, 1).astype(np.uint8) if len(sub.points) else np.zeros(0, np.uint8)
    if "points" in makes and len(pts_m):
        GL.write_ply(paths["points"], pts_m, pcol, phow)
        written["points"] = paths["points"]
    cams = []
    for i, v in enumerate(views):
        e = {"name": v.name, "path": v.path, "placed": bool(i in members), "how": sub.how[i] if i in members
             else None, "colour": SV.HOW_COLOUR.get(sub.how[i] if i in members else None),
             "how_text": SV.HOW_TEXT.get(sub.how[i], "") if i in members else "",
             "reason": "" if i in members else (sol.reason[i] or "placed with another object"),
             "quality": sub.quality[i] if i in members else None, "size": [int(v.shape[1]), int(v.shape[0])]}
        if i in members:
            Rmv = sub.R[i] @ Rm.T
            tmv = k * (sub.R[i] @ origin + sub.t[i])
            e.update(R=[[round(float(x), 9) for x in r] for r in Rmv], t=[round(float(x), 9) for x in tmv],
                     centre=[round(float(x), 6) for x in (-Rmv.T @ tmv)], f_px=round(float(sub.f[i]), 3),
                     cx=round(float(v.cx), 3), cy=round(float(v.cy), 3), focal_from=v.f_from)
        cams.append(e)
    cam_doc = {"studio_cameras": 1, "written_by": f"Photographic Studio By Bill {__version__}",
               "axes": "model frame: glTF, Y up; R, t world-to-camera with x right, y down, z forward",
               "units": units, "k1": round(float(sub.k1), 6), "views": cams}
    if "cameras" in makes:
        _json(paths["cameras"], cam_doc)
        written["cameras"] = paths["cameras"]
    dataset_doc = None
    if "dataset" in makes:
        if gate_res[0] or not p.gate:
            dcams = []
            for i in order:
                v = views[i]
                e = next(c for c in cams if c["name"] == v.name)
                col = v.colour if v.colour is not None else np.dstack([np.clip(v.image * 255, 0, 255)] * 3).astype(
                    np.uint8)
                cs = col.shape[0] / float(v.shape[0])
                dv = dense[i]
                dcams.append({"name": v.name, "image": col, "depth": dv.z * k, "confidence": dv.conf,
                              "agreement": dv.label, "mask": v.mask, "R": np.array(e["R"]), "t": np.array(e["t"]),
                              "f": sub.f[i] * cs, "cx": (v.cx + 0.5) * cs - 0.5, "cy": (v.cy + 0.5) * cs - 0.5,
                              "how": e["how"], "quality": e["quality"]})
            dataset_doc = DS.write(paths["dataset"], dcams, pts_m, pcol, phow, written.get("glb"), units, mpu,
                                   quality, gate_res, {"coach": coach["lines"]})
            written["dataset"] = paths["dataset"]
        else:
            notes.append("The Blender dataset was not written: " + "; ".join(gate_res[1]) + ".")
    doc = {
        "studio_model": MODEL_VERSION,
        "written_by": f"Photographic Studio By Bill {__version__}",
        "files": {kk: os.path.basename(vv) for kk, vv in written.items()},
        "units": units,
        "scale": {"metres_per_unit": round(mpu, 6) if mpu else None, "how": scale_how,
                  "colour": "amber" if mpu else "none",
                  "about": "the shape, texture and cameras do not depend on the scale; only the size in metres does "
                           "(set it in Blender or Godot, or give the object's length with --length-m)"},
        "views": {"given": len(views), "placed": len(members),
                  "not_placed": [{"view": c["name"], "reason": c["reason"]} for c in cams if not c["placed"]],
                  "how": {c["name"]: c["how"] for c in cams if c["placed"]}},
        "quality": quality,
        "quality_about": "reprojection_px: how far the placed points land from where the views saw them; "
                         "depth_agreement: the points' distance against each view's own depth (share); "
                         "measured_share: surface measured (cyan) against filled from the outline (violet); "
                         "disagree_share: where the two depth measures disagree (red, kept out of the mesh)",
        "gate": {"passed": bool(gate_res[0]), "reasons": gate_res[1], "limits": DS.GATE,
                 "about": "the Blender dataset is written only when the scan passes"},
        "coach": coach,
        "ground_cut": bool(cut),
        "lods": [{"name": base if li == 0 else f"{base}_LOD{li}", "triangles": int(len(l[1]))}
                 for li, l in enumerate(lods)],
        "godot": {"import": "the .glb directly (Godot 4 reads its materials, normal map and LOD meshes)",
                  "normal_map": "OpenGL (no flip)", "scale_metres": round(mpu, 6) if mpu else None,
                  "collision": "use the LOD2 mesh (or LOD0) as a trimesh collider, or Godot's convex shape",
                  "up": "+Y"},
        "made_png": "cyan: measured; violet: filled from the outline (no view measured it)",
        "settings": p.to_dict(),
        "notes": notes,
    }
    _json(paths["json"], doc)
    written["json"] = paths["json"]
    prog(1.0, "model written")
    log.info("model %s: %d/%d views, %d triangles (game %d), measured %.0f%%, gate %s", base, len(members),
             len(views), len(tex.T), len(T0), 100 * measured_share, gate_res[0])
    return {"paths": written, "json": doc, "solution": sol, "gate": gate_res, "dataset": dataset_doc,
            "mesh": (to_model(tex.V), tex.T, tex.measured_face), "mesh_world": (tex.V, tex.T, tex.measured_face),
            "coach": coach}


def _point_colours(sol, views):
    if not len(sol.points):
        return np.zeros((0, 3), np.uint8)
    from scipy.ndimage import map_coordinates
    col = np.zeros((len(sol.points), 3))
    cnt = np.zeros(len(sol.points))
    for k in range(len(sol.obs_view)):
        pass
    for v in np.unique(sol.obs_view):
        s = sol.obs_view == v
        vw = views[v]
        img = vw.colour if vw.colour is not None else np.dstack([vw.image * 255] * 3)
        cs = img.shape[0] / float(vw.shape[0])
        xy = (sol.obs_xy[s] + 0.5) * cs - 0.5
        c = np.stack([map_coordinates(img[..., ch].astype(np.float32), [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
                      for ch in range(3)], 1)
        np.add.at(col, sol.obs_point[s], c)
        np.add.at(cnt, sol.obs_point[s], 1)
    out = col / np.maximum(cnt, 1)[:, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _json(path, doc):
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_plain(doc), fh, indent=1)
    os.replace(tmp, path)


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v

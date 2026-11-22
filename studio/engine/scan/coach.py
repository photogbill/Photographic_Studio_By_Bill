"""What to shoot next (BUILD_PLAN_DP3D H.4, E.10): from where the cameras stood and which parts of the surface were
measured, plain sentences a photographer can act on at the next shoot.

The object's surface is divided by the direction it faces (twelve 30-degree sectors round the up axis, plus up and
down); a sector whose faces are mostly filled (violet: no view measured them) is where to shoot. Hand-held is assumed
(Bill, 2026-11-21: "95% of the time I'll be hand-holding the camera"): advice is in steps and overlaps, never in
tripod positions.
"""
from __future__ import annotations

import numpy as np

SECTOR_DEG = 30


def frame(up: np.ndarray):
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    h1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    h1 /= np.linalg.norm(h1)
    return up, h1, np.cross(up, h1)


def advise(sol, V, T, measured_face, up, scale_known: bool) -> dict:
    up, h1, h2 = frame(up)
    centre = V.mean(0)
    cams = [i for i in range(len(sol.names)) if sol.placed[i]]
    az, el = [], []
    for i in cams:
        d = sol.centre(i) - centre
        az.append(float(np.degrees(np.arctan2(d @ h2, d @ h1)) % 360))
        el.append(float(np.degrees(np.arcsin(np.clip(d @ up / max(np.linalg.norm(d), 1e-12), -1, 1)))))
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(area, 1e-12)[:, None]
    vert = fn @ up
    faz = np.degrees(np.arctan2(fn @ h2, fn @ h1)) % 360
    sectors = []
    side = np.abs(vert) < 0.7
    for s in range(0, 360, SECTOR_DEG):
        m = side & (faz >= s) & (faz < s + SECTOR_DEG)
        a = area[m].sum()
        sectors.append({"from_deg": s, "to_deg": s + SECTOR_DEG, "share_of_surface": round(float(a / area.sum()), 4),
                        "measured": round(float(area[m & measured_face].sum() / a), 3) if a > 0 else None})
    top = vert >= 0.7
    bottom = vert <= -0.7
    meas_top = float(area[top & measured_face].sum() / max(area[top].sum(), 1e-12)) if top.any() else None
    meas_bot = float(area[bottom & measured_face].sum() / max(area[bottom].sum(), 1e-12)) if bottom.any() else None
    lines = []
    total = float(area[measured_face].sum() / area.sum())
    lines.append(f"{100 * total:.0f}% of the surface was measured (cyan); the rest is filled in smoothly from the "
                 "outline (violet).")
    # the unmeasured arc round the object
    weak = [s for s in sectors if s["measured"] is not None and s["measured"] < 0.4 and s["share_of_surface"] > 0.01]
    if weak:
        runs = []
        for s in weak:
            if runs and runs[-1][1] == s["from_deg"]:
                runs[-1][1] = s["to_deg"]
            else:
                runs.append([s["from_deg"], s["to_deg"]])
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 360:
            runs[0][0] = runs[-1][0] - 360
            runs.pop()
        for a0, a1 in runs:
            span = a1 - a0
            n = max(2, int(np.ceil(span / 30.0)) + 1)
            lines.append(f"The side facing {a0 % 360:.0f} to {a1 % 360:.0f} degrees was never measured: shoot about {n} "
                         f"more stacks round it, each overlapping its neighbour by about half (a step of 20-30 "
                         "degrees as you walk round).")
    else:
        lines.append("All the way round was measured.")
    if el:
        lo, hi = min(el), max(el)
        if meas_bot is not None and meas_bot < 0.4:
            lines.append(f"Every view was taken from {lo:.0f} to {hi:.0f} degrees above the object: the underside was "
                         "not seen. If the object can be turned over or raised, a few stacks from below finish it.")
        if meas_top is not None and meas_top < 0.4 and hi < 50:
            lines.append("The top was barely seen: add two or three stacks looking down on it.")
    if len(cams) < len(sol.names):
        miss = [sol.names[i] for i in range(len(sol.names)) if not sol.placed[i]]
        lines.append(f"{len(miss)} view(s) could not be placed ({', '.join(miss[:6])}{'...' if len(miss) > 6 else ''}): "
                     "neighbouring stacks should share about half of what they see.")
    if not scale_known:
        lines.append("No size was known, so the model is in relative units: its shape, texture and cameras are the "
                     "same either way; only its size in metres is unknown. Scale it once in Blender or Godot, or give "
                     "the object's length (a lens that records its focus distance does this by itself).")
    return {"lines": lines, "camera_azimuth_deg": [round(a, 1) for a in az],
            "camera_elevation_deg": [round(e, 1) for e in el], "sectors": sectors,
            "top_measured": None if meas_top is None else round(meas_top, 3),
            "bottom_measured": None if meas_bot is None else round(meas_bot, 3), "measured_share": round(total, 4)}

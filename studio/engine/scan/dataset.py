"""A Blender dataset of a scan (Bill, 2026-11-21: "process raw Dual Pixel files directly into Blender datasets, but
only if they are of high enough quality. I can then later create my own Blender add-on for said data").

One folder, ``NAME_dataset_Studio/``, readable by any Blender add-on with the standard library and NumPy (both ship
with Blender):

    dataset.json            what everything is (format "studio_blender_dataset", version 1): units, axes, every
                            camera (Blender's and OpenCV's conventions), the files, the quality numbers, the gate
    images/VIEW.png         the all-in-focus picture of each placed view (8-bit sRGB)
    depth/VIEW.npy          float32 distance along the camera's view axis, dataset units, NaN where nothing was
                            measured (a smaller size than the picture: the size is in dataset.json)
    depth/VIEW_preview.png  the same as 16-bit grey, near = white (for a quick look)
    confidence/VIEW.png     8-bit: 255 sure, 0 nothing measured
    agreement/VIEW.png      how each depth was made: cyan both ways, amber one way, red the two disagree, black none
    masks/VIEW.png          the subject (255) found by its measured depth
    points.ply              the solver's points (colour; "how": 2 triangulated, 1 from one view's depth)
    model.glb               the textured model (the same as NAME_model_Studio.glb)

Axes: Blender's, Z up, right-handed; units metres when the scale is known (``units`` says), else model units. A
camera's ``matrix_world`` is Blender's camera-to-world matrix (the camera looks down its local -Z, +Y up); ``opencv``
gives the same camera as world-to-camera R, t with x right, y down, z forward, and K in pixels of its picture.

**The quality gate**: the dataset is written only when the scan is good enough (``GATE``); otherwise nothing is
written and the reasons are given in plain words.
"""
from __future__ import annotations

import json
import os

import numpy as np

FORMAT = "studio_blender_dataset"
VERSION = 1
GATE = {"min_views": 3, "min_placed_share": 0.6, "max_reprojection_px": 1.5, "min_measured_share": 0.35,
        "max_disagree_share": 0.25}
# model frame (glTF: Y up) -> Blender (Z up): (x, y, z) -> (x, -z, y)
GLTF_TO_BLENDER = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
CV_TO_BLENDER_CAM = np.diag([1.0, -1.0, -1.0])          # camera axes: OpenCV (x right, y down, z ahead) -> Blender


def gate(quality: dict, limits: dict = GATE) -> tuple[bool, list]:
    """(passed, reasons in plain words)."""
    reasons = []
    given, placed = quality.get("views_given", 0), quality.get("views_placed", 0)
    if placed < limits["min_views"]:
        reasons.append(f"only {placed} view(s) could be placed; a dataset needs at least {limits['min_views']}")
    if given and placed / given < limits["min_placed_share"]:
        reasons.append(f"{given - placed} of {given} views could not be placed (the others share too little with "
                       "them)")
    rp = quality.get("reprojection_px")
    if rp is None or rp > limits["max_reprojection_px"]:
        reasons.append("the cameras do not agree closely enough with what they saw "
                       f"({'unknown' if rp is None else f'{rp:.2f} px'}; at most {limits['max_reprojection_px']} px)")
    ms = quality.get("measured_share")
    if ms is None or ms < limits["min_measured_share"]:
        reasons.append(f"only {100 * (ms or 0):.0f}% of the surface was measured; at least "
                       f"{100 * limits['min_measured_share']:.0f}% is needed (the rest would be guessed)")
    ds = quality.get("disagree_share")
    if ds is not None and ds > limits["max_disagree_share"]:
        reasons.append(f"the two depth measures disagree on {100 * ds:.0f}% of what was measured (something moved, "
                       "or the surface is shiny)")
    return not reasons, reasons


def blender_camera(Rm: np.ndarray, tm: np.ndarray, f: float, cx: float, cy: float, w: int, h: int,
                   sensor_mm: float = 36.0) -> dict:
    """A camera in the model frame (world-to-camera Rm, tm, OpenCV axes) as Blender wants it."""
    C = -Rm.T @ tm
    cam_to_world_cv = Rm.T                                   # columns: camera x, y, z in the model frame
    Mb = GLTF_TO_BLENDER @ cam_to_world_cv @ CV_TO_BLENDER_CAM
    Cb = GLTF_TO_BLENDER @ C
    mw = np.eye(4)
    mw[:3, :3] = Mb
    mw[:3, 3] = Cb
    big = max(w, h)
    return {"matrix_world": [[round(float(x), 9) for x in row] for row in mw],
            "location": [round(float(x), 9) for x in Cb],
            "lens_mm": round(float(f / w * sensor_mm), 6), "sensor_width_mm": sensor_mm, "sensor_fit": "HORIZONTAL",
            "shift_x": round(float(-(cx - (w - 1) / 2.0) / big), 6),
            "shift_y": round(float((cy - (h - 1) / 2.0) / big), 6),
            "resolution": [int(w), int(h)]}


def write(folder: str, cameras: list, points=None, point_colours=None, point_how=None, glb_path: str | None = None,
          units: str = "model units", metres_per_unit: float | None = None, quality: dict | None = None,
          gate_result=None, extras: dict | None = None) -> dict:
    """Write the dataset folder. ``cameras``: dicts with name, image (uint8 RGB), depth (float32, dataset units,
    NaN none), confidence (0..1), agreement (uint8 label), mask (bool), R, t (model frame, OpenCV axes, dataset
    units), f, cx, cy (pixels of ``image``), how, quality. Returns the doc written."""
    from ... import __version__ as ver
    from ..io.export import write_png_any
    from ..io import gltf as GL
    os.makedirs(folder, exist_ok=True)
    for sub in ("images", "depth", "confidence", "agreement", "masks"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    cams = []
    lab_rgb = {0: (0, 0, 0), 1: (0xF5, 0x9E, 0x0B), 2: (0x22, 0xD3, 0xEE), 3: (0xF8, 0x71, 0x71)}
    for c in cameras:
        nm = c["name"]
        img = c["image"]
        h, w = img.shape[:2]
        write_png_any(os.path.join(folder, "images", nm + ".png"), img)
        files = {"image": f"images/{nm}.png"}
        if c.get("depth") is not None:
            d = np.asarray(c["depth"], np.float32)
            np.save(os.path.join(folder, "depth", nm + ".npy"), d)
            fin = np.isfinite(d)
            prev = np.zeros(d.shape, np.uint16)
            if fin.any():
                lo, hi = np.percentile(d[fin], [1, 99])
                prev[fin] = np.rint(np.clip((hi - d[fin]) / max(hi - lo, 1e-12), 0, 1) * 65535)
            write_png_any(os.path.join(folder, "depth", nm + "_preview.png"), prev)
            files["depth"] = f"depth/{nm}.npy"
            files["depth_preview"] = f"depth/{nm}_preview.png"
            files["depth_size"] = [int(d.shape[1]), int(d.shape[0])]
        if c.get("confidence") is not None:
            write_png_any(os.path.join(folder, "confidence", nm + ".png"),
                          np.rint(np.clip(c["confidence"], 0, 1) * 255).astype(np.uint8))
            files["confidence"] = f"confidence/{nm}.png"
        if c.get("agreement") is not None:
            lab = np.asarray(c["agreement"])
            rgb = np.zeros(lab.shape + (3,), np.uint8)
            for k, col in lab_rgb.items():
                rgb[lab == k] = col
            write_png_any(os.path.join(folder, "agreement", nm + ".png"), rgb)
            files["agreement"] = f"agreement/{nm}.png"
        if c.get("mask") is not None:
            write_png_any(os.path.join(folder, "masks", nm + ".png"), (np.asarray(c["mask"]) * 255).astype(np.uint8))
            files["mask"] = f"masks/{nm}.png"
        R, t = np.asarray(c["R"]), np.asarray(c["t"])
        K = [[c["f"], 0, c["cx"]], [0, c["f"], c["cy"]], [0, 0, 1]]
        # OpenCV numbers in Blender's world axes (the dataset's world is Blender's)
        Rb = R @ GLTF_TO_BLENDER.T
        cams.append({"name": nm, "how": c.get("how"), "quality": c.get("quality"), "files": files,
                     "blender": blender_camera(R, t, c["f"], c["cx"], c["cy"], w, h),
                     "opencv": {"R": [[round(float(x), 9) for x in r] for r in Rb],
                                "t": [round(float(x), 9) for x in t], "K": K, "size": [int(w), int(h)]}})
    files = {}
    if points is not None and len(points):
        Pb = np.asarray(points) @ GLTF_TO_BLENDER.T
        GL.write_ply(os.path.join(folder, "points.ply"), Pb, point_colours, point_how)
        files["points"] = "points.ply"
    if glb_path and os.path.exists(glb_path):
        import shutil
        shutil.copyfile(glb_path, os.path.join(folder, "model.glb"))
        files["model"] = "model.glb"
        files["model_axes"] = "glTF (Y up): Blender's glTF importer turns it Z up, matching the cameras"
    doc = {"format": FORMAT, "version": VERSION, "written_by": f"Photographic Studio By Bill {ver}",
           "units": units, "metres_per_unit": metres_per_unit,
           "axes": "Blender: right-handed, Z up; cameras look down their local -Z with +Y up (matrix_world)",
           "opencv_axes": "world-to-camera R, t in the same world; camera x right, y down, z forward; K in pixels",
           "depth": "float32 distance along each camera's view axis, dataset units, NaN = not measured; "
                    "agreement colours: cyan both ways, amber one way, red disagree, black none",
           "cameras": cams, "files": files, "quality": quality or {},
           "gate": {"passed": True if gate_result is None else bool(gate_result[0]),
                    "reasons": [] if gate_result is None else list(gate_result[1]), "limits": GATE},
           "extras": extras or {}}
    tmp = os.path.join(folder, "dataset.json.partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, os.path.join(folder, "dataset.json"))
    return doc

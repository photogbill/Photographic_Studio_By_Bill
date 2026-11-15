"""The engine API (section 2.2): a small set of calls the GUI, the CLI, the Photoshop bridge, ATK and the
tests all use.

    img = open(path)                         -> Image
    arr = render(img, edit, region, scale)   -> display-referred linear ProPhoto float32 (upright)
    out = export(img, edit, spec, out_path)  -> path of the written file

Sidecars (section 5): ``<name>.<ext>.psb.json`` beside the raw, plain JSON. ``load_edit`` returns the
sidecar's edit if one exists, else defaults. ``save_edit`` writes it (atomically).
"""
from __future__ import annotations

import json
import os

from .. import __version__, log as slog
from .io.export import EXT, ExportSpec, write
from .pipeline.edit import Edit
from .pipeline.pipeline import Image, open_image, output_size, render, render_view, resize  # noqa: F401 (API)

log = slog.get("engine.api")

SIDECAR_SUFFIX = ".psb.json"


def open(path: str) -> Image:          # noqa: A001 - the plan names this call open()
    return open_image(path)


def sidecar_path(raw_path: str, central_dir: str | None = None) -> str:
    name = os.path.basename(raw_path) + SIDECAR_SUFFIX
    return os.path.join(central_dir, name) if central_dir else raw_path + SIDECAR_SUFFIX


def load_edit(raw_path: str, central_dir: str | None = None) -> Edit:
    p = sidecar_path(raw_path, central_dir)
    if os.path.isfile(p):
        try:
            with open_text(p) as fh:
                d = json.load(fh)
            return Edit.from_dict(d.get("edit", {}))
        except Exception:
            slog.log_exception(f"sidecar {p} unreadable; using defaults", log)
    return Edit()


def load_doc(raw_path: str, central_dir: str | None = None) -> dict:
    p = sidecar_path(raw_path, central_dir)
    if os.path.isfile(p):
        try:
            with open_text(p) as fh:
                return json.load(fh)
        except Exception:
            slog.log_exception(f"sidecar {p} unreadable", log)
    return {}


def load_library_meta(raw_path: str, central_dir: str | None = None) -> dict:
    """Rating (0-5), pick (1 / 0 / -1 reject) and colour label kept in the sidecar."""
    m = load_doc(raw_path, central_dir).get("library", {})
    return {"rating": int(m.get("rating", 0)), "pick": int(m.get("pick", 0)), "label": m.get("label", "")}


def save_library_meta(raw_path: str, meta: dict, central_dir: str | None = None) -> str:
    doc = load_doc(raw_path, central_dir)
    edit = Edit.from_dict(doc.get("edit", {})) if doc.get("edit") else Edit()
    return save_edit(raw_path, edit, central_dir, library=meta)


def save_edit(raw_path: str, edit: Edit, central_dir: str | None = None, library: dict | None = None) -> str:
    """Write the master edit. Everything else already in the sidecar (ratings, virtual copies, snapshots,
    history) is kept."""
    doc = load_doc(raw_path, central_dir)
    doc.update({"app": "Photographic Studio By Bill", "app_version": __version__, "format": 1,
                "edit": edit.to_dict()})
    if library is not None:
        doc["library"] = library
    return write_doc(raw_path, doc, central_dir)


def write_doc(raw_path: str, doc: dict, central_dir: str | None = None) -> str:
    p = sidecar_path(raw_path, central_dir)
    doc.setdefault("app", "Photographic Studio By Bill")
    doc["app_version"] = __version__
    doc.setdefault("format", 1)
    tmp = p + ".partial"
    with open_text(tmp, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp, p)
    return p


# ------------------------------------------------------------------------------------------ virtual copies
def list_versions(raw_path: str, central_dir: str | None = None) -> list:
    """Names of the edits kept for one raw: index 0 is the master, then the virtual copies."""
    doc = load_doc(raw_path, central_dir)
    return ["Master"] + [v.get("name") or f"Copy {i + 1}" for i, v in enumerate(doc.get("versions", []))]


def load_version(raw_path: str, index: int, central_dir: str | None = None) -> Edit:
    if index == 0:
        return load_edit(raw_path, central_dir)
    v = load_doc(raw_path, central_dir).get("versions", [])
    return Edit.from_dict(v[index - 1].get("edit", {}))


def save_version(raw_path: str, index: int, edit: Edit, central_dir: str | None = None) -> str:
    if index == 0:
        return save_edit(raw_path, edit, central_dir)
    doc = load_doc(raw_path, central_dir)
    doc["versions"][index - 1]["edit"] = edit.to_dict()
    return write_doc(raw_path, doc, central_dir)


def add_version(raw_path: str, edit: Edit, name: str | None = None, central_dir: str | None = None) -> int:
    """New virtual copy (starting from ``edit``). Returns its index."""
    doc = load_doc(raw_path, central_dir)
    if "edit" not in doc:
        doc["edit"] = Edit().to_dict()
    vs = doc.setdefault("versions", [])
    vs.append({"name": name or f"Copy {len(vs) + 1}", "edit": edit.to_dict()})
    write_doc(raw_path, doc, central_dir)
    return len(vs)


def delete_version(raw_path: str, index: int, central_dir: str | None = None) -> None:
    if index <= 0:
        raise ValueError("the master edit cannot be deleted (use Reset all instead)")
    doc = load_doc(raw_path, central_dir)
    doc.get("versions", []).pop(index - 1)
    write_doc(raw_path, doc, central_dir)


# ------------------------------------------------------------------------------------------ snapshots, history
def list_snapshots(raw_path: str, central_dir: str | None = None) -> list:
    return load_doc(raw_path, central_dir).get("snapshots", [])


def add_snapshot(raw_path: str, name: str, edit: Edit, central_dir: str | None = None) -> None:
    import datetime as _dt
    doc = load_doc(raw_path, central_dir)
    doc.setdefault("snapshots", []).append({"name": name, "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                            "edit": edit.to_dict()})
    write_doc(raw_path, doc, central_dir)


def delete_snapshot(raw_path: str, index: int, central_dir: str | None = None) -> None:
    doc = load_doc(raw_path, central_dir)
    doc.get("snapshots", []).pop(index)
    write_doc(raw_path, doc, central_dir)


HISTORY_KEEP = 40


def save_history(raw_path: str, version: int, steps: list, central_dir: str | None = None) -> None:
    """Undo history for one version: [(label, Edit), ...], oldest first, the last HISTORY_KEEP kept."""
    doc = load_doc(raw_path, central_dir)
    doc["history"] = {"version": int(version),
                      "steps": [{"label": lb, "edit": e.to_dict()} for lb, e in steps[-HISTORY_KEEP:]]}
    write_doc(raw_path, doc, central_dir)


def load_history(raw_path: str, version: int, central_dir: str | None = None) -> list:
    h = load_doc(raw_path, central_dir).get("history") or {}
    if int(h.get("version", -1)) != int(version):
        return []
    out = []
    for st in h.get("steps", []):
        try:
            out.append((st.get("label", ""), Edit.from_dict(st.get("edit", {}))))
        except Exception:
            continue
    return out


def open_text(path, mode="r"):
    import builtins
    return builtins.open(path, mode, encoding="utf-8")


def export(image: Image, edit: Edit | None, spec: ExportSpec, out_path: str | None = None,
           out_dir: str | None = None) -> str:
    edit = edit or Edit()
    spec.validate()
    if out_path is None:
        base = os.path.splitext(os.path.basename(image.path))[0] + spec.suffix + EXT[spec.format]
        out_path = os.path.join(out_dir or os.path.dirname(image.path), base)
    arr = render(image, edit)
    if spec.long_edge:
        h, w = arr.shape[:2]
        k = spec.long_edge / max(h, w)
        if k < 1:
            arr = resize(arr, max(1, round(w * k)), max(1, round(h * k)))
    if spec.sharpen_for != "none":
        from .io.export import output_sharpen
        arr = output_sharpen(arr, spec)
    return write(out_path, arr, spec, export_metadata(image, edit))


def export_metadata(image: Image, edit: Edit) -> dict:
    """What goes into an exported file: camera, date, exposure and the lens (the user's tag for contactless
    glass). Never serial numbers or GPS (they are not read in the first place)."""
    from .lens.library import effective_lens
    m = image.meta
    lens, _ = effective_lens(m, edit)
    focal = m.focal_mm
    if edit.lens.tag:
        from .lens.library import get as get_lens
        ent = get_lens(edit.lens.tag)
        if ent and ent.focal_mm and not focal:
            focal = ent.focal_mm
    return {"make": m.make, "model": m.model, "datetime": m.datetime, "lens": lens, "exposure_s": m.exposure_s,
            "fnumber": m.fnumber, "iso": m.iso, "focal_mm": focal}

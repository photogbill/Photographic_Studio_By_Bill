"""The engine API (section 2.2): a small set of calls the GUI, the CLI, the Photoshop bridge, ATK and the
tests all use.

    img = open(path)                         -> Image
    arr = render(img, edit, region, scale)   -> display-referred linear ProPhoto float32 (upright)
    out = export(img, edit, spec, out_path)  -> path of the written file

Sidecars (section 5): ``<name>.<ext>.psb.json`` beside the raw, plain JSON. ``load_edit`` returns the
sidecar's edit if one exists, else defaults. ``save_edit`` writes it (atomically).
"""
from __future__ import annotations

import functools
import json
import os
import threading

from .. import __version__, log as slog
from .io.export import EXT, ExportSpec, write  # noqa: F401 (EXT: API)
from .pipeline.edit import Edit
from .pipeline.pipeline import Image, open_image, output_size, render, render_view, resize  # noqa: F401 (API)

log = slog.get("engine.api")

SIDECAR_SUFFIX = ".psb.json"
# One sidecar write at a time: the Library saves ratings in the background while Develop saves edits, and each is a
# read-modify-write of the same file (without this, one could undo the other).
_SIDECAR_LOCK = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with _SIDECAR_LOCK:
            return fn(*a, **k)
    return wrapper


def open(path: str) -> Image:          # noqa: A001 - the plan names this call open()
    return open_image(path)


def sidecar_path(raw_path: str, central_dir: str | None = None) -> str:
    """Where a photo's sidecar lives: beside the raw, or in the central sidecar folder when the photo's folder was
    switched to it (a read-only drive, plan section 5 / 24.7 C), or in ``central_dir`` when one is given."""
    name = os.path.basename(raw_path) + SIDECAR_SUFFIX
    if central_dir:
        return os.path.join(central_dir, name)
    c = central_folder_for(os.path.dirname(os.path.abspath(raw_path)))
    return os.path.join(c, name) if c else raw_path + SIDECAR_SUFFIX


# ------------------------------------------------------------------------------------------ central sidecars
# Folders whose sidecars are kept on this PC instead of beside the photos (read-only drives, plan section 5).
# settings/sidecar_folders.json maps each such folder to settings/sidecars/<id>/. Private, never shared.
_CENTRAL: dict | None = None


def _central_file() -> str:
    from .. import paths
    return str(paths.get("SETTINGS") / "sidecar_folders.json")


def _central_map() -> dict:
    global _CENTRAL
    if _CENTRAL is None:
        try:
            with open_text(_central_file()) as fh:
                _CENTRAL = {os.path.normcase(os.path.abspath(k)): v for k, v in json.load(fh).items()}
        except (OSError, ValueError):
            _CENTRAL = {}
    return _CENTRAL


def central_folder_for(folder: str) -> str:
    """The central sidecar folder used for photos in ``folder`` ('' when their sidecars sit beside them)."""
    m = _central_map()
    return m.get(os.path.normcase(os.path.abspath(folder)), "") if m else ""


def use_central_sidecars(folder: str, on: bool = True) -> str:
    """Keep the sidecars of photos in ``folder`` on this PC (settings/sidecars/<id>/) from now on, or stop doing so.
    Sidecars already beside the photos are copied in once, so ratings and edits made earlier are not lost.
    Returns the central folder ('' when switched off)."""
    import hashlib
    import shutil
    from .. import paths
    key = os.path.normcase(os.path.abspath(folder))
    m = dict(_central_map())
    if on:
        cid = hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        c = str(paths.get("SETTINGS") / "sidecars" / cid)
        os.makedirs(c, exist_ok=True)
        with open_text(os.path.join(c, "_folder.txt"), "w") as fh:
            fh.write(os.path.abspath(folder) + "\n")
        try:
            for n in os.listdir(folder):
                if n.endswith(SIDECAR_SUFFIX) and not os.path.exists(os.path.join(c, n)):
                    shutil.copy2(os.path.join(folder, n), os.path.join(c, n))
        except OSError:
            pass
        m[key] = c
    else:
        m.pop(key, None)
        c = ""
    global _CENTRAL
    _CENTRAL = m
    tmp = _central_file() + ".partial"
    with open_text(tmp, "w") as fh:
        json.dump({k: v for k, v in m.items()}, fh, indent=2)
    os.replace(tmp, _central_file())
    return c


def reload_central_sidecars() -> None:
    global _CENTRAL
    _CENTRAL = None


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


def edit_is_default(edit_dict: dict | None) -> bool:
    """True when a sidecar's edit is missing or holds only default settings (the photo has no edit yet)."""
    if not edit_dict:
        return True
    try:
        return Edit.from_dict(edit_dict).to_dict() == Edit().to_dict()
    except Exception:                                    # noqa: BLE001  (unreadable: treat as edited, leave it be)
        return False


def has_edit(raw_path: str, central_dir: str | None = None) -> bool:
    """Has this photo been edited (anything but the default settings in its master edit)?"""
    return not edit_is_default(load_doc(raw_path, central_dir).get("edit"))


@_locked
def save_library_meta(raw_path: str, meta: dict, central_dir: str | None = None) -> str:
    doc = load_doc(raw_path, central_dir)
    edit = Edit.from_dict(doc.get("edit", {})) if doc.get("edit") else Edit()
    return save_edit(raw_path, edit, central_dir, library=meta)


@_locked
def save_edit(raw_path: str, edit: Edit, central_dir: str | None = None, library: dict | None = None) -> str:
    """Write the master edit. Everything else already in the sidecar (ratings, virtual copies, snapshots,
    history) is kept."""
    doc = load_doc(raw_path, central_dir)
    doc.update({"app": "Photographic Studio By Bill", "app_version": __version__, "format": 1,
                "edit": edit.to_dict()})
    if library is not None:
        doc["library"] = library
    return write_doc(raw_path, doc, central_dir)


@_locked
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


@_locked
def save_version(raw_path: str, index: int, edit: Edit, central_dir: str | None = None) -> str:
    if index == 0:
        return save_edit(raw_path, edit, central_dir)
    doc = load_doc(raw_path, central_dir)
    doc["versions"][index - 1]["edit"] = edit.to_dict()
    return write_doc(raw_path, doc, central_dir)


@_locked
def add_version(raw_path: str, edit: Edit, name: str | None = None, central_dir: str | None = None) -> int:
    """New virtual copy (starting from ``edit``). Returns its index."""
    doc = load_doc(raw_path, central_dir)
    if "edit" not in doc:
        doc["edit"] = Edit().to_dict()
    vs = doc.setdefault("versions", [])
    vs.append({"name": name or f"Copy {len(vs) + 1}", "edit": edit.to_dict()})
    write_doc(raw_path, doc, central_dir)
    return len(vs)


@_locked
def delete_version(raw_path: str, index: int, central_dir: str | None = None) -> None:
    if index <= 0:
        raise ValueError("the master edit cannot be deleted (use Reset all instead)")
    doc = load_doc(raw_path, central_dir)
    doc.get("versions", []).pop(index - 1)
    write_doc(raw_path, doc, central_dir)


# ------------------------------------------------------------------------------------------ snapshots, history
def list_snapshots(raw_path: str, central_dir: str | None = None) -> list:
    return load_doc(raw_path, central_dir).get("snapshots", [])


@_locked
def add_snapshot(raw_path: str, name: str, edit: Edit, central_dir: str | None = None) -> None:
    import datetime as _dt
    doc = load_doc(raw_path, central_dir)
    doc.setdefault("snapshots", []).append({"name": name, "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                            "edit": edit.to_dict()})
    write_doc(raw_path, doc, central_dir)


@_locked
def delete_snapshot(raw_path: str, index: int, central_dir: str | None = None) -> None:
    doc = load_doc(raw_path, central_dir)
    doc.get("snapshots", []).pop(index)
    write_doc(raw_path, doc, central_dir)


HISTORY_KEEP = 40


@_locked
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
           out_dir: str | None = None, taken: set | None = None) -> str:
    """Render and write one photo; returns the path written. The file is named after the photo with _Studio added
    (io/export.export_name: IMG_0001.CR2 -> IMG_0001_Studio.tif), so it never has the original's name. As a safety
    net it also never replaces the photo itself or a file the Studio did not write (io/export.free_target: written
    as NAME-2.ext instead). Pass the same ``taken`` set for every photo of a batch, so two photos with the same
    name do not replace each other."""
    from .io.export import export_name, free_target
    edit = edit or Edit()
    spec.validate()
    if out_path is None:
        out_path = os.path.join(out_dir or os.path.dirname(image.path), export_name(image.path, spec))
    out_path = free_target(out_path, image.path, taken)
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

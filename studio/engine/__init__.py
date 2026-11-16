"""Photographic Studio By Bill - the engine. No GUI imports, ever (section 2.2).

    from studio import engine
    img = engine.open("IMG_0001.CR3")
    arr = engine.render(img, engine.Edit(exposure_ev=0.3), scale=0.25)
    engine.export(img, engine.Edit(), engine.ExportSpec(format="tiff", space="adobergb"))

Names are loaded lazily, so light tools (the Raw Analyzer reading metadata) can import a sub-module such as
studio.engine.io.metadata without pulling in SciPy and the whole pipeline.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "open": ".api", "render": ".api", "export": ".api", "load_edit": ".api", "save_edit": ".api",
    "sidecar_path": ".api", "load_library_meta": ".api", "save_library_meta": ".api", "load_doc": ".api",
    "render_view": ".api", "output_size": ".api", "export_metadata": ".api",
    "list_versions": ".api", "load_version": ".api", "save_version": ".api", "add_version": ".api",
    "delete_version": ".api", "list_snapshots": ".api", "add_snapshot": ".api", "delete_snapshot": ".api",
    "save_history": ".api", "load_history": ".api", "has_edit": ".api", "edit_is_default": ".api",
    "central_folder_for": ".api", "use_central_sidecars": ".api",
    "make_it_good": ".presets.auto", "make_it_good_settings": ".presets.auto",
    "ExportSpec": ".io.export",
    "RAW_EXTENSIONS": ".io.rawfile", "UnsupportedRaw": ".io.rawfile",
    "Edit": ".pipeline.edit", "Tone": ".pipeline.edit", "WhiteBalance": ".pipeline.edit",
    "Geometry": ".pipeline.edit", "Sharpen": ".pipeline.edit", "Microcontrast": ".pipeline.edit",
    "Light": ".pipeline.edit", "Colour": ".pipeline.edit", "Finish": ".pipeline.edit",
    "Image": ".pipeline.pipeline",
    "view_to_s": ".pipeline.pipeline", "render_mask": ".pipeline.pipeline", "mask_features": ".pipeline.pipeline",
    "object_mask": ".pipeline.pipeline",
    "s_to_view": ".pipeline.pipeline", "view_window": ".pipeline.pipeline", "render_window": ".pipeline.pipeline", "auto_source": ".pipeline.pipeline",
    "find_dust": ".pipeline.pipeline", "dust_from_map": ".pipeline.pipeline", "learn_dust_map": ".pipeline.pipeline",
    "glow_report": ".pipeline.pipeline", "measure_glow_profile": ".pipeline.pipeline",
    "glow_profiles": ".pipeline.pipeline", "pick_glow_profile": ".pipeline.pipeline",
    "find_faces": ".pipeline.pipeline", "face_from_eyes": ".pipeline.pipeline", "face_outlines": ".pipeline.pipeline",
    "find_blemishes": ".pipeline.pipeline", "sculpt_source": ".pipeline.pipeline", "shape_up": ".pipeline.pipeline",
    "find_people": ".pipeline.pipeline", "suggest_crops": ".pipeline.pipeline",
    "add_irises": ".pipeline.pipeline", "find_red_eyes": ".pipeline.pipeline",
    "composition_subject": ".pipeline.pipeline",
    "detail_balance": ".pipeline.pipeline", "hdr_report": ".pipeline.pipeline",
    "AutoHDR": ".pipeline.edit",
    "Detail": ".pipeline.edit",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'studio.engine' has no attribute {name!r}")
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return __all__

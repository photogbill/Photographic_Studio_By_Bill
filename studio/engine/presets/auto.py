""""Make it good": the one-press automatic clean-up (plan section 6.1), in the engine so Develop's button and the
Library's Process picks (section 24.7 C) do exactly the same thing (rule 2.2: no logic in the interface).

What it does: noise reduction set for the photo's ISO, banding and gamut fixes, gentle capture sharpening (not for
character lenses), and the lens fixes (Fix lens). Bill's rule: it never touches brightness.
"""
from __future__ import annotations

import math

from ... import log as slog

log = slog.get("engine.auto")


def make_it_good_settings(meta, edit):
    """The settings part of Make it good (no pixels needed). Returns (new edit, character lens?)."""
    from ..lens.library import effective_lens, is_character
    from ..pipeline.edit import Denoise, Edit
    e = Edit.from_dict(edit.to_dict())
    iso = getattr(meta, "iso", None) or 100
    if iso >= 800:
        e.denoise.enabled = True
        s = min(1.6, max(0.6, 0.6 + 0.25 * math.log2(iso / 800.0)))
        e.denoise.method, e.denoise.luma, e.denoise.chroma = "hybrid", round(Denoise().luma * s, 2), \
            round(Denoise().chroma * s, 2)
    e.banding = True
    e.gamut_compress = True
    lens, _ = effective_lens(meta, e)
    character = is_character(lens)
    if not character:
        e.sharpen.enabled = True
        # when Crisp has already raised the finest scale, sharpen less: noise is not sharpened twice (23.11)
        crisp = 0.0 if "crisp" in (e.detail.off or ()) else max(0.0, e.detail.crisp) / 100.0
        e.sharpen.amount = round(0.4 * (1.0 - 0.5 * crisp), 2)
    return e, character


def make_it_good(image, edit):
    """Make it good on an opened photo: the settings above, then Fix lens. Returns (edit, notes) where notes is a
    short list of plain sentences saying what was done."""
    from ..lens.fix import fix_lens
    e, character = make_it_good_settings(image.meta, edit)
    notes = [f"noise set for {image.meta.iso_label}" if e.denoise.enabled else "low ISO: no noise reduction needed",
             "character lens: sharpening left off" if character else "gentle capture sharpening"]
    try:
        e, rep = fix_lens(image, e)
        notes += [f"{r['what']}: {r['text']}" for r in rep if r.get("tier") in ("cyan", "violet", "amber")][:3]
    except Exception as err:                      # noqa: BLE001  (lens fixes are a bonus; the rest still applies)
        log.warning("make it good: fix lens failed on %s: %s", getattr(image, "path", "?"), err)
        notes.append("lens fixes skipped (could not measure)")
    return e, notes

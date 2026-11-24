"""Tone > Curve colour (tone.apply_curve_rgb natural=..., pipeline.natural_curve)."""
from __future__ import annotations

import numpy as np

from studio.engine.color import tone as T


def _skin(k=1.0):
    from studio.engine.color.spaces import apply_matrix
    r = np.array([[[0.20, 0.11, 0.07]]], np.float32) * k            # warm skin in the render space
    return apply_matrix(r, T.RENDER_TO_PP)


def _ratios(pp):
    from studio.engine.color.spaces import apply_matrix
    r = apply_matrix(pp, T.PP_TO_RENDER)[0, 0]
    return r[1] / r[0], r[2] / r[0]


def test_natural_keeps_skin_colour_and_rich_pushes_it_redder():
    xs, ys = T.curve_lut(lambda v: T.standard_curve(v))
    src = _ratios(_skin())
    nat = _ratios(T.apply_curve_rgb(_skin(), xs, ys, natural=True))
    rich = _ratios(T.apply_curve_rgb(_skin(), xs, ys, natural=False))
    assert abs(nat[0] - src[0]) < 0.01 and abs(nat[1] - src[1]) < 0.01
    assert rich[0] < src[0] - 0.05 and rich[1] < src[1] - 0.05


def test_natural_eases_bright_colours_to_white_instead_of_clipping():
    xs, ys = T.curve_lut(lambda v: T.standard_curve(v))
    out = T.apply_curve_rgb(_skin(8.0), xs, ys, natural=True)
    from studio.engine.color.spaces import apply_matrix
    r = apply_matrix(out, T.PP_TO_RENDER)[0, 0]
    assert r.max() <= 1.0 + 1e-5 and r.min() > 0.3                    # bright, desaturated, not clipped orange


def test_new_edits_are_natural_and_old_ones_rich():
    from studio.engine.pipeline.edit import Edit
    from studio.engine.pipeline.pipeline import natural_curve
    assert natural_curve(Edit())
    assert not natural_curve(Edit.from_dict({"stage_versions": {"tone": 2}}))
    e = Edit()
    e.tone.curve_colour = "rich"
    assert not natural_curve(e)

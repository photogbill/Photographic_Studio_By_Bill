"""Export colour spaces and Save for web (io/export.py)."""
from __future__ import annotations

import io

import numpy as np

from studio.engine.io import export as ex


def _photo(h=600, w=900, seed=3):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    base = np.stack([0.5 + 0.4 * np.sin(6 * x), 0.4 + 0.3 * np.cos(5 * y), 0.3 + 0.2 * np.sin(9 * x * y)], -1)
    from scipy.ndimage import gaussian_filter
    tex = gaussian_filter(rng.normal(0, 0.04, (h, w)), 1.2)[..., None]     # photo-like fine texture
    return (np.clip(base + tex, 0, 1) * 255).astype(np.uint8)


def test_srgb_is_the_default_and_the_other_spaces_stay():
    assert ex.ExportSpec().space == "srgb"
    for s in ("adobergb", "prophoto", "displayp3", "srgb"):
        ex.ExportSpec(space=s).validate()


def test_save_for_web_is_an_srgb_jpeg_sized_and_sharpened_for_screens():
    s = ex.ExportSpec(web=True, format="tiff", space="prophoto")
    s.validate()
    assert (s.format, s.space, s.long_edge, s.sharpen_for) == ("jpeg", "srgb", ex.WEB_LONG_EDGE, "screen")


def test_web_jpeg_is_much_smaller_and_still_looks_the_same():
    from PIL import Image
    arr = _photo()
    icc = ex.icc.profile_bytes("srgb")
    data, info = ex.web_jpeg_bytes(arr, icc)
    best = io.BytesIO()
    Image.fromarray(arr).save(best, "JPEG", quality=95, subsampling=0, optimize=True, icc_profile=icc)
    assert len(data) < 0.8 * len(best.getvalue())
    ref = np.asarray(Image.open(io.BytesIO(best.getvalue())).convert("RGB"))
    got = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert ex._ssim_y(ref, got) >= ex.WEB_SSIM
    im = Image.open(io.BytesIO(data))
    assert im.info.get("progressive") or im.info.get("progression")
    assert im.info.get("icc_profile") == icc
    assert 60 <= info["quality"] <= 95

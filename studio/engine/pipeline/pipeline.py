"""The develop pipeline (section 3): stages 1-3, 5, 7, 8, 10-16 and 21 so far.

render(image, edit, region, scale)   -> display-referred linear ProPhoto float32 (HxWx3), oriented upright
render_view(image, edit, u, v, w, h) -> a w x h window of the finished photo at 100% (the loupe)
output_size(image, edit)             -> size of the finished photo (after straighten / perspective / crop)

Order: linearise, banding, raw denoise, white balance, lateral CA (raw planes), demosaic, camera matrix,
lens falloff (off by default), purple-fringe removal, capture sharpening (12), then geometry (13: distortion,
perspective, straighten, crop in one resample), exposure (15), microcontrast (16), tone curve (15), output.
Fringing runs before geometry rather than after it (plan order 13 -> 14): it is detected better on pixels that
have not been resampled yet.

Caching: everything up to the camera matrix (the expensive part: decode, linearise, white balance,
demosaic) is cached per image and keyed by Edit.raw_key(), so moving exposure or tone re-runs only the
cheap stages. Tiling: demosaic runs in strips (demosaic.py); the stages after it run strip by strip in
place, so peak memory stays near one float32 RGB frame.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict

import numpy as np

from ... import log as slog
from ..color import camera as cam
from ..color import tone as tone_mod
from ..color.spaces import apply_matrix
from ..io.rawfile import RawImage, open_raw
from ..raw import demosaic as dm
from ..raw import linearize as lin
from .edit import Edit, Light

log = slog.get("engine.pipeline")
_VIEW = threading.local()          # render_mask / mask_features ask _post for a mask or its features instead

STRIP = 512


class Image:
    """An opened raw plus its per-image caches. Create with ``studio.engine.open(path)``."""

    def __init__(self, raw: RawImage):
        self.raw = raw
        self._levels: dict = {}
        self._cache_key = None
        self._cache_val = None
        self._lock = threading.Lock()
        self._jpeg = None
        self._noise = {}
        self._rows = None
        self._ca = None
        self._blur = {}
        self._warp = None
        self._memo: dict = {}
        self.denoise_report: dict | None = None    # last whole-frame denoise: photosites kept by the safeguard

    def highlight_maps(self, mult, use_optical_black: bool = True):
        """Rebuilt highlight colour for the whole frame at this white balance (raw/highlights.py), computed once."""
        key = ("hl", tuple(np.round(np.asarray(mult, float), 6)), use_optical_black)
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        from ..raw.highlights import maps_for
        r = maps_for(self.raw, self.levels(use_optical_black), mult)
        with self._lock:
            self._memo = {k: v for k, v in self._memo.items() if k[0] != "hl"}
            self._memo[key] = r
        return r

    def glow_maps(self, mult, use_optical_black: bool = True, profile: str = ""):
        """Highlights measured from the lens glow (raw/glow.py) for the whole frame: (maps, report), computed once."""
        from ..raw import glow as gl
        prof = gl.load_profile(profile) if profile else None
        stamp = 0.0
        if prof is not None:
            try:
                stamp = os.path.getmtime(os.path.join(gl.profile_dir(), profile + ".json"))
            except OSError:
                stamp = 0.0
        key = ("glow", tuple(np.round(np.asarray(mult, float), 6)), use_optical_black, profile, stamp)
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        r = gl.maps_for(self.raw, self.levels(use_optical_black), mult, prof)
        with self._lock:
            self._memo = {k: v for k, v in self._memo.items() if k[0] != "glow"}
            self._memo[key] = r
        return r

    def haze_airlight(self, edit: "Edit"):
        """Colour of the haze for Dehaze, measured once on the whole frame (scene-linear, before exposure)."""
        key = ("haze", edit.raw_key())
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        from ..detail.finish import airlight
        e = Edit.from_dict(edit.to_dict())
        e.sharpen.enabled = False
        base = linear_rgb(self, e, None, half_size=True, cache=False)
        k = min(1.0, 800.0 / max(base.shape[:2]))
        if k < 1:
            base = resize(base, max(8, round(base.shape[1] * k)), max(8, round(base.shape[0] * k)))
        A = airlight(base)
        with self._lock:
            self._memo = {kk: v for kk, v in self._memo.items() if kk[0] != "haze"}
            self._memo[key] = A
        log.debug("dehaze airlight %s", np.round(A, 4).tolist())
        return A

    def detail_proxy(self, edit: "Edit"):
        """A small copy of the whole frame (scene-linear, before exposure, long edge about 1024 px) on which
        "Detail by scale" measures its veil floor, tonal-separation regions and broad field, so that a 100% view
        and the export agree."""
        key = ("dproxy", edit.raw_key())
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        from ..detail.scales import PROXY_LONG
        e = Edit.from_dict(edit.to_dict())
        e.sharpen.enabled = False
        base = linear_rgb(self, e, None, half_size=True, cache=False)
        k = min(1.0, float(PROXY_LONG) / max(base.shape[:2]))
        if k < 1:
            base = resize(base, max(8, round(base.shape[1] * k)), max(8, round(base.shape[0] * k)))
        base = np.ascontiguousarray(base, np.float32)
        with self._lock:
            self._memo = {kk: v for kk, v in self._memo.items() if kk[0] != "dproxy"}
            self._memo[key] = base
        return base

    @property
    def path(self) -> str:
        return self.raw.path

    @property
    def meta(self):
        return self.raw.meta

    def levels(self, use_optical_black: bool = True) -> lin.Levels:
        if use_optical_black not in self._levels:
            with slog.stage("linearize.levels", log):
                self._levels[use_optical_black] = lin.levels(self.raw, use_optical_black)
        return self._levels[use_optical_black]

    @property
    def output_size(self) -> tuple[int, int]:
        """(width, height) of a full-size render, after the camera crop and orientation."""
        _, _, w, h = self.raw.crop
        return (h, w) if self.raw.flip in (5, 6) else (w, h)

    def embedded_jpeg(self) -> bytes | None:
        if self._jpeg is None:
            self._jpeg = self.raw.embedded_jpeg() or b""
        return self._jpeg or None

    def noise_model(self, use_optical_black: bool = True):
        """Noise model for this frame: estimated from the image itself (amber tier) with the read noise
        pinned to the masked border. Calibrated profiles (cyan) will take precedence once they exist."""
        if use_optical_black not in self._noise:
            from ..calib.noise import estimate_from_image
            from ..calib.profile import find_profile
            lv = self.levels(use_optical_black)
            prof = find_profile(self.meta.make, self.meta.model)
            pm = prof.noise_model(self.meta.iso, lv.range) if prof else None
            if pm is not None and np.all(np.isfinite(pm.a)):
                log.debug("noise model from %s profile (%s)", pm.source, prof.model)
                self._noise[use_optical_black] = pm          # cyan (this body) or violet (shared)
                return pm
            with slog.stage("noise.estimate", log):
                mos = lin.normalise(self.raw, lv)
                rn = None if lv.read_noise_dn is None else lv.read_noise_dn / lv.range
                if pm is not None:                          # Quick profile: calibrated read noise, image gain
                    rn = np.sqrt(pm.b)
                m = estimate_from_image(mos, rn, self.meta.iso)
                if pm is not None:
                    m.source = "calibrated read noise + estimated gain"
                self._noise[use_optical_black] = m
                del mos
        return self._noise[use_optical_black]

    def row_banding(self, use_optical_black: bool = True):
        if self._rows is None:
            from ..raw.banding import measure_rows
            self._rows = measure_rows(self.raw, self.levels(use_optical_black)) or False
        return self._rows or None

    def lateral_ca(self):
        """Lateral CA measured once on the whole frame (as-shot white balance), then reused for every region."""
        if self._ca is None:
            from ..lens.auto import estimate_lateral_ca
            lv = self.levels(True)
            with slog.stage("lens.estimate_ca", log):
                mos = lin.normalise(self.raw, lv)
                m = self.raw.wb_as_shot / self.raw.wb_as_shot.min()
                for i in (0, 1):
                    for j in (0, 1):
                        mos[i::2, j::2] *= np.float32(m[self.raw.cfa_rgb[i, j]])
                self._ca = estimate_lateral_ca(np.minimum(mos, 1.0), self.raw.cfa_rgb)
                del mos
        return self._ca

    def capture_blur(self, edit: Edit | None = None):
        """Capture blur (lens + AA filter + demosaic) measured once from the sharpest edges of the most detailed
        tiles (detail/sharpen.py). Cached per demosaic algorithm."""
        algo = (edit or Edit()).demosaic
        algo = "menon" if algo == "half" else algo
        if algo not in self._blur:
            with slog.stage("sharpen.measure_blur", log, demosaic=algo):
                self._blur[algo] = _measure_capture_blur(self, algo)
            b = self._blur[algo]
            log.info("capture blur %.2f px (%s, %d edges)", b.sigma, b.confidence, b.edges)
        return self._blur[algo]

    def drop_cache(self):
        self._cache_key = self._cache_val = None


SAFEGUARD_K = 4.5           # denoise safeguard: photosites moved by more than this many sigma are restored


def open_image(path: str) -> Image:
    return Image(open_raw(path))


# ------------------------------------------------------------------------------------------ white balance
def wb_multipliers(image: Image, edit: Edit) -> np.ndarray:
    """R, G, B multipliers for the edit's white balance, normalised so the smallest is 1."""
    wb = edit.white_balance
    r = image.raw
    if wb.mode == "as_shot":
        m = r.wb_as_shot
    elif wb.mode == "daylight":
        m = r.wb_daylight
    elif wb.mode == "temp_tint":
        m = cam.multipliers_for_white(r.cam_xyz, cam.temp_tint_to_xy(wb.temperature, wb.tint))
    elif wb.mode == "multipliers" and wb.multipliers:
        m = np.asarray(wb.multipliers, float)[:3]
    else:
        raise ValueError(f"unknown white balance mode {wb.mode!r}")
    m = np.asarray(m, float)
    if not (np.isfinite(m).all() and m.min() > 0):
        raise ValueError(f"bad white balance multipliers {m}")
    return m / m.min()


# ------------------------------------------------------------------------------------------ stages 1-11
def _crop_box(image: Image, region):
    """Region in crop-frame coordinates (unrotated) -> (x, y, w, h) in visible-area coordinates."""
    cx, cy, cw, ch = image.raw.crop
    if region is None:
        return cx, cy, cw, ch
    x, y, w, h = (int(v) for v in region)
    x, y = max(0, x), max(0, y)
    w, h = min(w, cw - x), min(h, ch - y)
    if w <= 0 or h <= 0:
        raise ValueError(f"region {region} is outside the image")
    return cx + x, cy + y, w, h


def linear_rgb(image: Image, edit: Edit, region=None, half_size: bool = False, cache: bool = True,
               sharpen: bool = True) -> np.ndarray:
    """Stages 1-12 (+ lens falloff and fringing): camera mosaic -> scene-linear ProPhoto RGB (1.0 = sensor clip of
    a neutral), in the camera-crop frame. ``sharpen=False`` skips capture sharpening (callers that sharpen only
    the part they show, see _render_plain)."""
    sharpen_on = sharpen and edit.sharpen.enabled and not half_size and edit.sharpen.amount > 0
    key = (edit.raw_key(), tuple(region) if region else None, half_size, sharpen_on)
    with image._lock:
        if cache and image._cache_key == key:
            return image._cache_val
    raw = image.raw
    lv = image.levels(edit.use_optical_black)
    x, y, w, h = _crop_box(image, region)
    # read a margin around the region so the demosaic has context; keep everything CFA-aligned
    m = 0 if half_size else 16
    if sharpen_on:
        m = 32                                     # deconvolution reaches a few blur widths
    if edit.denoise.enabled:
        m = 256                                    # coarse denoise scales need context around a region
    x0, y0 = max(0, (x - m) // 2 * 2), max(0, (y - m) // 2 * 2)
    x1, y1 = min(raw.width, x + w + m), min(raw.height, y + h + m)
    with slog.stage("linearize.normalise", log, w=x1 - x0, h=y1 - y0):
        mos = lin.normalise(raw, lv, (x0, y0, x1 - x0, y1 - y0))
    if edit.banding:
        from ..raw.banding import remove_rows
        rb = image.row_banding(edit.use_optical_black)
        if rb is not None and rb.weight > 0.01:
            with slog.stage("banding.rows", log, weight=round(rb.weight, 2)):
                remove_rows(mos, rb, y0)
    if edit.denoise.enabled:
        from ..raw.denoise import DenoiseParams, denoise_mosaic
        dn = edit.denoise
        model = image.noise_model(edit.use_optical_black)
        with slog.stage("denoise", log, method=dn.method, luma=dn.luma, chroma=dn.chroma, src=model.source):
            st = {}
            mos = denoise_mosaic(mos, raw.cfa_rgb, model,
                                 DenoiseParams(method=dn.method, luma=dn.luma, chroma=dn.chroma, grain=dn.grain,
                                               safeguard=SAFEGUARD_K if dn.safeguard else 0.0), stats=st)
            if region is None and st:              # whole frame: what the Clean panel reports
                image.denoise_report = dict(st, half=half_size)
    mult = wb_multipliers(image, edit)
    with slog.stage("white_balance", log, mult=np.round(mult, 4).tolist(), mode=edit.white_balance.mode):
        pat = raw.cfa_rgb
        for i in (0, 1):
            for j in (0, 1):
                mos[i::2, j::2] *= np.float32(mult[pat[i, j]])
        if edit.highlight_mode == "clip":
            np.minimum(mos, 1.0, out=mos)          # neutral clip: every channel saturates together
        elif edit.highlight_mode == "rebuild":     # stage 9: clipped channels get the colour of the area's edge
            from ..raw import highlights as hl
            maps = image.highlight_maps(mult, edit.use_optical_black)
            if maps is not None:
                with slog.stage("highlights.rebuild", log):
                    mos = hl.apply(mos, pat, mult, maps, y0, x0)
        elif edit.highlight_mode == "glow":        # stage 9: blown lights measured from the lens glow (raw/glow.py)
            from ..raw import highlights as hl
            maps, _ = image.glow_maps(mult, edit.use_optical_black, edit.glow_profile)
            if maps is not None:
                with slog.stage("highlights.glow", log):
                    mos = hl.apply(mos, pat, mult, maps, y0, x0)
    if edit.lens.auto_ca:
        ca = image.lateral_ca()
        if ca.confidence != "none":
            from ..lens.auto import correct_lateral_ca
            with slog.stage("lens.lateral_ca", log, corner_px=round(ca.max_shift_px, 2)):
                mos = correct_lateral_ca(mos, pat, ca, full_shape=(raw.height, raw.width), origin=(y0, x0))
    algo = "half" if half_size else edit.demosaic
    if getattr(edit, "green_eq", False):
        with slog.stage("green_equilibrate", log):
            mos = dm.green_equilibrate(mos, pat)
    with slog.stage("demosaic", log, algorithm=algo, mp=round(mos.size / 1e6, 1)):
        rgb = dm.demosaic(mos, pat, algo)
    del mos
    prof = None
    if edit.profile:
        from ..color import dcp
        try:
            prof = dcp.load(edit.profile)
        except (OSError, ValueError) as err:
            log.warning("camera profile %r unusable (%s); using the camera matrix", edit.profile, err)
    if prof is not None:
        from ..color import dcp
        M, g = dcp.camera_to_working(prof, mult)
        with slog.stage("camera_matrix", log, profile=prof.name, weight=round(g, 3)):
            _strips(rgb, lambda s: apply_matrix(s, M, out=s))
            if prof.hsm or prof.look is not None:
                _strips(rgb, lambda s: s.__setitem__(Ellipsis, dcp.apply_tables(s, prof, g)))
    else:
        M = cam.camera_to_working(raw.cam_xyz).astype(np.float32)
        with slog.stage("camera_matrix", log):
            _strips(rgb, lambda s: apply_matrix(s, M, out=s))
    if edit.lens.vignette and edit.lens.vignette_model and edit.lens.vignette_amount:
        from ..lens import falloff as fo           # part of stage 13, done here where sensor positions are known
        cx0, cy0, cw0, ch0 = raw.crop
        step = 2.0 if half_size else 1.0
        with slog.stage("lens.falloff", log, corner_ev=round(fo.corner_ev(edit.lens.vignette_model), 2)):
            g = fo.gain_map(edit.lens.vignette_model, edit.lens.vignette_amount, rgb.shape[0], rgb.shape[1],
                            (x0 - cx0, y0 - cy0), (cw0, ch0), step)
            rgb *= g[..., None]
    if edit.lens.defringe and edit.lens.defringe_strength > 0:
        from ..lens.auto import defringe
        with slog.stage("lens.defringe", log):        # before the crop, so region renders match the full frame
            rgb = _overlapped(rgb, lambda s: defringe(s, edit.lens.defringe_strength), pad=16)
    if sharpen_on:                                 # stage 12 (full resolution only: invisible at half size)
        rgb = apply_sharpen(image, edit, rgb)
    if half_size:
        ox, oy = (x - x0) // 2, (y - y0) // 2
        rgb = rgb[oy:oy + h // 2, ox:ox + w // 2]
    else:
        rgb = rgb[y - y0:y - y0 + h, x - x0:x - x0 + w]
    rgb = np.ascontiguousarray(rgb)
    if cache:
        with image._lock:
            image._cache_key, image._cache_val = key, rgb
    return rgb


def apply_sharpen(image: Image, edit: Edit, rgb: np.ndarray) -> np.ndarray:
    """Capture sharpening (stage 12) on a full-resolution scene-linear block. Local (support under 32 px), so a
    block with a 32 px margin gives the same pixels as the full frame."""
    from ..detail import sharpen as shp
    sig = edit.sharpen.radius if edit.sharpen.radius > 0 else image.capture_blur(edit).sigma
    return shp.capture_sharpen(rgb, shp.SharpenParams(amount=edit.sharpen.amount, sigma=sig,
                                                      threshold=edit.sharpen.threshold), luma_noise_fn(image, edit))


def _sharpen_inner(image: Image, edit: Edit, base: np.ndarray, inner) -> np.ndarray:
    """Sharpen only ``inner`` (x0, y0, x1, y1 in base's pixels) plus a 32 px margin; the rest is context."""
    h, w = base.shape[:2]
    x0, y0, x1, y1 = inner
    a0, b0, a1, b1 = max(0, x0 - 32), max(0, y0 - 32), min(w, x1 + 32), min(h, y1 + 32)
    out = base.copy()
    out[b0:b1, a0:a1] = apply_sharpen(image, edit, np.ascontiguousarray(base[b0:b1, a0:a1]))
    return out


# ------------------------------------------------------------------------------------------ noise for later stages
def luma_noise_fn(image: Image, edit: Edit):
    """sigma(Y) of scene-linear luminance (pre-exposure, 1.0 = clip) from this frame's noise model, after
    white balance and denoising (approximate: used by the noise gates of stages 12 and 16, not by the denoiser)."""
    try:
        nm = image.noise_model(edit.use_optical_black)
    except Exception:
        slog.log_exception("noise model unavailable; detail stages run without a noise gate", log)
        return None
    pat = image.raw.cfa_rgb
    gpos = [(i, j) for i in (0, 1) for j in (0, 1) if pat[i, j] == 1]
    a = float(np.mean([nm.a[p] for p in gpos]))
    b = float(np.mean([nm.b[p] for p in gpos]))
    mg = float(wb_multipliers(image, edit)[1])
    resid = 1.0
    if edit.denoise.enabled:
        resid = float(np.clip(1.0 - 0.5 * min(edit.denoise.luma, 1.0), 0.45, 1.0))
    k = 0.8 * resid                                # luminance mixes channels; demosaic smooths a little

    def sigma(y):
        y = np.maximum(np.asarray(y, np.float32), 0.0)
        return k * mg * np.sqrt(a * y / mg + b)
    return sigma


# Fraction of the luminance noise that each denoise method leaves, by pyramid level (0 = pixel level), at
# luminance strengths 0.3 / 0.6 / 1.0. Measured on flat synthetic patches with the R5 ISO 4000 noise model
# (docs/RESULTS.md, session 10): the denoisers clean the pixel level hard and leave most of the slow blotches.
_RESIDUAL = {
    "hybrid": ((0.3, 0.6, 1.0), ((0.18, 0.31, 0.51, 0.76, 0.90), (0.16, 0.28, 0.49, 0.74, 0.89),
                                 (0.15, 0.26, 0.48, 0.73, 0.88))),
    "starlet": ((0.3, 0.6, 1.0), ((0.36, 0.60, 0.68, 0.76, 0.84), (0.20, 0.36, 0.47, 0.60, 0.74),
                                  (0.10, 0.19, 0.31, 0.45, 0.64))),
}
_RESIDUAL["nlm"] = _RESIDUAL["hybrid"]            # hybrid's luminance is the non-local means
_RESIDUAL["mppca"] = _RESIDUAL["starlet"]         # not measured yet: the more cautious table


def denoise_residual(edit: Edit) -> list:
    """Noise left after the denoiser as a fraction of the original, per pyramid level (1.0 everywhere when the
    denoiser is off). Keep-grain puts a fine fraction back, mostly at pixel level."""
    dn = edit.denoise
    if not dn.enabled or dn.luma <= 0:
        return [1.0] * 5
    xs, rows = _RESIDUAL.get(dn.method, _RESIDUAL["starlet"])
    xs, rows = (0.0,) + tuple(xs), ((1.0,) * 5,) + tuple(rows)
    r = [float(np.interp(min(dn.luma, xs[-1]), xs, [row[i] for row in rows])) for i in range(5)]
    g = float(np.clip(dn.grain, 0.0, 1.0))
    return [ri + g * (1.0 - ri) * w for ri, w in zip(r, (1.0, 0.5, 0.2, 0.0, 0.0))]


def detail_noise_fn(image: Image, edit: Edit):
    """sigma(Y) for "Detail by scale": the frame's luminance noise before denoising, carrying ``resid_levels``,
    the measured fraction the denoiser leaves at each scale (plan 23.11: the noise toe must use the noise left
    AFTER the denoiser, or noisy shadows get nothing while highlights get the most)."""
    try:
        nm = image.noise_model(edit.use_optical_black)
    except Exception:
        slog.log_exception("noise model unavailable; detail stages run without a noise gate", log)
        return None
    pat = image.raw.cfa_rgb
    gpos = [(i, j) for i in (0, 1) for j in (0, 1) if pat[i, j] == 1]
    a = float(np.mean([nm.a[p] for p in gpos]))
    b = float(np.mean([nm.b[p] for p in gpos]))
    mg = float(wb_multipliers(image, edit)[1])
    g = 2.0 ** edit.exposure_ev                    # the array carries the exposure by the time stage 16 runs

    def sigma(y):
        y = np.maximum(np.asarray(y, np.float32), 0.0) / g
        return g * 0.8 * mg * np.sqrt(a * y / mg + b)
    sigma.resid_levels = denoise_residual(edit)
    return sigma


def _measure_capture_blur(image: Image, algo: str):
    from ..detail import sharpen as shp
    raw = image.raw
    lv = image.levels(True)
    cx, cy, cw, ch = raw.crop
    mos = lin.normalise(raw, lv, (cx, cy, cw // 2 * 2, ch // 2 * 2))
    pat = raw.cfa_rgb
    g = 0.5 * (mos[0::2, 1::2] + mos[1::2, 0::2]) if pat[0, 1] == 1 else 0.5 * (mos[0::2, 0::2] + mos[1::2, 1::2])
    del mos
    tiles = shp.select_tiles(g, 2, n=8, tile=384)
    if not tiles:
        return shp.DEFAULT_BLUR
    e = Edit(demosaic=algo)
    e.lens.defringe = False
    e.lens.auto_ca = True
    nf = luma_noise_fn(image, e)
    samples = []
    for (x, y, w, h) in tiles:
        x, y = min(x, cw - w) // 2 * 2, min(y, ch - h) // 2 * 2
        rgb = linear_rgb(image, e, (x, y, w, h), cache=False)
        Y = rgb @ shp.LUMA
        ns = float(nf(np.median(Y))) if nf else 0.0
        samples.append(shp.edge_blur_samples(Y, ns, max_points=800))
    return shp.summarise_blur(np.concatenate(samples) if samples else np.zeros(0))


def _overlapped(img: np.ndarray, fn, rows: int = STRIP, pad: int = 16) -> np.ndarray:
    """Apply a neighbourhood filter strip by strip with overlap (bounded memory on 45 MP frames), in parallel."""
    from ..parallel import strips
    out = np.empty_like(img)
    h = img.shape[0]

    def job(y0, y1):
        a, b = max(0, y0 - pad), min(h, y1 + pad)
        out[y0:y1] = fn(img[a:b])[y0 - a:y0 - a + (y1 - y0)]
    strips(h, rows, job)
    return out


def _strips(img: np.ndarray, fn, rows: int = STRIP):
    """Apply a per-pixel function to every strip in place, in parallel threads."""
    from ..parallel import strips
    strips(img.shape[0], rows, lambda y0, y1: fn(img[y0:y1]))


# ------------------------------------------------------------------------------------------ stage 15
def tone_lut(image: Image, edit: Edit, lin_img: np.ndarray | None = None):
    """(xs, ys) tone curve for stage 15, including the Light panel's highlights / shadows / whites / blacks and
    point curves. ys is (3, N) when the colour curves differ. None = no curve (linear mode, nothing to do)."""
    from ..color import light as lt
    t = edit.tone
    li = edit.light
    if t.mode == "standard":
        mid = _mid_in(image, edit)
        if lt.is_identity(li):
            return tone_mod.curve_lut(lambda v: tone_mod.standard_curve(v, mid_in=mid, contrast=t.contrast,
                                                                        shoulder=t.shoulder, white=t.white,
                                                                        toe=t.toe))
        white = max(t.white + lt.white_shift(li.whites), t.shoulder + 0.3)
        base = lambda v: tone_mod.standard_curve(v, mid_in=mid, contrast=t.contrast,          # noqa: E731
                                                 shoulder=t.shoulder, white=white, toe=t.toe)
        xs, _ = tone_mod.curve_lut(lambda v: v)
        return xs, lt.build_light_lut(base, li, mid, white, xs)
    if t.mode == "camera_match":
        jpg = image.embedded_jpeg()
        if jpg is None or lin_img is None:
            log.warning("camera_match: no embedded JPEG; using Studio Standard")
            e = Edit.from_dict(edit.to_dict())
            e.tone = type(t)()
            return tone_lut(image, e, lin_img)
        xs_m, ys_m = tone_mod.camera_match_curve(lin_img, jpg)
        if lt.is_identity(li):
            return xs_m, ys_m
        base = _with_whites(lambda v: np.interp(v, xs_m, ys_m), li.whites)
        xs, _ = tone_mod.curve_lut(lambda v: v)
        return xs, lt.build_light_lut(base, li, _mid_in(image, edit), 3.0, xs)
    if t.mode == "linear":
        if lt.is_identity(li):
            return None
        base = _with_whites(lambda v: np.clip(v, 0.0, 1.0), li.whites)
        xs, _ = tone_mod.curve_lut(lambda v: v)
        return xs, lt.build_light_lut(base, li, _mid_in(image, edit), 3.0, xs)
    raise ValueError(f"unknown tone mode {t.mode!r}")


def _with_whites(fn, whites: float):
    """Whites for the curve modes without a shoulder: brighten (or dim) the top of the range, monotonically."""
    if not whites:
        return fn
    k = 0.3 * whites / 100.0

    def g(v):
        y = np.asarray(fn(v), np.float64)
        t = np.clip((y - 0.2) / 0.8, 0.0, 1.0)
        return np.clip(y * (1.0 + k * t * t * (3 - 2 * t)), 0.0, 1.0)
    return g


def apply_orientation(img: np.ndarray, flip: int) -> np.ndarray:
    if flip == 3:
        return img[::-1, ::-1]
    if flip == 5:
        return np.rot90(img, 1)       # 90 degrees counter-clockwise
    if flip == 6:
        return np.rot90(img, -1)      # 90 degrees clockwise
    return img


def _mid_in(image: Image, edit: Edit) -> float:
    mid = edit.tone.mid_grey
    if edit.tone.match_camera_brightness:
        mid = mid / 2.0 ** tone_mod.body_brightness_ev(image.meta.model)
    return mid


def _mc_params(edit: Edit):
    """Stage 16 parameters ("Detail by scale", detail/scales.py): every method's amount, switched-off ones at 0."""
    from ..detail.scales import ScaleParams
    m, d = edit.microcontrast, edit.detail
    off = set(d.off or ())
    on = lambda k, v: 0.0 if k in off else float(v)
    return ScaleParams(tonal=on("tonal", d.tonal) / 100.0, local=on("local", m.amount),
                       crisp=on("crisp", d.crisp) / 100.0, separation=on("separation", d.separation) / 100.0,
                       veil=on("veil", d.veil) / 100.0, broad=on("broad", d.broad) / 100.0, shadows=m.shadows,
                       midtones=m.midtones, highlights=m.highlights, scale=m.scale, range_ev=m.range_ev).clamped()


def _detail_fields(image: Image, edit: Edit, sp=None):
    """Veil floor, tonal-separation table and broad field, measured once on the whole frame (detail/scales.py)."""
    from ..detail import scales as sc
    sp = sp or _mc_params(edit)
    if not sp.needs_proxy:
        return None
    mid = _mid_in(image, edit)
    key = ("dfields", edit.raw_key(), round(edit.exposure_ev, 4), round(mid, 6),
           tuple(round(getattr(sp, k), 4) for k in ("separation", "veil", "broad", "shadows", "midtones",
                                                    "highlights")))
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    proxy = image.detail_proxy(edit)
    nf = detail_noise_fn(image, edit)
    g = 2.0 ** edit.exposure_ev
    k = min(1.0, max(proxy.shape[:2]) / _long_edge(image))           # the proxy averages pixel noise away
    nfs = (lambda y: k * nf(y)) if nf else None
    f = sc.measure_fields(proxy, sp, g, mid, nfs)
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "dfields"}
        image._memo[key] = f
    return f


def _hdr_params(edit: Edit):
    from ..color.hdr import HDRParams
    h = edit.hdr
    return HDRParams(strength=h.strength, amount=float(np.clip(h.amount, 0.0, 150.0)) / 100.0 if h.enabled else 0.0,
                     engine=h.engine)


def _hdr_fields(image: Image, edit: Edit):
    """Auto HDR's brightness field, measured once on the whole frame (color/hdr.py). None when it is off."""
    from ..color import hdr
    hp = _hdr_params(edit)
    if not hp.active:
        return None
    mid = _mid_in(image, edit)
    sp = _mc_params(edit)
    key = ("hdr", edit.raw_key(), round(edit.exposure_ev, 4), round(mid, 6), hp.strength, round(hp.amount, 4),
           hp.engine, round(sp.veil, 4))
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    g = 2.0 ** edit.exposure_ev
    proxy = image.detail_proxy(edit) * np.float32(g)
    df = _detail_fields(image, edit, sp)
    if df is not None and df.veil is not None:          # the veil comes off first: lifting a haze floor gives mush
        from ..detail.scales import apply_veil
        proxy = apply_veil(proxy, df)
    nf = detail_noise_fn(image, edit)                   # the lift is judged by the noise of the full-size photo
    f = hdr.measure(proxy, hp, 1.0, mid, nf)
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "hdr"}
        image._memo[key] = f
    log.debug("auto hdr: %s", f.report)
    return f


def hdr_report(image: Image, edit: Edit) -> dict | None:
    """What Auto HDR measured and did for this photo (scene range, stops lifted and pulled), for the Light panel."""
    f = _hdr_fields(image, edit)
    return dict(f.report) if f is not None else None


def _proxy_xy(image: Image, frame: "Frame", fields, h: int, w: int):
    """Position of every pixel of an (h, w) render on the whole-frame proxy."""
    xs, ys = frame.s_coords(h, w)
    _, _, sw, sh = image.raw.crop
    ph, pw = fields.shape
    return (np.asarray(xs, np.float32) + 0.5) * np.float32(pw / sw) - 0.5, \
           (np.asarray(ys, np.float32) + 0.5) * np.float32(ph / sh) - 0.5


def detail_balance(image: Image, edit: Edit, size: int = 1024):
    """(before, after): the balance number of "Detail by scale" (medium-scale tonal variation over pixel-level
    contrast, detail/scales.balance_ratio), measured at 100% on the middle of the photo without and with the
    panel's methods. Higher = modelled, lower = etched."""
    from ..detail import scales as sc
    _, _, cw, ch = image.raw.crop
    w, h = min(size, cw // 2 * 2), min(size, ch // 2 * 2)
    region = ((cw - w) // 4 * 2, (ch - h) // 4 * 2, w, h)
    plain = Edit.from_dict(edit.to_dict())
    plain.microcontrast.amount = 0.0
    for k in ("tonal", "crisp", "separation", "veil", "broad"):
        setattr(plain.detail, k, 0.0)
    plain.geometry = Edit().geometry
    cur = Edit.from_dict(edit.to_dict())
    cur.geometry = Edit().geometry
    le = _long_edge(image)
    lum = lambda a: a @ np.array([0.2880402, 0.7118741, 0.0000857], np.float32)
    before = sc.balance_ratio(lum(_render_plain(image, plain, region, 1.0)), 1.0, le)
    if not _mc_params(edit).active:
        return before, before
    return before, sc.balance_ratio(lum(_render_plain(image, cur, region, 1.0)), 1.0, le)


def _long_edge(image: Image) -> float:
    _, _, w, h = image.raw.crop
    return float(max(w, h))


class Frame:
    """Where the pixels of an array in stages 15-17 sit. Pixel (i, j) is at full-resolution position
    (ox + (j + 0.5) / sx - 0.5, oy + (i + 0.5) / sy - 0.5) of the S frame (camera crop, before orientation) for
    plain renders, or of the output frame for renders with geometry (``warp`` set). Grain and masks use it, so a
    100% region matches the full render exactly."""

    def __init__(self, ox: float, oy: float, sx: float, sy: float, warp=None):
        self.ox, self.oy, self.sx, self.sy, self.warp = float(ox), float(oy), float(sx), float(sy), warp

    @property
    def scale(self) -> float:
        return min(self.sx, self.sy)

    def axes(self, h: int, w: int):
        """1-D full-resolution coordinates (xs, ys) of the columns and rows, in this frame."""
        return (self.ox + (np.arange(w) + 0.5) / self.sx - 0.5, self.oy + (np.arange(h) + 0.5) / self.sy - 0.5)

    def s_coords(self, h: int, w: int, step: int = 16):
        """2-D S-frame coordinates (xs, ys) of every pixel (for masks, which are attached to the photo)."""
        xa, ya = self.axes(h, w)
        if self.warp is None:
            return np.broadcast_to(xa[None, :], (h, w)), np.broadcast_to(ya[:, None], (h, w))
        from scipy.ndimage import map_coordinates
        gx = np.unique(np.concatenate([np.arange(0, w, step), [w - 1]]))
        gy = np.unique(np.concatenate([np.arange(0, h, step), [h - 1]]))
        X, Y = np.meshgrid(xa[gx], ya[gy])
        sx, sy = self.warp.output_to_s(X, Y)
        fy = np.interp(np.arange(h), gy, np.arange(len(gy)))
        fx = np.interp(np.arange(w), gx, np.arange(len(gx)))
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        return (map_coordinates(np.asarray(sx, np.float64), [FY, FX], order=1),
                map_coordinates(np.asarray(sy, np.float64), [FY, FX], order=1))


def _post(image: Image, edit: Edit, out: np.ndarray, frame: Frame, lut) -> np.ndarray:
    """Stages 15-17 on a scene-linear array: exposure, Dehaze, Smart lighting, microcontrast, Texture, tone
    curve, colour, grain."""
    from ..color import grade
    eff = frame.scale
    gain = np.float32(2.0 ** edit.exposure_ev)
    mc = _mc_params(edit)
    fin = edit.finish
    fields = _detail_fields(image, edit, mc)
    if fields is not None and fields.veil is not None:      # veil first: before any haze removal or shadow lift
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
            gain = np.float32(1.0)
        from ..detail.scales import apply_veil
        _strips(out, lambda s: s.__setitem__(Ellipsis, apply_veil(s, fields)))
    if edit.light.dehaze:
        A = image.haze_airlight(edit) * gain
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
            gain = np.float32(1.0)
        from ..detail.finish import dehaze
        out = dehaze(out, edit.light.dehaze, A, eff, _long_edge(image))
    if edit.light.smart > 0:
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
            gain = np.float32(1.0)
        from ..color.light import smart_lighting
        out = smart_lighting(out, edit.light.smart, _mid_in(image, edit), eff, _long_edge(image))
    hf = _hdr_fields(image, edit)
    if hf is not None:                                      # Auto HDR: after the veil, before local contrast
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
            gain = np.float32(1.0)
        from ..color import hdr as hdrm
        hx, hy = _proxy_xy(image, frame, hf, out.shape[0], out.shape[1])
        with slog.stage("hdr.apply", log, engine=hf.report["engine"], strength=hf.report["strength"]):
            mid_h = _mid_in(image, edit)
            for y0 in range(0, out.shape[0], STRIP):
                out[y0:y0 + STRIP] = hdrm.apply(out[y0:y0 + STRIP], hf, mid_h, hx[y0:y0 + STRIP], hy[y0:y0 + STRIP])
    if mc.beyond_veil:
        g = float(gain) if mc.only_local else 2.0 ** edit.exposure_ev     # (the first engine's own scaling, kept)
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
        from ..detail import scales as mcm
        if mc.only_local:                                   # the first engine with its own noise estimate
            nf = luma_noise_fn(image, edit)
            nfs = (lambda y: g * nf(np.asarray(y) / g)) if nf else None
        else:
            nfs = detail_noise_fn(image, edit)
        pxy = _proxy_xy(image, frame, fields, out.shape[0], out.shape[1]) if fields is not None else None
        out = mcm.apply(out, mc, _mid_in(image, edit) * 1.0, nfs, eff, _long_edge(image), fields, pxy,
                        veil_done=True)
        gain = np.float32(1.0)
    if fin.texture:
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
        from ..detail.finish import texture
        nf = luma_noise_fn(image, edit)
        g = 2.0 ** edit.exposure_ev                         # the array now carries the full exposure
        nfs = (lambda y: g * nf(np.asarray(y) / g)) if nf else None
        out = texture(out, fin.texture, eff, nfs)
        gain = np.float32(1.0)
    from ..local import masks as mk
    act = mk.active(edit.masks)
    want = getattr(_VIEW, "what", None)
    alphas, ctx = [], None
    if act or want:
        if gain != 1:
            _strips(out, lambda s: s.__imul__(gain))
            gain = np.float32(1.0)
        h, w = out.shape[:2]
        xs, ys = frame.s_coords(h, w)
        _, _, sw, sh = image.raw.crop
        ctx = mk.Ctx(out, xs, ys, sw, sh, eff, _lum_fn(lut))
        if want is not None:
            if want[0] == "features":
                return ctx.features()
            return np.repeat(mk.mask_alpha(edit.masks[want[1]], ctx)[..., None], 3, axis=2)
        with slog.stage("masks", log, n=len(act)):
            alphas = [mk.mask_alpha(m, ctx) for m in act]
            air = image.haze_airlight(edit) * np.float32(2.0 ** edit.exposure_ev) if mk.uses(act, ("dehaze",)) \
                else None
            for m, a in zip(act, alphas):
                if any(abs(float(m["adjust"].get(k, 0))) > 1e-6 for k in mk.SCENE_KEYS):
                    out = mk.apply_scene(out, m, a, ctx, air)
    tone_masks = [(m, a) for m, a in zip(act, alphas)
                  if any(abs(float(m["adjust"].get(k, 0))) > 1e-6 for k in mk.TONE_KEYS)]
    with slog.stage("exposure+tone", log, ev=edit.exposure_ev, mode=edit.tone.mode):
        def st(s):
            if gain != 1:
                s *= gain
            if lut is not None:
                s[...] = tone_mod.apply_curve_rgb(s, *lut, compress=edit.gamut_compress)
        if not tone_masks:
            _strips(out, st)
        else:                                               # local tone: blend the global and local curves
            luts = [_local_lut(image, edit, lut, m) for m, _ in tone_masks]
            for y0 in range(0, out.shape[0], STRIP):
                s = out[y0:y0 + STRIP]
                x = s.copy()
                st(s)
                for (m, a), lm in zip(tone_masks, luts):
                    ym = tone_mod.apply_curve_rgb(x.copy(), *lm, compress=edit.gamut_compress)
                    s += a[y0:y0 + STRIP, :, None] * (ym - s)
    if not grade.is_identity(edit.colour):                  # stage 17 (per pixel, in OKLCh)
        c = edit.colour
        _strips(out, lambda s: s.__setitem__(Ellipsis, grade.apply(s, c)))
    for m, a in zip(act, alphas):                           # local colour, then local detail
        cm = mk.colour_of(m)
        if not grade.is_identity(cm):
            for y0 in range(0, out.shape[0], STRIP):
                s = out[y0:y0 + STRIP]
                s += a[y0:y0 + STRIP, :, None] * (grade.apply(s, cm) - s)
        if any(abs(float(m["adjust"].get(k, 0))) > 1e-6 for k in mk.DETAIL_KEYS):
            out = mk.apply_detail(out, m, a, eff)
    if fin.grain > 0:
        from ..detail.finish import grain
        out = grain(out, fin.grain, fin.grain_size, fin.grain_roughness, frame.ox, frame.oy, frame.sx, frame.sy,
                    seed=1, colour=fin.grain_colour)
    return out


def _context(image: Image, edit: Edit, near=None) -> tuple[int, int]:
    """(pad, align) of full-resolution context a region render needs for the neighbourhood stages after
    linear_rgb (Smart lighting's base layer, microcontrast's pyramid, body sculpting's reach). (0, 2) when none is
    on. ``near``: the S box (x, y, w, h) being rendered, so sculpting far away adds nothing."""
    pad, align = 0, 2
    from ..local import portrait as pt
    if pt.sculpt_active(edit.portrait):
        _, _, sw, sh = image.raw.crop
        ws = pt.warps(edit.portrait, sw, sh)
        if near is not None:
            x, y, w, h = near
            hit = []
            for wp in ws:
                (a0, b0, a1, b1), mv = pt._reach(wp)
                if a0 - mv < x + w and a1 + mv > x and b0 - mv < y + h and b1 + mv > y:
                    hit.append(wp)
            ws = hit
        if ws:
            pad = max(pad, int(math.ceil(pt.margin(ws))) + 8)
    mc = _mc_params(edit)
    if mc.active:
        from ..detail.scales import context
        cp, align = context(mc, _long_edge(image))
        pad = max(pad, cp)
    if edit.light.smart > 0:
        from ..color.light import smart_radius
        pad = max(pad, 3 * smart_radius(1.0, _long_edge(image)) + 8)
    if edit.light.dehaze > 0:
        from ..detail.finish import dehaze_context
        pad = max(pad, dehaze_context(_long_edge(image)))
    if edit.finish.texture:
        from ..detail.finish import texture_radius
        pad = max(pad, 4 * texture_radius(1.0) + 4)
    from ..local import masks as mk
    act = mk.active(edit.masks)
    if act:
        pad = max(pad, 48)
        if mk.uses(act, ("clarity",)):
            pad = max(pad, 4 * mk.clarity_radius(1.0) + 8)
        if mk.uses(act, ("dehaze",)):
            from ..detail.finish import dehaze_context
            pad = max(pad, dehaze_context(_long_edge(image)))
        for m in act:
            if float((m.get("adjust") or {}).get("smooth_skin", 0)) > 0:
                r = mk.skin_radius(1.0, _long_edge(image), float(m["adjust"].get("skin_size", 0)))
                pad = max(pad, 4 * r + 8)
    return pad, align


def _lum_fn(lut):
    """Global tone curve as a function of scene luminance (for masks that select by screen brightness)."""
    if lut is None:
        return lambda y: np.clip(y, 0.0, 1.0)
    xs, ys = lut
    ys1 = ys if np.ndim(ys) == 1 else np.asarray(ys).mean(axis=0)
    return lambda y: np.interp(y, xs, ys1).astype(np.float32)


def _local_lut(image: Image, edit: Edit, lut, m: dict):
    """The global curve with one mask's tone controls laid on top (highlights / shadows / whites / blacks in
    stops, contrast around mid grey)."""
    from ..color import light as lt
    from ..local import masks as mk
    mid = _mid_in(image, edit)
    if lut is None:
        base = lambda v: np.clip(v, 0.0, 1.0)              # noqa: E731
    else:
        xs_g, ys_g = lut
        ys1 = ys_g if np.ndim(ys_g) == 1 else np.asarray(ys_g).mean(axis=0)
        base = lambda v: np.interp(v, xs_g, ys1)           # noqa: E731
    deltas, con = mk.tone_light(m, Light(), 1.0)
    if con != 1.0:
        b0 = base
        base = lambda v: b0(np.where(v > 0, mid * np.exp2(np.log2(np.maximum(v, 1e-12) / mid) * con), 0.0))  # noqa
    if deltas.whites:
        base = _with_whites(base, deltas.whites)
    xs, _ = tone_mod.curve_lut(lambda v: v)
    return xs, lt.build_light_lut(base, deltas, mid, 3.0, xs)


def make_warp(image: Image, edit: Edit):
    """Stage 13 geometry for this image and edit (cached on the image for the last geometry used)."""
    from ..lens.geometry import GeomParams, Warp
    g = edit.geometry
    key = json.dumps({k: v for k, v in asdict(g).items() if k != "aspect"}, sort_keys=True)
    if image._warp is not None and image._warp[0] == key:
        return image._warp[1]
    _, _, cw, ch = image.raw.crop
    w = Warp(cw, ch, image.raw.flip, GeomParams(
        distortion=g.distortion, distortion_amount=g.distortion_amount, distortion_manual=g.distortion_manual,
        rotate=g.rotate, vertical=g.vertical, horizontal=g.horizontal,
        crop=tuple(g.crop) if g.crop else None, constrain=g.constrain))
    image._warp = (key, w)
    return w


def output_size(image: Image, edit: Edit | None = None) -> tuple[int, int]:
    """(width, height) of a full-size render of this edit: after orientation, warps and crop."""
    return make_warp(image, edit or Edit()).output_size


def render(image: Image, edit: Edit | None = None, region=None, scale: float = 1.0) -> np.ndarray:
    """Develop to display-referred linear ProPhoto RGB, float32, upright.

    Without ``region``: the whole photo, with every stage including geometry and crop (stage 13).
    With ``region`` (x, y, w, h in the camera-crop frame before rotation): that part of the sensor, without
    geometry or crop, for analysis tools and comparisons. For a 100% view of the finished photo use render_view.
    ``scale`` <= 0.5 uses the fast half-size path."""
    edit = edit or Edit()
    if region is None:
        warp = make_warp(image, edit)
        if warp.warped:
            return _render_warp(image, edit, warp, None, scale)
        if not warp.identity:                       # crop only: no resampling, just the rectangle
            return _render_plain(image, edit, _window_to_s(image, warp, None), scale)
    return _render_plain(image, edit, region, scale)


def render_view(image: Image, edit: Edit | None, u: float, v: float, w: int = 720, h: int = 480) -> np.ndarray:
    """A w x h window of the finished photo at 100%, centred at (u, v) (0..1 of the displayed photo)."""
    edit = edit or Edit()
    warp = make_warp(image, edit)
    x, y, w, h = view_window(image, edit, u, v, w, h)
    if warp.warped:
        return _render_warp(image, edit, warp, (x, y, w, h), 1.0)
    return _render_plain(image, edit, _window_to_s(image, warp, (x, y, w, h)), 1.0)


def _window_to_s(image: Image, warp, window):
    """An output window (identity or crop-only geometry) as an S-frame region."""
    from ..lens.geometry import d_to_s
    x, y, w, h = window or (0, 0, warp.out_w, warp.out_h)
    xs, ys = d_to_s(np.array([x + warp.ox, x + warp.ox + w - 1]), np.array([y + warp.oy, y + warp.oy + h - 1]),
                    warp.sw, warp.sh, warp.flip)
    x0, y0 = int(min(xs)), int(min(ys))
    return x0, y0, int(max(xs)) - x0 + 1, int(max(ys)) - y0 + 1


def _render_plain(image: Image, edit: Edit, region, scale: float) -> np.ndarray:
    half = scale <= 0.5 or edit.demosaic == "half"
    work = region
    pad, align = _context(image, edit, region)
    if pad and region is not None:                  # neighbourhood stages need context around a region
        _, _, cw, ch = image.raw.crop
        x, y, w, h = (int(v) for v in region)
        px0, py0 = max(0, (x - pad) // align * align), max(0, (y - pad) // align * align)
        px1, py1 = min(cw, x + w + pad), min(ch, y + h + pad)
        work = (px0, py0, px1 - px0, py1 - py0)
    if work is not region and edit.sharpen.enabled and not half:
        base = linear_rgb(image, edit, work, half_size=half, sharpen=False)
        ix0, iy0 = region[0] - work[0], region[1] - work[1]
        base = _sharpen_inner(image, edit, base, (ix0, iy0, ix0 + region[2], iy0 + region[3]))
    else:
        base = linear_rgb(image, edit, work, half_size=half)
    base = apply_retouch(image, edit, base, work if work is not None else (0, 0), half)
    base = apply_portrait(image, edit, base, work if work is not None else (0, 0), half)
    base = apply_sculpt(image, edit, base, work if work is not None else (0, 0), half)
    lut = tone_lut(image, edit, base)
    eff = scale * (2.0 if half else 1.0)
    if abs(eff - 1.0) > 1e-6:
        # previews: shrink the scene-linear image first, then tone (tone is per pixel, so this is ~10x faster
        # and visually identical); full-size exports never take this path
        out = resize(base, max(1, round(base.shape[1] * eff)), max(1, round(base.shape[0] * eff)))
    else:
        out = base.copy()
    _, _, cw0, ch0 = image.raw.crop
    wx, wy, ww, wh = work if work is not None else (0, 0, cw0, ch0)
    frame = Frame(wx, wy, out.shape[1] / float(ww), out.shape[0] / float(wh))
    out = _post(image, edit, out, frame, lut)
    if work is not region:
        k = out.shape[1] / float(work[2])
        ox, oy = round((region[0] - work[0]) * k), round((region[1] - work[1]) * k)
        out = out[oy:oy + max(1, round(region[3] * k)), ox:ox + max(1, round(region[2] * k))]
    return np.ascontiguousarray(apply_orientation(out, image.raw.flip))


def _render_warp(image: Image, edit: Edit, warp, window, scale: float) -> np.ndarray:
    """Stage 13 with resampling: distortion, perspective, straighten, crop in one pass. Output is upright."""
    win = window or (0, 0, warp.out_w, warp.out_h)
    pwin = win
    pad, align = _context(image, edit, warp.source_box(win) if window is not None else None)
    if pad and window is not None:
        x, y, w, h = win
        px0, py0 = max(0, (x - pad) // align * align), max(0, (y - pad) // align * align)
        px1, py1 = min(warp.out_w, x + w + pad), min(warp.out_h, y + h + pad)
        pwin = (px0, py0, px1 - px0, py1 - py0)
    half = scale <= 0.5 or edit.demosaic == "half"
    box = warp.source_box(pwin)
    if pwin != win and edit.sharpen.enabled and not half:
        base = linear_rgb(image, edit, box, half_size=half, sharpen=False)
        ib = warp.source_box(win)
        base = _sharpen_inner(image, edit, base, (ib[0] - box[0], ib[1] - box[1], ib[0] - box[0] + ib[2],
                                                  ib[1] - box[1] + ib[3]))
    else:
        base = linear_rgb(image, edit, box, half_size=half)
    base = apply_retouch(image, edit, base, box, half)
    base = apply_portrait(image, edit, base, box, half)
    base = apply_sculpt(image, edit, base, box, half)
    src_scale = 0.5 if half else 1.0
    if scale < 0.75 * src_scale:                    # small previews: anti-alias before resampling
        f = scale / src_scale
        nw, nh = max(8, round(base.shape[1] * f)), max(8, round(base.shape[0] * f))
        src_scale = src_scale * nw / base.shape[1]
        base = resize(base, nw, nh)
    with slog.stage("geometry.resample", log, zoom=round(warp.zoom, 3), out=f"{pwin[2]}x{pwin[3]}@{scale:.2f}"):
        lin_out = warp.resample(base, box, src_scale, pwin, scale, order=3 if scale >= 0.75 else 1)
    lut = tone_lut(image, edit, lin_out)
    frame = Frame(pwin[0], pwin[1], lin_out.shape[1] / float(pwin[2]), lin_out.shape[0] / float(pwin[3]), warp)
    out = _post(image, edit, lin_out, frame, lut)
    if pwin != win:
        ox, oy = round((win[0] - pwin[0]) * scale), round((win[1] - pwin[1]) * scale)
        out = out[oy:oy + max(1, round(win[3] * scale)), ox:ox + max(1, round(win[2] * scale))]
    return np.ascontiguousarray(out)


# ------------------------------------------------------------------------------------------ automatic helpers
def auto_light(image: Image, edit: Edit) -> dict:
    """Auto for the Light panel: highlights / whites / blacks for this photo. Never exposure (Bill's rule)."""
    from ..color import light as lt
    sc = min(1.0, 900.0 / max(output_size(image, edit)))
    e = Edit.from_dict(edit.to_dict())
    e.light = type(edit.light)(smart=edit.light.smart, curve=None)
    base = linear_rgb(image, e, None, half_size=True, cache=False)
    k = min(1.0, (sc * 2.0))
    if k < 1.0:
        base = resize(base, max(8, round(base.shape[1] * k)), max(8, round(base.shape[0] * k)))
    base = base * np.float32(2.0 ** edit.exposure_ev)
    if e.light.smart > 0:
        base = lt.smart_lighting(base, e.light.smart, _mid_in(image, edit), sc, _long_edge(image))
    xs, ys = tone_lut(image, Edit(tone=edit.tone), base) or (None, None)
    if xs is None:
        fn = lambda v: np.clip(v, 0.0, 1.0)                 # noqa: E731
    else:
        fn = lambda v: np.interp(v, xs, ys)                 # noqa: E731
    Y = tone_mod.luminance(np.ascontiguousarray(base))
    return lt.auto_tone(Y, fn, _mid_in(image, edit))


def auto_colour(image: Image, edit: Edit) -> dict:
    """Auto colour: vibrance from the photo's measured colourfulness (Hasler and Susstrunk 2003). Never touches
    white balance or brightness and never lowers colour. Returns {"vibrance": v, "colourfulness": M}."""
    from ..color import grade
    from ..io.export import ExportSpec, to_output
    e = Edit.from_dict(edit.to_dict())
    e.colour.vibrance = 0.0
    e.colour.saturation = 0.0
    if e.colour.bw:
        return {"vibrance": 0.0, "colourfulness": 0.0}
    e.finish.grain = 0.0
    sc = min(1.0, 600.0 / max(output_size(image, e)))
    a8 = to_output(render(image, e, scale=sc), ExportSpec(format="jpeg", space="srgb", bits=8))
    m = grade.colourfulness(a8)
    return {"vibrance": round(grade.auto_vibrance(m), 1), "colourfulness": round(m, 1)}


def auto_white_balance(image: Image, method: str = "grey_edge", mood: float | None = None):
    """Auto white balance (plan 23.9). Returns (multipliers [R, G, B] or None, how, info).

    method: "grey_edge" or "grayness". mood: how far to correct from the as-shot balance, 0..1 (1 = fully
    neutral); None = the default for this photo (60% for a warm light in a low-light photo, else 90%).
    info: both estimates in kelvin, their agreement, the confidence, the mood used, the number of lights."""
    from ..color import autowb as aw
    est = aw.estimates(image)
    m = est["grayness"] if method == "grayness" and est["grayness"] is not None else est["grey_edge"]
    how = "grayness index" if (method == "grayness" and est["grayness"] is not None) else "grey edge"
    if m is None:
        m, how = aw.auto_multipliers(image, "grey_edge")          # grey world, the last resort
        if m is None:
            return None, how, {}
    cam_xyz = image.raw.cam_xyz
    m, pulled = aw.in_band(cam_xyz, m)
    T, _ = cam.temp_tint_from_multipliers(cam_xyz, m)
    k = aw.default_mood(T, image.meta.iso) if mood is None else float(np.clip(mood, 0.0, 1.0))
    as_shot = cam.normalise_wb(image.raw.wb_as_shot)
    out = aw.keep_mood(m, as_shot, k)
    info = {"mood": k, "confidence": est["confidence"], "agreement_deg": est["agreement_deg"],
            "lights": len(est["lights"]), "in_band": not pulled, "neutral_k": round(T)}
    for name in ("grey_edge", "grayness"):
        if est[name] is not None:
            info[name + "_k"] = round(cam.temp_tint_from_multipliers(cam_xyz, est[name])[0])
    if len(est["lights"]) > 1:
        info["lights_k"] = [round(cam.temp_tint_from_multipliers(cam_xyz, l)[0]) for l, _ in est["lights"]]
    return out, how, info


def wb_temp_tint(image: Image, mult) -> tuple[float, float]:
    """Nearest temperature (K) and tint for white-balance multipliers, so picked / auto WB can be shown (and then
    fine-tuned) on the Temperature and Tint sliders."""
    return cam.temp_tint_from_multipliers(image.raw.cam_xyz, mult)


def pick_white(image: Image, edit: Edit, u: float, v: float):
    """Multipliers that make the spot at (u, v) of the displayed photo neutral (the eyedropper)."""
    from ..color.autowb import patch_multipliers
    warp = make_warp(image, edit)
    xs, ys = warp.output_to_s(u * warp.out_w - 0.5, v * warp.out_h - 0.5)
    return patch_multipliers(image, float(xs), float(ys))


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """High-quality float resize (Lanczos per channel via Pillow 'F' mode)."""
    from PIL import Image as PI
    if (img.shape[1], img.shape[0]) == (w, h):
        return img
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        ch = PI.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32))
        out[..., c] = np.asarray(ch.resize((w, h), PI.Resampling.LANCZOS, reducing_gap=3.0))
    return out


# ------------------------------------------------------------------------------------------ masks (local/masks.py)
def view_to_s(image: Image, edit: Edit, u, v):
    """Positions on the displayed photo (fractions 0..1) -> mask coordinates (fractions of the S frame)."""
    warp = make_warp(image, edit)
    xs, ys = warp.output_to_s(np.asarray(u, float) * warp.out_w - 0.5, np.asarray(v, float) * warp.out_h - 0.5)
    _, _, sw, sh = image.raw.crop
    return np.asarray(xs, float) / sw, np.asarray(ys, float) / sh


def render_mask(image: Image, edit: Edit, index: int, scale: float = 0.25) -> np.ndarray:
    """Mask ``index`` of the edit as the displayed photo shows it (0..1, H x W), for the overlay."""
    _VIEW.what = ("mask", int(index))
    try:
        a = render(image, edit, scale=scale)
    finally:
        _VIEW.what = None
    return np.ascontiguousarray(a[..., 0])


def mask_features(image: Image, edit: Edit, u: float, v: float) -> list:
    """The auto-mask features (screen brightness, a/L, b/L) under a point of the displayed photo."""
    e = Edit.from_dict(edit.to_dict())
    e.masks = None
    _VIEW.what = ("features",)
    try:
        f = render_view(image, e, u, v, 9, 9)
    finally:
        _VIEW.what = None
    return [round(float(np.median(f[..., k])), 4) for k in range(3)]


def _unorient(img: np.ndarray, flip: int) -> np.ndarray:
    if flip == 5:
        return np.rot90(img, -1)
    if flip == 6:
        return np.rot90(img, 1)
    if flip == 3:
        return img[::-1, ::-1]
    return img


def object_mask(image: Image, edit: Edit, fg: list, bg: list | None = None, size: int = 480) -> dict:
    """'Click an object': a mask shape for the object under the foreground points (S-frame fractions), not
    reaching the background points. Computed on a small render of the whole photo, stored as a bitmap."""
    from ..lens.geometry import GeomParams  # noqa: F401  (geometry is reset below)
    from ..local import masks as mk
    from .edit import Geometry
    e = Edit.from_dict(edit.to_dict())
    e.masks = None
    e.geometry = Geometry()
    _, _, sw, sh = image.raw.crop
    sc = min(1.0, size / float(max(sw, sh)))
    _VIEW.what = ("features",)
    try:
        f = render(image, e, scale=sc)
    finally:
        _VIEW.what = None
    f = np.ascontiguousarray(_unorient(f, image.raw.flip))
    with slog.stage("masks.object", log, size=f"{f.shape[1]}x{f.shape[0]}", fg=len(fg), bg=len(bg or [])):
        bm = mk.object_bitmap(f, fg, bg or [])
    log.info("object mask: %.1f%% of the photo", 100.0 * (bm > 127).mean())
    return {"type": "object", "op": "add", "w": int(bm.shape[1]), "h": int(bm.shape[0]),
            "data": mk.encode_bitmap(bm), "fg": [list(map(float, p)) for p in fg],
            "bg": [list(map(float, p)) for p in (bg or [])]}


# ------------------------------------------------------------------------------------------ retouch (local/retouch.py)
def _grid_origin(image: Image, x: int, y: int, half: bool):
    """S position of the first pixel cell of linear_rgb(region starting at x, y): half-size pixels always sit on
    the raw file's even 2 x 2 cells."""
    if not half:
        return int(x), int(y)
    cx, cy, _, _ = image.raw.crop
    return (cx + int(x)) // 2 * 2 - cx, (cy + int(y)) // 2 * 2 - cy


def spot_patch(image: Image, edit: Edit, spot: dict, half: bool):
    """One retouch spot's patch (cached per spot, raw settings and size)."""
    from ..local import retouch as rt
    key = ("rt", edit.raw_key(), bool(half), json.dumps(spot, sort_keys=True))
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    cx, cy, sw, sh = image.raw.crop
    step = 2 if half else 1
    parity = (cx % 2, cy % 2) if half else (0, 0)

    def fetch(x0, y0, x1, y1):
        return linear_rgb(image, edit, (x0, y0, x1 - x0, y1 - y0), half_size=half, cache=False)
    res = rt.patch(spot, fetch, sw, sh, step, parity)
    with image._lock:
        old = [k for k in image._memo if k[0] == "rt"]
        if len(old) > 400:
            for k in old[:200]:
                image._memo.pop(k, None)
        image._memo[key] = res
    return res


def apply_retouch(image: Image, edit: Edit, base: np.ndarray, box, half: bool) -> np.ndarray:
    """Heal / clone / red-eye spots laid into a scene-linear block (box = its S-frame region, x and y used)."""
    from ..local import retouch as rt
    spots = rt.active(getattr(edit, "retouch", None))
    if not spots:
        return base
    _, _, sw, sh = image.raw.crop
    step = 2 if half else 1
    gx, gy = _grid_origin(image, box[0], box[1], half)
    sbox = (gx, gy, gx + base.shape[1] * step, gy + base.shape[0] * step)
    hits = [s for s in spots if rt.overlaps(s, sbox, sw, sh)]
    if not hits:
        return base
    out = base.copy()
    with slog.stage("retouch", log, n=len(hits), half=bool(half)):
        for s in hits:
            rt.paste(out, gx, gy, step, spot_patch(image, edit, s, half))
    return out


def s_to_view(image: Image, edit: Edit, xs, ys):
    """S-frame fractions (masks, spots) -> positions on the displayed photo (fractions 0..1)."""
    warp = make_warp(image, edit)
    _, _, sw, sh = image.raw.crop
    X, Y = warp.s_to_output(np.asarray(xs, float) * sw, np.asarray(ys, float) * sh)
    return (np.asarray(X) + 0.5) / warp.out_w, (np.asarray(Y) + 0.5) / warp.out_h


def view_window(image: Image, edit: Edit, u: float, v: float, w: int, h: int):
    """The output window (x, y, w, h) that render_view(u, v, w, h) shows."""
    warp = make_warp(image, edit)
    ow, oh = warp.output_size
    w, h = min(int(w), ow), min(int(h), oh)
    x = int(min(max(round(u * ow - w / 2.0), 0), ow - w))
    y = int(min(max(round(v * oh - h / 2.0), 0), oh - h))
    return x, y, w, h


def _spot_avoid(spots, sw, sh):
    from ..local import retouch as rt
    out = []
    for s in spots or []:
        try:
            shp = rt.Shape(s, sw, sh)
        except (KeyError, ValueError, TypeError):
            continue
        out += [(float(x), float(y), shp.R) for x, y in zip(shp.px, shp.py)]
        out += [(float(x + shp.ox), float(y + shp.oy), shp.R) for x, y in zip(shp.px, shp.py)]
    return out


def auto_source(image: Image, edit: Edit, spot: dict, others: list | None = None) -> list:
    """Where a heal or clone spot should take its source from (offset, S fractions)."""
    from ..local import retouch as rt
    _, _, sw, sh = image.raw.crop
    x0, y0, x1, y1 = rt.neighbourhood_box(spot, sw, sh)
    half = (x1 - x0) * (y1 - y0) > 2.5e6
    gx, gy = _grid_origin(image, x0, y0, half)
    with slog.stage("retouch.auto_source", log, half=half):
        img = linear_rgb(image, edit, (x0, y0, x1 - x0, y1 - y0), half_size=half, cache=False)
        return rt.best_source(img, gx, gy, 2 if half else 1, spot, sw, sh, _spot_avoid(others, sw, sh))


def _dust_base(image: Image, edit: Edit):
    base = linear_rgb(image, edit, None, half_size=True, cache=False)
    gx, gy = _grid_origin(image, 0, 0, True)
    return base, gx, gy


def _with_sources(image, spots, base, gx, gy, others):
    """Choose each found spot's source on the half-size frame already in memory (a window around each spot)."""
    from ..local import retouch as rt
    _, _, sw, sh = image.raw.crop
    placed = list(others or [])
    h, w = base.shape[:2]
    for s in spots:
        x0, y0, x1, y1 = rt.neighbourhood_box(s, sw, sh)
        j0, i0 = max(0, (x0 - gx) // 2), max(0, (y0 - gy) // 2)
        j1, i1 = min(w, (x1 - gx) // 2 + 1), min(h, (y1 - gy) // 2 + 1)
        win = base[i0:i1, j0:j1]
        s["src"] = rt.best_source(win, gx + 2 * j0, gy + 2 * i0, 2, s, sw, sh, _spot_avoid(placed + spots, sw, sh))
        placed.append(s)
    return spots


def find_dust(image: Image, edit: Edit, limit: int = 60) -> list:
    """Dust-like spots found in this photo alone (dark, round, soft blobs in smooth areas), as heal spots with
    their sources chosen. Spots already in the edit are skipped. Review them: a bird in the sky looks the same."""
    from ..local import dust as du
    from ..local import retouch as rt
    _, _, sw, sh = image.raw.crop
    long = float(max(sw, sh))
    with slog.stage("dust.find", log):
        base, gx, gy = _dust_base(image, edit)
        blobs = du.find_blobs(base @ du.LUMA, 2, gx, gy, r_min=max(3.0, 0.0006 * long), r_max=max(24.0, 0.009 * long),
                              limit=limit * 2)
        have = rt.active(edit.retouch)
        av = _spot_avoid(have, sw, sh)
        blobs = [b for b in blobs if all(math.hypot(b["x"] - x, b["y"] - y) > r + b["r"] for x, y, r in av)][:limit]
        spots = _with_sources(image, du.blobs_to_spots(blobs, sw, sh, "dust"), base, gx, gy, have)
    log.info("find dust: %d spot(s)", len(spots))
    return spots


def dust_from_map(image: Image, edit: Edit) -> list | None:
    """Heal spots for this camera's dust map, found again in this photo (only spots the photo actually shows).
    None when the camera has no dust map yet."""
    from ..local import dust as du
    from ..local import retouch as rt
    md = image.meta
    m = du.load_map(getattr(md, "make", "") or "", getattr(md, "model", "") or "")
    if m is None:
        return None
    _, _, sw, sh = image.raw.crop
    long = float(max(sw, sh))
    with slog.stage("dust.from_map", log, spots=len(m.get("spots", []))):
        base, gx, gy = _dust_base(image, edit)
        blobs = du.find_blobs(base @ du.LUMA, 2, gx, gy, r_min=max(3.0, 0.0005 * long), r_max=max(32.0, 0.02 * long),
                              min_contrast=0.03, k_texture=2.5, limit=600)
        found = du.match_map(m, blobs, sw, sh, getattr(md, "fnumber", None))
        have = rt.active(edit.retouch)
        av = _spot_avoid(have, sw, sh)
        found = [b for b in found if all(math.hypot(b["x"] - x, b["y"] - y) > r for x, y, r in av)]
        spots = _with_sources(image, du.blobs_to_spots(found, sw, sh, "map"), base, gx, gy, have)
    log.info("dust map: %d of %d spot(s) visible in this photo", len(spots), len(m.get("spots", [])))
    return spots


def learn_dust_map(paths: list, progress=None) -> dict:
    """Build this camera's dust map from several photos of different scenes (sky or a plain wall at f/11 or
    smaller is ideal). Only blobs that recur in the same place on the sensor are kept. Saved privately in
    calibration_local; returns a report."""
    from ..local import dust as du
    per, used, skipped, fns, thumbs = [], [], [], [], []
    camera = sw = sh = None
    for k, p in enumerate(paths):
        if progress:
            progress(k, len(paths), os.path.basename(p))
        try:
            img = open_image(p)
        except Exception as err:                                  # noqa: BLE001  (unreadable files are skipped)
            skipped.append((p, f"unreadable ({err})"))
            continue
        md = img.meta
        cam = (getattr(md, "make", "") or "", getattr(md, "model", "") or "")
        _, _, w, h = img.raw.crop
        if camera is None:
            camera, sw, sh = cam, w, h
        elif cam != camera or (w, h) != (sw, sh):
            skipped.append((p, "another camera"))
            continue
        base, gx, gy = _dust_base(img, Edit())
        Y = base @ du.LUMA
        # the same scene twice (a focus stack, a burst) would make its own dark details look like dust: compare
        # the photos' detail (not their overall light, so frames of plain sky or a wall all count)
        from scipy.ndimage import gaussian_filter as _gf
        th = resize(np.log2(np.maximum(Y, 0) + 1e-3)[..., None].repeat(3, axis=2), 96, 64)[..., 0]
        th = (th - _gf(th, 3.0)).ravel()
        th = (th - th.mean()) / (th.std() + 1e-6)
        if any(float(np.mean(th * t)) > 0.8 for t in thumbs):
            skipped.append((p, "same scene as another photo"))
            continue
        thumbs.append(th)
        long = float(max(sw, sh))
        blobs = du.find_blobs(Y, 2, gx, gy, r_min=max(3.0, 0.0005 * long), r_max=max(32.0, 0.02 * long),
                              min_contrast=0.05, k_texture=4.0, limit=200)
        per.append([{"x": b["x"] / sw, "y": b["y"] / sh, "size": b["r"] / long, "contrast": b["contrast"]}
                    for b in blobs])
        used.append(p)
        if getattr(md, "fnumber", None):
            fns.append(float(md.fnumber))
        img.drop_cache()
    if progress:
        progress(len(paths), len(paths), "")
    if len(per) < 3:
        return {"ok": False, "reason": f"need at least 3 different scenes from one camera (got {len(per)})",
                "used": used, "skipped": skipped}
    min_seen = max(3, int(math.ceil(0.35 * len(per))))       # three chance matches in one place are rare
    spots = du.cluster(per, sw, sh, min_seen)
    name = " ".join(x for x in camera if x).strip() or "camera"
    m = du.new_map(name, spots, len(per), float(np.median(fns)) if fns else None)
    path = du.save_map(camera[0], camera[1], m)
    log.info("dust map for %s: %d spot(s) from %d photos (seen in >= %d)", name, len(spots), len(per), min_seen)
    return {"ok": True, "camera": name, "spots": len(spots), "photos": len(per), "min_seen": min_seen,
            "used": used, "skipped": skipped, "path": path}


# ------------------------------------------------------------------------------------------ lens glow (raw/glow.py)
def glow_report(image: Image, edit: Edit) -> list:
    """The blown lights of this photo and how each was recovered (for the interface), at the edit's settings."""
    mult = wb_multipliers(image, edit)
    _, report = image.glow_maps(mult, edit.use_optical_black, edit.glow_profile)
    return report


def measure_glow_profile(paths: list, lens: str = "", progress=None) -> dict:
    from ..raw import glow as gl
    return gl.measure_profile(paths, lens, progress)


def glow_profiles() -> list:
    """[{"name", "lens", "camera", "fnumber", "made", "frames", "stops"}] stored on this PC."""
    from ..raw import glow as gl
    return [{"name": p.name, "lens": p.lens, "camera": p.camera, "fnumber": p.fnumber, "made": p.made,
             "frames": p.frames, "stops": p.stops} for p in gl.list_profiles()]


def pick_glow_profile(image: Image, edit: Edit) -> str:
    """The stored glow profile for this photo's lens (the lens picked by hand first), nearest aperture."""
    from ..raw import glow as gl
    md = image.meta
    lens = edit.lens.tag or getattr(md, "lens", "") or ""
    return gl.pick_profile(lens, getattr(md, "fnumber", None))


# ------------------------------------------------------------------------------------------ portrait (local/portrait.py)
def _face_copy(image: Image, edit: Edit, box, half: bool, k: int = 1):
    """A retouched, scene-linear copy of an S box (x0, y0, x1, y1): (rgb, gx, gy, q) with pixel (0, 0) centred at S
    (gx + (q - 1) / 2, gy + (q - 1) / 2), pixels q apart."""
    x0, y0, x1, y1 = box
    rgb = linear_rgb(image, edit, (x0, y0, x1 - x0, y1 - y0), half_size=half, cache=False)
    rgb = apply_retouch(image, edit, rgb, (x0, y0), half)
    gx, gy = _grid_origin(image, x0, y0, half)
    if k > 1:
        from ..local.portrait import _down
        rgb = _down(rgb, k)
    return rgb, gx, gy, (2 if half else 1) * k


def face_gain(image: Image, edit: Edit, index: int):
    """One face's gain map (G, gx, gy, q), worked out once per face and settings and kept on the image."""
    from ..local import portrait as pt
    from ..local import retouch as rt
    faces = pt.faces_of(edit.portrait)
    if not (0 <= index < len(faces)):
        return None
    _, _, sw, sh = image.raw.crop
    fc = pt.Face(faces[index], sw, sh)
    box, half, k = pt.fetch_plan(fc, sw, sh)
    amt = pt.amounts(edit.portrait)
    hits = [s for s in rt.active(edit.retouch) if rt.overlaps(s, box, sw, sh)]
    key = ("pt", edit.raw_key(), json.dumps(hits, sort_keys=True), json.dumps(faces[index]["pts"]),
           json.dumps({a: round(v, 4) for a, v in amt.items() if a != "jawline"}, sort_keys=True),
           edit.stage_versions.get("portrait", 1))
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    with slog.stage("portrait.face", log, face=index, width=round(fc.W), half=half, k=k):
        rgb, gx, gy, q = _face_copy(image, edit, box, half, k)
        G = pt.face_gain(rgb, gx, gy, q, fc, amt)
    res = (G, gx, gy, q)
    with image._lock:
        old = [kk for kk in image._memo if kk[0] == "pt"]
        if len(old) > 12:
            for kk in old[:6]:
                image._memo.pop(kk, None)
        image._memo[key] = res
    return res


def apply_portrait(image: Image, edit: Edit, base: np.ndarray, box, half: bool) -> np.ndarray:
    """The face sliders laid into a scene-linear block (box = its S-frame region, x and y used)."""
    from ..local import portrait as pt
    if not pt.face_active(edit.portrait):
        return base
    _, _, sw, sh = image.raw.crop
    step = 2 if half else 1
    gx, gy = _grid_origin(image, box[0], box[1], half)
    sbox = (gx, gy, gx + base.shape[1] * step, gy + base.shape[0] * step)
    out = None
    for i, f in enumerate(pt.faces_of(edit.portrait)):
        fc = pt.Face(f, sw, sh)
        x0, y0, x1, y1 = fc.window(0.6, sw, sh)
        if x0 >= sbox[2] or x1 <= sbox[0] or y0 >= sbox[3] or y1 <= sbox[1]:
            continue
        g = face_gain(image, edit, i)
        if g is None:
            continue
        if out is None:
            out = base.copy()
        pt.paste_gain(out, gx, gy, step, g)
    return base if out is None else out


def chin_skin(image: Image, edit: Edit, index: int):
    """Where the skin is round one face (portrait.chin_skin), for the double-chin warp: (M, gx, gy, q), worked out
    once per face and kept on the image, so every region render warps the same way."""
    from ..local import portrait as pt
    from ..local import retouch as rt
    faces = pt.faces_of(edit.portrait)
    if not (0 <= index < len(faces)):
        return None
    _, _, sw, sh = image.raw.crop
    fc = pt.Face(faces[index], sw, sh)
    box, half, k = pt.fetch_plan(fc, sw, sh)
    hits = [s for s in rt.active(edit.retouch) if rt.overlaps(s, box, sw, sh)]
    key = ("chinskin", edit.raw_key(), json.dumps(hits, sort_keys=True), json.dumps(faces[index]["pts"]),
           edit.stage_versions.get("portrait", 1))
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    with slog.stage("portrait.chin_skin", log, face=index, width=round(fc.W)):
        rgb, gx, gy, q = _face_copy(image, edit, box, half, k)
        res = (pt.chin_skin(rgb, gx, gy, q, fc), gx, gy, q)
    with image._lock:
        old = [kk for kk in image._memo if kk[0] == "chinskin"]
        for kk in old[:-6]:
            image._memo.pop(kk, None)
        image._memo[key] = res
    return res


def portrait_warps(image: Image, edit: Edit) -> list:
    """The edit's warps (portrait.warps), with the skin maps the double-chin warp needs."""
    from ..local import portrait as pt
    p = edit.portrait
    if not p:
        return []
    _, _, sw, sh = image.raw.crop
    skin = None
    if pt.amounts(p).get("double_chin", 0.0) > 1e-4:
        skin = [chin_skin(image, edit, i) for i in range(len(pt.faces_of(p)))]
    return pt.warps(p, sw, sh, skin=skin)


def apply_sculpt(image: Image, edit: Edit, base: np.ndarray, box, half: bool) -> np.ndarray:
    """Body sculpting, the jawline and the double chin: the block warped (box = its S-frame region, x and y
    used)."""
    from ..local import portrait as pt
    p = edit.portrait
    if not p:
        return base
    ws = portrait_warps(image, edit)
    if not ws:
        return base
    step = 2 if half else 1
    gx, gy = _grid_origin(image, box[0], box[1], half)
    with slog.stage("portrait.sculpt", log, warps=len(ws), half=bool(half)):
        return pt.apply_warps(base, gx, gy, step, ws, cubic=not half)


def sculpt_source(image: Image, edit: Edit, xs, ys):
    """S fractions as shown (after sculpting) -> where those pixels come from (before sculpting). Spots and faces
    placed by clicking on a sculpted photo go through this, so they land on what was clicked."""
    from ..local import portrait as pt
    _, _, sw, sh = image.raw.crop
    ws = portrait_warps(image, edit)
    if not ws:
        return np.asarray(xs, float), np.asarray(ys, float)
    X, Y = pt.source_of(ws, np.asarray(xs, float) * sw, np.asarray(ys, float) * sh)
    return X / sw, Y / sh


def _face_view(image: Image, edit: Edit):
    """The photo as the face finder sees it: upright, no straightening or crop, no portrait work, display-encoded
    sRGB 0..1, long edge up to 1600 px; plus the scale (render pixels per full-size pixel). Kept on the image."""
    from ..io.export import ExportSpec, to_output
    from .edit import Geometry
    e = Edit.from_dict(edit.to_dict())
    e.portrait = None
    e.geometry = Geometry()
    e.masks = None
    e.finish.grain = 0.0
    key = ("faceview", e.to_json())
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    _, _, sw, sh = image.raw.crop
    sc = min(1.0, 1600.0 / max(sw, sh))
    img = render(image, e, scale=sc)
    rgb = to_output(img, ExportSpec(format="jpeg", space="srgb", bits=8)).astype(np.float32) / 255.0
    dw = sh if image.raw.flip in (5, 6) else sw
    res = (np.ascontiguousarray(rgb), rgb.shape[1] / float(dw))
    with image._lock:
        image._memo = {k: v for k, v in image._memo.items() if k[0] != "faceview"}
        image._memo[key] = res
    return res


def _view_pts_to_s(image: Image, P: np.ndarray, k: float):
    """Face-view image coordinates (pixel i covers i..i+1) -> S fractions."""
    from ..lens.geometry import d_to_s
    _, _, sw, sh = image.raw.crop
    xd = P[:, 0] / k - 0.5
    yd = P[:, 1] / k - 0.5
    xs, ys = d_to_s(xd, yd, sw, sh, image.raw.flip)
    return np.stack([np.asarray(xs, float) / sw, np.asarray(ys, float) / sh], -1)


def _s_to_view_pts(image: Image, xs, ys, k: float):
    from ..lens.geometry import s_to_d
    _, _, sw, sh = image.raw.crop
    xd, yd = s_to_d(np.asarray(xs, float) * sw, np.asarray(ys, float) * sh, sw, sh, image.raw.flip)
    return np.stack([(np.asarray(xd) + 0.5) * k, (np.asarray(yd) + 0.5) * k], -1)


def _face_entry(image: Image, f: dict, k: float, how: str) -> dict:
    S = _view_pts_to_s(image, f["pts"], k)
    return {"pts": [[round(float(x), 6), round(float(y), 6)] for x, y in S], "conf": round(float(f["conf"]), 3),
            "how": how, "enabled": True}


def find_faces(image: Image, edit: Edit) -> list:
    """Every face in the photo, found on this PC: [{"pts": 468 S-fraction points, "conf", "how": "found"}]."""
    from ..local import faces as fz
    with slog.stage("faces.find", log):
        img, k = _face_view(image, edit)
        found = fz.find(img)
    log.info("faces: %d found (%s)", len(found), ", ".join(f"{f['conf']:.2f} via {f['variant']}" for f in found))
    return [_face_entry(image, f, k, "found") for f in found]


def face_from_eyes(image: Image, edit: Edit, u1: float, v1: float, u2: float, v2: float) -> dict | None:
    """A face from two clicks on the eyes (positions on the displayed photo, either order). None when the face
    mesh cannot make a face of it."""
    from ..local import faces as fz
    xs, ys = view_to_s(image, edit, [u1, u2], [v1, v2])
    xs, ys = sculpt_source(image, edit, xs, ys)
    img, k = _face_view(image, edit)
    E = _s_to_view_pts(image, xs, ys, k)
    best = None
    for a, b in ((E[0], E[1]), (E[1], E[0])):          # the person's right eye is usually on the left: try both
        f = fz.from_eyes(img, a, b)
        if best is None or f["conf"] > best["conf"]:
            best = f
    log.info("face from eyes: mesh confidence %.2f", best["conf"])
    if best["conf"] < 0.2:
        return None
    return _face_entry(image, best, k, "eyes")


def face_outlines(image: Image, edit: Edit) -> list:
    """Each face's outline, eyes, brows and lips as polylines on the displayed photo (fractions 0..1)."""
    from ..local import faces as fz
    out = []
    for f in (edit.portrait or {}).get("faces") or []:
        P = np.asarray(f.get("pts") or [], float)
        if P.shape != (468, 2):
            continue
        lines = []
        for idx in (fz.FACE_OVAL, fz.RIGHT_EYE, fz.LEFT_EYE, fz.RIGHT_BROW, fz.LEFT_BROW, fz.LIPS_OUTER):
            u, v = s_to_view(image, edit, P[idx + idx[:1], 0], P[idx + idx[:1], 1])
            lines.append([(float(a), float(b)) for a, b in zip(u, v)])
        out.append({"lines": lines, "how": f.get("how", "found"), "enabled": f.get("enabled", True),
                    "conf": f.get("conf", 0)})
    return out


def find_blemishes(image: Image, edit: Edit, limit: int = 60) -> dict:
    """Blemishes on every face, as heal spots with their sources chosen (auto = "blemish"). Spots already in the
    edit are skipped. Returns {"spots": [...], "moles": n left alone, "faces": n}."""
    from ..local import dust as du
    from ..local import portrait as pt
    from ..local import retouch as rt
    _, _, sw, sh = image.raw.crop
    faces = pt.faces_of(edit.portrait)
    have = rt.active(edit.retouch)
    placed = list(have)
    spots, moles = [], 0
    for f in faces:
        fc = pt.Face(f, sw, sh)
        half = fc.W > 1600
        box = fc.window(0.25, sw, sh)
        with slog.stage("portrait.blemishes", log, width=round(fc.W), half=half):
            rgb, gx, gy, q = _face_copy(image, edit, box, half)
            blobs, nm = pt.find_blemishes(rgb, gx, gy, q, fc, limit)
        moles += nm
        av = _spot_avoid(have, sw, sh)
        blobs = [b for b in blobs if all(math.hypot(b["x"] - x, b["y"] - y) > r + b["r"] for x, y, r in av)]
        new = du.blobs_to_spots(blobs, sw, sh, "blemish")
        for s in new:
            s["size"] = round(max(1.7 * float(s["size"]) / 1.4, float(s["size"])), 6)
        feats = pt.feature_avoid(fc)
        z = pt.zones(fc)
        for s in new:
            cx, cy = s["pts"][0][0] * sw, s["pts"][0][1] * sh

            def pen(ox, oy, cx=cx, cy=cy):
                v = float(pt.sample_zone(z, "blem", np.array([[cx + ox]]), np.array([[cy + oy]]))[0, 0])
                return 0.0 if v > 0.5 else 50.0
            s["src"] = rt.best_source(rgb, gx, gy, q, s, sw, sh, _spot_avoid(placed, sw, sh) + feats, penalty=pen)
            placed.append(s)
        spots += new
    log.info("blemishes: %d spot(s) on %d face(s); %d probable mole(s) left alone", len(spots), len(faces), moles)
    return {"spots": spots, "moles": moles, "faces": len(faces)}


def shape_up(image: Image, edit: Edit, u: float, v: float) -> list:
    """The displayed photo's 'up' at a point, as a unit vector in the S frame (for Lift)."""
    xs, ys = view_to_s(image, edit, [u, u], [v, v - 0.01])
    _, _, sw, sh = image.raw.crop
    d = np.array([(xs[1] - xs[0]) * sw, (ys[1] - ys[0]) * sh])
    n = float(np.hypot(*d)) or 1.0
    return [round(float(d[0] / n), 6), round(float(d[1] / n), 6)]


# ------------------------------------------------------------------------------------------ people (local/people.py)
def find_people(image: Image, edit: Edit, size: int = 640) -> dict | None:
    """'Select a person': a Person mask shape for everyone in the photo (person mask on a small grid of the S frame,
    33 body points per person, the face outlines from Find faces). None when nobody is found."""
    from ..local import faces as fz
    from ..local import masks as mk
    from ..local import people as pp
    img, k = _face_view(image, edit)
    key = ("people", id(img))
    with image._lock:
        res = image._memo.get(key)
    if res is None:
        face_pts = []
        for f in (edit.portrait or {}).get("faces") or []:
            P = np.asarray(f.get("pts") or [], float)
            if P.shape == (468, 2):
                face_pts.append(_s_to_view_pts(image, P[:, 0], P[:, 1], k))
        with slog.stage("people.find", log):
            res = pp.find(img, face_pts)
        with image._lock:
            image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "people"}
            image._memo[key] = res
    if not res["people"] and res["mask"].max() < 0.5:
        log.info("people: nobody found")
        return None
    from ..lens.geometry import s_to_d
    _, _, sw, sh = image.raw.crop
    sc = min(1.0, size / float(max(sw, sh)))
    bw, bh = max(8, round(sw * sc)), max(8, round(sh * sc))
    X, Y = np.meshgrid((np.arange(bw) + 0.5) / bw * sw - 0.5, (np.arange(bh) + 0.5) / bh * sh - 0.5)
    xd, yd = s_to_d(X, Y, sw, sh, image.raw.flip)
    from scipy.ndimage import map_coordinates
    m = map_coordinates(res["mask"], [(np.asarray(yd) + 0.5) * k - 0.5, (np.asarray(xd) + 0.5) * k - 0.5], order=1,
                        mode="nearest")
    people = []
    for p in res["people"]:
        S = _view_pts_to_s(image, p["pts"][:, :2], k)
        people.append([[round(float(x), 6), round(float(y), 6), round(float(v), 3)]
                       for (x, y), v in zip(S, p["pts"][:, 2])])
    faces = []
    for f in (edit.portrait or {}).get("faces") or []:
        P = np.asarray(f.get("pts") or [], float)
        if P.shape == (468, 2):
            faces.append([[round(float(x), 6), round(float(y), 6)] for x, y in P[fz.FACE_OVAL]])
    log.info("people: %d person(s), person mask %.0f%% of the photo", len(people), 100.0 * float((m > 0.5).mean()))
    shape = {"type": "person", "op": "add", "part": "person", "w": int(bw), "h": int(bh),
             "data": mk.encode_bitmap(np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)), "people": people,
             "faces": faces}
    if "classes" in res:                                # the parts model's labels, on a smaller grid (it sees 256 px)
        from ..local.people import CLASSES
        sc2 = min(1.0, 384.0 / float(max(sw, sh)))
        cw, ch = max(8, round(sw * sc2)), max(8, round(sh * sc2))
        X2, Y2 = np.meshgrid((np.arange(cw) + 0.5) / cw * sw - 0.5, (np.arange(ch) + 0.5) / ch * sh - 0.5)
        xd2, yd2 = s_to_d(X2, Y2, sw, sh, image.raw.flip)
        fy, fx = (np.asarray(yd2) + 0.5) * k - 0.5, (np.asarray(xd2) + 0.5) * k - 0.5
        shape["cw"], shape["ch"], shape["classes"] = int(cw), int(ch), {}
        for i, name in enumerate(CLASSES):
            if name in ("background", "others"):
                continue
            v = map_coordinates(res["classes"][..., i], [fy, fx], order=1, mode="nearest")
            shape["classes"][name] = mk.encode_bitmap(np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8))
        log.info("people: parts labelled by the multiclass model")
    from ..local import people as pp2
    if pp2.hair_available() and res["people"]:
        shape["hair_maps"] = _hair_maps(image, edit, res["people"], img.shape[1], img.shape[0], k)
    return shape


def _hair_maps(image: Image, edit: Edit, people: list, vw: int, vh: int, k: float) -> list:
    """The hair model on a sharper render of each head (a crop of the photo, about 768 px across): its hair map on a
    grid of the S frame over that head."""
    from ..io.export import ExportSpec, to_output
    from ..lens.geometry import d_to_s, s_to_d
    from ..local import masks as mk
    from ..local import people as pp
    from .edit import Geometry
    from scipy.ndimage import map_coordinates
    _, _, sw, sh = image.raw.crop
    e = Edit.from_dict(edit.to_dict())
    e.portrait, e.masks, e.geometry = None, None, Geometry()
    e.finish.grain = 0.0
    out = []
    for p in people:
        x0, y0, x1, y1 = pp.head_box(p["pts"], vw, vh)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        # the head's box in the S frame (full-resolution pixels), then a render of just that part
        cx = np.array([x0, x1, x0, x1]) / k - 0.5
        cy = np.array([y0, y0, y1, y1]) / k - 0.5
        xs, ys = d_to_s(cx, cy, sw, sh, image.raw.flip)
        sx0, sy0 = max(0, int(np.floor(np.min(xs)))), max(0, int(np.floor(np.min(ys))))
        sx1, sy1 = min(sw, int(np.ceil(np.max(xs))) + 1), min(sh, int(np.ceil(np.max(ys))) + 1)
        if sx1 - sx0 < 8 or sy1 - sy0 < 8:
            continue
        sc = min(1.0, 768.0 / max(sx1 - sx0, sy1 - sy0))
        with slog.stage("people.hair", log, box=f"{sx1 - sx0}x{sy1 - sy0}", scale=round(sc, 3)):
            crop = render(image, e, region=(sx0, sy0, sx1 - sx0, sy1 - sy0), scale=sc)
            rgb = to_output(crop, ExportSpec(format="jpeg", space="srgb", bits=8)).astype(np.float32) / 255.0
            prob = pp.hair_prob(rgb)
        # back to the S frame: a grid over the box, each cell looked up in the (upright) crop
        gw = max(8, min(512, int(round((sx1 - sx0) * sc))))
        gh = max(8, min(512, int(round((sy1 - sy0) * sc))))
        GX, GY = np.meshgrid(sx0 + (np.arange(gw) + 0.5) / gw * (sx1 - sx0) - 0.5,
                             sy0 + (np.arange(gh) + 0.5) / gh * (sy1 - sy0) - 0.5)
        dx, dy = s_to_d(GX, GY, sw, sh, image.raw.flip)
        # the crop's own display-frame origin: its corner is the box corner seen upright
        cxs, cys = s_to_d(np.array([sx0, sx1 - 1.0]), np.array([sy0, sy1 - 1.0]), sw, sh, image.raw.flip)
        ox, oy = float(np.min(cxs)), float(np.min(cys))
        kx = prob.shape[1] / float(np.max(cxs) - ox + 1.0)
        ky = prob.shape[0] / float(np.max(cys) - oy + 1.0)
        v = map_coordinates(prob, [(np.asarray(dy) - oy + 0.5) * ky - 0.5, (np.asarray(dx) - ox + 0.5) * kx - 0.5],
                            order=1, mode="constant", cval=0.0)
        out.append({"box": [round(sx0 / sw, 6), round(sy0 / sh, 6), round(sx1 / sw, 6), round(sy1 / sh, 6)],
                    "w": int(gw), "h": int(gh),
                    "data": mk.encode_bitmap(np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8))})
    log.info("people: hair model on %d head(s)", len(out))
    return out


# ------------------------------------------------------------------------------------------ composition
def composition_subject(image: Image, edit: Edit) -> dict:
    """What a crop should be built round, on the displayed photo before cropping (fractions 0..1): faces (from the
    edit, else found now), people (33 body points), else the part that stands out and is sharp. See
    local/composition.py for the keys."""
    from ..local import faces as fz
    from ..local import composition as cp
    e0 = Edit.from_dict(edit.to_dict())
    e0.geometry.crop = None
    faces = [f for f in ((e0.portrait or {}).get("faces") or []) if len(f.get("pts") or []) == 468]
    if not faces:
        try:
            faces = find_faces(image, e0)
        except Exception:                                    # noqa: BLE001  (no face finder: carry on without)
            log.warning("composition: the face finder failed", exc_info=True)
            faces = []
    ow, oh = output_size(image, e0)
    sub = {"keep": [], "others": [], "joints": [], "necks": []}
    fv = []
    for f in faces:
        P = np.asarray(f["pts"], float)
        u, v = s_to_view(image, e0, P[:, 0], P[:, 1])
        Q = np.stack([np.asarray(u) * ow, np.asarray(v) * oh], 1)          # display pixels, so angles are true
        er, el = fz.eye_centres(Q)
        oval = Q[fz.FACE_OVAL]
        x0, y0 = oval.min(0)
        x1, y1 = oval.max(0)
        fh = y1 - y0
        cx = 0.5 * (x0 + x1)
        gaze = float(np.clip((Q[fz.NOSE_TIP][0] - cx) / max(0.5 * (x1 - x0), 1.0) * 2.5, -1.0, 1.0))
        top = y0 - 0.3 * fh                                                  # hair above the hairline
        box = ((x0 - 0.08 * (x1 - x0)) / ow, top / oh, (x1 + 0.08 * (x1 - x0)) / ow, (y1 + 0.05 * fh) / oh)
        fv.append({"eyes": (0.5 * (er[0] + el[0]) / ow, 0.5 * (er[1] + el[1]) / oh), "box": box, "size": fh / oh,
                   "gaze": gaze, "top": top / oh, "chin": (Q[fz.CHIN][0] / ow, Q[fz.CHIN][1] / oh)})
    people = []
    try:
        shape = find_people(image, e0)
    except Exception:                                        # noqa: BLE001
        log.warning("composition: the person finder failed", exc_info=True)
        shape = None
    for pts in (shape or {}).get("people") or []:
        A = np.asarray(pts, float)
        u, v = s_to_view(image, e0, A[:, 0], A[:, 1])
        vis = A[:, 2] > 0.5
        people.append({"u": np.asarray(u), "v": np.asarray(v), "vis": vis})
    for pl in people:
        for j in cp.JOINTS:
            if pl["vis"][j] and 0.0 < pl["u"][j] < 1.0 and 0.0 < pl["v"][j] < 1.0:
                sub["joints"].append((float(pl["u"][j]), float(pl["v"][j])))
        if pl["vis"][[9, 10, 11, 12]].all():
            sub["necks"].append((float(0.5 * (pl["u"][[9, 10]].mean() + pl["u"][[11, 12]].mean())),
                                 float(0.45 * pl["v"][[9, 10]].mean() + 0.55 * pl["v"][[11, 12]].mean())))
    if fv:
        big = max(f["size"] for f in fv)
        main_faces = [f for f in fv if f["size"] > 0.6 * big]
        sub["main"] = tuple(np.mean([f["eyes"] for f in main_faces], axis=0))
        sub["keep"] = [f["box"] for f in main_faces]
        sub["others"] = [f["box"] for f in fv if f not in main_faces]
        sub["head_top"] = min(f["top"] for f in main_faces)
        sub["gaze"] = float(np.mean([f["gaze"] for f in main_faces])) if len(main_faces) == 1 else 0.0
        sub["portrait"] = big > 0.08
        sub["what"] = "the eyes" if len(main_faces) == 1 else "the faces"
        xs = [b[0] for b in sub["keep"]] + [b[2] for b in sub["keep"]]
        ys = [b[1] for b in sub["keep"]] + [b[3] for b in sub["keep"]]
        sub["box"] = (min(xs), min(ys), max(xs), max(ys))
    elif people:
        pl = max(people, key=lambda q: float(np.ptp(q["v"][q["vis"]])) if q["vis"].any() else 0.0)
        uu, vv = pl["u"][pl["vis"]], pl["v"][pl["vis"]]
        sub["main"] = (float(pl["u"][0]), float(pl["v"][0])) if pl["vis"][0] else (float(uu.mean()), float(vv.mean()))
        sub["box"] = (float(uu.min()), float(vv.min()), float(uu.max()), float(vv.max()))
        sub["what"] = "the person"
    else:
        from ..io.export import ExportSpec, to_output
        sc = min(1.0, 384.0 / max(ow, oh))
        small = to_output(render(image, e0, scale=sc), ExportSpec(format="jpeg", space="srgb", bits=8))
        sub.update(cp.salient_subject(small))
    log.info("composition: %s at (%.2f, %.2f); %d face(s), %d person(s), %d joint(s)", sub.get("what"),
             sub["main"][0], sub["main"][1], len(fv), len(people), len(sub["joints"]))
    return sub


def suggest_crops(image: Image, edit: Edit, ratio: float | None = None, method: str | None = None, n: int = 3):
    """Suggested crops for the displayed photo (fractions of the uncropped frame, like Geometry.crop), best first:
    [{"crop", "method", "orient", "score", "text"}]. ``ratio``: the crop's width / height (None: the photo's own);
    ``method``: a composition method (local/composition.py METHODS) or None for any."""
    from ..local import composition as cp
    e0 = Edit.from_dict(edit.to_dict())
    e0.geometry.crop = None
    ow, oh = output_size(image, e0)
    with slog.stage("composition.suggest", log, method=method or "any"):
        sub = composition_subject(image, e0)
        return cp.suggest(sub, ow / oh, float(ratio or ow / oh), method, n)


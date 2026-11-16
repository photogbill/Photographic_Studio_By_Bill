"""The develop pipeline (section 3): stages 1-3, 5, 7, 8, 10-16 and 21 so far.

render(image, edit, region, scale)   -> display-referred linear ProPhoto float32 (HxWx3), oriented upright
render_view(image, edit, u, v, w, h) -> a w x h window of the finished photo at 100%
render_window(image, edit, window, scale) -> any window of the finished photo at any scale (Develop's zoom)
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
        # stages 1-12, kept for the last whole frame and the last region separately: Develop zoomed renders both
        # (the part on screen, then the whole-photo preview), and one slot made each evict the other, so every
        # tone-slider change redid the denoise on both
        self._lin_cache: dict = {}
        self._lock = threading.Lock()
        self._jpeg = None
        self._noise = {}
        self._rows = None
        self._ca = None
        self._blur = {}
        self._warp = None
        self._memo: dict = {}
        self.denoise_report: dict | None = None    # last whole-frame denoise: photosites kept by the safeguard
        self.scratch_report: dict | None = None    # last whole-frame dust and scratch pass: marks found
        self.colour_report: dict | None = None     # the colourizer's last run (time, map size)

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
        if use_optical_black not in self._noise and getattr(self.raw, "is_scan", False):
            self._noise[use_optical_black] = _scan_noise_model(self)
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
        if getattr(self.raw, "is_scan", False):
            return None
        if self._rows is None:
            from ..raw.banding import measure_rows
            self._rows = measure_rows(self.raw, self.levels(use_optical_black)) or False
        return self._rows or None

    def lateral_ca(self):
        """Lateral CA measured once on the whole frame (as-shot white balance), then reused for every region."""
        if self._ca is None and getattr(self.raw, "is_scan", False):
            from ..lens.auto import LateralCA
            self._ca = LateralCA(red=(0.0, 0.0), blue=(0.0, 0.0), tiles=0, confidence="none", max_shift_px=0.0)
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
        if getattr(self.raw, "is_scan", False):
            from ..detail import sharpen as shp
            return shp.DEFAULT_BLUR                    # a scan's blur is the scanner's; no mosaic to measure
        if algo not in self._blur:
            with slog.stage("sharpen.measure_blur", log, demosaic=algo):
                self._blur[algo] = _measure_capture_blur(self, algo)
            b = self._blur[algo]
            log.info("capture blur %.2f px (%s, %d edges)", b.sigma, b.confidence, b.edges)
        return self._blur[algo]

    def drop_cache(self):
        self._lin_cache = {}


SAFEGUARD_K = 4.5           # denoise safeguard: photosites moved by more than this many sigma are restored


def open_image(path: str) -> Image:
    from ..io.scanfile import is_scan, open_scan
    if is_scan(path):
        return Image(open_scan(path))
    return Image(open_raw(path))


def is_scan_image(image: Image) -> bool:
    """True for a TIFF / JPEG / PNG scan (io/scanfile.py), False for a raw."""
    return bool(getattr(image.raw, "is_scan", False))


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
    if getattr(image.raw, "is_scan", False):
        sharpen_on = False                         # capture sharpening undoes a lens; a scan's blur is not measured
    key = (edit.raw_key(), tuple(region) if region else None, half_size, sharpen_on)
    slot = "whole" if region is None else "region"
    with image._lock:
        hit = image._lin_cache.get(slot) if cache else None
        if hit is not None and hit[0] == key:
            return hit[1]
        whole = image._lin_cache.get("whole") if (cache and region is not None and half_size) else None
    if whole is not None and whole[0] == (edit.raw_key(), None, True, sharpen_on):
        # a half-size region (a zoomed view up to 60%) is part of the half-size whole frame the preview already
        # made: cut it out instead of decoding and denoising that part again (seconds with Clean on)
        x, y, w, h = _crop_box(image, region)
        cut = whole[1][y // 2:y // 2 + h // 2, x // 2:x // 2 + w // 2]
        if cut.shape[:2] == (h // 2, w // 2):
            return np.ascontiguousarray(cut)
    raw = image.raw
    if getattr(raw, "is_scan", False):
        rgb = _scan_linear(image, edit, region, half_size)
        if cache:
            with image._lock:
                image._lin_cache[slot] = (key, rgb)
        return rgb
    lv = image.levels(edit.use_optical_black)
    x, y, w, h = _crop_box(image, region)
    # read a margin around the region so the demosaic has context; keep everything CFA-aligned
    m = 0 if half_size else 16
    if sharpen_on:
        m = 32                                     # deconvolution reaches a few blur widths
    if edit.denoise.enabled:
        m = 256                                    # coarse denoise scales need context around a region
    if _old_on(edit):
        m = max(m, _old_photo_pad(image, edit) * (2 if half_size else 1))
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
    film_on = _film_on(edit)
    if film_on:                                    # a negative: balance the light through the film neutrally;
        user_mult, mult = mult, cam.normalise_wb(raw.wb_as_shot)    # the user's balance acts on the positive
        mult = np.asarray(mult, float) / float(np.min(mult))
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
    if film_on:                                    # film negative -> positive (local/film.py)
        from ..local import film as fm
        params = film_params(image, edit)
        with slog.stage("film.positive", log, kind=params.get("kind")):
            rgb = fm.positive(rgb, params, scene=True)
            r = np.asarray(user_mult, float) / np.asarray(mult, float)
            r = r / r[1]
            if np.any(np.abs(r - 1.0) > 1e-6):
                rgb *= _cam_ratio_in_pp(raw, r).astype(np.float32)
    if _halftone_on(edit):                         # printing dots (local/halftone.py)
        rgb = apply_halftone(image, edit, rgb, 2 if half_size else 1)
    if _stains_on(edit):                           # stains and corrosion, unmixed (local/stains.py)
        cx0, cy0 = raw.crop[0], raw.crop[1]
        rgb = apply_stains(image, edit, rgb, x0 - cx0, y0 - cy0, 2 if half_size else 1)
    if _scratches_on(edit):                        # dust and scratches (local/scratches.py)
        cx0, cy0 = raw.crop[0], raw.crop[1]
        rgb = apply_scratches(image, edit, rgb, x0 - cx0, y0 - cy0, 2 if half_size else 1, whole=region is None)
    if _colour_on(edit):                           # black and white -> colour (local/colourize.py)
        cx0, cy0 = raw.crop[0], raw.crop[1]
        rgb = apply_colourize(image, edit, rgb, x0 - cx0, y0 - cy0, 2 if half_size else 1)
    if _relight_on(edit):                          # the light moved (local/relight.py), before the haze
        cx0, cy0 = raw.crop[0], raw.crop[1]
        rgb = apply_relight(image, edit, rgb, x0 - cx0, y0 - cy0, 2 if half_size else 1)
    if _depth_on(edit):                            # haze by distance (local/depth.py)
        cx0, cy0 = raw.crop[0], raw.crop[1]
        rgb = apply_depth(image, edit, rgb, x0 - cx0, y0 - cy0, 2 if half_size else 1)
    if half_size:
        ox, oy = (x - x0) // 2, (y - y0) // 2
        rgb = rgb[oy:oy + h // 2, ox:ox + w // 2]
    else:
        rgb = rgb[y - y0:y - y0 + h, x - x0:x - x0 + w]
    rgb = np.ascontiguousarray(rgb)
    if cache:
        with image._lock:
            image._lin_cache[slot] = (key, rgb)
    return rgb


# ------------------------------------------------------------------------------------------ scans (io/scanfile.py)
_INV_STD = None


def _inverse_standard():
    """(ys, xs): the default Studio Standard curve sampled finely, for undoing it (display-linear -> scene)."""
    global _INV_STD
    if _INV_STD is None:
        t = Edit().tone
        xs = np.concatenate([[0.0], np.exp2(np.linspace(-20.0, math.log2(t.mid_grey) + t.white, 8192))])
        ys = tone_mod.standard_curve(xs, mid_in=t.mid_grey, contrast=t.contrast, shoulder=t.shoulder,
                                     white=t.white, toe=t.toe).astype(np.float64)
        keep = np.concatenate([[True], np.diff(ys) > 1e-9])
        _INV_STD = (ys[keep], xs[keep])
    return _INV_STD


def scan_to_scene(disp_pp: np.ndarray) -> np.ndarray:
    """Display-linear ProPhoto (a scan as the file shows it) -> scene-linear ProPhoto that the default Studio
    Standard curve turns back into exactly the file: per channel in the render space, as the curve is applied."""
    ys, xs = _inverse_standard()
    r = apply_matrix(disp_pp, tone_mod.PP_TO_RENDER)
    np.clip(r, 0.0, 1.0, out=r)
    flat = r.reshape(-1)
    for a in range(0, flat.size, 1 << 22):
        flat[a:a + (1 << 22)] = np.interp(flat[a:a + (1 << 22)], ys, xs)
    return apply_matrix(r, tone_mod.RENDER_TO_PP)


def _film_on(edit: Edit) -> bool:
    from ..local import film as fm
    return fm.active(edit.film)


def _cam_ratio_in_pp(raw, r) -> np.ndarray:
    """White-balance ratios given in camera RGB, as per-channel gains on linear ProPhoto (the neutral's change)."""
    M = cam.camera_to_working(raw.cam_xyz)
    g = M @ np.asarray(r, float) / (M @ np.ones(3))
    return g / g[1]


def film_params(image: Image, edit: Edit) -> dict:
    """The edit's film settings with anything still to measure (base, density range) measured on the whole
    negative, once per image and setting (local/film.py)."""
    from ..local import film as fm
    f = dict(edit.film or {})
    if not fm.needs_measure(f):
        return f
    key = ("film", f.get("kind"), json.dumps(f.get("base")), edit.raw_key() if not is_scan_image(image) else "")
    with image._lock:
        hit = image._memo.get(key)
    if hit is None:
        T = film_light(image, edit)
        hit = fm.measure(T, f["kind"], f.get("base"))
        with image._lock:
            image._memo = {k: v for k, v in image._memo.items() if k[0] != "film"}
            image._memo[key] = hit
    out = dict(hit)
    out.update(f)                                  # what the edit already holds wins over what was measured
    return out


def film_light(image: Image, edit: Edit, long_edge: int = 1200) -> np.ndarray:
    """A small copy of the whole negative as linear light let through (before the positive), for measuring."""
    if is_scan_image(image):
        T = image.raw.display_linear(None, half=True)
    else:
        e = Edit.from_dict(edit.to_dict())
        e.film = None
        e.sharpen.enabled = False
        e.white_balance = type(e.white_balance)()          # as shot
        T = linear_rgb(image, e, None, half_size=True, cache=False)
    k = min(1.0, float(long_edge) / max(T.shape[:2]))
    if k < 1:
        T = resize(np.ascontiguousarray(T), max(8, round(T.shape[1] * k)), max(8, round(T.shape[0] * k)))
    return np.maximum(np.ascontiguousarray(T, np.float32), 0.0)


def film_measure(image: Image, edit: Edit, kind: str, base=None) -> dict:
    """Film settings for a negative of this kind, measured now (the interface keeps them in the edit, so the
    photo renders the same later). ``base``: the film base picked by the user, or None to find it."""
    from ..local import film as fm
    with slog.stage("film.measure", log, kind=kind, picked=base is not None):
        return fm.measure(film_light(image, edit), kind, base)


def film_base_at(image: Image, edit: Edit, u: float, v: float, r: int = 6) -> list:
    """The film base colour at a spot of the displayed photo (fractions), for "pick the film base"."""
    warp = make_warp(image, edit)
    xs, ys = warp.output_to_s(u * warp.out_w - 0.5, v * warp.out_h - 0.5)
    _, _, sw, sh = image.raw.crop
    x, y = int(np.clip(xs, 0, sw - 1)), int(np.clip(ys, 0, sh - 1))
    box = (max(0, x - r), max(0, y - r), min(sw, x + r + 1) - max(0, x - r), min(sh, y + r + 1) - max(0, y - r))
    if is_scan_image(image):
        T = image.raw.display_linear(box)
    else:
        e = Edit.from_dict(edit.to_dict())
        e.film = None
        e.sharpen.enabled = False
        e.white_balance = type(e.white_balance)()
        T = linear_rgb(image, e, box, cache=False)
    return [round(float(np.median(T[..., c])), 7) for c in range(3)]


def _scratches_on(edit: Edit) -> bool:
    from ..local import scratches as scr
    return scr.active(edit.scratches)


def _stains_on(edit: Edit) -> bool:
    from ..local import stains as stn
    return stn.active(edit.stains)


def _halftone_on(edit: Edit) -> bool:
    from ..local import halftone as hft
    return hft.active(edit.halftone)


def _depth_on(edit: Edit) -> bool:
    from ..local import depth as dp
    return dp.active(edit.depth) and dp.available()


def _relight_on(edit: Edit) -> bool:
    from ..local import depth as dp
    from ..local import relight as rl
    return rl.active(edit.relight) and dp.available()


def _old_on(edit: Edit) -> bool:
    """Any of the old-photo and Cognitive stages that work on a block (with context round it where needed)."""
    return (_halftone_on(edit) or _stains_on(edit) or _scratches_on(edit) or _colour_on(edit) or _depth_on(edit)
            or _relight_on(edit))


def _colour_on(edit: Edit) -> bool:
    from ..local import colourize as cz
    return cz.active(edit.colourize)


def _old_photo_pad(image: Image, edit: Edit) -> int:
    """Context (full-resolution pixels) the dust and scratch pass and the colourizer need round a block."""
    pad = 8 if _stains_on(edit) else 0
    if _halftone_on(edit):
        from ..local import halftone as hft
        pad = max(pad, hft.pad_for())
    if _scratches_on(edit):
        from ..local import scratches as scr
        pad = max(pad, scr.pad_for(edit.scratches))
    if _colour_on(edit):
        from ..local import colourize as cz
        _, _, cw, ch = image.raw.crop
        pad = max(pad, cz.pad_for((256, 256), (ch, cw)))
    return pad


def _std_display(Y: np.ndarray) -> np.ndarray:
    """The default Studio Standard curve on scene-linear luminance (a neutral's display-linear value)."""
    t = Edit().tone
    return tone_mod.standard_curve(Y, mid_in=t.mid_grey, contrast=t.contrast, shoulder=t.shoulder, white=t.white,
                                   toe=t.toe)


def colour_map(image: Image, edit: Edit):
    """DDColor's colour map (Lab a, b at the model's size) for the whole photo, once per image and base settings:
    (ab, crop width, crop height, milliseconds)."""
    from ..local import colourize as cz
    e = Edit.from_dict(edit.to_dict())
    e.colourize = None
    e.sharpen.enabled = False
    key = ("colour", e.raw_key())
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    with slog.stage("colourize.predict", log):
        base = linear_rgb(image, e, None, half_size=True, cache=False)
        k = min(1.0, 1024.0 / max(base.shape[:2]))
        if k < 1:
            base = resize(np.ascontiguousarray(base), max(16, round(base.shape[1] * k)),
                          max(16, round(base.shape[0] * k)))
        Y = tone_mod.luminance(np.ascontiguousarray(base)).reshape(base.shape[:2])
        L = cz.L_of_Y(_std_display(Y)).astype(np.float32)
        if edit.stage_versions.get("colourize", 1) >= 2:
            L = cz.even_tones(L)               # a faded or badly exposed photo is shown to the model evened out
        ab, ms = cz.predict(L)
    _, _, cw, ch = image.raw.crop
    res = (ab, cw, ch, ms)
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "colour"}
        image._memo[key] = res
    image.colour_report = {"ms": ms, "size": list(ab.shape[:2])}
    log.info("DDColor: %dx%d colour map in %.0f ms", ab.shape[1], ab.shape[0], ms)
    return res


def apply_colourize(image: Image, edit: Edit, rgb: np.ndarray, ox: int, oy: int, step: int) -> np.ndarray:
    """Colour on a scene-linear block whose top-left is at crop-frame (ox, oy) (local/colourize.py): DDColor's
    automatic colour (when on), then each colour hint laid into its own area."""
    from ..local import colourize as cz
    from ..local import masks as mk
    st = cz.settings(edit.colourize)
    _, _, cw, ch = image.raw.crop
    blk, k_map = None, max(cw, ch) / 256.0
    if st["on"]:
        ab, cw, ch, _ = colour_map(image, edit)
        blk = np.stack([_prior_block((ab[..., c], cw, ch), ox, oy, rgb.shape[0], rgb.shape[1], step)
                        for c in range(2)], axis=-1)
        k_map = cw / float(ab.shape[1])
    with slog.stage("colourize.apply", log, w=rgb.shape[1], h=rgb.shape[0], hints=len(cz.hints_of(edit.colourize))):
        target, Yd, guide = cz.display_target(rgb, blk, float(st["amount"]), step, k_map, _std_display,
                                              warm=float(st["warm"]), tint=float(st["tint"]))
        for h in cz.hints_of(edit.colourize):
            bm = mk.decode_bitmap(h["data"], int(h["w"]), int(h["h"])).astype(np.float32) / 255.0
            alpha = _prior_block((bm, cw, ch), ox, oy, rgb.shape[0], rgb.shape[1], step)
            if float(alpha.max()) <= 0.002:
                continue
            r = int(round(1.2 * max(cw, ch) / float(max(bm.shape)) / step))
            target = cz.blend_hint(target, Yd, guide, alpha, h["rgb"], r)
        return scan_to_scene(target)


def colour_hint_area(image: Image, edit: Edit, fg: list, bg: list | None = None, spread: float = 0.08,
                     size: int = 640, model: str | None = None) -> dict:
    """The area of a colour hint: what is reachable from the clicks (S-frame fractions) without crossing a sharp
    edge in brightness (local/colourize.py, geodesic_area), on a small copy of the whole photo as it is shown
    without colour. ``spread``: the brightness step (L*/100) that stops it. {"w", "h", "data", "fg", "bg",
    "spread"} with the bitmap in the S frame."""
    from ..local import colourize as cz
    from ..local import masks as mk
    from ..local import sam
    from .edit import Geometry
    if model in sam.SIZES and sam.available(model) and fg:        # SAM 2.1 outlines the part (spread unused)
        with slog.stage("colourize.hint_area_sam", log, model=model, fg=len(fg), bg=len(bg or [])):
            bm = sam_bitmap(image, edit, fg, bg, model)
        return {"w": int(bm.shape[1]), "h": int(bm.shape[0]), "data": mk.encode_bitmap(bm),
                "fg": [list(map(float, p)) for p in fg], "bg": [list(map(float, p)) for p in (bg or [])],
                "spread": round(float(spread), 4), "model": f"sam2.1-{model}"}
    e = Edit.from_dict(edit.to_dict())
    e.masks = None
    e.geometry = Geometry()
    e.colourize = None
    _, _, sw, sh = image.raw.crop
    sc = min(1.0, size / float(max(sw, sh)))
    a = np.ascontiguousarray(_unorient(render(image, e, scale=sc), image.raw.flip))
    L = cz.L_of_Y(tone_mod.luminance(a).reshape(a.shape[:2])) / 100.0
    with slog.stage("colourize.hint_area", log, size=f"{a.shape[1]}x{a.shape[0]}", fg=len(fg), bg=len(bg or [])):
        bm = cz.geodesic_area(L, fg, bg or [], edge=float(spread))
    log.info("colour hint area: %.1f%% of the photo", 100.0 * (bm > 127).mean())
    return {"w": int(bm.shape[1]), "h": int(bm.shape[0]), "data": mk.encode_bitmap(bm),
            "fg": [list(map(float, p)) for p in fg], "bg": [list(map(float, p)) for p in (bg or [])],
            "spread": round(float(spread), 4)}


def colour_at(image: Image, edit: Edit, u: float, v: float, r: int = 3) -> list:
    """The colour (sRGB 0..1) DDColor suggests at a spot of the displayed photo: where the colour picker starts."""
    from ..io.export import ExportSpec, to_output
    e = Edit.from_dict(edit.to_dict())
    e.colourize = {"on": True, "amount": float(((edit.colourize or {}).get("amount", 1.0)))}
    a8 = to_output(render(image, e, scale=min(1.0, 900.0 / max(output_size(image, e)))),
                   ExportSpec(format="jpeg", space="srgb", bits=8))
    h, w = a8.shape[:2]
    x, y = min(w - 1, int(u * w)), min(h - 1, int(v * h))
    patch = a8[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
    return [round(float(c) / 255.0, 4) for c in np.median(patch, axis=0)]


def _srgb_preview(image: Image, e: Edit, max_side: int):
    """The whole photo small (long side at most ``max_side``) in the S frame: (scene-linear ProPhoto, display sRGB
    0..1 with the standard tone curve) - what the learned models look at."""
    _, _, cw, ch = image.raw.crop
    base = linear_rgb(image, e, None, half_size=max(cw, ch) >= 2 * max_side, cache=False)
    k = min(1.0, float(max_side) / max(base.shape[:2]))
    if k < 1:
        base = resize(np.ascontiguousarray(base), max(16, round(base.shape[1] * k)), max(16, round(base.shape[0] * k)))
    lut = tone_lut(image, Edit(), base)
    disp = tone_mod.apply_curve_rgb(np.ascontiguousarray(base), *lut) if lut is not None else base
    from ..color import spaces as sp
    srgb = sp.srgb_encode(np.clip(apply_matrix(disp, sp.conversion_matrix(sp.PROPHOTO, sp.SRGB)), 0, 1))
    return base, np.ascontiguousarray(srgb, np.float32)


def sam_features(image: Image, edit: Edit, size: str):
    """SAM 2.1's look at the whole photo (local/sam.py), once per photo, model size and base settings: (features,
    preview width, preview height)."""
    from ..local import sam
    e = Edit.from_dict(edit.to_dict())
    e.masks = None
    e.colourize = None
    e.depth = None
    e.sharpen.enabled = False
    key = ("sam", size, e.raw_key())
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    with slog.stage("sam.encode", log, size=size):
        _, srgb = _srgb_preview(image, e, sam.SIZE)
        feats, ms = sam.encode(srgb, size)
    res = (feats, srgb.shape[1], srgb.shape[0])
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "sam"}
        image._memo[key] = res
    image.sam_report = {"size": size, "ms": ms}
    log.info("SAM 2.1 %s looked at the photo (%dx%d) in %.0f ms", size, srgb.shape[1], srgb.shape[0], ms)
    return res


def sam_bitmap(image: Image, edit: Edit, fg: list, bg: list | None, size: str) -> np.ndarray:
    """SAM 2.1's outline of the thing under the clicks (S-frame fractions), as a soft uint8 bitmap in the S frame."""
    from ..local import sam
    feats, w, h = sam_features(image, edit, size)
    t = __import__("time").perf_counter()
    logits, score = sam.decode(feats, size, fg, bg or [])
    _, _, cw, ch = image.raw.crop
    k = min(1.0, 2048.0 / max(cw, ch))                       # the outline kept at up to 2048 px (soft edge)
    bm = sam.bitmap(logits, max(16, round(ch * k)), max(16, round(cw * k)))
    log.info("SAM 2.1 %s outline: %d+%d clicks, score %.2f, %.1f%% of the photo, %.0f ms", size, len(fg),
             len(bg or []), score, 100.0 * float((bm > 127).mean()), (__import__("time").perf_counter() - t) * 1e3)
    return bm


def colour_match_solve(image: Image, edit: Edit, ref_image: Image, ref_edit: Edit | None = None,
                       size: int = 320, ref_weight=None):
    """Colour match (local/colourmatch.py): the setting changes that give the photo the reference photo's colour
    statistics, solved on small renders of both. (p, base multipliers, report); apply them with
    colour_match_apply. ``ref_weight``: optional per-pixel weight over the reference render (e.g. a subject)."""
    from ..local import colourmatch as cm
    ref_edit = ref_edit or Edit()
    with slog.stage("colour_match.solve", log):
        rs = min(0.5, size / float(max(output_size(ref_image, ref_edit))))
        R = render(ref_image, ref_edit, scale=rs)
        target = cm.stats(R, ref_weight)
        sc = min(0.5, size / float(max(output_size(image, edit))))
        base_mult = wb_multipliers(image, edit)
        p, rep = cm.solve(lambda e: render(image, e, scale=sc), edit, target, base_mult, log=log)
    rep["text"] = cm.describe(p)
    rep["target"] = target.tolist()
    log.info("colour match: %s (residual %.3f -> %.3f, %d renders, %.0f ms)", rep["text"], rep["start"], rep["end"],
             rep["renders"], rep["ms"])
    return p, base_mult, rep


def colour_match_apply(edit: Edit, p, base_mult, amount: float = 1.0, image: Image | None = None) -> Edit:
    """The edit with a colour match's setting changes applied at ``amount`` (0..1.5). With ``image``, a changed
    white balance also gets its temperature and tint filled in, for the sliders."""
    from ..local import colourmatch as cm
    e = cm.apply(edit, np.asarray(p, float) * float(amount), base_mult)
    if image is not None and e.white_balance.mode == "multipliers" and e.white_balance.multipliers:
        try:
            tt = wb_temp_tint(image, np.asarray(e.white_balance.multipliers, float))
            if tt is not None:
                e.white_balance.temperature, e.white_balance.tint = round(tt[0]), round(tt[1])
        except Exception:
            slog.log_exception("colour match: temperature/tint for the sliders failed", log)
    return e


def depth_map(image: Image, edit: Edit):
    """Relative depth of the whole photo (local/depth.py; near = 1), once per image and base settings, from the
    photo as displayed without depth effects: (D, crop width, crop height, milliseconds, haze colour, scene-linear
    luminance of the copy it was measured on)."""
    from ..local import depth as dp
    e = Edit.from_dict(edit.to_dict())
    e.depth = None
    e.relight = None
    e.colourize = None
    e.sharpen.enabled = False
    key = ("depth", e.raw_key())
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    with slog.stage("depth.predict", log):
        base, srgb = _srgb_preview(image, e, 1024)
        D, ms = dp.predict(srgb)
        Y = tone_mod.luminance(np.ascontiguousarray(base)).reshape(base.shape[:2])
        far = D < np.percentile(D, 15)                     # the haze colour: the brightest of the far tenth
        sky = base[far] if far.sum() > 50 else base.reshape(-1, 3)
        ys = Y[far] if far.sum() > 50 else Y.ravel()
        A = np.median(sky[ys >= np.percentile(ys, 70)], axis=0)
        A = A / max(float(A.max()), 1e-6) * float(np.clip(np.percentile(Y, 97), 0.2, 1.0))
    _, _, cw, ch = image.raw.crop
    res = (D, cw, ch, ms, A.astype(np.float32), Y.astype(np.float32))
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "depth"}
        image._memo[key] = res
    image.depth_report = {"ms": ms, "size": list(D.shape)}
    log.info("depth: %dx%d map in %.0f ms", D.shape[1], D.shape[0], ms)
    return res


def apply_depth(image: Image, edit: Edit, rgb: np.ndarray, ox: int, oy: int, step: int) -> np.ndarray:
    """Haze by distance on a scene-linear block whose top-left is at crop-frame (ox, oy) (local/depth.py)."""
    from ..local import depth as dp
    D, cw, ch, _, A, _ = depth_map(image, edit)
    blk = _prior_block((D, cw, ch), ox, oy, rgb.shape[0], rgb.shape[1], step)
    with slog.stage("depth.haze", log, w=rgb.shape[1], h=rgb.shape[0]):
        return dp.haze(rgb, blk, float(dp.settings(edit.depth)["haze"]), A)


def relight_map(image: Image, edit: Edit):
    """The relight gain over the whole photo (local/relight.py) from the depth map's normals and the light measured
    on the photo: (G, crop width, crop height, report). The normals and the measured light are kept per photo."""
    from ..local import relight as rl
    D, cw, ch, _, _, Y = depth_map(image, edit)
    e = Edit.from_dict(edit.to_dict())
    e.depth, e.relight, e.colourize = None, None, None
    e.sharpen.enabled = False
    key = ("relight_fit", e.raw_key())
    with image._lock:
        hit = image._memo.get(key)
    if hit is None:
        with slog.stage("relight.fit", log):
            N = rl.normals(D)
            coef = rl.fit_light(Y, N)
        hit = (N, coef)
        with image._lock:
            image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "relight_fit"}
            image._memo[key] = hit
        c = coef[1:]
        az = math.degrees(math.atan2(-c[1], c[0])) if np.hypot(c[0], c[1]) > 1e-9 else 0.0
        el = math.degrees(math.atan2(c[2], float(np.hypot(c[0], c[1])) + 1e-12))
        image.relight_report = {"azimuth": az, "elevation": el, "strength": float(np.linalg.norm(c) /
                                                                                   max(abs(coef[0]), 1e-9))}
        log.info("relight: measured light %s (azimuth %.0f, elevation %.0f)", rl.direction_name(az, el), az, el)
    N, coef = hit
    s = rl.settings(edit.relight)
    gk = ("relight_gain", key, tuple(round(s[k], 2) for k in sorted(s)))
    with image._lock:
        G = image._memo.get(gk)
    if G is None:
        G = rl.gain(N, coef, edit.relight)
        with image._lock:
            image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "relight_gain"}
            image._memo[gk] = G
    return G, cw, ch


def apply_relight(image: Image, edit: Edit, rgb: np.ndarray, ox: int, oy: int, step: int) -> np.ndarray:
    """The light moved on a scene-linear block whose top-left is at crop-frame (ox, oy) (local/relight.py)."""
    G, cw, ch = relight_map(image, edit)
    blk = _prior_block((G, cw, ch), ox, oy, rgb.shape[0], rgb.shape[1], step)
    with slog.stage("relight.apply", log, w=rgb.shape[1], h=rgb.shape[0]):
        return (rgb * blk[..., None]).astype(np.float32)


def depth_shape(image: Image, edit: Edit, lo: float = 0.0, hi: float = 0.4, soft: float = 0.1) -> dict:
    """A depth mask shape for the photo (masks.py, "depth"): the depth map as a bitmap in the S frame, and the
    distance range (near = 1)."""
    from ..local import masks as mk
    D = depth_map(image, edit)[0]
    return {"type": "depth", "op": "add", "w": int(D.shape[1]), "h": int(D.shape[0]),
            "data": mk.encode_bitmap(np.round(D * 255).astype(np.uint8)), "lo": lo, "hi": hi, "soft": soft}


def depth_view(image: Image, edit: Edit, scale: float = 0.25) -> np.ndarray:
    """The depth map as the displayed photo shows it (crop, rotation and geometry applied), coloured warm near and
    cool far (RGBA uint8), for the "Show depth map" overlay."""
    from ..local import depth as dp
    shp = depth_shape(image, edit)
    shp["raw"] = True
    e = Edit.from_dict(edit.to_dict())
    e.masks = [{"name": "depth", "enabled": True, "adjust": {}, "shapes": [shp]}]
    return dp.colour_map(render_mask(image, e, 0, scale))


def halftone_map(image: Image, edit: Edit) -> dict:
    """The printing screen found in the photo (local/halftone.py, measure), once per image and base settings,
    from a full-resolution crop of its middle (the dots are finer than any preview)."""
    from ..local import halftone as hft
    e = Edit.from_dict(edit.to_dict())
    e.halftone = None
    e.stains = None
    e.scratches = None
    e.colourize = None
    e.sharpen.enabled = False
    key = ("halftone", e.raw_key())
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    _, _, cw, ch = image.raw.crop
    s = int(min(hft.CROP, cw, ch))
    box = ((cw - s) // 2, (ch - s) // 2, s, s)
    with slog.stage("halftone.measure", log, side=s):
        crop = linear_rgb(image, e, box, half_size=False, cache=False)
        m = hft.measure(np.ascontiguousarray(crop, np.float32))
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "halftone"}
        image._memo[key] = m
    image.halftone_report = dict(m)
    if m["peaks"]:
        log.info("halftone: screen every %.2f px at %.0f degrees, %d peaks, %.0f dB", m["period"], m["angle"],
                 len(m["peaks"]), m["strength"])
    else:
        log.info("halftone: no printing screen found")
    return m


def apply_halftone(image: Image, edit: Edit, rgb: np.ndarray, step: int) -> np.ndarray:
    """Printing dots taken out of a scene-linear block (local/halftone.py): the same notches in every block."""
    from ..local import halftone as hft
    m = halftone_map(image, edit)
    if not m["peaks"]:
        return rgb
    with slog.stage("halftone.clean", log, w=rgb.shape[1], h=rgb.shape[0], peaks=len(m["peaks"])):
        return hft.clean(rgb, m["peaks"], edit.halftone, step)


def stain_map(image: Image, edit: Edit):
    """The stain measurement over the whole photo (local/stains.py, measure), once per image and base settings:
    (measurement, crop width, crop height). Made from the photo as it is before stains, dust and colour."""
    from ..local import stains as stn
    e = Edit.from_dict(edit.to_dict())
    sens = float(stn.settings(e.stains)["sensitivity"])
    e.stains = None
    e.scratches = None
    e.colourize = None
    e.sharpen.enabled = False
    key = ("stain", e.raw_key(), round(sens, 3))      # (printing dots, when on, are already out of this base)
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    with slog.stage("stains.measure", log):
        base = linear_rgb(image, e, None, half_size=True, cache=False)
        k = min(1.0, 1024.0 / max(base.shape[:2]))
        if k < 1:
            base = resize(np.ascontiguousarray(base), max(16, round(base.shape[1] * k)),
                          max(16, round(base.shape[0] * k)))
        m = stn.measure(np.ascontiguousarray(base, np.float32), sens)
    _, _, cw, ch = image.raw.crop
    res = (m, cw, ch)
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "stain"}
        image._memo[key] = res
    image.stain_report = {"stained": m["stained"], "lost": m["lost_frac"], "coef": m["coef"]}
    log.info("stains: %.1f%% of the photo stained, %.1f%% bloom cores (picture may be gone); fit %s",
             100 * m["stained"], 100 * m["lost_frac"], [round(c, 3) for c in m["coef"]])
    return res


def apply_stains(image: Image, edit: Edit, rgb: np.ndarray, ox: int, oy: int, step: int) -> np.ndarray:
    """Stains and corrosion unmixed on a scene-linear block whose top-left is at crop-frame (ox, oy), full-resolution
    pixels ``step`` apart (local/stains.py). The colour backgrounds come from the whole-photo measurement, so a 100%
    view and the export are cleaned exactly alike."""
    from ..local import stains as stn
    m, cw, ch = stain_map(image, edit)
    h, w = rgb.shape[:2]
    mw = _prior_block((m["med_w"], cw, ch), ox, oy, h, w, step)
    mg = _prior_block((m["med_g"], cw, ch), ox, oy, h, w, step)
    with slog.stage("stains.clean", log, w=w, h=h):
        return stn.clean(rgb, mw, mg, m["coef"], edit.stains, step, angle=m.get("angle", 0.0))


def stain_lost_shape(image: Image, edit: Edit) -> dict | None:
    """Where the picture may be gone under a bloom core (local/stains.py), as an object-mask shape in the S frame
    (for the "Show what's lost" overlay); None when stains are off."""
    from ..local import masks as mk
    if not _stains_on(edit):
        return None
    m, _, _ = stain_map(image, edit)
    lost = m["lost"]
    return {"type": "object", "op": "add", "w": int(lost.shape[1]), "h": int(lost.shape[0]),
            "data": mk.encode_bitmap(lost)}


def scratch_prior(image: Image, edit: Edit):
    """The learned scratch detector's damage map over the whole photo (local/scratchnet.py), measured once per
    image and base settings: (P, crop width, crop height), or None when the detector is not installed."""
    from ..local import scratchnet as sn
    if not sn.available():
        return None
    e = Edit.from_dict(edit.to_dict())
    e.scratches = None
    e.sharpen.enabled = False
    key = ("scrnet", e.raw_key())
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    with slog.stage("scratches.learned_map", log):
        base = linear_rgb(image, e, None, half_size=True, cache=False)
        k = min(1.0, 1024.0 / max(base.shape[:2]))
        if k < 1:
            base = resize(np.ascontiguousarray(base), max(16, round(base.shape[1] * k)),
                          max(16, round(base.shape[0] * k)))
        lut = tone_lut(image, Edit(), base)
        disp = tone_mod.apply_curve_rgb(np.ascontiguousarray(base), *lut) if lut is not None else base
        from ..color import spaces as sp
        srgb = sp.srgb_encode(np.clip(apply_matrix(disp, sp.conversion_matrix(sp.PROPHOTO, sp.SRGB)), 0, 1))
        grey = (srgb @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.float32)
        _, _, cw, ch = image.raw.crop
        P = sn.damage_map(grey, sn.short_for(ch, cw))
    res = (P, cw, ch)
    with image._lock:
        image._memo = {kk: v for kk, v in image._memo.items() if kk[0] != "scrnet"}
        image._memo[key] = res
    log.info("learned scratch map: %dx%d, %.2f%% marked", P.shape[1], P.shape[0], 100.0 * float((P > 0.35).mean()))
    return res


def _prior_block(prior, ox: int, oy: int, h: int, w: int, step: int) -> np.ndarray:
    """The damage map sampled at a block's pixels (block top-left at crop-frame (ox, oy), ``step`` apart)."""
    from scipy.ndimage import map_coordinates
    P, cw, ch = prior
    ys = (oy + (np.arange(h) + 0.5) * step) * (P.shape[0] / float(ch)) - 0.5
    xs = (ox + (np.arange(w) + 0.5) * step) * (P.shape[1] / float(cw)) - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(P, [yy, xx], order=1, mode="nearest").astype(np.float32)


def _protect_block(edit: Edit, ox: int, oy: int, h: int, w: int, step: int, sw: int, sh: int):
    """Eyes, brows, lips and nostrils of the faces in the edit (their catchlights and highlights look like dust)."""
    from ..local import portrait as pt
    faces = pt.faces_of(edit.portrait)
    if not faces:
        return None
    m = np.zeros((h, w), bool)
    ys = oy + (np.arange(h) + 0.5) * step
    xs = ox + (np.arange(w) + 0.5) * step
    for f in faces:
        try:
            circles = pt.feature_avoid(pt.Face(f, sw, sh))
        except Exception:
            continue
        for cx, cy, r in circles:
            r = r * 1.3
            iy = (ys > cy - r) & (ys < cy + r)
            ix = (xs > cx - r) & (xs < cx + r)
            if iy.any() and ix.any():
                sub = (ys[iy][:, None] - cy) ** 2 + (xs[ix][None, :] - cx) ** 2 <= r * r
                m[np.ix_(iy, ix)] |= sub
    return m if m.any() else None


def apply_scratches(image: Image, edit: Edit, rgb: np.ndarray, ox: int, oy: int, step: int,
                    whole: bool = False) -> np.ndarray:
    """Dust and scratch removal on a scene-linear block whose top-left is at crop-frame (ox, oy), full-resolution
    pixels, ``step`` apart (local/scratches.py, with the learned detector's map when it is installed)."""
    from ..local import scratches as scr
    prior = None
    try:
        pm = scratch_prior(image, edit)
        if pm is not None:
            prior = _prior_block(pm, ox, oy, rgb.shape[0], rgb.shape[1], step)
    except Exception:
        slog.log_exception("learned scratch detector failed; using the filter alone", log)
    _, _, sw, sh = image.raw.crop
    protect = _protect_block(edit, ox, oy, rgb.shape[0], rgb.shape[1], step, sw, sh)
    st = {}
    out = scr.clean(rgb, edit.scratches, ox, oy, step, stats=st, protect=protect, prior=prior)
    if whole:
        image.scratch_report = dict(st, learned=prior is not None, half=step > 1)
    return out


def jpeg_repairable(image: Image) -> bool:
    raw = getattr(image, "raw", None)
    return bool(getattr(raw, "is_scan", False)) and str(getattr(raw, "path", "")).lower().endswith((".jpg", ".jpeg"))


def _scan_source(image: Image, edit: Edit):
    """The scan's pixels as the pipeline reads them: the file as decoded; with JPEG repair, repaired inside its
    quantisation intervals (local/jpegfix.py); with Clean, its measured noise removed (local/scanclean.py). Each
    once per photo and setting (kept with it)."""
    src = _scan_repaired(image, edit)
    if not edit.denoise.enabled or src.data.dtype.kind == "f":
        return src
    dn = edit.denoise
    key = ("scanclean", round(float(dn.luma), 3), round(float(dn.chroma), 3), round(float(dn.grain), 3),
           bool(dn.safeguard), src is not image.raw)
    with image._lock:
        hit = image._memo.get(key)
    if hit is not None:
        return hit
    import dataclasses
    from ..local import scanclean as sc
    top = float(np.iinfo(src.data.dtype).max)
    enc = src.data.astype(np.float32) / top
    with slog.stage("scan.clean", log, w=src.width, h=src.height, luma=dn.luma, chroma=dn.chroma):
        nk = ("scannoise", src is not image.raw)
        with image._lock:
            noise = image._memo.get(nk)
        if noise is None:
            noise = sc.measure(enc)
            with image._lock:
                image._memo[nk] = noise
        st: dict = {}
        out = sc.clean(enc, noise, float(dn.luma), float(dn.chroma), float(dn.grain), bool(dn.safeguard), st)
        fixed = dataclasses.replace(src, data=np.round(out * 65535.0).astype(np.uint16), bits=16, _luts=None)
    image.scan_noise_report = {"levels": noise["levels"], "protected": st.get("protected", 0),
                               "sites": st.get("sites", 1)}
    log.info("scan clean: noise about %.2f levels, %d pixels kept by the safeguard", noise["levels"],
             st.get("protected", 0))
    with image._lock:
        image._memo = {k: v for k, v in image._memo.items() if not (isinstance(k, tuple) and k and k[0] == "scanclean")}
        image._memo[key] = fixed
    return fixed


def _scan_repaired(image: Image, edit: Edit):
    """The scan's pixels, JPEG-repaired when that is on (local/jpegfix.py)."""
    raw = image.raw
    if not ((edit.jpeg or {}).get("on") and jpeg_repairable(image)):
        return raw
    with image._lock:
        hit = image._memo.get(("jpegfix",))
    if hit is not None:
        return hit
    import dataclasses
    from PIL import Image as PI
    from ..io.scanfile import _upright
    from ..local import jpegfix as jf
    with slog.stage("jpeg.repair", log, file=os.path.basename(raw.path)):
        a, rep = jf.repair_file(raw.path)
        orient = 1
        try:
            with PI.open(raw.path) as im:
                orient = int(im.getexif().get(274, 1) or 1)
        except Exception:
            pass
        a = _upright(a, orient)
        if a.shape[:2] != raw.data.shape[:2] or a.ndim != raw.data.ndim:
            log.warning("JPEG repair: %s came back %s, the file is %s; repair skipped", os.path.basename(raw.path),
                        a.shape, raw.data.shape)
            return raw
        fixed = dataclasses.replace(raw, data=a, bits=16, _luts=None)
    image.jpeg_report = rep
    log.info("JPEG repair: quality about %d, %d steps, %.0f ms", rep["quality"], rep["iterations"], rep["ms"])
    with image._lock:
        image._memo[("jpegfix",)] = fixed
    return fixed


def _scan_linear(image: Image, edit: Edit, region, half_size: bool) -> np.ndarray:
    """Stages 1-12 for a scan: decode the box, undo the file's colour space (scanfile.py) and the Standard curve
    (or turn a negative into a positive), then white balance in linear ProPhoto (the scan's "camera")."""
    raw = _scan_source(image, edit)
    x, y, w, h = _crop_box(image, region)
    if half_size:
        x, y, w, h = x // 2 * 2, y // 2 * 2, w // 2 * 2, h // 2 * 2
    bx, by, bw, bh = x, y, w, h                    # the box asked for, inside the block read (context round it)
    if _old_on(edit):
        pd = _old_photo_pad(image, edit) // 2 * 2 * (2 if half_size else 1)
        x0, y0 = max(0, x - pd), max(0, y - pd)
        x1, y1 = min(raw.width, x + w + pd), min(raw.height, y + h + pd)
        if half_size:
            x0, y0 = x0 // 2 * 2, y0 // 2 * 2
            x1, y1 = x0 + (x1 - x0) // 2 * 2, y0 + (y1 - y0) // 2 * 2
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
    with slog.stage("scan.decode", log, w=w, h=h, half=half_size):
        disp = raw.display_linear((x, y, w, h), half_size)
    from ..local import film as fm
    if fm.active(edit.film):
        params = film_params(image, edit)
        with slog.stage("film.positive", log, kind=params.get("kind")):
            rgb = fm.positive(disp, params, scene=True)
    else:
        rgb = scan_to_scene(disp)
    mult = wb_multipliers(image, edit)
    if np.any(np.abs(mult - mult[1]) > 1e-6):
        rgb *= (mult / mult[1]).astype(np.float32)
    if _old_on(edit):
        st = 2 if half_size else 1
        if _halftone_on(edit):
            rgb = apply_halftone(image, edit, rgb, st)
        if _stains_on(edit):
            rgb = apply_stains(image, edit, rgb, x, y, st)
        if _scratches_on(edit):
            rgb = apply_scratches(image, edit, rgb, x, y, st, whole=region is None)
        if _colour_on(edit):
            rgb = apply_colourize(image, edit, rgb, x, y, st)
        if _relight_on(edit):
            rgb = apply_relight(image, edit, rgb, x, y, st)
        if _depth_on(edit):
            rgb = apply_depth(image, edit, rgb, x, y, st)
        ix, iy = (bx - x) // st, (by - y) // st
        rgb = rgb[iy:iy + bh // st, ix:ix + bw // st]
    return np.ascontiguousarray(rgb, np.float32)


def _scan_noise_model(image: Image):
    """A scan has no sensor model: one constant noise level, measured as the robust spread of the finest
    differences of its scene-linear luminance (grain included), in the noise-model shape the gates expect."""
    from ..calib.noise import NoiseModel
    raw = image.raw
    k = max(1, int(math.sqrt(raw.width * raw.height / 4.0e6)))
    disp = raw.display_linear((0, 0, raw.width // k * k, raw.height // k * k))[::k, ::k] if k > 1 \
        else raw.display_linear()
    Y = tone_mod.luminance(np.ascontiguousarray(scan_to_scene(np.ascontiguousarray(disp)))).reshape(disp.shape[:2])
    d = (Y[1:-1, 1:-1] * 4 - Y[:-2, 1:-1] - Y[2:, 1:-1] - Y[1:-1, :-2] - Y[1:-1, 2:]) / math.sqrt(20.0)
    mid = (Y[1:-1, 1:-1] > 0.03) & (Y[1:-1, 1:-1] < 0.5)
    v = d[mid] if mid.sum() > 1000 else d.reshape(-1)
    sig = float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else 1e-3
    sig = max(sig, 1e-5)
    log.info("scan noise: sigma %.5f (scene-linear, constant)", sig)
    return NoiseModel(a=np.zeros((2, 2)), b=np.full((2, 2), sig * sig), source="scan (measured)",
                      confidence="low")


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
            return _with_subjects(image, edit, _render_warp(image, edit, warp, None, scale), None)
        if not warp.identity:                       # crop only: no resampling, just the rectangle
            return _with_subjects(image, edit, _render_plain(image, edit, _window_to_s(image, warp, None), scale),
                                  None)
        return _with_subjects(image, edit, _render_plain(image, edit, region, scale), None)
    return _render_plain(image, edit, region, scale)


def _with_subjects(image: Image, edit: Edit, out: np.ndarray, window) -> np.ndarray:
    """The composite's subjects laid on a finished picture (pipeline/composite.py); not on masks or analysis views."""
    if not edit.composite or getattr(_VIEW, "what", None) is not None:
        return out
    from . import composite as cp
    if not cp.active(edit):
        return out
    with slog.stage("composite", log, layers=len(cp.layers(edit))):
        return cp.apply(image, edit, out, window)


def render_view(image: Image, edit: Edit | None, u: float, v: float, w: int = 720, h: int = 480) -> np.ndarray:
    """A w x h window of the finished photo at 100%, centred at (u, v) (0..1 of the displayed photo)."""
    edit = edit or Edit()
    warp = make_warp(image, edit)
    x, y, w, h = view_window(image, edit, u, v, w, h)
    if warp.warped:
        return _with_subjects(image, edit, _render_warp(image, edit, warp, (x, y, w, h), 1.0), (x, y, w, h))
    return _with_subjects(image, edit, _render_plain(image, edit, _window_to_s(image, warp, (x, y, w, h)), 1.0),
                          (x, y, w, h))


def render_window(image: Image, edit: Edit | None, window, scale: float = 1.0) -> np.ndarray:
    """A window of the finished photo, (x, y, w, h) in full-size output pixels, at ``scale`` (<= 1): the sharp
    part of Develop's zoomed view. The same pixels as that part of render(scale=scale); at scale 1 the same as
    render_view. ``scale`` <= 0.5 takes the fast half-size path, as previews do."""
    edit = edit or Edit()
    warp = make_warp(image, edit)
    ow, oh = warp.output_size
    x, y, w, h = (int(round(v)) for v in window)
    x, y = min(max(x, 0), ow - 1), min(max(y, 0), oh - 1)
    w, h = max(1, min(w, ow - x)), max(1, min(h, oh - y))
    scale = float(min(max(scale, 0.02), 1.0))
    with slog.stage("render.window", log, window=f"{w}x{h}+{x}+{y}", scale=round(scale, 3)):
        if warp.warped:
            return _with_subjects(image, edit, _render_warp(image, edit, warp, (x, y, w, h), scale), (x, y, w, h))
        return _with_subjects(image, edit, _render_plain(image, edit, _window_to_s(image, warp, (x, y, w, h)), scale),
                              (x, y, w, h))


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

    method: "grey_edge", "grayness" or "night". mood: how far to correct from the as-shot balance, 0..1 (1 = fully
    neutral); None = the default for this photo (60% for a warm light in a low-light photo, else 90%).
    info: both estimates in kelvin, their agreement, the confidence, the mood used, the number of lights."""
    from ..color import autowb as aw
    est = aw.estimates(image)
    if method in ("grayness", "night") and est.get(method) is not None:
        m, how = est[method], {"grayness": "grayness index", "night": "night (SGP-LRD)"}[method]
    else:
        m, how = est["grey_edge"], "grey edge"
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
    for name in ("grey_edge", "grayness", "night"):
        if est.get(name) is not None:
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


def change_map(image: Image, edit: Edit, scale: float = 0.25) -> np.ndarray:
    """How much the automatic tools (printing dots, stains, dust and scratches, colourize) changed each pixel of the
    photo as displayed: 0 = untouched, 1 = a large change (H x W), for the "Show what changed" overlay. Both renders
    use every other setting as it is, so only those tools' work shows."""
    e0 = Edit.from_dict(edit.to_dict())
    e0.halftone = None
    e0.stains = None
    e0.scratches = None
    e0.colourize = None
    with slog.stage("change_map", log, scale=round(scale, 3)):
        a = render(image, edit, scale=scale)
        b = render(image, e0, scale=scale)
    if a.shape != b.shape:
        b = resize(np.ascontiguousarray(b), a.shape[1], a.shape[0])
    d = np.abs(np.log(np.maximum(a, 0.0) + 1e-3) - np.log(np.maximum(b, 0.0) + 1e-3)).max(axis=2)
    return np.clip((d - 0.02) / 0.25, 0.0, 1.0).astype(np.float32)


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


def object_mask(image: Image, edit: Edit, fg: list, bg: list | None = None, size: int = 480,
                model: str | None = None) -> dict:
    """'Click an object': a mask shape for the object under the foreground points (S-frame fractions), not
    reaching the background points, stored as a bitmap. ``model``: "tiny" / "large" = SAM 2.1 of that size
    (local/sam.py); None or "classic" = the Studio's own colour-and-edge method on a small render."""
    from ..lens.geometry import GeomParams  # noqa: F401  (geometry is reset below)
    from ..local import masks as mk
    from ..local import sam
    from .edit import Geometry
    if model in sam.SIZES and sam.available(model) and fg:
        with slog.stage("masks.object_sam", log, model=model, fg=len(fg), bg=len(bg or [])):
            bm = sam_bitmap(image, edit, fg, bg, model)
        return {"type": "object", "op": "add", "w": int(bm.shape[1]), "h": int(bm.shape[0]),
                "data": mk.encode_bitmap(bm), "fg": [list(map(float, p)) for p in fg],
                "bg": [list(map(float, p)) for p in (bg or [])], "model": f"sam2.1-{model}"}
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
    from ..local.eyes import colour_settings
    colour = colour_settings(edit.portrait)
    hits = [s for s in rt.active(edit.retouch) if rt.overlaps(s, box, sw, sh)]
    key = ("pt", edit.raw_key(), json.dumps(hits, sort_keys=True), json.dumps(faces[index]["pts"]),
           json.dumps(faces[index].get("irises") or [], sort_keys=True), json.dumps(colour, sort_keys=True),
           json.dumps({a: round(v, 4) for a, v in amt.items() if a != "jawline"}, sort_keys=True),
           edit.stage_versions.get("portrait", 1))
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    with slog.stage("portrait.face", log, face=index, width=round(fc.W), half=half, k=k):
        rgb, gx, gy, q = _face_copy(image, edit, box, half, k)
        G = pt.face_gain(rgb, gx, gy, q, fc, amt, colour)
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


def _face_entry(image: Image, f: dict, k: float, how: str, img: np.ndarray | None = None) -> dict:
    S = _view_pts_to_s(image, f["pts"], k)
    out = {"pts": [[round(float(x), 6), round(float(y), 6)] for x, y in S], "conf": round(float(f["conf"]), 3),
           "how": how, "enabled": True}
    if f.get("mesh"):
        out["mesh"] = int(f["mesh"])
    if img is not None:
        out["irises"] = _iris_entries(image, img, np.asarray(f["pts"], float), k, f.get("iris"))
    return out


def _iris_entries(image: Image, img: np.ndarray, P: np.ndarray, k: float, iris10=None) -> list:
    """Both irises of a face (eyes.find_irises on the face view), stored as S fractions: centre and an edge point."""
    from ..local import eyes as ey
    out = []
    try:
        found = ey.find_irises(img, P, iris10)
    except Exception:                                    # noqa: BLE001  (a face without irises still works)
        slog.log_exception("iris finding failed", log)
        return out
    for e in found:
        cx, cy = e["c"]
        Q = np.array([[cx, cy], [cx + e["r"], cy]], float)
        S = _view_pts_to_s(image, Q, k)
        out.append({"side": e["side"], "c": [round(float(S[0, 0]), 6), round(float(S[0, 1]), 6)],
                    "e": [round(float(S[1, 0]), 6), round(float(S[1, 1]), 6)], "how": e["how"]})
    return out


def add_irises(image: Image, edit: Edit) -> list:
    """The edit's faces with their irises found where they have none yet (faces mapped before the eye tools)."""
    faces = [dict(f) for f in (edit.portrait or {}).get("faces") or []]
    if not any(not f.get("irises") for f in faces):
        return faces
    img, k = _face_view(image, edit)
    for f in faces:
        P = np.asarray(f.get("pts") or [], float)
        if f.get("irises") or P.shape != (468, 2):
            continue
        V = _s_to_view_pts(image, P[:, 0], P[:, 1], k)
        f["irises"] = _iris_entries(image, img, V, k)
    return faces


def find_red_eyes(image: Image, edit: Edit) -> dict:
    """Red pupils and blown-out pupils in every face's irises, as Red eye / Pupil spots (auto = "redeye"). Spots
    already in the edit near a pupil are kept and not added twice. {"spots", "red", "blown", "faces"}."""
    from ..local import eyes as ey
    from ..local import portrait as pt
    from ..local import retouch as rt
    _, _, sw, sh = image.raw.crop
    long = float(max(sw, sh))
    have = [s for s in rt.active(edit.retouch) if s.get("kind") in ("redeye", "pupil")]
    spots, red, blown = [], 0, 0
    faces = pt.faces_of(edit.portrait)
    for f in faces:
        fc = pt.Face(f, sw, sh)
        if not fc.irises:
            continue
        box = fc.window(0.05, sw, sh)
        half = fc.W > 1600
        with slog.stage("eyes.red", log, width=round(fc.W)):
            rgb, gx, gy, q = _face_copy(image, edit, box, half)
            from ..local import faces as fz
            found = ey.find_red_pupils(rgb, gx, gy, q, fc.irises, {"r": fc.P[fz.RIGHT_EYE], "l": fc.P[fz.LEFT_EYE]})
        for b in found:
            if any(math.hypot(b["x"] - s["pts"][0][0] * sw, b["y"] - s["pts"][0][1] * sh) < b["r"] for s in have):
                continue
            spots.append({"kind": b["kind"], "pts": [[round(b["x"] / sw, 6), round(b["y"] / sh, 6)]],
                          "size": round(b["r"] / long, 6), "feather": rt.DEFAULT_FEATHER[b["kind"]], "opacity": 100,
                          "src": [0.0, 0.0], "enabled": True, "auto": "redeye", "v": 2})
            red += b["kind"] == "redeye"
            blown += b["kind"] == "pupil"
    log.info("red eye: %d red, %d blown pupil(s) on %d face(s)", red, blown, len(faces))
    return {"spots": spots, "red": red, "blown": blown, "faces": len(faces)}


def find_faces(image: Image, edit: Edit) -> list:
    """Every face in the photo, found on this PC: [{"pts": 468 S-fraction points, "conf", "how": "found"}]."""
    from ..local import faces as fz
    with slog.stage("faces.find", log):
        img, k = _face_view(image, edit)
        found = fz.find(img)
    log.info("faces: %d found (%s)", len(found), ", ".join(f"{f['conf']:.2f} via {f['variant']}" for f in found))
    return [_face_entry(image, f, k, "found", img) for f in found]


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
    return _face_entry(image, best, k, "eyes", img)


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
        _, _, sw, sh = image.raw.crop
        from ..local.eyes import iris_list
        for _, cx, cy, r, _ in iris_list(f, sw, sh):            # the irises, as circles
            t = np.linspace(0, 2 * np.pi, 33)
            u, v = s_to_view(image, edit, (cx + r * np.cos(t)) / sw, (cy + r * np.sin(t)) / sh)
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


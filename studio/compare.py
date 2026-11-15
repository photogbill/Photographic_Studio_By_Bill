"""Camera-JPEG comparison: the Phase 1 acceptance check, made visible.

For each raw: the camera's own embedded JPEG and the Studio render side by side with a wipe slider, plus
measured numbers: brightness offset (EV), median colour difference (CIE76 delta E), saturation relative to
the camera, and where the black level came from. Writes a folder with the images and an index.html.
"""
from __future__ import annotations

import datetime as _dt
import html
import io
import json
import os

import numpy as np

from . import __version__, log as slog
from .engine import Edit
from .engine.color import spaces as S
from .engine.color import tone as T
from .engine.io.export import ExportSpec, to_output
from .engine.pipeline.pipeline import Image, apply_orientation, linear_rgb, render, resize

log = slog.get("compare")

PREVIEW_LONG_EDGE = 1600


def measure(image: Image, edit: Edit, ours_pp: np.ndarray, cam_srgb: np.ndarray) -> dict:
    """Numbers comparing our render to the camera JPEG (both already the same size, upright)."""
    out = {}
    try:
        lin = linear_rgb(image, edit, half_size=True)
        xs, ys = T.camera_match_curve(lin, image.embedded_jpeg())
        out["ev_vs_camera"] = T.ev_offset_vs_standard(xs, ys, mid_in=edit.tone.mid_grey, contrast=edit.tone.contrast)
    except Exception:
        slog.log_exception("camera-match curve failed", log)
    # colour is compared on ~400 px versions so noise and sharpening differences do not count as colour
    k = 400.0 / max(ours_pp.shape[:2])
    if k < 1:
        w, h = max(1, round(ours_pp.shape[1] * k)), max(1, round(ours_pp.shape[0] * k))
        ours_pp = resize(ours_pp, w, h)
        cam_srgb = resize(np.ascontiguousarray(cam_srgb, dtype=np.float32), w, h)
    lab_o = S.xyz_to_lab(ours_pp.reshape(-1, 3) @ S.PROPHOTO.to_xyz_d50.T)
    lab_c = S.xyz_to_lab(S.srgb_decode(np.clip(cam_srgb, 0, 1)).reshape(-1, 3) @ S.SRGB.to_xyz_d50.T)
    m = (lab_c[:, 0] > 15) & (lab_c[:, 0] < 90) & (lab_o[:, 0] > 15)
    if m.sum() > 500:
        co = np.hypot(lab_o[m, 1], lab_o[m, 2])
        cc = np.hypot(lab_c[m, 1], lab_c[m, 2])
        sel = cc > 8
        out["delta_e_median"] = float(np.median(np.sqrt(((lab_o[m] - lab_c[m]) ** 2).sum(1))))
        out["lightness_diff"] = float(np.median(lab_o[m, 0] - lab_c[m, 0]))
        out["chroma_ratio"] = float(np.median(co[sel] / cc[sel])) if sel.sum() > 100 else float("nan")
    return out


def _fmt(v, f="{:.1f}"):
    return f.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"


class CompareBuilder:
    """Add images one at a time (so only one raw is in memory), then write() the page."""

    def __init__(self, out_dir: str, edit: Edit | None = None):
        self.out_dir = out_dir
        self.edit = edit or Edit()
        self.cards: list = []
        os.makedirs(out_dir, exist_ok=True)
        # resumable: a page can be built up over several runs (long folders, archive drives)
        mj = os.path.join(out_dir, "measurements.json")
        if os.path.isfile(mj):
            try:
                with open(mj, encoding="utf-8") as fh:
                    self.cards = json.load(fh)
            except Exception:
                self.cards = []

    def has(self, path: str) -> bool:
        return any(c.get("file") == os.path.basename(path) and not c.get("error") for c in self.cards)

    def add(self, image: Image, export_path: str | None = None) -> dict:
        from PIL import Image as PI
        edit, out_dir = self.edit, self.out_dir
        name = os.path.splitext(os.path.basename(image.path))[0]
        try:
            with slog.stage("compare.render", log, file=name):
                arr = render(image, edit, scale=0.5)
            h, w = arr.shape[:2]
            k = PREVIEW_LONG_EDGE / max(h, w)
            if k < 1:
                arr = resize(arr, round(w * k), round(h * k))
            h, w = arr.shape[:2]
            ours8 = to_output(arr, ExportSpec(format="jpeg", space="srgb", bits=8))
            PI.fromarray(ours8).save(os.path.join(out_dir, name + "_studio.jpg"), quality=92)
            jpg = image.embedded_jpeg()
            stats = {}
            has_cam = jpg is not None
            if has_cam:
                cam = np.asarray(PI.open(io.BytesIO(jpg)).convert("RGB"))
                cam = np.ascontiguousarray(apply_orientation(cam, image.raw.flip))
                cam = np.asarray(PI.fromarray(cam).resize((w, h), PI.Resampling.LANCZOS))
                PI.fromarray(cam).save(os.path.join(out_dir, name + "_camera.jpg"), quality=92)
                stats = measure(image, edit, arr, cam.astype(np.float32) / 255.0)
            crop = self._crop100(image, name) if edit.denoise.enabled else None
            m = image.meta
            lv = image.levels(edit.use_optical_black)
            card = dict(crop=crop, name=name, file=os.path.basename(image.path), camera=m.model or "Unknown camera",
                        iso=m.iso_label, iso_source=m.iso_source, lens=m.lens or "Manual / unknown lens",
                        exposure=(f"1/{round(1 / m.exposure_s)} s" if m.exposure_s and m.exposure_s < 0.5
                                  else (f"{m.exposure_s:g} s" if m.exposure_s else "")),
                        fnum=f"f/{m.fnumber:g}" if m.fnumber else "", temp=m.camera_temperature_c,
                        black_src=lv.source, size=f"{image.output_size[0]} x {image.output_size[1]}",
                        export=os.path.basename(export_path) if export_path else "", has_cam=has_cam,
                        w=w, h=h, **stats)
        except Exception as e:
            slog.log_exception(f"compare failed for {image.path}", log)
            card = dict(name=name, file=os.path.basename(image.path), error=str(e))
        self.cards = [c for c in self.cards if c.get("file") != card["file"]] + [card]
        self.write()                    # keep the page current after every photo
        return card

    def _crop100(self, image: Image, name: str, size=(900, 600)) -> dict | None:
        """100% crops from the middle of the frame: Studio without denoise, with denoise, and the camera JPEG."""
        from PIL import Image as PI
        try:
            _, _, cw, ch = image.raw.crop
            w, h = min(size[0], cw), min(size[1], ch)
            x, y = ((cw - w) // 2) // 2 * 2, ((ch - h) // 2) // 2 * 2
            off = Edit.from_dict(self.edit.to_dict())
            off.denoise.enabled = False
            spec = ExportSpec(format="jpeg", space="srgb", bits=8)
            with slog.stage("compare.crop100", log, file=name):
                a0 = render(image, off, region=(x, y, w, h))
                a1 = render(image, self.edit, region=(x, y, w, h))
            PI.fromarray(to_output(a0, spec)).save(os.path.join(self.out_dir, name + "_crop_raw.jpg"), quality=94)
            PI.fromarray(to_output(a1, spec)).save(os.path.join(self.out_dir, name + "_crop_dn.jpg"), quality=94)
            jpg = image.embedded_jpeg()
            has_cam = False
            if jpg is not None:
                cam = np.asarray(PI.open(io.BytesIO(jpg)).convert("RGB"))
                if cam.shape[1] >= cw - 8 and cam.shape[0] >= ch - 8:          # full-size preview only
                    c = np.ascontiguousarray(apply_orientation(cam[y:y + h, x:x + w], image.raw.flip))
                    PI.fromarray(c).save(os.path.join(self.out_dir, name + "_crop_cam.jpg"), quality=94)
                    has_cam = True
            model = image.noise_model(self.edit.use_optical_black)
            return dict(has_cam=has_cam, noise_src=model.source, noise_conf=model.confidence)
        except Exception:
            slog.log_exception(f"100% crop failed for {image.path}", log)
            return None

    def write(self) -> str:
        path = os.path.join(self.out_dir, "index.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_page(self.cards))
        with open(os.path.join(self.out_dir, "measurements.json"), "w", encoding="utf-8") as fh:
            json.dump(self.cards, fh, indent=2, default=float)
        return path


def build(images: list, out_dir: str, edit: Edit | None = None) -> str:
    """images: list of (Image, export_path or None). Returns the index.html path."""
    b = CompareBuilder(out_dir, edit)
    for image, p in images:
        b.add(image, p)
    return b.write()


CSS = """
:root{--bg:#07090f;--panel:#0e1320;--line:#1c2a4a;--text:#e6ecff;--dim:#8a97b8;--cyan:#22d3ee;--amber:#f59e0b;
--violet:#c084fc;--green:#22c55e;--red:#f87171}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 20% -10%,#14203d 0,#07090f 60%);
color:var(--text);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:28px 36px 8px}h1{margin:0;font-size:26px;letter-spacing:.5px}
h1 span{background:linear-gradient(90deg,var(--cyan),var(--violet));-webkit-background-clip:text;color:transparent}
.sub{color:var(--dim);margin-top:4px}.legend{display:flex;gap:18px;margin:14px 0 4px;color:var(--dim);font-size:13px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
main{padding:10px 36px 40px;display:grid;gap:28px}
.card{background:linear-gradient(180deg,#0f1628,#0b101c);border:1px solid var(--line);border-radius:16px;
overflow:hidden;box-shadow:0 20px 50px #0008}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;padding:16px 20px;border-bottom:1px solid var(--line)}
.title{font-size:18px;font-weight:600}.meta{color:var(--dim);font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:8px;padding:14px 20px}
.chip{border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:13px;background:#0a0f1a}
.chip b{font-weight:600}.c-cyan{border-color:#22d3ee66;color:#bff5fd}.c-amber{border-color:#f59e0b66;color:#ffe2b0}
.c-green{border-color:#22c55e66;color:#c6f6d5}.c-red{border-color:#f8717166;color:#fecaca}
.wipe{position:relative;margin:0 20px 20px;border-radius:10px;overflow:hidden;background:#000;user-select:none}
.wipe img{display:block;width:100%;height:auto}.wipe .over{position:absolute;inset:0;overflow:hidden}
.wipe .over img{position:absolute;inset:0}.bar{position:absolute;top:0;bottom:0;width:2px;background:var(--cyan);
box-shadow:0 0 12px var(--cyan)}.lab{position:absolute;top:10px;padding:3px 10px;border-radius:6px;font-size:12px;
background:#000a;border:1px solid #fff2}.lab.l{left:10px}.lab.r{right:10px}
input[type=range]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}
.err{padding:20px;color:var(--red)}.sub2{color:var(--dim);font-size:13px;padding:0 20px 10px}
.crops{display:grid;grid-template-columns:3fr 2fr;gap:14px;padding:0 20px 20px}.crops .wipe{margin:0}
.crops figure{margin:0}.crops figure img{width:100%;border-radius:10px;display:block}
.crops figcaption{color:var(--dim);font-size:12px;margin-top:6px}footer{color:var(--dim);padding:0 36px 30px;font-size:13px}
"""

JS = """
document.querySelectorAll('.wipe').forEach(function(w){
  var r=w.querySelector('input'),o=w.querySelector('.over'),b=w.querySelector('.bar');
  function set(v){o.style.clipPath='inset(0 '+(100-v)+'% 0 0)';b.style.left=v+'%';}
  r.addEventListener('input',function(){set(r.value)});set(r.value);
});
"""


def _crop_html(c) -> str:
    cr = c.get("crop")
    if not cr:
        return ""
    n = html.escape(c["name"])
    cam = (f"<figure><img src='{n}_crop_cam.jpg' alt='Camera JPEG 100%'><figcaption>Camera JPEG (Canon's own "
           f"noise reduction), 100%</figcaption></figure>") if cr.get("has_cam") else ""
    return (f"<div class=sub2>100% crop from the middle of the frame. Noise model: estimated from this image "
            f"(<span style='color:var(--amber)'>amber</span>, confidence {html.escape(cr['noise_conf'])}); "
            f"calibrated profiles will replace it.</div><div class=crops><div class=wipe>"
            f"<img src='{n}_crop_dn.jpg' alt='Studio denoised'><div class=over><img src='{n}_crop_raw.jpg' "
            f"alt='Studio without denoise'></div><div class=bar></div><span class='lab l'>Studio, no denoise</span>"
            f"<span class='lab r'>Studio, denoised</span><input type=range min=0 max=100 value=50 "
            f"aria-label='wipe'></div>{cam}</div>")


def _chip(label, value, cls):
    return f'<span class="chip {cls}">{html.escape(label)} <b>{html.escape(value)}</b></span>'


def _page(cards) -> str:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Studio vs camera JPEG</title>"
             f"<style>{CSS}</style></head><body><header><h1><span>Photographic Studio</span> vs the camera JPEG</h1>"
             f"<div class=sub>Phase 1 develop engine, Studio Standard look. Drag across each picture to wipe between "
             f"the camera's own JPEG (left label) and the Studio render (right label). {stamp} - v{__version__}</div>"
             f"<div class=legend><span><i class=dot style='background:var(--cyan)'></i>measured from the raw</span>"
             f"<span><i class=dot style='background:var(--amber)'></i>estimated</span>"
             f"<span><i class=dot style='background:var(--green)'></i>close to the camera</span></div></header><main>"]
    for c in cards:
        if c.get("error"):
            parts.append(f"<section class=card><div class=top><div class=title>{html.escape(c['file'])}</div></div>"
                         f"<div class=err>Could not develop this file: {html.escape(c['error'])}</div></section>")
            continue
        chips = [_chip("True ISO", c["iso"].replace("ISO ", ""), "c-cyan"),
                 _chip("Black level", "masked border" if c["black_src"] == "optical-black" else "LibRaw",
                       "c-cyan" if c["black_src"] == "optical-black" else "c-amber")]
        if c.get("temp") is not None:
            chips.append(_chip("Camera temp", f"{c['temp']:.0f} C", "c-cyan"))
        ev = c.get("ev_vs_camera")
        if isinstance(ev, float) and np.isfinite(ev) and abs(ev) > 2.5:
            chips.append(_chip("Studio vs camera", "not comparable (unusual light)", "c-amber"))
        elif isinstance(ev, float) and np.isfinite(ev):
            chips.append(_chip("Studio vs camera", f"{-ev:+.2f} EV", "c-green" if abs(ev) <= 0.35 else "c-amber"))
        de = c.get("delta_e_median")
        if de is not None:
            chips.append(_chip("Colour difference", f"dE {de:.1f}", "c-green" if de <= 5 else "c-amber"))
        cr = c.get("chroma_ratio")
        if cr is not None and np.isfinite(cr):
            chips.append(_chip("Saturation", f"{100 * cr:.0f}% of camera", "c-green" if 0.9 <= cr <= 1.1 else "c-amber"))
        meta = " - ".join(x for x in (c["camera"], c["lens"], c["exposure"], c["fnum"], c["size"]) if x)
        img_l = f"{c['name']}_camera.jpg" if c["has_cam"] else f"{c['name']}_studio.jpg"
        parts.append(
            f"<section class=card><div class=top><div><div class=title>{html.escape(c['file'])}</div>"
            f"<div class=meta>{html.escape(meta)}</div></div>"
            f"<div class=meta>{('Full-size export: ' + html.escape(c['export'])) if c['export'] else ''}</div></div>"
            f"<div class=chips>{''.join(chips)}</div>"
            f"<div class=wipe><img src='{html.escape(c['name'])}_studio.jpg' alt='Studio render'>"
            f"<div class=over><img src='{html.escape(img_l)}' alt='Camera JPEG'></div><div class=bar></div>"
            f"<span class='lab l'>Camera JPEG</span><span class='lab r'>Studio</span>"
            f"<input type=range min=0 max=100 value=50 aria-label='wipe'></div>{_crop_html(c)}</section>")
    parts.append("</main><footer>Pictures are reduced to 1600 px for this page. The full-size files are 16-bit TIFFs "
                 "in Adobe RGB. Colour differences are measured on the reduced pictures; Canon's picture style "
                 "adds its own saturation and sharpening, so a small difference is expected. The Studio applies no "
                 "sharpening yet (capture sharpening is stage 12, Phase 7), so Canon's JPEG looks crisper at "
                 "100% for that reason too.</footer>"
                 f"<script>{JS}</script></body></html>")
    return "".join(parts)

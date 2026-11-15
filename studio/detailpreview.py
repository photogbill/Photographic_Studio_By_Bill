"""Detail preview page: microcontrast and capture sharpening on your own raws, next to your DxO export when one
sits beside the raw (NAME_DxO.jpg). Built by 'Detail Preview.bat' into outputs\\detail_preview (never to C:).

For each photo: a before/after wipe of the whole frame (microcontrast off / on) and 100% crops of its most
detailed area: as developed, with "Recover lens sharpness", with microcontrast, with both, and DxO's version.
The "shades of shadow" figure is how much the medium-scale tonal variation in the shadows grew (log luminance,
band 4-32 px): the thing microcontrast is meant to add.
"""
from __future__ import annotations

import datetime as _dt
import html
import os

import numpy as np

from . import __version__, log as slog
from .engine import Edit
from .engine.io.export import ExportSpec, to_output

log = slog.get("detailpreview")
SPEC = ExportSpec(format="jpeg", space="srgb", bits=8)


def shadow_separation(rgb8: np.ndarray, s1: float = 6.0, s2: float = 30.0) -> tuple[float, float]:
    """(medium-scale tonal variation in the shadows, in the midtones), stops: std of a band-pass (Gaussian s1 minus
    Gaussian s2 pixels) of log2 luminance, split by the local tone."""
    from scipy.ndimage import gaussian_filter
    y = rgb8.astype(np.float32).mean(axis=2) / 255.0
    lin = np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)
    L = np.log2(np.maximum(lin, 1e-4))
    base = gaussian_filter(L, s2)
    band = gaussian_filter(L, s1) - base
    sh = band[(base < -3.5) & (base > -9.0)]
    mid = band[(base >= -3.5) & (base < -1.0)]
    return (float(sh.std()) if sh.size > 500 else float("nan"), float(mid.std()) if mid.size > 500 else float("nan"))


def _save(arr8, path):
    from PIL import Image as PI
    PI.fromarray(arr8).save(path, quality=92)
    return os.path.basename(path)


def build(files: list, out_dir: str, mc_amount: float = 0.8, sharpen_amount: float = 0.6, crop=(900, 600)) -> str:
    from PIL import Image as PI, ImageOps
    from . import engine as E
    from .blindtest import _detail_box, find_dxo_pairs
    os.makedirs(out_dir, exist_ok=True)
    dxo = {}
    for folder in {os.path.dirname(f) for f in files}:
        dxo.update(dict(find_dxo_pairs(folder)))
    cards = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            with slog.stage("detail_preview", log, file=name):
                img = E.open(f)
                m = img.meta
                ow, oh = img.output_size
                sc = min(1.0, 1400.0 / max(ow, oh))
                off = Edit()
                on = Edit()
                on.microcontrast.amount = mc_amount
                fit0 = to_output(E.render(img, off, scale=sc), SPEC)
                fit1 = to_output(E.render(img, on, scale=sc), SPEC)
                a = _save(fit0, os.path.join(out_dir, f"{name}_fit_off.jpg"))
                b = _save(fit1, os.path.join(out_dir, f"{name}_fit_mc.jpg"))
                k = 6.0 * sc                                   # the same feature sizes as the crops, at this scale
                s0, s1 = shadow_separation(fit0, k, 5 * k), shadow_separation(fit1, k, 5 * k)
                bx, by, bw, bh = _detail_box(fit0, int(crop[0] * sc), int(crop[1] * sc))
                u, v = (bx + bw / 2) / fit0.shape[1], (by + bh / 2) / fit0.shape[0]
                w, h = (crop if ow >= oh else crop[::-1])
                variants = []
                for label, e in (("As developed", Edit()),
                                 (f"Recover lens sharpness {sharpen_amount:.0%}", _with(sharpen=sharpen_amount)),
                                 (f"Microcontrast {mc_amount:.0%}", _with(mc=mc_amount)),
                                 ("Both", _with(mc=mc_amount, sharpen=sharpen_amount))):
                    c8 = to_output(E.render_view(img, e, u, v, w, h), SPEC)
                    sep = shadow_separation(c8)
                    variants.append((label, _save(c8, os.path.join(out_dir, f"{name}_{len(variants)}.jpg")), sep))
                if f in dxo:
                    d = ImageOps.exif_transpose(PI.open(dxo[f])).convert("RGB")
                    if d.size != (ow, oh):
                        d = d.resize((ow, oh), PI.Resampling.LANCZOS)
                    x = int(min(max(round(u * ow - w / 2), 0), ow - w))
                    y = int(min(max(round(v * oh - h / 2), 0), oh - h))
                    c8 = np.asarray(d.crop((x, y, x + w, y + h)))
                    variants.append(("Your DxO export", _save(c8, os.path.join(out_dir, f"{name}_dxo.jpg")),
                                     shadow_separation(c8)))
                blur = img.capture_blur()
                cards.append(dict(name=name, file=os.path.basename(f), camera=m.model, iso=m.iso_label,
                                  lens=m.lens or "manual lens", fit=(a, b), variants=variants, whole=(s0, s1),
                                  blur=f"{blur.sigma:.2f} px ({blur.confidence})"))
                img.drop_cache()
            print(f"  {name}: done")
        except Exception as e:
            slog.log_exception(f"detail preview failed for {f}", log)
            cards.append(dict(name=name, file=os.path.basename(f), error=str(e)))
            print(f"  {name}: PROBLEM {e}")
    page = os.path.join(out_dir, "index.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(_page(cards, mc_amount))
    return page


def _with(mc: float = 0.0, sharpen: float = 0.0) -> Edit:
    e = Edit()
    e.microcontrast.amount = mc
    if sharpen:
        e.sharpen.enabled = True
        e.sharpen.amount = sharpen
    return e


def _fmt_sep(sep, ref):
    sh, mid = sep
    if not np.isfinite(sh) or not np.isfinite(ref[0]) or ref[0] <= 0:
        return ""
    return f"shades of shadow {100 * (sh / ref[0] - 1):+.0f}%"


def _fmt_mid(sep, ref):
    if not np.isfinite(sep[1]) or not np.isfinite(ref[1]) or ref[1] <= 0:
        return ""
    return f", midtones {100 * (sep[1] / ref[1] - 1):+.0f}%"


def _page(cards, mc_amount) -> str:
    esc = html.escape
    parts = []
    for c in cards:
        if "error" in c:
            parts.append(f"<section class='card'><h2>{esc(c['file'])}</h2><p class='bad'>{esc(c['error'])}</p>"
                         "</section>")
            continue
        ref = c["variants"][0][2]
        crops = "".join(f"<figure><img src='{esc(fn)}' loading='lazy'><figcaption>{esc(lb)}"
                        f"<span>{esc(_fmt_sep(sep, ref)) if i else ''}</span></figcaption></figure>"
                        for i, (lb, fn, sep) in enumerate(c["variants"]))
        parts.append(f"""<section class='card'>
<h2>{esc(c['file'])}</h2>
<div class='chips'><span>{esc(c['camera'] or '')}</span><span>{esc(c['iso'])}</span><span>{esc(c['lens'])}</span>
<span class='amber'>measured blur {esc(c['blur'])}</span>
<span class='cyan'>whole photo: {esc(_fmt_sep(c['whole'][1], c['whole'][0]))}{esc(_fmt_mid(c['whole'][1], c['whole'][0]))}</span></div>
<div class='wipe'><img src='{esc(c['fit'][1])}'><div class='top'><img src='{esc(c['fit'][0])}'></div>
<input type='range' min='0' max='100' value='50' oninput='this.previousElementSibling.style.width=this.value+"%"'>
<span class='l'>microcontrast off</span><span class='r'>microcontrast {mc_amount:.0%}</span></div>
<h3>100% crops of the most detailed area</h3><div class='grid'>{crops}</div></section>""")
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Detail preview</title><style>
body{{background:#07090f;color:#e6ecff;font-family:Segoe UI,Inter,sans-serif;margin:0;padding:28px 40px}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#8a97b8;margin-bottom:24px;max-width:1100px;line-height:1.5}}
.card{{background:#0e1320;border:1px solid #1c2a4a;border-radius:16px;padding:20px 22px;margin-bottom:28px}}
h2{{margin:0 0 8px;font-size:18px}} h3{{color:#8a97b8;font-weight:500;font-size:14px;margin:18px 0 8px}}
.chips span{{display:inline-block;border:1px solid #1c2a4a;border-radius:999px;padding:3px 10px;margin:0 6px 6px 0;
color:#8a97b8;font-size:12px}} .chips .amber{{color:#f59e0b;border-color:rgba(245,158,11,.4)}}
.chips .cyan{{color:#22d3ee;border-color:rgba(34,211,238,.4)}}
.wipe{{position:relative;display:inline-block;max-width:100%}} .wipe img{{display:block;max-width:100%}}
.wipe .top{{position:absolute;top:0;left:0;width:50%;overflow:hidden;height:100%;border-right:2px solid #22d3ee}}
.wipe .top img{{max-width:none;height:100%}}
.wipe input{{position:absolute;left:0;right:0;bottom:8px;width:96%;margin:0 2%}}
.wipe .l,.wipe .r{{position:absolute;top:10px;background:rgba(0,0,0,.65);padding:4px 10px;border-radius:6px;
font-size:12px}} .wipe .l{{left:10px}} .wipe .r{{right:10px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}} figure{{margin:0}} figure img{{display:block;border-radius:8px;
max-width:100%}} figcaption{{color:#8a97b8;font-size:12px;padding:6px 2px}}
figcaption span{{color:#22d3ee;margin-left:8px}} .bad{{color:#f87171}}
</style></head><body><h1>Detail preview</h1><div class='sub'>Microcontrast (more shades of shadow, not edge sharpening)
and Recover lens sharpness (deconvolution to a target, measured blur) on your own raws. Crops are at 100%: view the page
at 100% browser zoom. Noise reduction is off here so the detail stages are judged on their own. Photographic Studio By
Bill {__version__}, {when}.</div>{''.join(parts)}</body></html>"""

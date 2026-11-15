"""Blind test (section 17): the final judge for denoise, microcontrast, stacking and signatures.

Two unlabelled 100% crops side by side; click the better one; the tool tallies and reveals at the end.
Phase 3 acceptance: Bill prefers the Studio result over his own DxO output on at least half of ten pairs.

DxO mode: for every raw that has a DxO export beside it (NAME_DxO.jpg, as in the Voigtlander folder), the Studio
develops the raw (denoise on) and the same region is cut from both. Left/right order is random per pair.
DxO applies no geometry correction to contactless vintage lenses, so the frames line up; if a DxO file has a
different size it is scaled to match (and the pair is marked as approximate).
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import os
import random

import numpy as np

from . import __version__, log as slog
from .engine import Edit
from .engine.io.export import ExportSpec, to_output

log = slog.get("blindtest")


def find_dxo_pairs(folder: str) -> list:
    from .engine import RAW_EXTENSIONS
    out = []
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in RAW_EXTENSIONS:
            for cand in (stem + "_DxO.jpg", stem + "_DxO.jpeg", stem + "_dxo.jpg", stem + "-DxO.jpg"):
                p = os.path.join(folder, cand)
                if os.path.isfile(p):
                    out.append((os.path.join(folder, f), p))
                    break
    return out


def _detail_box(rgb8: np.ndarray, w: int, h: int) -> tuple:
    """Most detailed w x h region (gradient energy on a coarse grid), avoiding the frame edges."""
    g = rgb8.astype(np.float32).mean(axis=2)[::8, ::8]
    gy, gx = np.gradient(g)
    e = np.hypot(gx, gy)
    kh, kw = max(1, h // 8), max(1, w // 8)
    best, by, bx = -1.0, 0, 0
    for y in range(e.shape[0] // 10, max(1, e.shape[0] - kh - e.shape[0] // 10), max(1, kh // 3)):
        for x in range(e.shape[1] // 10, max(1, e.shape[1] - kw - e.shape[1] // 10), max(1, kw // 3)):
            v = float(e[y:y + kh, x:x + kw].mean())
            if v > best:
                best, by, bx = v, y, x
    return bx * 8, by * 8, w, h


def build_dxo_test(folder: str, out_dir: str, max_pairs: int = 10, crop=(800, 560), edit: Edit | None = None,
                   seed: int | None = None) -> str:
    from PIL import Image as PI
    from . import engine as E
    from .engine.pipeline.pipeline import resize
    edit = edit or Edit()
    if not edit.denoise.enabled:
        edit.denoise.enabled = True
    pairs = find_dxo_pairs(folder)
    if not pairs:
        raise FileNotFoundError(f"no raw + NAME_DxO.jpg pairs in {folder}")
    rnd = random.Random(seed if seed is not None else int(_dt.datetime.now().timestamp()))
    rnd.shuffle(pairs)
    os.makedirs(out_dir, exist_ok=True)
    items = []
    spec = ExportSpec(format="jpeg", space="srgb", bits=8)
    for raw_path, dxo_path in pairs[:max_pairs]:
        name = os.path.splitext(os.path.basename(raw_path))[0]
        try:
            img = E.open(raw_path)
            preview = to_output(E.render(img, Edit(), scale=0.25), spec)       # to choose the region
            dx = np.asarray(PI.open(dxo_path).convert("RGB"))
            ow, oh = img.output_size
            approx = (dx.shape[1], dx.shape[0]) != (ow, oh)
            if approx:
                if abs(dx.shape[1] / dx.shape[0] - ow / oh) > 0.02:
                    log.warning("blind test: %s DxO aspect differs; skipped", name)
                    continue
                dx = np.asarray(PI.fromarray(dx).resize((ow, oh), PI.Resampling.LANCZOS))
            cw, ch = min(crop[0], ow), min(crop[1], oh)
            bx, by, _, _ = _detail_box(preview, cw // 4, ch // 4)
            x, y = min(ow - cw, bx * 4) // 2 * 2, min(oh - ch, by * 4) // 2 * 2
            # render() regions are in the unrotated frame; for rotated files take the region from a full render
            if img.raw.flip:
                full = E.render(img, edit)
                ours = to_output(np.ascontiguousarray(full[y:y + ch, x:x + cw]), spec)
            else:
                ours = to_output(E.render(img, edit, region=(x, y, cw, ch)), spec)
            theirs = dx[y:y + ch, x:x + cw]
            studio_left = rnd.random() < 0.5
            a, b = (ours, theirs) if studio_left else (theirs, ours)
            PI.fromarray(a).save(os.path.join(out_dir, f"{name}_A.jpg"), quality=95)
            PI.fromarray(b).save(os.path.join(out_dir, f"{name}_B.jpg"), quality=95)
            items.append(dict(name=name, iso=img.meta.iso_label, studio="A" if studio_left else "B", approx=approx))
            img.drop_cache()
            print(f"  pair {len(items)}: {name} ({img.meta.iso_label})")
        except Exception:
            slog.log_exception(f"blind test pair failed: {raw_path}", log)
    if not items:
        raise RuntimeError("no usable pairs")
    page = os.path.join(out_dir, "index.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(_page(items, edit))
    return page


def _page(items: list, edit: Edit) -> str:
    key = base64.b64encode(json.dumps([i["studio"] for i in items]).encode()).decode()   # revealed at the end
    pairs_js = json.dumps([dict(name=i["name"], iso=i["iso"], approx=i["approx"]) for i in items])
    settings = html.escape(f"denoise {edit.denoise.method}, luma {edit.denoise.luma}, chroma {edit.denoise.chroma}")
    css = ("body{margin:0;background:#07090f;color:#e6ecff;font:15px 'Segoe UI',sans-serif}header{padding:20px 30px}"
           "h1{margin:0;font-size:22px}h1 span{color:#22d3ee}.sub{color:#8a97b8}.pair{display:grid;"
           "grid-template-columns:1fr 1fr;gap:12px;padding:0 30px}.pair figure{margin:0;cursor:pointer;border:3px solid "
           "transparent;border-radius:12px;overflow:hidden}.pair figure:hover{border-color:#22d3ee}.pair img{width:100%;"
           "display:block;image-rendering:pixelated}.cap{text-align:center;color:#8a97b8;padding:6px}"
           "#res{padding:30px;font-size:18px}.big{font-size:42px;color:#22c55e}button{background:#0e1320;color:#e6ecff;"
           "border:1px solid #1c2a4a;border-radius:8px;padding:8px 14px;cursor:pointer}")
    js = f"""
const P={pairs_js}, K=JSON.parse(atob('{key}')); let i=0, picks=[];
function show(){{ if(i>=P.length) return done();
 document.getElementById('n').textContent=(i+1)+' of '+P.length+' - '+P[i].iso+(P[i].approx?' (DxO file rescaled)':'');
 document.getElementById('a').src=P[i].name+'_A.jpg'; document.getElementById('b').src=P[i].name+'_B.jpg'; }}
function pick(s){{ picks.push(s); i++; show(); }}
function done(){{ let st=0; picks.forEach((p,k)=>{{ if(p===K[k]) st++; }});
 document.getElementById('test').style.display='none';
 const r={{date:new Date().toISOString(), studio_preferred:st, of:P.length, settings:'{settings}',
   pairs:P.map((p,k)=>({{file:p.name, iso:p.iso, picked:(picks[k]===K[k])?'Studio':'DxO'}}))}};
 document.getElementById('res').innerHTML='<div>You preferred the <b>Studio</b> result in</div><div class=big>'+st+
  ' of '+P.length+'</div><div class=sub>Phase 3 passes at half or more. Settings: {settings}</div>'+
  '<p>'+r.pairs.map(p=>p.file+' ('+p.iso+'): '+p.picked).join('<br>')+'</p>'+
  '<button onclick="dl()">Save results</button>'; window.R=r; }}
function dl(){{ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(window.R,null,2)],
 {{type:'application/json'}})); a.download='blind_test_results.json'; a.click(); }}
document.addEventListener('keydown',e=>{{ if(e.key==='ArrowLeft'||e.key==='1') pick('A'); if(e.key==='ArrowRight'||e.key==='2') pick('B'); }});
show();"""
    return (f"<!doctype html><meta charset=utf-8><title>Blind test</title><style>{css}</style>"
            f"<header><h1><span>Blind test</span> - which do you prefer?</h1><div class=sub>Click the better picture "
            f"(or press the left / right arrow). One is the Studio, one is DxO; the side changes at random. "
            f"Shown at 100%. <span id=n></span></div></header><div id=test><div class=pair>"
            f"<figure onclick=\"pick('A')\"><img id=a><div class=cap>Left</div></figure>"
            f"<figure onclick=\"pick('B')\"><img id=b><div class=cap>Right</div></figure></div></div>"
            f"<div id=res></div><script>{js}</script><footer class=sub style='padding:20px 30px'>Photographic Studio "
            f"By Bill {__version__}. Results stay on this PC.</footer>")

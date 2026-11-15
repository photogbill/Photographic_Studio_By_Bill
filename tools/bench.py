"""Denoise benchmark harness (section 17, tools/bench).

Clean low-ISO raws + realistic noise synthesised from a measured noise model = ground truth.
Every denoiser candidate is run on the same noisy mosaics and scored:
  * PSNR on sqrt-encoded values (weights shadows roughly as the eye does). A sanity check only:
    section 22 change 2 warns PSNR barely tracks preference on fine texture.
  * Residual structure: correlation between what a method removed and the clean image's edges.
    0 = it removed only noise; higher = it is eating detail.
  * Bias: mean brightness shift (shadow tint shows up here first).
The blind test (section 17) remains the final judge.

Usage (from the app folder):
    python tools\\bench.py CLEAN_RAW [CLEAN_RAW...] --noise-from HIGH_ISO_RAW [HIGH_ISO_RAW...]
Writes outputs\\bench\\bench_<time>.html and .json (local only, never committed).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from studio import engine as E, log as slog, paths  # noqa: E402
from studio.engine.calib import noise as N  # noqa: E402
from studio.engine.raw import denoise as DN  # noqa: E402
from studio.engine.raw import linearize as L  # noqa: E402

CANDIDATES = {
    "starlet default (luma 0.6, chroma 1.5)": DN.DenoiseParams(method="starlet", luma=0.6, chroma=1.5),
    "starlet (luma 1, chroma 1.5)": DN.DenoiseParams(method="starlet", luma=1.0, chroma=1.5),
    "mppca (1.5 / 2)": DN.DenoiseParams(method="mppca", luma=1.5, chroma=2.0),
    "nlm (1 / 2)": DN.DenoiseParams(method="nlm", luma=1.0, chroma=2.0),
    "nlm (1.5 / 3)": DN.DenoiseParams(method="nlm", luma=1.5, chroma=3.0),
}


def enc(z):
    return np.sqrt(np.clip(z, 0, 1))


def psnr(a, b, m=16):
    return float(10 * np.log10(1.0 / np.mean((enc(a) - enc(b))[m:-m, m:-m] ** 2)))


def residual_structure(noisy, den, clean, m=16):
    from scipy.ndimage import gaussian_filter, sobel
    r = gaussian_filter(np.abs(noisy - den), 2.0)[m:-m, m:-m]
    g = np.hypot(sobel(clean, 0), sobel(clean, 1))[m:-m, m:-m]
    r, g = r.ravel() - r.mean(), g.ravel() - g.mean()
    return float((r @ g) / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-12))


def detailed_crop(img, size=1024):
    """Pick the most detailed size x size region (even origin) by gradient energy on a coarse grid."""
    from scipy.ndimage import sobel
    small = img.raw.visible[::16, ::16].astype(np.float32)
    g = np.hypot(sobel(small, 0), sobel(small, 1))
    k = size // 16
    best, by, bx = -1, 0, 0
    for y in range(0, max(1, g.shape[0] - k), max(1, k // 4)):
        for x in range(0, max(1, g.shape[1] - k), max(1, k // 4)):
            v = g[y:y + k, x:x + k].mean()
            if v > best:
                best, by, bx = v, y, x
    return (bx * 16) // 2 * 2, (by * 16) // 2 * 2, min(size, img.raw.width), min(size, img.raw.height)


def preview(mos, img, edit):
    """Half-size developed preview of a mosaic crop (same path for every method), as 8-bit sRGB."""
    from studio.engine.color import tone as T
    from studio.engine.color.camera import camera_to_working
    from studio.engine.io.export import ExportSpec, to_output
    from studio.engine.pipeline.pipeline import wb_multipliers
    from studio.engine.raw.demosaic import half
    m = mos.copy()
    mult = wb_multipliers(img, edit)
    for i in (0, 1):
        for j in (0, 1):
            m[i::2, j::2] *= np.float32(mult[img.raw.cfa_rgb[i, j]])
    rgb = half(np.minimum(m, 1.0), img.raw.cfa_rgb)
    from studio.engine.color.spaces import apply_matrix
    rgb = apply_matrix(rgb, camera_to_working(img.raw.cam_xyz).astype(np.float32))
    xs, ys = T.curve_lut(lambda v: T.standard_curve(v))
    rgb = T.apply_curve_rgb(rgb, xs, ys)
    return to_output(rgb, ExportSpec(format="jpeg", space="srgb", bits=8))


def run(clean_paths, noise_paths, out_dir, crop=1024, seed=0):
    from PIL import Image as PI
    os.makedirs(out_dir, exist_ok=True)
    models = []
    for p in noise_paths:
        im = E.open(p)
        models.append((os.path.basename(p), im.meta.iso, im.noise_model()))
    rows, cards = [], []
    rng = np.random.default_rng(seed)
    for cp in clean_paths:
        ci = E.open(cp)
        lv = ci.levels()
        box = detailed_crop(ci, crop)
        clean = L.normalise(ci.raw, lv, box)
        clean = np.clip(clean * (0.12 / max(float(np.median(clean)), 1e-4)), 0, 1).astype(np.float32)
        edit = E.Edit()
        for mname, iso, model in models:
            a, b = model.a, model.b
            noisy = clean.copy()
            for i in (0, 1):
                for j in (0, 1):
                    c = clean[i::2, j::2]
                    noisy[i::2, j::2] = rng.poisson(c / a[i, j]) * a[i, j] + rng.normal(0, np.sqrt(b[i, j]), c.shape)
            tag = f"{os.path.splitext(os.path.basename(cp))[0]}__{os.path.splitext(mname)[0]}"
            imgs = {"truth": preview(clean, ci, edit), "noisy": preview(noisy, ci, edit)}
            base = {"clean": os.path.basename(cp), "noise_from": mname, "iso": iso, "noisy_psnr": psnr(noisy, clean)}
            for cname, params in CANDIDATES.items():
                t0 = time.perf_counter()
                den = DN.denoise_mosaic(noisy, ci.raw.cfa_rgb, model, params)
                dt = time.perf_counter() - t0
                rows.append(dict(base, method=cname, psnr=psnr(den, clean),
                                 residual_structure=residual_structure(noisy, den, clean),
                                 bias=float((den - clean).mean()), seconds_per_mp=dt / (clean.size / 1e6)))
                imgs[cname] = preview(den, ci, edit)
            names = []
            for k, arr in imgs.items():
                fn = f"{tag}__{len(names)}.jpg"
                PI.fromarray(arr).save(os.path.join(out_dir, fn), quality=93)
                names.append((k, fn))
            cards.append((tag, base, names))
            print(f"  {tag}: noisy {base['noisy_psnr']:.2f} dB -> " +
                  ", ".join(f"{r['method']} {r['psnr']:.2f}" for r in rows if r['clean'] == base['clean']
                            and r['noise_from'] == mname))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, f"bench_{stamp}.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    page = os.path.join(out_dir, f"bench_{stamp}.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(_page(rows, cards))
    return page


def _page(rows, cards):
    css = ("body{background:#07090f;color:#e6ecff;font:14px 'Segoe UI',sans-serif;margin:30px}"
           "table{border-collapse:collapse;margin:10px 0 30px}td,th{border:1px solid #1c2a4a;padding:6px 10px}"
           "th{background:#0e1320}.best{color:#22c55e;font-weight:600}h1 span{color:#22d3ee}"
           ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px}"
           "figure{margin:0}figure img{width:100%;image-rendering:pixelated;border-radius:8px}"
           "figcaption{color:#8a97b8;font-size:12px}")
    out = [f"<!doctype html><meta charset=utf-8><title>Denoise benchmark</title><style>{css}</style>"
           "<h1><span>Denoise benchmark</span> - synthetic noise from measured models on clean low-ISO raws</h1>"
           "<p>PSNR is a sanity check only; the blind test decides. Residual structure: 0 means only noise was "
           "removed, higher means detail was removed too.</p><table><tr><th>Clean</th><th>Noise from</th>"
           "<th>ISO</th><th>Noisy</th><th>Method</th><th>PSNR</th><th>Residual structure</th><th>Bias</th>"
           "<th>s / MP</th></tr>"]
    for r in rows:
        grp = [x for x in rows if x["clean"] == r["clean"] and x["noise_from"] == r["noise_from"]]
        best = max(grp, key=lambda x: x["psnr"]) is r
        out.append(f"<tr><td>{html.escape(r['clean'])}</td><td>{html.escape(r['noise_from'])}</td>"
                   f"<td>{r['iso'] or ''}</td><td>{r['noisy_psnr']:.2f}</td><td>{html.escape(r['method'])}</td>"
                   f"<td class={'best' if best else ''}>{r['psnr']:.2f}</td><td>{r['residual_structure']:.3f}</td>"
                   f"<td>{r['bias']:+.1e}</td><td>{r['seconds_per_mp']:.2f}</td></tr>")
    out.append("</table>")
    for tag, base, names in cards:
        out.append(f"<h3>{html.escape(base['clean'])} with noise from {html.escape(base['noise_from'])}</h3>"
                   "<div class=grid>")
        for k, fn in names:
            out.append(f"<figure><img src='{fn}'><figcaption>{html.escape(k)}</figcaption></figure>")
        out.append("</div>")
    return "".join(out)


def main(argv=None):
    slog.setup()
    ap = argparse.ArgumentParser(description="Denoise benchmark harness")
    ap.add_argument("clean", nargs="*", help="clean low-ISO raw files")
    ap.add_argument("--noise-from", nargs="*", default=[], help="high-ISO raws whose noise model is used")
    ap.add_argument("--auto", action="store_true",
                    help="use the raws in the app folder and 'high iso raw': ISO <= 200 clean, ISO >= 1600 noise")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.auto:
        from studio.cli import collect
        from studio.engine.io.metadata import read_metadata
        files = collect([ROOT, os.path.join(ROOT, "high iso raw")])
        isos = {f: (read_metadata(f).iso or 0) for f in files}
        a.clean = a.clean or [f for f in files if 0 < isos[f] <= 200][:3]
        a.noise_from = a.noise_from or sorted([f for f in files if isos[f] >= 1600], key=lambda f: -isos[f])[:4]
    if not a.clean or not a.noise_from:
        print("Need at least one clean low-ISO raw (ISO 200 or lower) and one high-ISO raw (ISO 1600 or higher).")
        return 2
    out = a.out or str(paths.get("OUTPUTS") / "bench")
    page = run(a.clean, a.noise_from, out, a.crop)
    print(f"Benchmark page: {page}")
    try:
        import webbrowser
        webbrowser.open("file:///" + os.path.abspath(page).replace("\\", "/"))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

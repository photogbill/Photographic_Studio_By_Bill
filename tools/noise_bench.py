"""The truth harness (BUILD_PLAN_NOISE block A): every Clean method scored on this camera's REAL noise.

Clean truth (charts with a known answer, and clean photo crops when the Lab has prepared some) is given the
camera's noise at each ISO: photon noise by the measured gain plus patches of its own dark frames (calib/synth.py).
Each method then cleans it, with and without the detail guard, and is scored on what matters:

  * noise left      how much of the error that came in is still there (lower is better)
  * texture kept    share of the truth's fine detail that survived, on the dead-leaves chart (higher is better)
  * edges kept      the same on a slanted edge
  * blotches left   slow colour mottling left in flat patches, against what came in (lower is better)
  * residual        band energy of what was removed, 1.0 = exactly noise, above = detail went with it
  * seconds

    python tools\\noise_bench.py [--camera NAME] [--isos 400,3200,12800,51200] [--quick] [--photos FOLDER]

Two checks beside the scores:
  * **Is the harness honest?** (A.4) Each pair of this camera's real flat frames is compared with a flat of the
    same brightness made by the harness: the two noise figures should agree within 5%.
  * **Your own photos** (A.5, --photos or a folder dragged onto the .bat): no truth exists for them, so each
    method's error and the share of fine detail it keeps are *estimated* from the photo itself (Stein's lemma,
    raw/auto_strength.py; a Dual Pixel photo is judged by its two halves instead, raw/referee.py).
Writes outputs\\noisebench\\<time>\\index.html and results.json (this PC only).
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import glob
import html
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from studio import log as slog, paths  # noqa: E402
from studio.engine.calib import deep as D  # noqa: E402
from studio.engine.calib import synth as S  # noqa: E402
from studio.engine.calib.noise import anscombe  # noqa: E402
from studio.engine.calib.profile import CameraProfile  # noqa: E402
from studio.engine.raw import denoise as DN  # noqa: E402
from studio.engine.raw import guard as GD  # noqa: E402
from studio.engine.raw import learned as LN  # noqa: E402
from studio.lab import data as LD  # noqa: E402

log = slog.get("noisebench")


def scenes(quick: bool, n_photos: int = 4) -> dict:
    out = {"dead leaves": S.mosaic_of(S.dead_leaves(384 if quick else 640, 384 if quick else 640, seed=1) * 0.55),
           "dead leaves, shadows": S.mosaic_of(S.dead_leaves(384 if quick else 640, 384 if quick else 640, 2) * 0.06),
           # the hard one: texture barely above the noise, where over-cleaning shows first
           "dead leaves, faint": S.mosaic_of(0.3 + 0.10 * (S.dead_leaves(384 if quick else 640, 384 if quick else 640, 3)
                                                             - 0.4)),
           "flat patches": S.mosaic_of(S.flat_patches(384, 512)),
           "slanted edge": S.mosaic_of(S.slanted_edge(384, 384))}
    files = sorted(glob.glob(os.path.join(LD.clean_dir(), "*.npz")))[:: max(1, 7)][:0 if quick else n_photos]
    for k, f in enumerate(files):
        c = np.load(f)["crops"][0].astype(np.float32)
        out[f"photo {k + 1}"] = np.clip(c * 0.5, 0.0, 1.0)
    return out


def luminance(mos: np.ndarray, model) -> np.ndarray:
    """Stabilised luminance plane (unit noise) of a mosaic."""
    pl = [anscombe(mos[i::2, j::2], float(model.a[i, j]), float(model.b[i, j])) for i in (0, 1) for j in (0, 1)]
    return DN.to_yc(pl[0], pl[1], pl[2], pl[3])


def score(clean, noisy, den, model, name: str) -> dict:
    enc = lambda v: np.sqrt(np.maximum(v, 0.0) + 1e-4)                      # noqa: E731  (roughly what the eye sees)
    e_in, e_out = float(np.mean((enc(noisy) - enc(clean)) ** 2)), float(np.mean((enc(den) - enc(clean)) ** 2))
    yc, yn, yd = luminance(clean, model), luminance(noisy, model), luminance(den, model)
    out = dict(noise_left=round(float(np.sqrt(e_out / max(e_in, 1e-12))), 4))
    if name.startswith("dead leaves") or name.startswith("photo"):
        out["texture_kept"] = round(LD.texture_kept(yd[0], yn[0], yc[0]), 4)
    if name == "slanted edge":
        out["edges_kept"] = round(LD.texture_kept(yd[0], yn[0], yc[0]), 4)
    if name == "flat patches":
        from scipy.ndimage import uniform_filter
        blot = lambda a: float(np.mean(uniform_filter(a[1] - yc[1], 15) ** 2 + uniform_filter(a[2] - yc[2], 15) ** 2))  # noqa: E731
        out["blotches_left"] = round(float(np.sqrt(blot(yd) / max(blot(yn), 1e-12))), 4)
    out["residual"] = GD.whiteness(yn[0] - yd[0])["band_energy"][1]
    return out


def candidates(camera: tuple, quick: bool) -> dict:
    """What is scored. "auto": the strength is chosen from each picture (raw/auto_strength.py), as Clean does."""
    c = {"hybrid": dict(method="hybrid", luma=1.0, chroma=1.5),
         "hybrid, auto strength": dict(method="hybrid", auto=True),
         "fusion, auto strength": dict(method="fusion", auto=True),
         "collab (BM3D class), auto": dict(method="collab", auto=True),
         "starlet": dict(method="starlet", luma=1.0, chroma=1.5),
         "nlm": dict(method="nlm", luma=1.0, chroma=2.0),
         # deliberately too strong: what over-cleaning costs, and what the detail guard gives back
         "starlet x2.5": dict(method="starlet", luma=2.5, chroma=2.5)}
    if not quick:
        c["mppca"] = dict(method="mppca", luma=1.5, chroma=2.0)
    if LN.for_camera(camera) is not None:
        c["learned"] = dict(method="learned", luma=1.0, chroma=1.0, camera=camera)
    return c


def auto_strengths(noisy: np.ndarray, model, method: str, corr: float = 0.0) -> tuple:
    """(luminance, colour) strengths the photo itself asks for, exactly as the pipeline's clean_auto chooses.
    ``corr``: how much neighbouring photosites share noise at this ISO (the profile's figure)."""
    from studio.engine.raw import auto_strength as AS
    y = luminance(noisy, model)
    fn, fc = DN.method_fns(method)
    slow = method in ("mppca", "collab")
    n = 1 if max(y[0].shape) <= 2 * AS.PIECE else (3 if slow else 5)      # a small chart is one piece
    lu = AS.choose(fn, AS.pieces(y[0], n), 4, AS.CANDIDATES_SLOW if slow else AS.CANDIDATES, corr=corr)
    ch = AS.choose(fc, AS.pieces(y[1], n) + AS.pieces(y[2], n), 6, corr=corr)
    return lu["strength"], ch["strength"]


def honesty(prof: CameraProfile, src, seed: int = 0) -> list:
    """A.4: the harness against this camera's real flat pairs. One row per pair: the noise the two real frames
    differ by, and the noise two harness flats of the same brightness differ by. [{iso, level, real, harness,
    ratio, ok}], level as a share of the range."""
    from studio.engine.calib import frames as F
    flats = [s for s in F.load_store((prof.make, prof.model)) if s.kind == "flat" and s.crop is not None]
    rows, rng = [], np.random.default_rng(seed)
    by_iso = {}
    for s in flats:
        by_iso.setdefault(float(s.iso), []).append(s)
    for iso in sorted(by_iso):
        if src.gain(iso) is None or prof.entry_for(iso, 0.01) is None:
            continue
        group = sorted(by_iso[iso], key=lambda s: (s.datetime or "", s.name))
        used = 0
        for a, b in zip(group, group[1:]):
            if used >= 3 or a.crop.shape != b.crop.shape:
                continue
            ca, cb = a.crop.astype(np.float32), b.crop.astype(np.float32)
            black = np.asarray(prof.entry_for(iso, 0.01).black_dn, np.float32).reshape(2, 2)
            real, level = [], []
            for i in (0, 1):
                for j in (0, 1):
                    pa, pb = ca[i::2, j::2] - black[i, j], cb[i::2, j::2] - black[i, j]
                    if abs(float(pa.mean()) / max(float(pb.mean()), 1e-6) - 1.0) > 0.03:
                        real = None                               # the light changed between the two: not a pair
                        break
                    d = pa - pb * (float(pa.mean()) / max(float(pb.mean()), 1e-6))
                    d = d - np.asarray(_smooth(d), np.float32)    # what both frames share (falloff, dust) is out
                    real.append(float(np.var(d)) / 2.0)
                    level.append(float(pa.mean()))
                if real is None:
                    break
            if not real:
                continue
            used += 1
            rngdn = src.range_dn(iso)
            flat = np.empty(ca.shape, np.float32)
            for k, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                flat[i::2, j::2] = level[k] / rngdn
            n1, n2 = src.add(flat, iso, rng), src.add(flat, iso, rng)
            d = (n1 - n2) * rngdn
            harness = [float(np.var((d - _smooth(d))[i::2, j::2])) / 2.0 for i in (0, 1) for j in (0, 1)]
            r, h = float(np.sqrt(np.mean(real))), float(np.sqrt(np.mean(harness)))
            rows.append(dict(iso=iso, level=round(float(np.mean(level)) / rngdn, 4), real=round(r, 3),
                             harness=round(h, 3), ratio=round(h / max(r, 1e-9), 4), ok=abs(h / max(r, 1e-9) - 1) <= 0.05))
    return rows


def _smooth(d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    out = np.empty_like(d)
    for i in (0, 1):
        for j in (0, 1):
            out[i::2, j::2] = uniform_filter(d[i::2, j::2], 15, mode="mirror")
    return out


PHOTO_METHODS = ("hybrid", "fusion", "starlet", "nlm")


def real_photos(files: list, methods=PHOTO_METHODS, say=print, opener=None) -> list:
    """A.5: each method on real photographs, where no truth exists. What is reported is estimated from the photo
    itself, at the strength Clean's Auto chooses for it: the error left (1 = the untouched photo's noise) and the
    share of the finest detail kept. [{file, iso, method, kind, luma, chroma, noise_left, kept, note}]"""
    from studio import engine
    from studio.engine.pipeline.pipeline import clean_auto
    rows = []
    for f in files:
        try:
            image = (opener or engine.open)(f)
        except Exception as e:                                # noqa: BLE001  (an unreadable raw is said, not a fault)
            say(f"  {os.path.basename(f)}: could not be read ({e})")
            continue
        for m in methods:
            edit = engine.Edit()
            edit.denoise.enabled, edit.denoise.method = True, m
            t0 = time.time()
            out = clean_auto(image, edit)
            note = (image.auto_report or {}).get("note", "")
            if not out:
                say(f"  {os.path.basename(f)} {m}: {note or 'no estimate'}")
                rows.append(dict(file=os.path.basename(f), iso=image.meta.iso, method=m, note=note))
                continue
            risk = dict(out["luma_table"]).get(out["luma"])
            if out["kind"] == "referee":                      # the referee's error carries a constant: only its
                risk = None                                   # order means something, so it is not shown as a level
            row = dict(file=os.path.basename(f), iso=image.meta.iso, method=m, kind=out["kind"], luma=out["luma"],
                       chroma=out["chroma"], kept=out.get("kept"),
                       noise_left=None if risk is None else round(float(np.sqrt(max(risk, 0.0))), 4),
                       noise_ratio=out.get("noise_ratio"), seconds=round(time.time() - t0, 1), note=note)
            rows.append(row)
            say(f"  {row['file']} ISO {row['iso']:g} {m:<8} luminance {row['luma']:.2f} colour {row['chroma']:.2f}"
                + ("" if row["noise_left"] is None else f"  noise left about {100 * row['noise_left']:.0f}%")
                + ("" if row["kept"] is None else f"  fine detail kept about {100 * row['kept']:.0f}%")
                + ("  (judged by the two halves)" if out["kind"] == "referee" else ""))
    return rows


def crop_png(mos: np.ndarray, model, size: int = 192) -> str:
    from PIL import Image
    y = luminance(mos, model)[0]
    h, w = y.shape
    c = y[h // 2 - size // 2:h // 2 + size // 2, w // 2 - size // 2:w // 2 + size // 2]
    c = np.clip((c - np.percentile(y, 1)) / max(float(np.percentile(y, 99) - np.percentile(y, 1)), 1e-6), 0, 1)
    im = Image.fromarray((c * 255).astype(np.uint8)).resize((size * 2, size * 2), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(prof: CameraProfile, isos: list, quick: bool = False, seed: int = 0, say=print) -> dict:
    bank = D.find_bank(prof.make, prof.model)
    src = S.NoiseSource(prof, bank)
    camera = (prof.make, prof.model)
    cands, scs = candidates(camera, quick), scenes(quick)
    rows, crops, picked = [], {}, {}
    for iso in isos:
        model = src.model(iso)
        real = src.has_real_noise(iso)
        rng = np.random.default_rng(seed + int(iso))
        noisy = {n: src.add(c, iso, rng) for n, c in scs.items()}
        crops[(iso, "noisy")] = crop_png(noisy["dead leaves"], model)
        crops[(iso, "truth")] = crop_png(scs["dead leaves"], model)
        for cname, kw in cands.items():
            for guard in (False, True):
                agg, t0 = {}, time.time()
                for sname, clean in scs.items():
                    k2 = dict(kw)
                    if k2.pop("auto", False):
                        key = (iso, cname, sname)
                        if key not in picked:
                            e = prof.entry_for(iso, 0.01)
                            corr = max(float(e.neighbour_corr), 0.0) if e is not None and e.neighbour_corr else 0.0
                            from studio.engine.raw import auto_strength as AS
                            corr *= AS.read_share([noisy[sname]], model)      # only the read noise is shared
                            picked[key] = auto_strengths(noisy[sname], model, k2["method"], corr)
                        k2["luma"], k2["chroma"] = picked[key]
                    den = DN.denoise_mosaic(noisy[sname], S.RGGB, model, DN.DenoiseParams(guard=guard, **k2))
                    for k, v in score(clean, noisy[sname], den, model, sname).items():
                        agg.setdefault(k, []).append(v)
                    if sname == "dead leaves":
                        crops[(iso, cname, guard)] = crop_png(den, model)
                row = dict(iso=iso, method=cname, guard=guard, real_noise=real, seconds=round(time.time() - t0, 2),
                           **{k: round(float(np.mean(v)), 4) for k, v in agg.items()})
                rows.append(row)
                say(f"  ISO {iso:<7g} {cname:<8} guard {'on ' if guard else 'off'}  noise left "
                    f"{100 * row['noise_left']:5.1f}%  texture kept {100 * row.get('texture_kept', 0):5.1f}%  "
                    f"blotches {100 * row.get('blotches_left', 0):5.1f}%  residual {row['residual']:.2f}")
    return dict(camera=prof.model, made=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), quick=quick, rows=rows,
                crops={"|".join(str(x) for x in k): v for k, v in crops.items()})


def page(res: dict) -> str:
    cols = [("noise_left", "Noise left", False), ("texture_kept", "Texture kept", True),
            ("edges_kept", "Edges kept", True), ("blotches_left", "Colour blotches left", False),
            ("residual", "Residual (1 = only noise)", None), ("seconds", "Seconds", False)]
    out = [f"<html><head><meta charset='utf-8'><title>Clean on {html.escape(res['camera'])}</title><style>"
           "body{background:#07090f;color:#e6ecff;font-family:Segoe UI,sans-serif;margin:32px}"
           "table{border-collapse:collapse;margin:10px 0 28px}td,th{padding:6px 12px;border-bottom:1px solid #1c2a4a;"
           "text-align:right}th{color:#8a97b8;font-weight:500}td:first-child,th:first-child{text-align:left}"
           ".best{color:#22c55e;font-weight:600}.worst{color:#f59e0b}.dim{color:#8a97b8}"
           "figure{display:inline-block;margin:0 10px 14px 0}figcaption{color:#8a97b8;font-size:12px}"
           "img{border:1px solid #1c2a4a;border-radius:8px}</style></head><body>"
           f"<h1>Clean, scored on the {html.escape(res['camera'])}'s own noise</h1>"
           f"<p class=dim>{res['made']}. Truth is known for every picture here. Green is the best in its column "
           f"for that ISO, amber the worst. 'Residual' is what was removed, measured against pure noise: 1.0 means "
           f"only noise was taken; above that, detail went with it.</p>"]
    for iso in sorted({r["iso"] for r in res["rows"]}):
        rows = [r for r in res["rows"] if r["iso"] == iso]
        out.append(f"<h2>ISO {iso:g}</h2>" + ("" if rows[0]["real_noise"] else
                   "<p class=worst>No dark frames in the noise bank for this ISO: formula noise was used.</p>"))
        out.append("<table><tr><th>Method</th>" + "".join(f"<th>{c[1]}</th>" for c in cols) + "</tr>")
        for r in rows:
            cells = []
            for key, _, high in cols:
                vals = [x[key] for x in rows if key in x]
                v = r.get(key)
                cls = ""
                if v is not None and high is not None and len(vals) > 1:
                    cls = "best" if v == (max(vals) if high else min(vals)) else \
                          "worst" if v == (min(vals) if high else max(vals)) else ""
                txt = "-" if v is None else f"{v:.2f}" if key in ("residual", "seconds") else f"{100 * v:.1f}%"
                cells.append(f"<td class='{cls}'>{txt}</td>")
            out.append(f"<tr><td>{r['method']}{' + detail guard' if r['guard'] else ''}</td>{''.join(cells)}</tr>")
        out.append("</table>")
        figs = [("truth", "the truth"), ("noisy", "as the camera records it")]
        figs += [(f"{m}|{g}", f"{m}{' + guard' if g else ''}") for m in dict.fromkeys(r["method"] for r in rows)
                 for g in (False, True)]
        for key, cap in figs:
            src = res["crops"].get(f"{iso}|{key}")
            if src:
                out.append(f"<figure><img src='{src}' width=288><figcaption>{html.escape(cap)}</figcaption></figure>")
    hon = res.get("honesty")
    if hon is not None:
        out.append("<h2>Is this benchmark honest?</h2>")
        if not hon:
            out.append("<p class=worst>No flat pairs of this camera are stored on this PC, so the harness could not "
                       "be checked against real frames. Drop the flat pairs on the Cameras page.</p>")
        else:
            good = sum(1 for r in hon if r["ok"])
            top = max(r["level"] for r in hon)
            out.append(f"<p class=dim>Each row is a pair of this camera's real flat frames and a flat of the same "
                       f"brightness made by the harness. The two noise figures should agree within 5%: "
                       f"<b class='{'best' if good == len(hon) else 'worst'}'>{good} of {len(hon)} do</b>. The "
                       f"brightest real flat is at {100 * top:.0f}% of the range; above that the harness has not "
                       f"been checked" + (" (shoot six brighter pairs to check it)." if top < 0.35 else ".") + "</p>")
            out.append("<table><tr><th>ISO</th><th>Brightness</th><th>Real noise (DN)</th><th>Harness (DN)</th>"
                       "<th>Harness / real</th></tr>")
            for r in hon:
                out.append(f"<tr><td>{r['iso']:g}</td><td>{100 * r['level']:.1f}%</td><td>{r['real']:.2f}</td>"
                           f"<td>{r['harness']:.2f}</td><td class='{'best' if r['ok'] else 'worst'}'>"
                           f"{r['ratio']:.3f}</td></tr>")
            out.append("</table>")
    ph = res.get("photos")
    if ph:
        out.append("<h2>Your own photos (no truth: estimated from each photo)</h2><p class=dim>At the strength "
                   "Clean's Auto chooses for each method. 'Noise left' and 'fine detail kept' are estimates made "
                   "from the photo alone (amber), or measured by the two halves of a Dual Pixel photo (cyan).</p>"
                   "<table><tr><th>Photo</th><th>ISO</th><th>Method</th><th>Luminance</th><th>Colour</th>"
                   "<th>Noise left</th><th>Fine detail kept</th><th>How</th></tr>")
        for r in ph:
            if "luma" not in r:
                out.append(f"<tr><td>{html.escape(r['file'])}</td><td>{r['iso']:g}</td><td>{r['method']}</td>"
                           f"<td colspan=5 class=dim>{html.escape(r.get('note') or 'no estimate')}</td></tr>")
                continue
            col = "#22d3ee" if r["kind"] == "referee" else "#f59e0b"
            how = ("two halves, noise " + (f"{r['noise_ratio']:.2f}" if r.get("noise_ratio") else "?")
                   + " x the profile") if r["kind"] == "referee" else "estimated"
            nl = "-" if r["noise_left"] is None else f"{100 * r['noise_left']:.0f}%"
            kp = "-" if r["kept"] is None else f"{100 * r['kept']:.0f}%"
            out.append(f"<tr><td>{html.escape(r['file'])}</td><td>{r['iso']:g}</td><td>{r['method']}</td>"
                       f"<td>{r['luma']:.2f}</td><td>{r['chroma']:.2f}</td><td style='color:{col}'>{nl}</td>"
                       f"<td style='color:{col}'>{kp}</td><td class=dim>{how}</td></tr>")
        out.append("</table>")
    return "".join(out) + "</body></html>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera")
    ap.add_argument("--isos")
    ap.add_argument("--quick", action="store_true", help="smaller charts, no MP-PCA: a few minutes instead of many")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--photos", help="a folder of your own raws to score as well (estimated: they have no truth)")
    ap.add_argument("folder", nargs="?", help="the same as --photos (a folder dragged onto the .bat)")
    a = ap.parse_args(argv)
    slog.setup(console_level=None)
    from studio.lab.train import pick_profile
    prof = pick_profile(a.camera)
    src = S.NoiseSource(prof, D.find_bank(prof.make, prof.model))
    have = src.isos()
    if not have:
        print(f"The {prof.model} has no gain in its profile yet: shoot flat pairs (Cameras page, card 3).")
        return 1
    if D.find_bank(prof.make, prof.model) is None:
        print("No noise bank yet (Cameras page > Deep calibration): formula noise is used. Real noise is the point "
              "of this benchmark, so do that first if you can.")
    want = [float(x) for x in a.isos.split(",")] if a.isos else [400, 3200, 12800, 51200]
    isos = [min(have, key=lambda h: abs(np.log2(h / w))) for w in want]
    isos = sorted(set(isos))
    print(f"Clean on the {prof.model}, ISO {', '.join(f'{i:g}' for i in isos)}:")
    res = run(prof, isos, a.quick)
    res["honesty"] = honesty(prof, src)
    good = sum(1 for r in res["honesty"] if r["ok"])
    print(f"\nHarness against this camera's real flat pairs: {good} of {len(res['honesty'])} within 5%."
          if res["honesty"] else "\nNo flat pairs stored for this camera: the harness was not checked against them.")
    folder = a.photos or a.folder
    if folder:
        from studio.lab.data import list_raws
        files = list_raws(folder)[: 6 if a.quick else 24]
        print(f"\nYour photos in {folder} ({len(files)} scored):")
        res["photos"] = real_photos(files)
    out = paths.get("OUTPUTS") / "noisebench" / _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page(res), encoding="utf-8")
    json.dump({k: v for k, v in res.items() if k != "crops"}, open(out / "results.json", "w", encoding="utf-8"),
              indent=1)
    print(f"\nReport: {out / 'index.html'}")
    if not a.no_open:
        import webbrowser
        webbrowser.open((out / "index.html").as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

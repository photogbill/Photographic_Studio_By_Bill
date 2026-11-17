"""studio-cli: the command-line runner behind the .bat files and the tests (section 2.2 / 15).

    python -m studio.cli develop FILES_OR_FOLDERS... [options]   raw -> 16-bit Adobe RGB TIFF (default)
    python -m studio.cli info FILE                               metadata, true ISO, black/white levels
    python -m studio.cli import SOURCE [--to DIR] [--name TEXT]   copy photos from a card into the Photos folder
    python -m studio.cli diag                                    diagnostic report (for bug reports)
    python -m studio.cli version

Outputs go to the app's outputs\\develop folder unless --out is given. Nothing is written to C:.
"""
from __future__ import annotations

import argparse

import numpy as np
import datetime as _dt
import json
import os
import sys
import time
import traceback
import webbrowser

from . import __version__, log as slog, paths

log = slog.get("cli")
SKIP_DIRS = {".venv", "venv", "reports", "logs", "outputs", "cache", "__pycache__", ".git", "calibration_local"}


def collect(items: list[str], recursive: bool = False) -> list[str]:
    from .engine import RAW_EXTENSIONS
    out = []
    for it in items:
        it = it.strip().strip('"')
        if os.path.isdir(it):
            if recursive:
                for root, dirs, files in os.walk(it):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                    out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(RAW_EXTENSIONS)]
            else:
                out += [os.path.join(it, f) for f in sorted(os.listdir(it)) if f.lower().endswith(RAW_EXTENSIONS)]
        elif os.path.isfile(it):
            if it.lower().endswith(RAW_EXTENSIONS + (".tif", ".tiff", ".jpg", ".jpeg", ".png")):
                out.append(it)
            else:
                print(f"  skipped (not a raw file): {it}")
        else:
            print(f"  not found: {it}")
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def apply_flags(edit, a):
    """Command-line options override the sidecar / defaults."""
    if a.exposure is not None:
        edit.exposure_ev = a.exposure
    if a.demosaic:
        edit.demosaic = a.demosaic
    if getattr(a, "green_eq", False):
        edit.green_eq = True
    if a.tone:
        edit.tone.mode = a.tone
    if a.match_camera_brightness:
        edit.tone.match_camera_brightness = True
    if a.wb:
        edit.white_balance.mode = a.wb
    if a.denoise is not None:
        edit.denoise.enabled = True
        from .engine.pipeline.edit import Denoise
        edit.denoise.luma = Denoise().luma * a.denoise
        edit.denoise.chroma = a.denoise_chroma if a.denoise_chroma is not None else Denoise().chroma * a.denoise
    if a.temp:
        edit.white_balance.mode = "temp_tint"
        edit.white_balance.temperature = a.temp
        edit.white_balance.tint = a.tint or 0.0
    if getattr(a, "microcontrast", None) is not None:
        edit.microcontrast.amount = max(0.0, min(1.0, a.microcontrast))
    if getattr(a, "detail", None):
        from .engine.detail.scales import METHODS, PRESETS
        spec = a.detail.strip()
        vals = dict(PRESETS[spec]) if spec in PRESETS else {}
        if not vals:
            for part in spec.split(","):
                k, _, v = part.partition("=")
                k = k.strip().lower()
                if k not in METHODS:
                    raise SystemExit(f"--detail: unknown method {k!r}. Methods: {', '.join(METHODS)}; "
                                     f"presets: {', '.join(PRESETS)}")
                vals[k] = float(v)
        for k, v in vals.items():
            v = max(float(METHODS[k]["lo"]), min(float(METHODS[k]["hi"]), float(v)))
            if k == "local":
                edit.microcontrast.amount = v / 100.0
            else:
                setattr(edit.detail, k, v)
    if getattr(a, "hdr", None):
        from .engine.color.hdr import ENGINES, STRENGTHS
        strength, _, engine = a.hdr.strip().lower().partition(":")
        if strength not in STRENGTHS or (engine and engine not in ENGINES):
            raise SystemExit(f"--hdr: use one of {', '.join(STRENGTHS)}, optionally with an engine, e.g. "
                             f"medium:laplacian (engines: {', '.join(ENGINES)})")
        edit.hdr.enabled, edit.hdr.strength = True, strength
        edit.hdr.engine = engine or "fusion"
        if getattr(a, "hdr_amount", None) is not None:
            edit.hdr.amount = max(0.0, min(150.0, a.hdr_amount))
    if getattr(a, "sharpen", None) is not None:
        edit.sharpen.enabled = a.sharpen > 0
        edit.sharpen.amount = a.sharpen
    if getattr(a, "rotate", None) is not None:
        edit.geometry.rotate = a.rotate
    if getattr(a, "smart_lighting", None) is not None:
        edit.light.smart = max(0.0, min(100.0, a.smart_lighting))
    if getattr(a, "highlights", None):
        edit.highlight_mode = a.highlights
    if getattr(a, "glow_profile", None) is not None:
        edit.glow_profile = a.glow_profile
    if getattr(a, "dehaze", None) is not None:
        edit.light.dehaze = max(-100.0, min(100.0, a.dehaze))
    if getattr(a, "texture", None) is not None:
        edit.finish.texture = max(-100.0, min(100.0, a.texture))
    if getattr(a, "grain", None) is not None:
        edit.finish.grain = max(0.0, min(100.0, a.grain))
    if getattr(a, "profile", None):
        edit.profile = a.profile
    if getattr(a, "look", None):
        from .engine.presets import looks
        lk = looks.find_look(a.look)
        if lk is None:
            raise SystemExit(f"no look called {a.look!r}. Looks: {', '.join(x['name'] for x in looks.list_looks())}")
        edit = looks.apply_look(edit, lk)
    return edit


def cmd_develop(a) -> int:
    from . import engine
    from .compare import CompareBuilder
    files = collect(a.inputs or [str(paths.APP_ROOT)], a.recursive)
    if not files:
        print("No raw files found. Drag CR2/CR3 files (or a folder) onto 'Develop Raws.bat'.")
        return 2
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = a.out or str(paths.get("OUTPUTS") / "develop")
    os.makedirs(out_dir, exist_ok=True)
    spec = engine.ExportSpec(format=a.format, space=a.space, bits=a.bits, quality=a.quality, long_edge=a.long_edge,
                             compression=a.compression, sharpen_for=getattr(a, "sharpen_for", "none") or "none")
    spec.validate()
    print(f"Developing {len(files)} raw file(s) -> {out_dir}")
    done, failed = 0, 0
    taken: set = set()                      # two files with the same name do not replace each other
    builder = CompareBuilder(a.compare_dir or os.path.join(out_dir, f"compare_{stamp}"),
                             apply_flags(engine.Edit(), a)) if a.compare else None
    for i, f in enumerate(files, 1):
        t0 = time.perf_counter()
        if builder is not None and a.compare_only and builder.has(f):
            print(f"[{i}/{len(files)}] {os.path.basename(f)} (already on the page)")
            continue
        print(f"[{i}/{len(files)}] {os.path.basename(f)}")
        try:
            img = engine.open(f)
            edit = apply_flags(engine.load_edit(f) if a.use_sidecar else engine.Edit(), a)
            if getattr(a, "fix_lens", False):
                from .engine.lens.fix import fix_lens
                edit, rep = fix_lens(img, edit)
                for r in rep:
                    print(f"      {r['what']}: {r['text']}")
            if edit.highlight_mode == "glow":
                if not edit.glow_profile:
                    edit.glow_profile = engine.pick_glow_profile(img, edit)
                rep = engine.glow_report(img, edit)
                if rep:
                    tiers = [r["tier"] for r in rep]
                    print(f"      lens glow ({edit.glow_profile or 'no profile'}): {len(rep)} blown light(s), "
                          f"{tiers.count('lens')} measured from the profile, {tiers.count('photo')} from the photo")
            if getattr(a, "dust_map", False):
                spots = engine.dust_from_map(img, edit)
                if spots is None:
                    print("      dust map: none for this camera yet (Library: Learn dust map)")
                else:
                    edit.retouch = (edit.retouch or []) + spots
                    print(f"      dust map: {len(spots)} spot(s) healed")
            p = None
            if not a.compare_only:
                p = engine.export(img, edit, spec, out_dir=out_dir, taken=taken)
                print(f"      wrote {os.path.basename(p)}  ({img.meta.model or 'camera ?'}, {img.meta.iso_label}, "
                      f"{time.perf_counter() - t0:.1f} s)")
            if builder is not None:
                img.drop_cache()
                builder.add(img, p)
            img.drop_cache()
            del img
            done += 1
        except Exception as e:
            failed += 1
            slog.log_exception(f"develop failed: {f}", log)
            print(f"      PROBLEM with {os.path.basename(f)}: {e}")
            rep = None if slog.expected_problem(e) else slog.save_error_report(
                f"Develop Raws could not develop {os.path.basename(f)}")
            print(f"      (an error report was saved: {rep})" if rep else "      (details are in logs\\studio.log)")
    if builder is not None and builder.cards:
        page = builder.write()
        print(f"Comparison page: {page}")
        if not a.no_open:
            try:
                webbrowser.open("file:///" + os.path.abspath(page).replace("\\", "/"))
            except Exception:
                pass
    print(f"Done: {done} developed, {failed} failed.")
    return 0 if failed == 0 else 1


def cmd_info(a) -> int:
    from . import engine
    rc = 0
    for f in collect(a.inputs, False) or a.inputs:
        try:
            img = engine.open(f)
            lv = img.levels()
            r = img.raw
            d = {
                "file": os.path.basename(f),
                "metadata": slog.scrub(img.meta.to_dict()),
                "sensor": {"visible": [r.width, r.height], "margins": {"top": r.top, "left": r.left},
                           "camera_crop": list(r.crop), "cfa": r.color_desc, "flip": r.flip},
                "levels": {"black_source": lv.source, "black": lv.black.round(2).tolist(),
                           "black_libraw": r.black_for_positions(r.black_libraw).tolist(), "white": lv.white,
                           "read_noise_dn": None if lv.read_noise_dn is None else lv.read_noise_dn.round(2).tolist()},
                "white_balance_as_shot": [round(float(x), 4) for x in r.wb_as_shot],
                "notes": r.notes,
            }
            print(json.dumps(d, indent=2, default=str))
        except Exception as e:
            rc = 1
            slog.log_exception(f"info failed: {f}", log)
            print(f"PROBLEM with {f}: {e}")
    return rc


def cmd_calibrate(a) -> int:
    from .engine.calib.builder import build_profile
    from .engine.io.rawfile import open_raw
    files = collect(a.inputs, recursive=True)
    if not files:
        print("No raw files found. Drag the folder of calibration frames onto 'Calibrate Camera.bat'.")
        return 2
    print(f"Reading {len(files)} calibration frame(s)...")
    raws = []
    for i, f in enumerate(files, 1):
        try:
            raws.append(open_raw(f))
            print(f"  [{i}/{len(files)}] {os.path.basename(f)}")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {os.path.basename(f)}: could not read ({e})")
    prof, rep = build_profile(raws)
    del raws
    path = prof.save()
    print("")
    print(f"Camera: {prof.make} {prof.model}   Profile: {prof.method.upper()}   Quality score: {prof.quality['score']}")
    print(f"Frames: {rep.counts()}")
    for inf in rep.frames:
        if inf.kind == "other":
            print(f"  ignored {os.path.basename(inf.path)}: {inf.reason}")
    print("")
    print("  ISO      black (DN)   read noise (DN)  row band  col band  gain (e-/DN)  hot px")
    for e in prof.isos:
        rn = np.mean(e.read_noise_dn) if e.read_noise_dn else float("nan")
        bl = np.mean(e.black_dn) if e.black_dn else float("nan")
        g = f"{e.gain_e_per_dn:.4f}" if e.gain_e_per_dn else "  (quick)"
        print(f"  {e.iso:<8g} {bl:10.2f}   {rn:12.2f}   {e.row_banding_dn or 0:8.2f}  {e.col_banding_dn or 0:8.2f}"
              f"  {g:>12}  {e.hot_pixels:6d}")
    for w in rep.warnings:
        print(f"  note: {w}")
    print(f"\nSaved (private, on this PC only): {path}")
    print("Nothing is shared unless you choose to share it later.")
    return 0


def cmd_blindtest(a) -> int:
    from . import engine
    from .blindtest import build_dxo_test
    folder = a.folder or str(paths.APP_ROOT / "Voigtlander")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(str(paths.get("OUTPUTS")), "blindtest", f"dxo_{stamp}")
    edit = engine.Edit()
    edit.denoise.enabled = True
    if a.denoise is not None:
        edit.denoise.luma *= a.denoise
        edit.denoise.chroma *= a.denoise
    print(f"Building a blind test from {folder} ...")
    page = build_dxo_test(folder, out, max_pairs=a.pairs, edit=edit)
    print(f"Blind test page: {page}")
    if not a.no_open:
        try:
            webbrowser.open("file:///" + os.path.abspath(page).replace("\\", "/"))
        except Exception:
            pass
    return 0


def cmd_detail_preview(a) -> int:
    from .detailpreview import build
    files = collect(a.inputs or [str(paths.APP_ROOT / "Voigtlander")], False)
    if not files:
        files = collect([str(paths.APP_ROOT)], False)
    if not files:
        print("No raw files found. Drag CR2/CR3 files (or a folder) onto 'Detail Preview.bat'.")
        return 2
    files = files[:a.max]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(str(paths.get("OUTPUTS")), "detail_preview", stamp)
    print(f"Detail preview of {len(files)} photo(s) -> {out}")
    page = build(files, out, mc_amount=a.microcontrast, sharpen_amount=a.sharpen)
    print(f"Page: {page}")
    if not a.no_open:
        try:
            webbrowser.open("file:///" + os.path.abspath(page).replace("\\", "/"))
        except Exception:
            pass
    return 0


def cmd_diag(a) -> int:
    text = slog.diagnostic_report()
    p = paths.get("LOGS") / "diagnostic_report.txt"
    p.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {p}. Attach it to a bug report; it contains no photos, serial numbers or locations.")
    return 0


def _photos_dir_setting() -> str:
    try:
        d = json.loads((paths.get("SETTINGS") / "app.json").read_text(encoding="utf-8"))
        return str(d.get("photos_dir") or "")
    except (OSError, ValueError):
        return ""


def cmd_import(a) -> int:
    """Import from card (plan section 24): the same engine and rules as the Studio's Import from card dialog."""
    from .engine.io import ingest as I
    src = a.source.strip().strip('"')
    if not os.path.isdir(src):
        print(f"Not found: {src}")
        return 2
    dest = a.to or _photos_dir_setting()
    if not dest:
        print("No Photos folder set yet. Use --to FOLDER, or choose it once in the Studio "
              "(Library > Import from card).")
        return 2
    print(f"Looking at {src} ...")
    scan = I.scan_source(src, videos=a.videos)
    I.confirm_capture_times(scan)
    opts = I.ImportOptions(dest=dest, raws=not a.no_raws, jpegs=a.jpegs, videos=a.videos, second=a.second or "",
                           show_imported_before=a.show_imported, by_year=not a.no_year, day_start=a.day_start)
    if a.name:
        opts.one_folder, opts.names = True, {"*": a.name}
    plan = I.plan_import(scan, opts)
    print(f"{scan.photos} photos on the source ({scan.count(I.RAW)} raw, {scan.count(I.JPEG)} JPEG/HEIF, "
          f"{scan.count(I.VIDEO)} video).")
    for g in plan.groups:
        print(f"  {g.key}: {g.new} new of {g.total}  ->  {g.folder}")
    if plan.imported_before:
        print(f"  {plan.imported_before} imported before (skipped; --show-imported copies them again)")
    if plan.left_text():
        print("  " + plan.left_text())
    if plan.no_raws:
        print("  This card has no raw files, only JPEGs (add --jpegs to copy them).")
    print(f"Needs {I.human_bytes(plan.bytes_needed)}; free on the destination: {I.human_bytes(plan.free)}.")
    for w in plan.warnings:
        print("  Note: " + w)
    if plan.problem:
        print("Cannot import: " + plan.problem)
        return 1
    if not plan.fits or not plan.second_fits:
        print(f"Not enough free space: {I.human_bytes(plan.shortfall or plan.second_shortfall)} short "
              f"(the Studio keeps 2 GB or 5% of the drive free).")
        return 1
    if a.dry_run:
        for it in plan.items:
            where = it.dest or ""
            extra = f"  (if different: {it.alt})" if it.alt else ""
            print(f"  {it.action:9s} {it.src.rel}  {('-> ' + where) if where else ''}{extra}")
        print("Dry run: nothing was copied.")
        return 0
    if not plan.to_copy:
        print("Nothing new to copy.")
        return 0

    def progress(p):
        pct = 100.0 * p["bytes_done"] / p["bytes_total"] if p["bytes_total"] else 100.0
        print(f"\r  {p['done']}/{p['total']} files  {pct:5.1f}%  {p['current'][:40]:40s}", end="", flush=True)
    rep = I.run_import(plan, on_progress=progress)
    print()
    for line in rep.summary_lines():
        print(line)
    if rep.all_checked:
        print("All photos are copied and checked. You can take the card out. The Studio wrote nothing to the card.")
    print(f"Log: {rep.log_path}")
    return 0 if rep.all_checked else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="studio-cli", description="Photographic Studio By Bill - command line")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("develop", help="develop raw files to TIFF/PNG/JPEG")
    d.add_argument("inputs", nargs="*")
    d.add_argument("--out", help="output folder (default: outputs\\develop beside the app)")
    d.add_argument("--format", default="tiff", choices=["tiff", "png", "jpeg"])
    d.add_argument("--space", default="adobergb", choices=["adobergb", "srgb", "prophoto", "displayp3"])
    d.add_argument("--bits", type=int, default=16, choices=[8, 16])
    d.add_argument("--quality", type=int, default=95)
    d.add_argument("--long-edge", type=int, default=None)
    d.add_argument("--compression", default="none", choices=["none", "zip"])
    d.add_argument("--exposure", type=float, default=None, help="exposure in EV")
    d.add_argument("--demosaic", choices=["rcd", "menon", "lmmse", "dual", "mhc", "bilinear", "half"],
                   help="demosaic method (default rcd; edits made before 2026-11-15 keep menon)")
    d.add_argument("--green-eq", action="store_true", help="even out the two greens before demosaicing")
    d.add_argument("--tone", choices=["standard", "camera_match", "linear"])
    d.add_argument("--match-camera-brightness", action="store_true",
                   help="optional: shift brightness per camera body to match the camera JPEG (off by default)")
    d.add_argument("--wb", choices=["as_shot", "daylight"])
    d.add_argument("--denoise", type=float, nargs="?", const=1.0, default=None,
                   help="raw denoise (experimental, Phase 3): overall strength, 1 = default")
    d.add_argument("--denoise-chroma", type=float, default=None, help="colour-noise strength (default 1.5x)")
    d.add_argument("--temp", type=float, help="white balance temperature in K")
    d.add_argument("--tint", type=float)
    d.add_argument("--microcontrast", type=float, default=None, help="microcontrast 0..1 (shades of shadow)")
    d.add_argument("--hdr", default=None, metavar="STRENGTH[:ENGINE]",
                   help="Auto HDR on each raw: light, medium, heavy, highlights, lowkey or shadows; optional engine "
                        "fusion (default), laplacian or retinex, e.g. --hdr medium:laplacian")
    d.add_argument("--hdr-amount", type=float, default=None, help="Auto HDR amount in percent (0-150, default 100)")
    d.add_argument("--detail", default=None, metavar="SPEC",
                   help='Detail by scale: a preset ("Tonal", "Crisp", "Both", "Soft glow") or method amounts in '
                        'percent, e.g. "tonal=70,crisp=30,veil=40" (methods: tonal, local, crisp, separation, '
                        'veil, broad)')
    d.add_argument("--sharpen", type=float, nargs="?", const=0.5, default=None,
                   help="capture sharpening (recover lens sharpness), amount 0..1.5 (default 0.5)")
    d.add_argument("--fix-lens", action="store_true", help="run Fix lens on every photo (as the Develop button)")
    d.add_argument("--rotate", type=float, default=None, help="straighten, degrees (+ = counter-clockwise)")
    d.add_argument("--look", default=None, help='apply a look by name, e.g. "Warm film" (never changes exposure)')
    d.add_argument("--smart-lighting", type=float, default=None, help="Smart lighting 0..100 (opens deep shadows)")
    d.add_argument("--highlights", choices=["clip", "rebuild", "glow"], default=None,
                   help="clipped highlights: plain clip, rebuild colour, or measure from the lens glow")
    d.add_argument("--glow-profile", default=None, help="lens glow profile name (settings\\glow); default: by lens")
    d.add_argument("--dehaze", type=float, default=None, help="Dehaze -100..100")
    d.add_argument("--texture", type=float, default=None, help="Texture -100..100 (fine detail)")
    d.add_argument("--grain", type=float, default=None, help="film grain 0..100")
    d.add_argument("--dust-map", action="store_true", help="heal this camera's dust (its dust map) in every photo")
    d.add_argument("--profile", default=None, help="camera profile (.dcp file, or a name in settings\\camera_profiles)")
    d.add_argument("--sharpen-for", choices=["none", "screen", "glossy", "matte"], default="none",
                   help="output sharpening after resizing")
    d.add_argument("--compare", action="store_true", help="also build the camera-JPEG comparison page")
    d.add_argument("--compare-only", action="store_true", help="only build the comparison page (no exports)")
    d.add_argument("--compare-dir", help="add to an existing comparison page folder (resumes where it stopped)")
    d.add_argument("--recursive", action="store_true", help="include subfolders")
    d.add_argument("--no-sidecar", dest="use_sidecar", action="store_false", help="ignore .psb.json sidecars")
    d.add_argument("--no-open", action="store_true", help="do not open the comparison page")
    i = sub.add_parser("info", help="show metadata, true ISO and levels")
    i.add_argument("inputs", nargs="+")
    c = sub.add_parser("calibrate", help="build a camera profile from calibration frames (darks, flat pairs)")
    c.add_argument("inputs", nargs="+")
    b = sub.add_parser("blindtest", help="blind test: Studio vs your DxO exports (NAME_DxO.jpg beside the raw)")
    b.add_argument("folder", nargs="?")
    b.add_argument("--pairs", type=int, default=10)
    b.add_argument("--denoise", type=float, default=None, help="overall denoise strength (1 = default)")
    b.add_argument("--no-open", action="store_true")
    p = sub.add_parser("detail-preview", help="microcontrast and capture sharpening on your raws, next to DxO")
    p.add_argument("inputs", nargs="*")
    p.add_argument("--max", type=int, default=6, help="at most this many photos (default 6)")
    p.add_argument("--microcontrast", type=float, default=0.8)
    p.add_argument("--sharpen", type=float, default=0.6)
    p.add_argument("--no-open", action="store_true")
    m = sub.add_parser("import", help="copy photos from a card (or a folder) into your Photos folder, checked")
    m.add_argument("source", help="the card (e.g. F:\\) or a folder")
    m.add_argument("--to", default=None, help="the Photos folder (default: the one chosen in the Studio)")
    m.add_argument("--name", default=None, help="put everything in one folder with this name (default: one "
                   "folder per shooting day, named by its date)")
    m.add_argument("--jpegs", action="store_true", help="also copy JPEG/HEIF files")
    m.add_argument("--videos", action="store_true", help="also copy videos")
    m.add_argument("--no-raws", action="store_true", help="do not copy raw files")
    m.add_argument("--second", default=None, help="also copy to a second drive (same folders)")
    m.add_argument("--show-imported", action="store_true", help="copy photos imported before again")
    m.add_argument("--no-year", action="store_true", help="no year level (Photos\\2026-11-14, not "
                   "Photos\\2026\\2026-11-14)")
    m.add_argument("--day-start", default="04:00", help="a new day starts at this time (default 04:00)")
    m.add_argument("--dry-run", action="store_true", help="show what would happen; copy nothing")
    sub.add_parser("diag", help="write the diagnostic report")
    sub.add_parser("version")
    return ap


def main(argv=None) -> int:
    slog.setup()
    ap = build_parser()
    a = ap.parse_args(argv)
    log.debug("studio-cli %s args=%s", __version__, argv if argv is not None else sys.argv[1:])
    try:
        if a.cmd == "develop":
            if a.compare_only:
                a.compare = True
            return cmd_develop(a)
        if a.cmd == "info":
            return cmd_info(a)
        if a.cmd == "calibrate":
            return cmd_calibrate(a)
        if a.cmd == "blindtest":
            return cmd_blindtest(a)
        if a.cmd == "detail-preview":
            return cmd_detail_preview(a)
        if a.cmd == "import":
            return cmd_import(a)
        if a.cmd == "diag":
            return cmd_diag(a)
        if a.cmd == "version":
            print(f"Photographic Studio By Bill {__version__}")
            return 0
        ap.print_help()
        return 2
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except Exception as e:
        log.error("unhandled error\n%s", traceback.format_exc())
        print(f"PROBLEM: {e}\nDetails are in logs\\studio.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())

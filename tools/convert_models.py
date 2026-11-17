"""Convert downloaded models for the Studio, on this PC (Convert Models.bat runs this).

    python tools/convert_models.py [--force] [--check-only]

Looks in the models folder (and the app folder) for downloads it knows, converts each for the Studio's NumPy runner
and checks the result runs:
    model.onnx (Depth Anything V2 Small, onnx-community)        -> models/depth/depth_anything_v2_small.npz
    sam2.1_hiera_tiny_*.zip  (SAM 2.1 Hiera-Tiny, ONNX export)   -> models/sam/sam2.1_tiny_{encoder,decoder}.npz
    sam2.1_hiera_large_*.zip (SAM 2.1 Hiera-Large, ONNX export)  -> models/sam/sam2.1_large_{encoder,decoder}.npz
    lama_fp32.onnx (LaMa, Carve/LaMa-ONNX, Apache-2.0)           -> models/inpaint/lama.npz (Retouch > Fill)
Anything else is left alone. A model already converted (and newer than its download) is skipped unless --force.
Work files go in the app's cache folder and are removed afterwards; nothing is written to C:\\ or the user profile.
Needs onnx, plus onnxsim (or onnxslim) to fold the graphs; onnxruntime, when present, is used to check the answers.
Every step is logged to logs/convert_models.log.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tools"))
MODELS = os.path.join(APP, "models")
WORK = os.path.join(APP, "cache", "convert_models")
LOG = os.path.join(APP, "logs", "convert_models.log")
log = logging.getLogger("convert_models")


def say(msg: str) -> None:
    print("  " + msg, flush=True)
    log.info(msg)


def fold(src: str, dst: str, shapes: dict) -> str:
    """Fold to fixed input shapes: onnxsim if it is installed, else onnxslim (pure Python)."""
    import onnx
    try:
        import onnxsim
        m, ok = onnxsim.simplify(onnx.load(src), overwrite_input_shapes=shapes)
        if not ok:
            raise RuntimeError("onnxsim: the folded model does not match the original")
        log.info("folded with onnxsim %s", getattr(onnxsim, "__version__", "?"))
    except ImportError:
        import onnxslim
        m = onnxslim.slim(src, input_shapes=[f"{k}:{','.join(map(str, v))}" for k, v in shapes.items()])
        log.info("folded with onnxslim %s", getattr(onnxslim, "__version__", "?"))
    onnx.save(m, dst)
    return dst


def convert(src: str, dst: str, f16: bool) -> None:
    import convert_onnx_model as cv
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    g = cv.convert(src, tmp, f16)
    os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, dst)
    log.info("wrote %s: %d layers, %.1f MB", dst, len(g["nodes"]), os.path.getsize(dst) / 1e6)


def check(npz: str, onnx_path: str, inputs: dict, outputs: list, region=None, limit: float = 0.0) -> str:
    """Run the converted model; against ONNX Runtime too when it is installed. A one-line verdict. ``region``: a
    boolean (h, w) area judged on its own as well (for LaMa: the hole, where the model makes things up); a
    difference there above ``limit`` (a fraction) is a failed conversion."""
    from studio.engine.local.onnxnp import Graph
    t = time.perf_counter()
    got = Graph.load(npz)(**inputs)
    secs = time.perf_counter() - t
    for k in outputs:
        if not np.all(np.isfinite(got[k])):
            raise RuntimeError(f"{os.path.basename(npz)}: {k} has invalid numbers")
    try:
        import onnxruntime as ort
    except ImportError:
        return f"runs ({secs:.1f} s); onnxruntime not installed, so not compared"
    ref = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(outputs, inputs)
    errs = [float(np.abs(got[k] - r).mean() / (np.abs(r).mean() + 1e-12)) for k, r in zip(outputs, ref)]
    txt = (f"runs ({secs:.1f} s); on a made-up test picture it differs from the original by "
           f"{100 * max(errs):.2f}% on average")
    if region is not None:
        k, r = outputs[0], ref[0]
        sel = np.broadcast_to(region, r.shape)
        e = float(np.abs(got[k][sel] - r[sel]).mean() / (np.abs(r[sel]).mean() + 1e-12))
        txt += f", {100 * e:.2f}% inside the hole"
        if limit and e > limit:
            raise RuntimeError(f"{os.path.basename(npz)}: differs from the original by {100 * e:.1f}% inside the "
                               f"hole (limit {100 * limit:.0f}%)")
    return txt


def _test_image(size: int) -> np.ndarray:
    """A smooth made-up picture (no photo needed), normalised as the models expect."""
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    rgb = np.stack([0.5 + 0.4 * np.sin(6 * xx + 2 * yy), 0.5 + 0.4 * np.cos(5 * yy), 0.3 + 0.5 * xx * yy], -1)
    x = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)


def stale(dst: str, src: str, force: bool) -> bool:
    return force or not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src)


def do_depth(src: str, force: bool) -> None:
    dst = os.path.join(MODELS, "depth", "depth_anything_v2_small.npz")
    if not stale(dst, src, force):
        say(f"Depth: already converted ({os.path.basename(dst)}); skipped.")
        return
    say(f"Depth: converting {os.path.basename(src)} (Depth Anything V2 Small)...")
    folded = fold(src, os.path.join(WORK, "depth_518.onnx"), {"pixel_values": [1, 3, 518, 518]})
    convert(folded, dst, f16=False)
    v = check(dst, folded, {"pixel_values": _test_image(518)}, ["predicted_depth"])
    say(f"Depth: done, {os.path.getsize(dst) / 1e6:.0f} MB; {v}.")


def do_sam(src: str, size: str, force: bool) -> None:
    enc_dst = os.path.join(MODELS, "sam", f"sam2.1_{size}_encoder.npz")
    dec_dst = os.path.join(MODELS, "sam", f"sam2.1_{size}_decoder.npz")
    if not (stale(enc_dst, src, force) or stale(dec_dst, src, force)):
        say(f"SAM 2.1 {size}: already converted; skipped.")
        return
    say(f"SAM 2.1 {size}: unpacking {os.path.basename(src)}...")
    d = os.path.join(WORK, f"sam_{size}")
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(d)
    found = {f: os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".onnx")}
    enc = next((p for f, p in found.items() if "encoder" in f), None)
    dec = next((p for f, p in found.items() if "decoder" in f), None)
    if not enc or not dec:
        raise RuntimeError(f"{os.path.basename(src)} does not hold an encoder and a decoder .onnx")
    t = time.perf_counter()
    say(f"SAM 2.1 {size}: folding the image encoder (Large takes a few minutes)...")
    folded = fold(enc, os.path.join(d, "encoder_1024.onnx"), {"image": [1, 3, 1024, 1024]})
    say(f"SAM 2.1 {size}: converting the image encoder (16-bit weights)...")
    convert(folded, enc_dst, f16=True)
    say(f"SAM 2.1 {size}: converting the click decoder...")
    convert(dec, dec_dst, f16=True)
    say(f"SAM 2.1 {size}: checking the encoder runs (Large: a minute or more)...")
    x = _test_image(1024)
    v = check(enc_dst, folded, {"image": x}, ["image_embed", "high_res_feats_0", "high_res_feats_1"])
    say(f"SAM 2.1 {size}: encoder {v}.")
    from studio.engine.local.onnxnp import Graph
    feats = Graph.load(enc_dst)(x)
    pts = {"point_coords": np.array([[[512.0, 512.0], [300.0, 700.0]]], np.float32),
           "point_labels": np.array([[1.0, 0.0]], np.float32),
           "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32)}
    v = check(dec_dst, dec, {**{k: feats[k] for k in ("image_embed", "high_res_feats_0", "high_res_feats_1")},
                             **pts}, ["masks", "iou_predictions"])
    say(f"SAM 2.1 {size}: decoder {v}.")
    say(f"SAM 2.1 {size}: done in {time.perf_counter() - t:.0f} s ({os.path.getsize(enc_dst) / 1e6:.0f} MB + "
        f"{os.path.getsize(dec_dst) / 1e6:.0f} MB). Choose it in Masks > Click-to-select.")


LAMA_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"   # Carve/LaMa-ONNX lama_fp32.onnx
LAMA_NOTICE = """LaMa inpainting model (Retouch > Fill, flare Rebuild)

Model: LaMa, "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (Suvorov, Logacheva, Mashikhin,
Remizova, Ashukha, Silvestrov, Kong, Goka, Park, Lempitsky; WACV 2022), Samsung AI Center: github.com/advimman/lama,
Apache License 2.0.
ONNX export: Carve/LaMa-ONNX (huggingface.co/Carve/LaMa-ONNX), lama_fp32.onnx, Apache License 2.0.
Trained on Places2 (Zhou et al.), CC BY 4.0: attribution to the Places2 authors.
Converted by the Studio's tools/convert_models.py (Convert Models.bat) into its NumPy format (lama.npz).
"""


def do_lama(src: str, force: bool) -> None:
    import hashlib
    dst = os.path.join(MODELS, "inpaint", "lama.npz")
    if not stale(dst, src, force):
        say(f"LaMa: already converted ({os.path.basename(dst)}); skipped.")
        return
    h = hashlib.sha256()
    with open(src, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != LAMA_SHA256:
        say("LaMa: this lama .onnx is not the Carve/LaMa-ONNX lama_fp32.onnx the Studio knows (checksum differs); "
            "converting it anyway, the check below says whether it runs.")
    say(f"LaMa: converting {os.path.basename(src)} (inpainting, 512 x 512)...")
    import onnx
    m = onnx.load(src)
    shapes = {i.name: ([1, 3, 512, 512] if k == 0 else [1, 1, 512, 512]) for k, i in enumerate(m.graph.input)}
    folded = fold(src, os.path.join(WORK, "lama_512.onnx"), shapes)
    # 16-bit weights, not 8-bit: folding turns this export's Fourier transforms into cos / sin tables, and 8-bit
    # tables bend the transform enough to paint coloured blobs inside the hole (measured against ONNX Runtime)
    convert(folded, dst, f16=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(os.path.join(os.path.dirname(dst), "NOTICE"), "w", encoding="utf-8") as fh:
        fh.write(LAMA_NOTICE)
    img = np.clip(_test_image(512) * np.array([0.229, 0.224, 0.225])[None, :, None, None] +
                  np.array([0.485, 0.456, 0.406])[None, :, None, None], 0, 1)       # 0..1, as LaMa takes it
    mask = np.zeros((1, 1, 512, 512), np.float32)
    mask[..., 180:330, 200:320] = 1.0
    names = list(shapes)
    v = check(dst, folded, {names[0]: img.astype(np.float32), names[1]: mask}, [m.graph.output[0].name],
              region=mask[0, 0] > 0.5, limit=0.02)
    say(f"LaMa: done, {os.path.getsize(dst) / 1e6:.0f} MB; {v}. Retouch > Fill now uses it.")


def find_downloads() -> list[tuple[str, str, str | None]]:
    """(kind, path, size) for every download recognised in models/ and the app folder."""
    out = []
    for folder in (MODELS, APP):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            low = f.lower()
            if not os.path.isfile(p):
                continue
            m = re.search(r"sam2(?:\.1)?_hiera_(tiny|large)", low)
            if m and low.endswith(".zip"):
                out.append(("sam", p, m.group(1)))
            elif low.endswith(".onnx") and "lama" in low:
                out.append(("lama", p, None))
            elif low.endswith(".onnx") and (low == "model.onnx" or "depth" in low):
                out.append(("depth", p, None))
    return out


def main(argv: list[str]) -> int:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    force = "--force" in argv
    log.info("---- convert_models start: %s (python %s)", " ".join(argv) or "(no options)", sys.version.split()[0])
    found = find_downloads()
    if not found:
        say("No model downloads found in the models folder. Put model.onnx (Depth Anything V2 Small), the")
        say("sam2.1_hiera_tiny / sam2.1_hiera_large zips or lama_fp32.onnx (LaMa) there and run this again.")
        return 1
    for kind, p, size in found:
        say(f"Found {os.path.basename(p)} ({os.path.getsize(p) / 1e6:.0f} MB): "
            f"{'depth model' if kind == 'depth' else ('LaMa inpainting' if kind == 'lama' else 'SAM 2.1 ' + size)}")
    if "--check-only" in argv:
        return 0
    os.makedirs(WORK, exist_ok=True)
    bad = 0
    for kind, p, size in found:
        try:
            if kind == "depth":
                do_depth(p, force)
            elif kind == "lama":
                do_lama(p, force)
            else:
                do_sam(p, size, force)
        except Exception as e:                                       # keep going with the others
            bad += 1
            say(f"PROBLEM with {os.path.basename(p)}: {e}")
            log.error("%s", traceback.format_exc())
            try:                                                     # rule 6: the fault reports itself
                from studio import log as slog
                rp = slog.save_error_report(f"Convert Models: converting {os.path.basename(p)}",
                                            traceback.format_exc())
                if rp:
                    say(f"An error report was saved: {rp}. Send it to Bill.")
            except Exception:                                        # noqa: BLE001 - the message above stands
                pass
    shutil.rmtree(WORK, ignore_errors=True)
    log.info("---- convert_models end: %d problem(s)", bad)
    if bad:
        say("Some conversions did not finish. Details: logs\\convert_models.log")
    else:
        say("All done. The downloads can stay or be deleted; the Studio only needs the converted files.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

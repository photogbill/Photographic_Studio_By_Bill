"""Train a camera's own denoising network (BUILD_PLAN_NOISE F.4). Run by "Train Camera Model.bat".

    python -m studio.lab.train --clean FOLDER [--camera "Canon EOS R5"] [--size balanced] [--hours 8]
    python -m studio.lab.train --clean FOLDER --general        (F.6: one network for every calibrated camera)

A camera's training starts from the general network when there is one of the same size (a fine-tune: it has
already learned what photographs look like and only has to learn this camera's noise), or from any checkpoint
given with --start-from.

Needs: that camera's Full profile (dark frames and flat pairs, Cameras page), its noise bank (Deep calibration,
made here if missing and the raws are still where they were), clean photos (see studio.lab.PHOTOS_WHY), PyTorch.
Writes calibration_local/<camera>/models/<camera>_<size>.onnx and its card. Can be stopped (Ctrl+C) and started
again: it carries on from the last checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

import numpy as np

from .. import __version__, log as slog, paths
from ..engine.calib import deep as D
from ..engine.calib import synth as S
from ..engine.calib.profile import CameraProfile
from ..engine.raw import learned as LN
from . import PHOTOS_MINIMUM, PHOTOS_WANTED, PHOTOS_WHY, data, photos_verdict

log = slog.get("lab.train")


def say(text: str = "") -> None:
    print(text, flush=True)
    if text.strip():
        log.info(text.strip())


def local_profiles() -> list:
    out = []
    for f in sorted(glob.glob(str(paths.get("CALIBRATION_LOCAL") / "profiles" / "*.json"))):
        try:
            out.append(CameraProfile.load(f, "local"))
        except Exception:                                    # noqa: BLE001
            slog.log_exception(f"profile unreadable: {f}", log)
    return out


def pick_profile(name: str | None) -> CameraProfile:
    profs = local_profiles()
    if not profs:
        raise SystemExit("No camera has been calibrated on this PC yet. Do that first: Cameras page, lens cap on.")
    if name:
        hit = [p for p in profs if name.lower() in (p.model or "").lower()]
        if not hit:
            raise SystemExit(f"No calibrated camera matches '{name}'. Calibrated: {[p.model for p in profs]}")
        return hit[0]
    if len(profs) == 1:
        return profs[0]
    say("Which camera?")
    for k, p in enumerate(profs, 1):
        say(f"  {k}. {p.model}")
    return profs[int(input("Number: ").strip()) - 1]


def ready(prof: CameraProfile) -> S.NoiseSource:
    """The camera's noise source, or a plain reason why training cannot start."""
    if not any(e.gain_e_per_dn for e in prof.isos):
        raise SystemExit(f"The {prof.model} has a Quick profile (dark frames only). Training needs its gain: shoot "
                         f"flat pairs and drop them on the Cameras page (card 3, 'Flat pairs').")
    bank = D.find_bank(prof.make, prof.model)
    if bank is None:
        raise SystemExit(f"The {prof.model} has no noise bank yet. On the Cameras page press 'Deep calibration' "
                         f"on its card and give it the folder of calibration raws, then start this again.")
    return S.NoiseSource(prof, bank)


def general_sources() -> tuple:
    """(noise sources, camera names, what was left out and why) over every camera calibrated on this PC."""
    sources, names, skipped = [], [], []
    for p in local_profiles():
        try:
            sources.append(ready(p))
            names.append(p.model)
        except SystemExit as e:
            skipped.append(str(e))
    return sources, names, skipped


def evaluate(net, torch, device, clean_set, source, seed: int = 1234, batches: int = 6, size: int = 256) -> dict:
    """``source``: one camera's noise, or a list of them (a general model is scored on all of them in turn)."""
    sources = list(source) if isinstance(source, (list, tuple)) else [source]
    rng = np.random.default_rng(seed)                         # the same held-out patches every time
    err, err_in, kept = [], [], []
    net.eval()
    with torch.no_grad():
        for k in range(max(batches, len(sources))):
            x, t = data.batch(clean_set, sources[k % len(sources)], rng, n=8, size=size, held_out=True)
            y = net(torch.from_numpy(x).to(device)).float().cpu().numpy()
            err.append(float(np.mean((y - t) ** 2))), err_in.append(float(np.mean((x - t) ** 2)))
            kept.append(data.texture_kept(y, x, t))
    net.train()
    return dict(noise_left=round(float(np.sqrt(np.mean(err) / max(np.mean(err_in), 1e-9))), 4),
                texture_kept=round(float(np.mean(kept)), 4))


def train(clean_folder: str, camera: str | None = None, size: str = "balanced", hours: float = 8.0,
          steps: int | None = None, batch_size: int = 16, patch: int = 256, eval_every: int = 500,
          patience: int = 12, force_few: bool = False, quiet: bool = False, general: bool = False,
          start_from: str | None = None) -> str:
    """Returns the path of the exported model. ``general``: one network over every calibrated camera's noise,
    kept in calibration_local/_general/models and used for any camera that has no network of its own.
    ``start_from``: a checkpoint (.pt) to start this training from instead of from nothing."""
    LN._add_lab_packages()
    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch is not installed. Run 'Train Camera Model.bat': it installs it beside the "
                         "Studio (lab_packages) the first time.") from e
    from .model import CameraNet, export_onnx
    if general:
        sources, names, skipped = general_sources()
        for why in skipped:
            say("Left out: " + why)
        if not sources:
            raise SystemExit("No camera on this PC is ready (each needs a Full profile and its noise bank: "
                             "Cameras page, flat pairs, then Deep calibration).")
        say(f"General network over {len(sources)} camera{'s' if len(sources) != 1 else ''}: {', '.join(names)}.")
        if len(sources) < 3:
            say("Note: a general network is only as general as the cameras it saw. With fewer than three it is "
                "mostly a copy of those cameras' own; it still gives an uncalibrated-for-training camera a start.")
        prof, source, model_name, make = None, sources[0], "general", ""
    else:
        prof = pick_profile(camera)
        source = ready(prof)
        sources, names, model_name, make = [source], [prof.model], prof.model, prof.make
        say(f"Camera: {prof.model}. ISO settings with gain and real noise: "
            f"{sum(1 for i in source.isos() if source.has_real_noise(i))} of {len(source.isos())}.")
    # ---- clean photos
    sv = data.survey(clean_folder, progress=None if quiet else lambda a, b: print(f"\r  looking at photos {a}/{b}",
                                                                                 end="", flush=True))
    print()
    tier, verdict = photos_verdict(sv["usable"])
    say(f"Clean photos in {clean_folder}: {sv['raws']} raw files, {sv['usable']} at ISO 400 or lower, from "
        f"{sv['sessions']} different hours on {sv['days']} days.")
    say(verdict)
    if tier != "good":
        say("")
        say("Why around 200 varied photos: " + PHOTOS_WHY)
        say("")
    if sv["usable"] < PHOTOS_MINIMUM and not force_few:
        raise SystemExit(f"Stopped: add clean photos (at least {PHOTOS_MINIMUM}; around {PHOTOS_WANTED} varied "
                         f"ones is the aim) and start again.")
    if sv["usable"] and sv["sessions"] < max(5, sv["usable"] // 40):
        say("Note: most of these were taken within a few hours. Photos from different days and places teach "
            "the network more than many frames of one scene.")
    crops = []
    for k, f in enumerate(sv["files"], 1):
        c = data.prepare_photo(f)
        if c:
            crops.append(c)
        if not quiet:
            print(f"\r  preparing clean crops {k}/{len(sv['files'])} ({len(crops)} usable)", end="", flush=True)
    print()
    if not crops:
        raise SystemExit("None of the clean photos could be used (they must be raws at ISO 400 or lower, not "
                         "blown out).")
    clean_set = data.CleanSet(crops)
    say(f"{len(clean_set.train)} photos to learn from, {len(clean_set.hold)} held back to score the result honestly.")
    # ---- network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        say("No NVIDIA graphics card found by PyTorch: training on the processor works but is very slow.")
    net = CameraNet(size).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    slug = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
    out_dir = LN.general_dir() if general else LN.model_dir(make, model_name)
    ckpt = os.path.join(out_dir, f"{slug}_{size}.checkpoint.pt")
    onnx_path = os.path.join(out_dir, f"{slug}_{size}.onnx")
    step, best, since_best, history, started = 0, None, 0, [], ""
    if os.path.isfile(ckpt):
        st = torch.load(ckpt, map_location=device)
        net.load_state_dict(st["net"]), opt.load_state_dict(st["opt"])
        step, best, history, started = st["step"], st.get("best"), st.get("history", []), st.get("started", "")
        say(f"Carrying on from step {step} (stopped earlier).")
    else:
        base = start_from or (None if general else os.path.join(LN.general_dir(), f"general_{size}.checkpoint.pt"))
        if base and os.path.isfile(base):
            try:
                net.load_state_dict(torch.load(base, map_location=device)["net"])
                started = os.path.basename(base)
                say(f"Starting from {started}: the network already knows what photographs look like, so this is "
                    f"a fine-tune to {model_name}'s noise and needs far fewer steps than starting from nothing.")
            except Exception as e:                            # noqa: BLE001  (another size, an old file)
                slog.log_exception(f"could not start from {base}", log)
                say(f"Could not start from {os.path.basename(base)} ({e}); starting from nothing.")
        elif start_from:
            say(f"The checkpoint to start from was not found ({start_from}); starting from nothing.")
    total = steps or 10 ** 9
    t_end = time.time() + hours * 3600.0
    rng = np.random.default_rng(step + 17)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def fine(v):
        return v - torch.nn.functional.avg_pool2d(torch.nn.functional.pad(v, (1, 1, 1, 1), mode="reflect"), 3, 1)

    def save(final: bool = False):
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, best=best, history=history,
                        started=started), ckpt)

    def card(**more):
        data.write_card(onnx_path, camera=model_name, make=make, size=size, step=step, studio=__version__,
                        made=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), clean_photos=len(clean_set),
                        held_out=len(clean_set.hold), clean_verdict=verdict, isos=source.isos(),
                        general=bool(general), cameras=names, started_from=started, **more)

    say(f"Training the '{size}' network on {device}. It stops by itself after {hours:g} hours, or sooner when the "
        f"held-back score stops improving. Ctrl+C stops it now; what it learned so far is kept.")
    t0, shown = time.time(), time.time()
    try:
        while step < total and time.time() < t_end:
            x, t = data.batch(clean_set, sources[int(rng.integers(len(sources)))], rng, n=batch_size, size=patch)
            x, t = torch.from_numpy(x).to(device), torch.from_numpy(t).to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                y = net(x)
            y = y.float()
            # error everywhere, plus the fine detail on its own so that losing texture costs more than leaving
            # a little noise. No adversarial term: nothing rewards inventing detail.
            loss = torch.sqrt((y - t) ** 2 + 1e-4).mean() + 0.5 * torch.sqrt((fine(y) - fine(t)) ** 2 + 1e-4).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt), scaler.update()
            step += 1
            if not quiet and time.time() - shown > 5:
                shown = time.time()
                print(f"\r  step {step}  loss {float(loss):.4f}  {step / max(time.time() - t0, 1e-6):.1f} steps/s   ",
                      end="", flush=True)
            if step % eval_every == 0 or step == total:
                ev = evaluate(net, torch, device, clean_set, sources, size=patch)
                history.append(dict(step=step, **ev))
                score = ev["noise_left"] + 0.5 * max(0.0, 1.0 - ev["texture_kept"])
                better = best is None or score < best - 1e-4
                print()
                say(f"  step {step}: noise left {100 * ev['noise_left']:.1f}% of what came in, texture kept "
                    f"{100 * ev['texture_kept']:.1f}%{'  (best so far: saved)' if better else ''}")
                if better:
                    best, since_best = score, 0
                    export_onnx(net, onnx_path)
                    net.to(device)
                    card(history=history[-20:], **ev)
                else:
                    since_best += 1
                save()
                if since_best >= patience:
                    say("The held-back score has stopped improving: finished.")
                    break
    except KeyboardInterrupt:
        print()
        say("Stopped by you. What was learned so far is kept; run this again to carry on.")
    save(final=True)
    if not os.path.isfile(onnx_path):                         # stopped before the first score: export what there is
        export_onnx(net, onnx_path)
        card(history=history, note="stopped before its first score: not a finished model")
    say(f"Model: {onnx_path}")
    if general:
        say("In the Studio: Develop > Clean > method 'learned' now works for every calibrated camera that has no "
            "network of its own, and each camera's own training starts from this one.")
    else:
        say("In the Studio: Develop > Clean > method 'learned'. It is always followed by the detail guard.")
    return onnx_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m studio.lab.train", description=__doc__.splitlines()[0])
    ap.add_argument("--clean", help="folder of clean photos (raws at ISO 400 or lower)")
    ap.add_argument("--camera")
    ap.add_argument("--size", choices=("fast", "balanced", "best"), default="balanced")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force-few", action="store_true", help="train even with fewer than the minimum of photos")
    ap.add_argument("--general", action="store_true",
                    help="one network over every calibrated camera's noise, used for cameras with none of their own")
    ap.add_argument("--start-from", help="a checkpoint (.pt) to start from (default: the general network's)")
    a = ap.parse_args(argv)
    slog.setup(console_level=None, filename="lab_train.log")
    clean = a.clean
    if not clean:
        try:
            clean = json.load(open(paths.get("SETTINGS") / "app.json", encoding="utf-8")).get("lab_clean_folder")
        except (OSError, ValueError):
            clean = None
    if not clean or not os.path.isdir(clean):
        say("No folder of clean photos was given. In the Studio: Cameras page > your camera's card > "
            "'Choose clean photos'. Or drag the folder onto 'Train Camera Model.bat'.")
        say("")
        say("What is needed: " + PHOTOS_WHY)
        return 2
    try:
        train(clean, a.camera, a.size, a.hours, a.steps, a.batch, force_few=a.force_few, general=a.general,
              start_from=a.start_from)
    except SystemExit as e:
        if e.code and not isinstance(e.code, int):
            say(str(e.code))
            return 1
        raise
    except Exception:                                         # noqa: BLE001
        slog.log_exception("training failed", log)
        slog.save_error_report("Training a camera model")
        say("Training stopped on a problem. An error report was saved in the error_reports folder.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

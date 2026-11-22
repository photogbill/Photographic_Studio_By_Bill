"""Dual Pixel speed benchmark (BUILD_PLAN_DP3D A.3.7): seconds per Dual Pixel frame of a focus stack, step by step,
the way the Studio did it before this build and the way it does it now, on the same frames.

Before (session 16):  develop the frame (one decode) | open it again for its halves (a second decode of the photo
                      and a temporary copy of the file for the second frame) | fit each half's shading | measure the
                      offsets (first implementation)
Now:                  develop the frame and read its halves from the same decode (the second frame from memory) |
                      the stack's shading fitted once and reused | measure the offsets (fast path) | several frames
                      at once on a PC with the cores and memory for it (shown separately: it is the same work)

The two ways must give the same offsets: the largest difference where the offsets are trusted is reported too.

    python tools\\dp_bench.py FOLDER_OR_FILES [--frames N] [--workers N]

Writes outputs\\dp_bench\\<time>\\results.json and prints a table (this PC only; no photos are copied).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def _frames(items, limit):
    out = []
    for it in items:
        if os.path.isdir(it):
            out += sorted(glob.glob(os.path.join(it, "*.CR3")) + glob.glob(os.path.join(it, "*.cr3")))
        elif os.path.isfile(it):
            out.append(it)
    from studio.engine.dualpixel.decode import is_dual_pixel
    out = [p for p in dict.fromkeys(out) if is_dual_pixel(p)]
    return out[:limit] if limit else out


def before(path, edit):
    from studio.engine.dualpixel import decode as DD, depth as DEP
    from studio.engine.stack import frames as F
    from studio.engine.stack.focus import DP_TILE, DP_STEP
    t = {}
    t0 = time.perf_counter()
    F.scene_rgb(path, edit, True)
    t["develop"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    from studio.engine import api                    # the first way: a second decode, the second frame via a copy
    tr = DD.cr3_raw_tracks(path)
    img = api.open(path)
    m1, black = DD._cr3_second_frame_via_copy(path, tr[0], tr[1])
    dp = DD.make_dual(img, m1, img.raw.black_for_positions(black), "dual pixel raw", path)
    t["halves_decode"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = DD.shading(dp)
    t["shading"] = time.perf_counter() - t0
    la, lb = DEP.halves_luma(dp, g)
    t0 = time.perf_counter()
    d, c = DEP.disparity_reference(la, lb, DP_TILE, DP_STEP)
    t["offsets"] = time.perf_counter() - t0
    return t, d, c


def now(path, edit, state):
    from studio.engine.dualpixel import depth as DEP
    from studio.engine.stack import frames as F
    from studio.engine.stack.focus import DP_TILE, DP_STEP
    t = {}
    t0 = time.perf_counter()
    rgb, la, lb = F.develop_and_halves(path, edit, state)
    t["develop_and_halves"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d, c = DEP.disparity(la, lb, DP_TILE, DP_STEP)
    t["offsets"] = time.perf_counter() - t0
    return t, d, c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--frames", type=int, default=6, help="frames to time (default 6)")
    ap.add_argument("--workers", type=int, default=None, help="frames at once for the parallel timing")
    a = ap.parse_args(argv)
    from studio import paths
    from studio.engine.stack import frames as F
    files = _frames(a.inputs, a.frames)
    if not files:
        print("No Dual Pixel RAW files found.")
        return 1
    ref = F.open_frame(files[len(files) // 2])
    edit = F.stack_edit(ref)
    del ref
    rows = []
    state = F.DualState()
    worst = 0.0
    for p in files:
        tb, d0, c0 = before(p, edit)
        tn, d1, c1 = now(p, edit, state)
        # where both are sure (confidence over 0.3): the typical and the 99th-percentile difference; a place whose
        # correlation has two near-equal peaks can flip between them on any change at all, so the largest single
        # difference says little and is kept only for the record
        sure = (c0 > 0.3) & (c1 > 0.3)
        dd = np.abs(d1 - d0)[sure]
        med = float(np.median(dd)) if dd.size else 0.0
        p99 = float(np.percentile(dd, 99)) if dd.size else 0.0
        worst = max(worst, p99)
        rows.append({"frame": os.path.basename(p), "before": tb, "now": tn, "before_s": sum(tb.values()),
                     "now_s": sum(tn.values()), "offset_difference_px": {"median": med, "p99": p99,
                                                                         "max": float(dd.max()) if dd.size else 0.0}})
        print(f"{os.path.basename(p)}: before {sum(tb.values()):6.1f} s  now {sum(tn.values()):6.1f} s  "
              f"(offsets where sure: typically {med:.3f} px apart, 99% within {p99:.3f} px; after the first frame "
              f"the stack's shared shading is used)")
    nb = float(np.mean([r["before_s"] for r in rows]))
    nn = float(np.mean([r["now_s"] for r in rows]))
    # the same work, several frames at once
    workers = F.stack_workers(limit=a.workers)
    t0 = time.perf_counter()
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        st2 = F.DualState()
        now(files[0], edit, st2)                     # the shading fitted first, as the stack does
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda q: now(q, edit, st2), files[1:]))
        par = (time.perf_counter() - t0) / len(files)
    else:
        par = nn
    res = {"date": _dt.datetime.now().isoformat(timespec="seconds"), "frames": len(files), "cpu_count":
           os.cpu_count(), "free_ram_gb": F.free_ram_gb(), "workers": workers,
           "seconds_per_frame": {"before": round(nb, 2), "now": round(nn, 2), "now_parallel": round(par, 2)},
           "speed_up": {"one_at_a_time": round(nb / max(nn, 1e-9), 2), "with_workers": round(nb / max(par, 1e-9), 2)},
           "offset_difference_p99_px": round(worst, 4), "rows": rows}
    d = paths.get("OUTPUTS") / "dp_bench" / _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print()
    print(f"Seconds per Dual Pixel frame: before {nb:.1f}, now {nn:.1f} one at a time "
          f"({nb / max(nn, 1e-9):.1f}x), {par:.1f} with {workers} at once ({nb / max(par, 1e-9):.1f}x).")
    print(f"Offsets: where both are sure, 99% agree within {worst:.3f} px.")
    print(f"Saved: {d / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

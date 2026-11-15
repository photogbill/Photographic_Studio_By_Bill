"""Library speed test (plan 24.7 B): how fast a big folder opens, cold and cached.

    python tools/bench_library.py                 a 2,000-raw test folder made of hard links of the sample raws
    python tools/bench_library.py "E:\\Photos\\2026\\2026-11-14"     one of your own folders (nothing is changed in it)

Measures: how long the window takes to answer, the first screen of thumbnails with an empty thumbnail cache (cold)
and again from the cache, the ratings/details pass (cold, then from the index), and the longest stall while
scrolling through the whole grid. The thumbnail cache and index used here are temporary (outputs\\bench_library),
so your real cache is not touched. Results: outputs\\bench_library\\results.txt.

Note: the hard-link test folder reads the same two files over and over, so Windows' own file cache is warm after the
first few: its "cold" means "no thumbnail cache yet", not "first read from the disk". Point it at a real folder on
a USB hard drive (after a restart) for the true first-visit number.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", ""))
WORK = os.path.join(ROOT, "outputs", "bench_library")


def make_test_folder(n: int = 2000) -> str:
    samples = [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.lower().endswith((".cr2", ".cr3"))]
    if not samples:
        raise SystemExit("No sample raws in the app folder: give a folder of raws instead.")
    d = os.path.join(WORK, f"folder_{n}")
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        src = samples[i % len(samples)]
        p = os.path.join(d, f"IMG_{i:04d}{os.path.splitext(src)[1]}")
        if not os.path.exists(p):
            try:
                os.link(src, p)
            except OSError:
                shutil.copy(src, p)
    return d


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    folder = argv[0] if argv else make_test_folder(2000)
    for k, sub in (("STUDIO_CACHE_DIR", "cache"), ("STUDIO_SETTINGS_DIR", "settings")):
        os.environ[k] = os.path.join(WORK, sub)
    shutil.rmtree(os.path.join(WORK, "cache"), ignore_errors=True)
    from PySide6.QtWidgets import QApplication
    from studio.app import common, theme as T
    from studio.app.library import LibraryPage
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(T.QSS)
    st = common.Settings()
    st.d = {}
    st.set = lambda k, v: st.d.__setitem__(k, v)
    lines = [f"Library speed test, folder {folder}"]

    def pump_until(cond, timeout=600.0):
        t0 = time.perf_counter()
        worst = 0.0
        last = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            app.processEvents()
            now = time.perf_counter()
            worst = max(worst, now - last)
            last = now
            if cond():
                return now - t0, worst
            time.sleep(0.002)
        return None, worst

    def visit(label: str):
        page = LibraryPage(st)
        page.resize(1366 - 240, 768 - 60)
        page.show()
        app.processEvents()
        t0 = time.perf_counter()
        page.load_folder(folder)
        answer = time.perf_counter() - t0
        listed, _ = pump_until(lambda: not page.loading)
        n = len(page.model.all)
        first, worst_wait = pump_until(lambda: not page.loading and page.visible_rows(0) and all(
            page.model.rows[r]["path"] in page.pix for r in page.visible_rows(0)))
        details, _ = pump_until(lambda: not page.meta_pending)
        first_s = (first + listed) if first is not None else float("nan")
        lines.append(f"{label}: {n} photos | window answered in {answer:.3f} s | listed in {listed:.2f} s | first "
                     f"screen of thumbnails in {first_s:.2f} s | ratings and details in {details:.2f} s | longest "
                     f"wait for the window while loading {1000 * worst_wait:.0f} ms")
        return page

    p1 = visit("Cold (no thumbnail cache, no index)")
    # scroll through everything, a screen at a time, measuring how long the window is ever unresponsive
    sb = p1.grid.verticalScrollBar()
    worst = 0.0
    t0 = time.perf_counter()
    while sb.value() < sb.maximum():
        a = time.perf_counter()
        sb.setValue(sb.value() + p1.grid.viewport().height())
        app.processEvents()
        worst = max(worst, time.perf_counter() - a)
        for _ in range(5):
            app.processEvents()
            time.sleep(0.01)
    lines.append(f"Scrolling to the end, a screen at a time: {time.perf_counter() - t0:.1f} s, longest stall "
                 f"{1000 * worst:.0f} ms")
    p1.close()
    p2 = visit("Second visit (thumbnails cached, index filled)")
    p2.close()
    from studio.app.main import finish_jobs
    finish_jobs()
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "results.txt"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M") + "\n" + "\n".join(lines) + "\n\n")
    print("\n".join(lines))
    print(f"\nSaved to {os.path.join(WORK, 'results.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run independent horizontal strips of an image in parallel threads.

NumPy and SciPy's filters release Python's global lock while they compute, so threads give real speed-ups
(about 1.7x on 2 cores, more on Bill's 14-core PC). Each job writes its own rows, so no locking is needed.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def strips(h: int, rows: int, job, n_workers: int | None = None) -> None:
    """Call job(y0, y1) for every strip [y0, y1) of an image h rows high."""
    starts = [(y, min(h, y + rows)) for y in range(0, h, rows)]
    n = n_workers or workers()
    if n <= 1 or len(starts) <= 1:
        for a, b in starts:
            job(a, b)
        return
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="strips") as ex:
        for f in [ex.submit(job, a, b) for a, b in starts]:
            f.result()                       # re-raise any error from a worker

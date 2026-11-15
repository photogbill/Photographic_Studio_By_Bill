"""Camera previews and the Library's thumbnail cache (plan sections 6.2 and 24.7 B). No GUI imports (rule 2.2).

* ``embedded_preview``: the JPEG the camera embedded in the raw, oriented (the Review view's "Camera preview").
* ``ThumbCache``: small JPEGs (about 320 px on the long side) in ``cache/thumbs/``, keyed on the photo's full path,
  size and modified time, so a folder opens from the cache in a fraction of a second the second time. Size-capped
  (1 GB by default; the least recently used are removed first). It lives in the app's cache folder, never beside
  the photos, so read-only drives work. Deleting it at any time is harmless.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading

import numpy as np

from ... import log as slog, paths

log = slog.get("engine.thumbs")

THUMB_SIZE = 320
DEFAULT_CAP = 1024 ** 3                      # 1 GB


def _orient(a: np.ndarray, flip: int) -> np.ndarray:
    from ..pipeline.pipeline import apply_orientation
    return np.ascontiguousarray(apply_orientation(a, flip))


def embedded_info(path: str) -> dict:
    """{'format': 'jpeg'|'bitmap'|'', 'width', 'height' (upright), 'flip'} of the largest embedded preview, without
    decoding it fully. rawpy (LibRaw) returns the largest preview the file holds."""
    import rawpy
    from PIL import Image
    with rawpy.imread(path) as r:
        flip = r.sizes.flip
        th = r.extract_thumb()
    if th.format == rawpy.ThumbFormat.JPEG:
        im = Image.open(io.BytesIO(th.data))
        w, h = im.size
        fmt = "jpeg"
    else:
        h, w = th.data.shape[:2]
        fmt = "bitmap"
    if flip in (5, 6):
        w, h = h, w
    return {"format": fmt, "width": int(w), "height": int(h), "flip": int(flip)}


def embedded_preview(path: str, max_side: int | None = None) -> np.ndarray:
    """The camera's embedded preview as an upright uint8 RGB array. ``max_side`` decodes a reduced copy quickly
    (JPEG draft mode); None gives the full size the file holds."""
    import rawpy
    from PIL import Image
    with rawpy.imread(path) as r:
        flip = r.sizes.flip
        th = r.extract_thumb()
    if th.format == rawpy.ThumbFormat.JPEG:
        im = Image.open(io.BytesIO(th.data))
        if max_side:
            im.draft("RGB", (max_side, max_side))
        im = im.convert("RGB")
    else:
        im = Image.fromarray(np.asarray(th.data))
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return _orient(np.asarray(im), flip)


class ThumbCache:
    """Disk cache of small JPEG thumbnails. Thread-safe; several loader threads may use one instance."""

    def __init__(self, folder: str | None = None, cap_bytes: int = DEFAULT_CAP, size: int = THUMB_SIZE):
        self.folder = folder or str(paths.get("CACHE") / "thumbs")
        self.cap = int(cap_bytes)
        self.size = int(size)
        self._lock = threading.Lock()
        self._used: int | None = None            # bytes on disk, counted once in the background

    # ------------------------------------------------------------------------------------ keys
    def key(self, path: str, size: int, mtime: float) -> str:
        k = f"{os.path.normcase(os.path.abspath(path))}|{int(size)}|{int(mtime * 1000)}|{self.size}"
        return hashlib.sha1(k.encode("utf-8", "surrogatepass")).hexdigest()

    def file_for(self, path: str, size: int, mtime: float) -> str:
        k = self.key(path, size, mtime)
        return os.path.join(self.folder, k[:2], k + ".jpg")

    # ------------------------------------------------------------------------------------ use
    def get(self, path: str, size: int | None = None, mtime: float | None = None) -> np.ndarray:
        """The thumbnail of a raw (upright uint8 RGB, long side ``self.size``), from the cache or made and kept."""
        from PIL import Image
        if size is None or mtime is None:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime
        f = self.file_for(path, size, mtime)
        try:
            with Image.open(f) as im:
                a = np.asarray(im.convert("RGB"))
            try:
                os.utime(f, None)                # least recently *used* goes first
            except OSError:
                pass
            return a
        except (OSError, ValueError):
            pass
        a = embedded_preview(path, max_side=self.size * 2)
        im = Image.fromarray(a)
        im.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        a = np.ascontiguousarray(np.asarray(im))
        try:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f + f".{threading.get_ident()}.partial"
            im.save(tmp, "JPEG", quality=85)
            os.replace(tmp, f)
            self._add(os.path.getsize(f))
        except OSError as e:
            log.debug("thumbnail cache write failed (%s): %s", f, e)
        return a

    def has(self, path: str, size: int, mtime: float) -> bool:
        return os.path.isfile(self.file_for(path, size, mtime))

    # ------------------------------------------------------------------------------------ size cap
    def usage(self) -> int:
        """Bytes the cache uses on disk (counted the first time it is asked)."""
        with self._lock:
            if self._used is not None:
                return self._used
        n = 0
        for d, _, files in os.walk(self.folder):
            for x in files:
                try:
                    n += os.path.getsize(os.path.join(d, x))
                except OSError:
                    pass
        with self._lock:
            self._used = n
        return n

    def _add(self, n: int) -> None:
        with self._lock:
            if self._used is None:
                return
            self._used += n
            over = self._used > self.cap
        if over:
            self.prune()

    def prune(self, target: float = 0.9) -> int:
        """Remove the least recently used thumbnails until the cache is under ``target`` x the cap. Returns the
        number removed."""
        entries = []
        for d, _, files in os.walk(self.folder):
            for x in files:
                p = os.path.join(d, x)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, p))
        total = sum(e[1] for e in entries)
        entries.sort()
        removed = 0
        goal = int(self.cap * target)
        for _, sz, p in entries:
            if total <= goal:
                break
            try:
                os.remove(p)
                total -= sz
                removed += 1
            except OSError:
                pass
        with self._lock:
            self._used = total
        if removed:
            log.info("thumbnail cache pruned: %d removed, %.0f MB left", removed, total / 1e6)
        return removed

    def clear(self) -> int:
        """The "Clear cache" button: remove every thumbnail. Returns the bytes freed."""
        freed = 0
        for d, _, files in os.walk(self.folder, topdown=False):
            for x in files:
                p = os.path.join(d, x)
                try:
                    freed += os.path.getsize(p)
                    os.remove(p)
                except OSError:
                    pass
            if d != self.folder:
                try:
                    os.rmdir(d)
                except OSError:
                    pass
        with self._lock:
            self._used = 0
        log.info("thumbnail cache cleared: %.0f MB", freed / 1e6)
        return freed


_DEFAULT: ThumbCache | None = None


def default_cache() -> ThumbCache:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ThumbCache()
    return _DEFAULT


def drive_is_ssd(path: str) -> bool | None:
    """Windows: does the drive holding ``path`` have no seek penalty (an SSD)? None when it cannot be told (other
    systems, network drives). Used to pick how many thumbnail readers to run: 2 on a spinning drive, more on an SSD."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        if not drive:
            return None
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        h = k32.CreateFileW(f"\\\\.\\{drive}", 0, 3, None, 3, 0, None)
        if h in (None, wintypes.HANDLE(-1).value):
            return None
        try:
            class Query(ctypes.Structure):
                _fields_ = [("PropertyId", wintypes.DWORD), ("QueryType", wintypes.DWORD),
                            ("AdditionalParameters", ctypes.c_ubyte * 1)]

            class SeekPenalty(ctypes.Structure):
                _fields_ = [("Version", wintypes.DWORD), ("Size", wintypes.DWORD),
                            ("IncursSeekPenalty", ctypes.c_ubyte)]
            q = Query(7, 0)                      # StorageDeviceSeekPenaltyProperty, PropertyStandardQuery
            out = SeekPenalty()
            got = wintypes.DWORD(0)
            ok = k32.DeviceIoControl(h, 0x2D1400, ctypes.byref(q), ctypes.sizeof(q), ctypes.byref(out),
                                     ctypes.sizeof(out), ctypes.byref(got), None)
            return (not bool(out.IncursSeekPenalty)) if ok else None
        finally:
            k32.CloseHandle(h)
    except Exception:                            # noqa: BLE001  (unknown: the caller uses 2 readers)
        return None


def reader_count(path: str) -> int:
    """How many thumbnail readers suit the drive: 2 on a spinning (or unknown USB) drive, up to 4 on an SSD."""
    ssd = drive_is_ssd(path)
    if ssd is None:
        return 2 if os.name == "nt" else min(4, os.cpu_count() or 2)
    return min(4, os.cpu_count() or 2) if ssd else 2


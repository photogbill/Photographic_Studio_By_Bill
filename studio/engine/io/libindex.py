"""The Library's folder listing and its small index (plan section 24.7 B). No GUI imports (rule 2.2).

* ``list_photos``: the raws in a folder (optionally its subfolders), from the directory listing only, so the window
  can answer at once whatever the folder size. The sidecar beside each raw is noticed in the same listing.
* ``LibraryIndex``: ``cache/library_index.sqlite`` (Python's own sqlite3) remembers each photo's rating, pick,
  lens tag and camera details, keyed on path, size and modified time (and the sidecar's modified time). The sidecar
  stays the truth: an index row is used only while both still match, and the index can be deleted at any time.
* ``load_entries``: fills in ratings and details in the background, from the index or by reading the sidecar and the
  first 512 KB of the raw, in batches the interface shows as they arrive.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from ... import log as slog, paths

log = slog.get("engine.libindex")

SIDECAR_SUFFIX = ".psb.json"
# Folders the Library never walks into when "Include subfolders" is on: the import record, hidden and system folders.
# (The app folder's own outputs/logs/cache are skipped by the command line's collect(); in a photographer's own tree
# a folder called "outputs" may well hold photos.)
SKIP_ALWAYS = {"_studio", "$recycle.bin", "system volume information", "__pycache__", ".git", ".venv"}
FIELDS = ("rating", "pick", "label", "lens_tag", "has_edit", "taken", "camera", "lens", "iso", "exposure", "fnumber",
          "focal", "stack")


def _skip_dir(name: str) -> bool:
    return name.startswith(".") or name.lower() in SKIP_ALWAYS


def list_photos(folder: str, recursive: bool = False) -> list:
    """[{path, name, size, mtime, side_mtime}] for the raws and scans (TIFF / JPEG / PNG) in ``folder`` (sorted by name within each folder).
    ``side_mtime`` is the sidecar's modified time, 0 when there is none. Reads the listing only."""
    from .. import api
    from .rawfile import RAW_EXTENSIONS
    from .scanfile import SCAN_EXTENSIONS
    out = []
    todo = [folder]
    while todo:
        d = todo.pop(0)
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except OSError as e:
            log.warning("cannot list %s: %s", d, e)
            continue
        central = api.central_folder_for(d)
        sides = {}
        if central:
            try:
                with os.scandir(central) as it:
                    for e in it:
                        if e.name.endswith(SIDECAR_SUFFIX):
                            sides[e.name[:-len(SIDECAR_SUFFIX)].lower()] = e.stat().st_mtime
            except OSError:
                pass
        subdirs = []
        raws = []
        for e in entries:
            n = e.name
            try:
                if e.is_dir():
                    if recursive and not _skip_dir(n):
                        subdirs.append(e.path)
                    continue
                low = n.lower()
                if not central and low.endswith(SIDECAR_SUFFIX):
                    sides[low[:-len(SIDECAR_SUFFIX)]] = e.stat().st_mtime
                elif low.endswith(RAW_EXTENSIONS + SCAN_EXTENSIONS) and not n.startswith("._"):
                    st = e.stat()
                    raws.append({"path": e.path, "name": n, "size": int(st.st_size), "mtime": float(st.st_mtime)})
            except OSError:
                continue
        # a camera's JPEG (or a TIFF made from the raw) beside its raw is the same photo: show the raw only
        stems = {os.path.splitext(r["name"])[0].lower() for r in raws if r["name"].lower().endswith(RAW_EXTENSIONS)}
        raws = [r for r in raws if r["name"].lower().endswith(RAW_EXTENSIONS)
                or os.path.splitext(r["name"])[0].lower() not in stems]
        for r in raws:
            r["side_mtime"] = float(sides.get(r["name"].lower(), 0.0))
        out += raws
        todo = subdirs + todo
    return out


def camera_name(make: str, model: str) -> str:
    """'Canon' + 'Canon EOS R5' -> 'Canon EOS R5'; 'SONY' + 'ILCE-7M3' -> 'SONY ILCE-7M3'."""
    make, model = (make or "").strip(), (model or "").strip()
    if not make or not model:
        return model or make
    if model.lower().startswith(make.split()[0].lower()):
        return model
    return f"{make} {model}"


def read_entry(e: dict) -> dict:
    """Ratings, pick, lens tag (from the sidecar) and camera details (from the raw's first 512 KB) of one photo."""
    from .. import api
    from .metadata import read_metadata
    out = {k: None for k in FIELDS}
    doc = api.load_doc(e["path"]) if e.get("side_mtime") else {}
    lib = doc.get("library", {}) if isinstance(doc, dict) else {}
    out["rating"] = int(lib.get("rating", 0) or 0)
    out["pick"] = int(lib.get("pick", 0) or 0)
    out["label"] = str(lib.get("label", "") or "")
    ed = doc.get("edit") if isinstance(doc, dict) else None
    out["lens_tag"] = str(((ed or {}).get("lens") or {}).get("tag", "") or "")
    out["has_edit"] = 0 if api.edit_is_default(ed) else 1
    try:
        m = read_metadata(e["path"], head_bytes=512 * 1024)
        dt = m.datetime.strip()
        out["taken"] = (dt[:4] + "-" + dt[5:7] + "-" + dt[8:10] + dt[10:19]) if len(dt) >= 19 else ""
        out["camera"] = camera_name(m.make, m.model)
        out["lens"] = m.lens or ""
        out["iso"] = float(m.iso) if m.iso else None
        out["exposure"] = m.exposure_s
        out["fnumber"] = m.fnumber
        out["focal"] = m.focal_mm
    except Exception as err:                         # noqa: BLE001  (a damaged file still shows, with no details)
        log.debug("library: no metadata for %s: %s", e["path"], err)
        out.update(taken="", camera="", lens="")
    if e["path"].lower().endswith((".tif", ".tiff")):  # a stacking tool's result says what it is on its card
        from .intermediate import label, read_info
        out["stack"] = label(read_info(e["path"])) or None
    return out


class LibraryIndex:
    """The small, throw-away index in ``cache/library_index.sqlite``. One connection per thread."""

    def __init__(self, path: str | None = None):
        self.path = path or str(paths.get("CACHE") / "library_index.sqlite")
        self._local = threading.local()
        self.ok = True

    def _db(self):
        db = getattr(self._local, "db", None)
        if db is None:
            try:
                db = sqlite3.connect(self.path, timeout=5)
                db.execute("CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, "
                           "side_mtime REAL, data TEXT, updated REAL)")
                db.commit()
            except sqlite3.Error as e:                # a damaged index: start a new one (it is only a cache)
                log.warning("library index unusable (%s); starting a new one", e)
                try:
                    if db is not None:
                        db.close()
                    os.remove(self.path)
                    db = sqlite3.connect(self.path, timeout=5)
                    db.execute("CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, "
                               "side_mtime REAL, data TEXT, updated REAL)")
                    db.commit()
                except (OSError, sqlite3.Error):
                    self.ok = False
                    raise
            self._local.db = db
        return db

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def lookup_many(self, entries: list) -> dict:
        """{path: data} for the entries whose size, modified time and sidecar time still match."""
        out = {}
        try:
            db = self._db()
            for i in range(0, len(entries), 500):
                chunk = entries[i:i + 500]
                keys = [self._key(e["path"]) for e in chunk]
                q = "SELECT path, size, mtime, side_mtime, data FROM photos WHERE path IN (%s)" % ",".join("?" * len(keys))
                rows = {r[0]: r for r in db.execute(q, keys)}
                for e, k in zip(chunk, keys):
                    r = rows.get(k)
                    if r and r[1] == e["size"] and abs(r[2] - e["mtime"]) < 1e-3 and \
                            abs((r[3] or 0) - (e.get("side_mtime") or 0)) < 1e-3:
                        try:
                            out[e["path"]] = json.loads(r[4])
                        except ValueError:
                            pass
        except sqlite3.Error as err:
            log.warning("library index read failed: %s", err)
        return out

    def put_many(self, items: list) -> None:
        """items: [(entry, data)]"""
        if not items:
            return
        try:
            db = self._db()
            now = time.time()
            db.executemany("INSERT OR REPLACE INTO photos VALUES (?, ?, ?, ?, ?, ?)",
                           [(self._key(e["path"]), e["size"], e["mtime"], e.get("side_mtime") or 0,
                             json.dumps(d), now) for e, d in items])
            db.commit()
        except sqlite3.Error as err:
            log.warning("library index write failed: %s", err)

    def update(self, path: str, size: int, mtime: float, side_mtime: float, data: dict) -> None:
        self.put_many([({"path": path, "size": size, "mtime": mtime, "side_mtime": side_mtime}, data)])

    def close(self) -> None:
        db = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            self._local.db = None


def load_entries(entries: list, index: LibraryIndex | None = None, on_batch=None, cancel=None,
                 batch_s: float = 0.15) -> int:
    """Fill in each entry's ratings and details, from the index or by reading. ``on_batch([(path, data), ...])``
    is called every ``batch_s`` seconds with what is new. Returns how many were read from the files (not the index).
    ``cancel``: a callable or an Event; stops between photos."""
    stop = cancel.is_set if hasattr(cancel, "is_set") else (cancel or (lambda: False))
    index = index or LibraryIndex()
    cached = index.lookup_many(entries)
    pending = [(e["path"], cached[e["path"]]) for e in entries if e["path"] in cached]
    if pending and on_batch:
        on_batch(pending)
    pending, fresh, read = [], [], 0
    last = time.perf_counter()
    for e in entries:
        if e["path"] in cached:
            continue
        if stop():
            break
        d = read_entry(e)
        read += 1
        pending.append((e["path"], d))
        fresh.append((e, d))
        if time.perf_counter() - last >= batch_s:
            if on_batch:
                on_batch(pending)
            index.put_many(fresh)
            pending, fresh, last = [], [], time.perf_counter()
    if pending and on_batch:
        on_batch(pending)
    index.put_many(fresh)
    index.close()
    return read

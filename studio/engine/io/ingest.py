"""Import from card (BUILD_PLAN.md section 24): copy photos from a memory card, or any folder, into the
photographer's Photos folder, every file checked, the card never changed. No GUI imports (rule 2.2).

    cards = find_cards()                                  # every drive with a DCIM folder at its root
    scan = scan_source(cards[0].root)                     # what is on it, from the directory listing (fast)
    confirm_capture_times(scan)                           # capture times from the photos (one reader on the card)
    plan = plan_import(scan, ImportOptions(dest=r"E:\\Photos"))   # the dry run: decides everything, writes nothing
    report = run_import(plan, on_progress, cancel)        # copy, check, then name

The copy rules (24.6), each with a named test in tests/test_ingest.py:
 1. The card is never changed: source files are opened read-only, nothing on the source is written, renamed,
    moved or deleted. There is no "delete after import".
 2. Copy, check, then name: ``<name>.partial`` is written, flushed and fsync'ed, checked, given the source's
    modified time, and only then renamed to its final name. A crash can only leave ``.partial`` files; the next
    import removes the ones it finds in the folders it writes to.
 3. Every file is checked: hashed (BLAKE2b) while reading the card, then read back from the destination and
    compared (hash and size). A mismatch: the copy is removed and tried once more, then listed as failed.
 4. Never overwrite. Same name and same content: skipped ("already imported"). Same name, different content:
    both kept, the new one as ``<stem>_2<ext>`` (camera counters wrap at 9999; two bodies can share a prefix).
 5. Only what is new: ``<Photos>/_studio/imports.jsonl`` records every imported file (name, size, capture time,
    hash, where it went, when). A file it lists is "already imported", even if its copy was moved or deleted
    since (deleted rejects do not come back). It is a convenience: without it, rule 4 still holds.
 6. One bad file does not stop the rest (two retries, then failed). A card pulled out stops the import cleanly.
 7. One reader on the card: 8 MB blocks, one thread. The second copy is written from the same blocks.
 8. Everything is logged (logs/import_<date>_<time>.log), one line per file. No serial numbers, no GPS.
 9. The dry run (``plan_import``) decides every destination, skip and rename; the dialog shows exactly that.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field

from ... import log as slog, paths
from .metadata import read_capture_time
from .rawfile import RAW_EXTENSIONS

log = slog.get("engine.ingest")

RAW, JPEG, VIDEO, OTHER = "raw", "jpeg", "video", "other"
JPEG_EXTENSIONS = (".jpg", ".jpeg", ".heic", ".heif", ".hif")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mts", ".m2ts", ".crm", ".mxf", ".3gp", ".mpg", ".wmv")
KIND_WORDS = {RAW: ("raw file", "raw files"), JPEG: ("JPEG", "JPEGs"), VIDEO: ("video", "videos"),
              OTHER: ("other file", "other files")}
# Videos some brands keep outside DCIM (looked for only when Videos is ticked). Verify per brand.
VIDEO_FOLDERS = ("PRIVATE/M4ROOT/CLIP", "PRIVATE/AVCHD/BDMV/STREAM", "PRIVATE/XDROOT/Clip", "MP_ROOT")
RECORD_DIR = "_studio"
RECORD_NAME = "imports.jsonl"
BLOCK = 8 * 1024 * 1024                       # rule 7: 4-8 MB blocks
RESERVE_MIN = 2 * 1024 ** 3                   # keep 2 GB or 5 % free, whichever is larger
RESERVE_FRACTION = 0.05
LONG_PATH = 240
MAX_NAME = 80
UNDATED = "Undated"
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class ImportStopped(Exception):
    """The whole import must stop (card pulled out, destination gone or full)."""


# ============================================================================================ files and times
def kind_of(name: str) -> str:
    n = name.lower()
    if n.endswith(RAW_EXTENSIONS):
        return RAW
    if n.endswith(JPEG_EXTENSIONS):
        return JPEG
    if n.endswith(VIDEO_EXTENSIONS):
        return VIDEO
    return OTHER


def parse_time(text: str) -> _dt.datetime | None:
    """'2026:11:14 21:03:55' (EXIF), '2026-11-14 21:03:55' or '2026-11-14T21:03:55' -> datetime, else None.
    Sub-seconds and time-zone suffixes are ignored: capture times stay in camera local time (24.4)."""
    m = re.match(r"\s*(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})", text or "")
    if not m:
        return None
    try:
        dt = _dt.datetime(*map(int, m.groups()))
    except ValueError:
        return None
    return dt if 1990 <= dt.year <= 2100 else None


def fmt_time(dt: _dt.datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def file_time(mtime: float) -> _dt.datetime | None:
    """A file's modified time as local time (cameras set it to the capture time). None when unusable."""
    try:
        dt = _dt.datetime.fromtimestamp(mtime)
    except (OverflowError, OSError, ValueError):
        return None
    return dt if 1990 <= dt.year <= 2100 else None


def day_start_minutes(text: str | int) -> int:
    """'04:00' -> 240. Bad values give the default (04:00)."""
    if isinstance(text, int):
        return max(0, min(23 * 60 + 59, text))
    m = re.match(r"\s*(\d{1,2})(?::(\d{2}))?\s*$", str(text or ""))
    if not m:
        return 240
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    return h * 60 + mi if 0 <= h <= 23 and 0 <= mi <= 59 else 240


def day_key(dt: _dt.datetime | None, day_start: str | int = "04:00") -> str:
    """The shooting day a capture time belongs to: a new day starts at ``day_start`` (04:00 by default), so a night
    shoot that runs past midnight stays under the date it started (24.4). 'YYYY-MM-DD', or 'Undated'."""
    if dt is None:
        return UNDATED
    return (dt - _dt.timedelta(minutes=day_start_minutes(day_start))).strftime("%Y-%m-%d")


def safe_folder_name(text: str, fallback: str) -> str:
    """Make a folder name safe for Windows (24.4): no \\ / : * ? " < > | or control characters, no trailing dots or
    spaces, no reserved device names, at most 80 characters. Empty -> the fallback (the date)."""
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f\x7f]', "", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s[:MAX_NAME].rstrip(". ").strip()
    if s.split(".")[0].strip().upper() in _RESERVED:
        s = "_" + s
    return s or fallback


@dataclass
class SourceFile:
    path: str                     # full path on the source
    rel: str                      # relative to the source root (forward slashes)
    size: int
    mtime: float
    kind: str
    taken: str = ""               # 'YYYY-MM-DD HH:MM:SS', camera local time ('' = unknown)
    taken_from: str = ""          # 'photo' (its own metadata), 'file time', or ''

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def taken_dt(self) -> _dt.datetime | None:
        return parse_time(self.taken)


@dataclass
class Scan:
    root: str
    files: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    confirmed: bool = False       # capture times read from the photos themselves
    videos_outside: bool = False  # brand video folders outside DCIM were looked at

    def count(self, kind: str) -> int:
        return sum(1 for f in self.files if f.kind == kind)

    @property
    def photos(self) -> int:
        return sum(1 for f in self.files if f.kind in (RAW, JPEG))


# ============================================================================================ cards and drives
@dataclass
class Card:
    root: str                     # 'F:\\' (or any folder in tests)
    label: str = ""               # volume label, e.g. EOS_DIGITAL
    letter: str = ""              # 'F:'

    @property
    def title(self) -> str:
        if self.letter:
            return f"{self.label or 'Card'} ({self.letter})"
        return self.label or os.path.basename(os.path.normpath(self.root)) or self.root


@dataclass
class Drive:
    root: str
    letter: str = ""
    label: str = ""
    free: int = 0
    total: int = 0
    kind: str = ""                # 'fixed', 'removable', ...
    card: bool = False            # has DCIM at its root

    @property
    def title(self) -> str:
        name = self.label or ("Local disk" if self.kind == "fixed" else "Drive")
        return f"{self.letter} {name}".strip() if self.letter else (self.label or self.root)


_DRIVE_TYPES = {2: "removable", 3: "fixed", 4: "network", 5: "cd", 6: "ram"}


def _win_drives() -> list:
    """(root, letter, type) for drive letters C: to Z: that may hold photos. Network and CD drives are skipped: a
    disconnected network drive can hang for seconds every time the card list is refreshed. No 'removable' test:
    many CFexpress readers (the R5's card) show up as fixed disks (24.5)."""
    import ctypes
    k32 = ctypes.windll.kernel32
    mask = k32.GetLogicalDrives()
    out = []
    for i in range(2, 26):                                   # skip A: and B:
        if mask & (1 << i):
            root = f"{chr(65 + i)}:\\"
            t = _DRIVE_TYPES.get(k32.GetDriveTypeW(root), "")
            if t in ("removable", "fixed", "ram"):
                out.append((root, root[:2], t))
    return out


def _win_label(root: str) -> str:
    import ctypes
    buf = ctypes.create_unicode_buffer(261)
    try:
        if ctypes.windll.kernel32.GetVolumeInformationW(root, buf, 261, None, None, None, None, 0):
            return buf.value
    except Exception:                                        # noqa: BLE001
        pass
    return ""


class _quiet_errors:
    """Windows: no 'There is no disk in the drive' box while probing an empty card-reader slot."""

    def __enter__(self):
        self.old = None
        if os.name == "nt":
            try:
                import ctypes
                old = ctypes.c_uint(0)
                ctypes.windll.kernel32.SetThreadErrorMode(0x0001 | 0x8000, ctypes.byref(old))
                self.old = old.value
            except Exception:                                # noqa: BLE001
                self.old = None
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadErrorMode(self.old, None)
            except Exception:                                # noqa: BLE001
                pass
        return False


def _candidate_roots() -> list:
    """(root, letter, kind) of the places a card can be mounted. STUDIO_CARD_ROOTS (os.pathsep-separated folders)
    replaces the system's list, so tests can fake cards on any system (24.8)."""
    env = os.environ.get("STUDIO_CARD_ROOTS")
    if env is not None:
        return [(p, "", "test") for p in env.split(os.pathsep) if p]
    if os.name == "nt":
        try:
            return _win_drives()
        except Exception:                                    # noqa: BLE001
            slog.log_exception("could not list drives", log)
            return []
    roots = []
    bases = ["/Volumes"] if sys.platform == "darwin" else [
        f"/media/{os.environ.get('USER', '')}", f"/run/media/{os.environ.get('USER', '')}", "/media", "/mnt"]
    for b in bases:
        try:
            for n in sorted(os.listdir(b)):
                p = os.path.join(b, n)
                if os.path.isdir(p):
                    roots.append((p, "", ""))
        except OSError:
            continue
    return roots


def _label(root: str, letter: str) -> str:
    if os.name == "nt" and letter:
        return _win_label(root)
    return os.path.basename(os.path.normpath(root))


def find_cards() -> list:
    """Every drive with a DCIM folder at its root, whatever Windows calls the drive type (24.5)."""
    cards = []
    with _quiet_errors():
        for root, letter, _kind in _candidate_roots():
            try:
                if os.path.isdir(os.path.join(root, "DCIM")):
                    cards.append(Card(root=root, label=_label(root, letter), letter=letter))
            except OSError:
                continue
    return cards


class CardWatcher:
    """Notices a card being put in, cheaply enough to ask every 2 seconds while the Studio is open (the Library's
    "Card found" notice, 24.5). Card readers (removable drives) are looked at each time; fixed drives only when a new
    drive letter appears (a CFexpress reader that shows up as a fixed disk), so an external archive drive is never
    kept spinning. Cards already in when the Studio starts are not announced; a card is announced again only after it
    was taken out. ``poll()`` returns the cards that are new since the last call."""

    def __init__(self):
        self.present: dict = {}                  # root -> Card
        self.letters: set = set()
        self.first = True

    def _scan(self) -> dict:
        if os.environ.get("STUDIO_CARD_ROOTS") is not None or os.name != "nt":
            return {c.root: c for c in find_cards()}
        try:
            drives = _win_drives()
        except Exception:                        # noqa: BLE001
            return dict(self.present)
        letters = {d[1] for d in drives}
        new_letters = letters - self.letters
        self.letters = letters
        found = {}
        with _quiet_errors():
            for root, letter, kind in drives:
                if kind == "removable" or letter in new_letters or self.first:
                    try:
                        if os.path.isdir(os.path.join(root, "DCIM")):
                            found[root] = Card(root=root, label=_label(root, letter), letter=letter)
                    except OSError:
                        continue
                elif root in self.present:       # a fixed-disk card still there: not looked at again
                    found[root] = self.present[root]
        return found

    def poll(self) -> list:
        now = self._scan()
        new = [c for r, c in now.items() if r not in self.present]
        self.present = now
        if self.first:
            self.first = False
            return []
        return new


def list_drives() -> list:
    """Drives the Photos folder can go on, with their free space (the quick-pick buttons, 24.3).
    STUDIO_DRIVE_ROOTS replaces the system's list in tests."""
    env = os.environ.get("STUDIO_DRIVE_ROOTS")
    roots = [(p, "", "test") for p in env.split(os.pathsep) if p] if env is not None else _candidate_roots()
    out = []
    with _quiet_errors():
        for root, letter, kind in roots:
            try:
                du = shutil.disk_usage(root)
            except OSError:
                continue
            out.append(Drive(root=root, letter=letter, label=_label(root, letter), free=du.free, total=du.total,
                             kind=kind, card=os.path.isdir(os.path.join(root, "DCIM"))))
    return out


def free_space(path: str) -> tuple:
    """(free, total) bytes on the drive that holds ``path`` (or would: the nearest folder that exists).
    (None, None) when the drive is not there."""
    p = os.path.abspath(path)
    with _quiet_errors():
        while p and not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p:
                return None, None
            p = parent
        try:
            du = shutil.disk_usage(p)
        except OSError:
            return None, None
    return du.free, du.total


def reserve_for(total: int | None) -> int:
    return max(RESERVE_MIN, int(RESERVE_FRACTION * (total or 0)))


def drive_of(path: str) -> str:
    """'E:' for 'E:\\Photos' on Windows; the first folder of the path elsewhere ('/media')."""
    p = os.path.abspath(path)
    d = os.path.splitdrive(p)[0]
    if d:
        return d
    parts = p.split(os.sep)
    return os.sep + parts[1] if len(parts) > 1 and parts[1] else os.sep


def drive_connected(path: str) -> bool:
    """Is the drive a folder lives on connected? (A Photos folder on an external drive that is unplugged.)"""
    d = os.path.splitdrive(os.path.abspath(path))[0]
    with _quiet_errors():
        if d:
            return os.path.isdir(d + os.sep)
        p = os.path.abspath(path)
        while p and not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p:
                return False
            p = parent
        return True


def inside_onedrive(path: str) -> bool:
    """Is this folder inside a folder that OneDrive syncs? (Raw files would fill cloud storage quickly; 24.3.)"""
    p = os.path.normcase(os.path.abspath(path))
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        root = os.environ.get(var)
        if not root:
            continue
        r = os.path.normcase(os.path.abspath(root))
        try:
            if os.path.commonpath([p, r]) == r:
                return True
        except ValueError:                                   # different drives
            continue
    return False


def default_photos_dir() -> str:
    """The suggested Photos folder: 'Photos' beside the app (inside the app folder on a portable or git copy,
    rule 0.3.3; the MSI copy's data folder otherwise). STUDIO_PHOTOS_DIR overrides. Nothing is created here."""
    return str(paths.photos_dir())


# ============================================================================================ scanning
def _walk(top: str, root: str, out: list, errors: list) -> None:
    def onerror(e):
        errors.append(f"{getattr(e, 'filename', top)}: {e.strerror or e}")
    for d, dirs, files in os.walk(top, onerror=onerror):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for n in sorted(files):
            if n.startswith("._") or n.startswith("."):     # macOS resource forks and hidden files
                continue
            p = os.path.join(d, n)
            try:
                st = os.stat(p)
            except OSError as e:
                errors.append(f"{p}: {e.strerror or e}")
                continue
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            dt = file_time(st.st_mtime)
            out.append(SourceFile(path=p, rel=rel, size=int(st.st_size), mtime=float(st.st_mtime), kind=kind_of(n),
                                  taken=fmt_time(dt), taken_from="file time" if dt else ""))


def scan_source(root: str, videos: bool = False) -> Scan:
    """List what is on a card (everything under DCIM) or in any folder ("Choose a folder instead"), from the
    directory listing only: name, size, modified time. Capture times are filled in from the file times and
    confirmed from the photos by ``confirm_capture_times``. ``videos`` also looks in the brand video folders
    outside DCIM (Sony PRIVATE/M4ROOT...). Reads nothing but the listing; writes nothing."""
    root = os.path.abspath(root)
    sc = Scan(root=root)
    with _quiet_errors():
        dcim = os.path.join(root, "DCIM")
        _walk(dcim if os.path.isdir(dcim) else root, root, sc.files, sc.errors)
        if videos and os.path.isdir(dcim):
            sc.videos_outside = True
            for sub in VIDEO_FOLDERS:
                p = os.path.join(root, *sub.split("/"))
                if os.path.isdir(p):
                    found: list = []
                    _walk(p, root, found, sc.errors)
                    sc.files += [f for f in found if f.kind == VIDEO]
    log.info("scan %s: %d files (%d raw, %d JPEG/HEIF, %d video, %d other), %d errors", root, len(sc.files),
             sc.count(RAW), sc.count(JPEG), sc.count(VIDEO), sc.count(OTHER), len(sc.errors))
    return sc


def confirm_capture_times(scan: Scan, on_progress=None, cancel=None) -> Scan:
    """Read each photo's own capture time (EXIF), one file at a time (one reader on the card, rule 7). Files that
    record none keep their modified time; videos keep theirs. ``on_progress(done, total)``."""
    stop = _stopper(cancel)
    todo = [f for f in scan.files if f.kind in (RAW, JPEG)]
    for i, f in enumerate(todo):
        if stop():
            return scan
        dt = parse_time(read_capture_time(f.path))
        if dt is not None:
            f.taken, f.taken_from = fmt_time(dt), "photo"
        if on_progress and (i % 10 == 9 or i == len(todo) - 1):
            on_progress(i + 1, len(todo))
    scan.confirmed = True
    return scan


# ============================================================================================ the import record
def record_path(dest: str) -> str:
    return os.path.join(dest, RECORD_DIR, RECORD_NAME)


class ImportRecord:
    """``<Photos>/_studio/imports.jsonl``: one line per imported file (rule 5). Travels with the Photos folder.
    Missing or unreadable: an empty record (import still works by looking at the destination, rule 4).
    Never holds a camera or card serial number."""

    def __init__(self, path: str):
        self.path = path
        self.by_key: dict = {}
        self.lines = 0
        self.problem = ""
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        self._index(e)
                    except (ValueError, KeyError, TypeError):
                        continue                              # one damaged line does not lose the rest
        except FileNotFoundError:
            pass
        except OSError as e:
            self.problem = str(e)
            log.warning("import record %s unreadable (%s): checking the destination folders only", path, e)

    def _index(self, e: dict) -> None:
        k = (str(e["name"]).lower(), int(e["size"]))
        self.by_key.setdefault(k, []).append(e)
        self.lines += 1

    def seen(self, f: SourceFile) -> bool:
        """Name and size match a line, and so does the capture time (within 2 s: FAT keeps file times to 2 s) when
        both sides know it from the photo itself."""
        for e in self.by_key.get((f.name.lower(), f.size), ()):
            known = e.get("taken", "")
            if not known or f.taken_from != "photo":
                return True
            a, b = parse_time(known), f.taken_dt
            if a is None or b is None or abs((a - b).total_seconds()) <= 2:
                return True
        return False

    def add(self, entry: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        line = json.dumps(slog.scrub(entry), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._index(entry)


# ============================================================================================ the plan (dry run)
@dataclass
class ImportOptions:
    dest: str                                    # the Photos folder (or "another folder for this import only")
    raws: bool = True                            # raws only by default (Bill)
    jpegs: bool = False
    videos: bool = False
    by_year: bool = True                         # <Photos>/<year>/<folder>
    day_start: str = "04:00"                     # a new day starts at 04:00 (night shoots stay together)
    one_folder: bool = False                     # "Put everything in one folder"
    names: dict = field(default_factory=dict)    # day key -> folder name typed by the photographer
    skip_days: list = field(default_factory=list)  # day keys whose tick is off
    second: str = ""                             # "Also copy to a second drive" root ('' = off)
    show_imported_before: bool = False           # override rule 5
    record: str = ""                             # default <dest>/_studio/imports.jsonl

    def wants(self, kind: str) -> bool:
        return {RAW: self.raws, JPEG: self.jpegs, VIDEO: self.videos}.get(kind, False)

    def to_dict(self) -> dict:
        return asdict(self)


COPY, EXISTS, IMPORTED, LEFT, DAY_OFF = "copy", "exists", "imported", "left", "day off"


@dataclass
class PlanItem:
    src: SourceFile
    action: str                  # copy | exists (same name and size there: compared when copying) | imported | left
    day: str                     # | day off
    folder: str = ""             # destination folder
    name: str = ""               # destination file name
    alt: str = ""                # for 'exists': the name used if the content turns out different
    renamed: bool = False
    note: str = ""

    @property
    def dest(self) -> str:
        return os.path.join(self.folder, self.name) if self.folder else ""


@dataclass
class DayGroup:
    key: str                     # 'YYYY-MM-DD' or 'Undated'
    total: int = 0               # files of the chosen kinds taken that day
    new: int = 0                 # of those, not imported before (what will be copied)
    name: str = ""               # the folder name that will be used (made safe)
    folder: str = ""             # the full destination folder
    chosen: bool = True


@dataclass
class ImportPlan:
    source: str
    options: ImportOptions
    items: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    left_behind: dict = field(default_factory=dict)  # kind -> count of files left on the card by choice
    days_off: int = 0                                # files of unticked days left on the card
    imported_before: int = 0
    bytes_needed: int = 0
    free: int | None = None
    total: int | None = None
    reserve: int = 0
    second_free: int | None = None
    second_total: int | None = None
    second_reserve: int = 0
    warnings: list = field(default_factory=list)
    problem: str = ""                                # set when the import cannot start (shown in red)
    no_raws: bool = False                            # the card holds JPEGs but no raw files
    record_lines: int = 0

    @property
    def to_copy(self) -> list:
        return [it for it in self.items if it.action in (COPY, EXISTS)]

    @property
    def fits(self) -> bool:
        return self.free is not None and self.bytes_needed <= max(0, self.free - self.reserve)

    @property
    def shortfall(self) -> int:
        if self.free is None:
            return self.bytes_needed
        return max(0, self.bytes_needed - max(0, self.free - self.reserve))

    @property
    def second_fits(self) -> bool:
        if not self.options.second:
            return True
        return self.second_free is not None and self.bytes_needed <= max(0, self.second_free - self.second_reserve)

    @property
    def second_shortfall(self) -> int:
        if not self.options.second:
            return 0
        if self.second_free is None:
            return self.bytes_needed
        return max(0, self.bytes_needed - max(0, self.second_free - self.second_reserve))

    @property
    def can_start(self) -> bool:
        return bool(self.to_copy) and not self.problem and self.fits and self.second_fits

    def left_text(self, where: str = "the card") -> str:
        """'214 JPEGs and 3 videos stay on the card.' ('' when nothing is left behind)."""
        parts = [count_words(n, k) for k, n in self.left_behind.items() if n]
        if self.days_off:
            parts.append(f"{self.days_off} file{'s' if self.days_off != 1 else ''} from unticked days")
        if not parts:
            return ""
        text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        many = len(parts) > 1 or not text.startswith("1 ")
        return f"{text} {'stay' if many else 'stays'} on {where}."


def count_words(n: int, kind: str) -> str:
    one, many = KIND_WORDS.get(kind, ("file", "files"))
    return f"{n} {one if n == 1 else many}"


def default_name(key: str) -> str:
    return key


def folder_for(dest: str, key: str, name: str, by_year: bool) -> str:
    """<dest>/<year>/<name> ('Undated' has no year level)."""
    if by_year and key != UNDATED:
        return os.path.join(dest, key[:4], name)
    return os.path.join(dest, name)


def _next_free(name: str, taken: set) -> str:
    stem, ext = os.path.splitext(name)
    i = 2
    while f"{stem}_{i}{ext}".lower() in taken:
        i += 1
    return f"{stem}_{i}{ext}"


def _listing(folder: str) -> dict:
    """lower-case name -> size of the files already in a destination folder ({} if it does not exist)."""
    out = {}
    try:
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file():
                    try:
                        out[e.name.lower()] = e.stat().st_size
                    except OSError:
                        out[e.name.lower()] = -1
    except OSError:
        pass
    return out


def _inside(child: str, parent: str) -> bool:
    c, p = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent))
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:
        return False


def plan_import(scan: Scan, options: ImportOptions, record: "ImportRecord | None" = None) -> ImportPlan:
    """The dry run (rule 9): decide every destination folder and file name, every skip and every rename, without
    writing anything. Reads the import record and the destination folders' listings. The dialog shows exactly
    what this returns, and ``run_import`` does exactly this."""
    o = options
    plan = ImportPlan(source=scan.root, options=o)
    dest = os.path.abspath(o.dest) if o.dest else ""
    if not dest:
        plan.problem = "Choose where your photos should be kept."
    elif _inside(dest, scan.root) or _inside(scan.root, dest):
        plan.problem = "The photos to import are already inside that folder. Choose a card or another folder."
    elif not drive_connected(dest):
        plan.problem = f"The drive {drive_of(dest)} is not connected."
    second = os.path.abspath(o.second) if o.second else ""
    if second and dest and (_inside(second, dest) or _inside(dest, second)):
        plan.problem = plan.problem or "The second copy must go to a different folder (ideally another drive)."
    if second and (_inside(second, scan.root) or _inside(scan.root, second)):
        plan.problem = plan.problem or "The second copy cannot go onto the card."
    rec = record if record is not None else (ImportRecord(o.record or record_path(dest)) if dest else None)
    plan.record_lines = rec.lines if rec else 0
    if rec and rec.problem:
        plan.warnings.append(f"The import record could not be read ({rec.problem}); photos already in the "
                             f"destination are still never copied twice.")

    # days, from the capture times
    wanted = [f for f in scan.files if o.wants(f.kind)]
    for f in scan.files:
        if not o.wants(f.kind) and f.kind != OTHER:
            plan.left_behind[f.kind] = plan.left_behind.get(f.kind, 0) + 1
    plan.no_raws = o.raws and scan.count(RAW) == 0 and scan.count(JPEG) > 0
    keys = {}
    for f in wanted:
        keys[id(f)] = day_key(f.taken_dt, o.day_start)
    days = sorted({k for k in keys.values()}, key=lambda k: (k == UNDATED, k))
    skip = set(o.skip_days or ())
    groups = {}
    for k in days:
        g = DayGroup(key=k, chosen=k not in skip)
        groups[k] = g
    chosen = [k for k in days if k not in skip]
    if o.one_folder and chosen:
        first = chosen[0]
        name = safe_folder_name(o.names.get("*", o.names.get(first, default_name(first))), default_name(first))
        folder = folder_for(dest, first, name, o.by_year) if dest else ""
        for k in chosen:
            groups[k].name, groups[k].folder = name, folder
    else:
        for k in chosen:
            name = safe_folder_name(o.names.get(k, default_name(k)), default_name(k))
            groups[k].name = name
            groups[k].folder = folder_for(dest, k, name, o.by_year) if dest else ""

    # files: skip, copy, or rename (rule 4), in card order (sequential reads)
    existing: dict = {}          # folder -> {lower name: size} already there
    used: dict = {}              # folder -> lower names taken (there, planned, or reserved as an alternative)
    planned: dict = {}           # folder -> lower names this plan writes
    for f in sorted(wanted, key=lambda x: x.rel.lower()):
        k = keys[id(f)]
        g = groups[k]
        g.total += 1
        if not g.chosen:
            plan.items.append(PlanItem(src=f, action=DAY_OFF, day=k))
            plan.days_off += 1
            continue
        if rec is not None and not o.show_imported_before and rec.seen(f):
            plan.items.append(PlanItem(src=f, action=IMPORTED, day=k, folder=g.folder, note="imported before"))
            plan.imported_before += 1
            continue
        g.new += 1
        plan.bytes_needed += f.size
        if not g.folder:
            plan.items.append(PlanItem(src=f, action=COPY, day=k))
            continue
        fk = os.path.normcase(g.folder)
        there = existing.setdefault(fk, _listing(g.folder))
        taken = used.setdefault(fk, set(there))
        mine = planned.setdefault(fk, set())
        it = PlanItem(src=f, action=COPY, day=k, folder=g.folder, name=f.name)
        low = f.name.lower()
        if low in mine or (low in taken and low not in there):
            it.name, it.renamed = _next_free(f.name, taken), True
            it.note = "another photo on the card has the same name"
        elif low in there and there[low] == f.size:
            it.action, it.alt = EXISTS, _next_free(f.name, taken)
            it.note = "a file with this name and size is already there: compared when copying"
            taken.add(it.alt.lower())
        elif low in there:
            it.name, it.renamed = _next_free(f.name, taken), True
            it.note = "a different photo with this name is already there"
        taken.add(it.name.lower())
        mine.add(it.name.lower())
        plan.items.append(it)
    for k in days:
        if groups[k].total:
            plan.groups.append(groups[k])

    # space (rule: keep 2 GB or 5 % free) and long paths
    if dest:
        plan.free, plan.total = free_space(dest)
        plan.reserve = reserve_for(plan.total)
    if second:
        plan.second_free, plan.second_total = free_space(second)
        plan.second_reserve = reserve_for(plan.second_total)
    long = [it for it in plan.to_copy if len(it.dest) > LONG_PATH]
    if long:
        plan.warnings.append(f"{len(long)} file path{'s are' if len(long) != 1 else ' is'} longer than {LONG_PATH} "
                             f"characters, which some Windows programs cannot open. A shorter folder name helps.")
    if dest and inside_onedrive(dest):
        plan.warnings.append("This folder is inside OneDrive: raw files will fill your cloud storage quickly.")
    return plan


# ============================================================================================ copying
def _stopper(cancel):
    if cancel is None:
        return lambda: False
    if hasattr(cancel, "is_set"):
        return cancel.is_set
    return cancel


class _KeepAwake:
    """Windows: keep the PC awake while the import runs (SetThreadExecutionState, ES_CONTINUOUS|ES_SYSTEM_REQUIRED).
    Per thread, so it is set and cleared on the thread that copies."""

    def __enter__(self):
        self.on = False
        if os.name == "nt":
            try:
                import ctypes
                self.on = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
            except Exception:                                # noqa: BLE001
                self.on = False
        return self

    def __exit__(self, *exc):
        if self.on:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:                                # noqa: BLE001
                pass
        return False


def _hash_cached(path: str) -> tuple:
    h = hashlib.blake2b(digest_size=32)
    n = 0
    buf = bytearray(BLOCK)
    mv = memoryview(buf)
    with open(path, "rb", buffering=0) as fh:
        while True:
            k = fh.readinto(buf)
            if not k:
                break
            h.update(mv[:k])
            n += k
    return h.hexdigest(), n


def _hash_uncached(path: str) -> tuple:
    """Windows: read a file back from the disk itself, past the Windows file cache (FILE_FLAG_NO_BUFFERING), so the
    check sees what was really written. Raises OSError if it cannot."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                             wintypes.LPVOID]
    k32.ReadFile.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.VirtualAlloc.restype = wintypes.LPVOID
    k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    invalid = wintypes.HANDLE(-1).value
    h = k32.CreateFileW(path, 0x80000000, 0x00000001, None, 3, 0x20000000 | 0x08000000, None)
    if h in (None, invalid):
        raise OSError(ctypes.get_last_error(), "CreateFileW failed", path)
    size = 4 * 1024 * 1024                                     # a multiple of every sector size
    buf = k32.VirtualAlloc(None, size, 0x1000 | 0x2000, 0x04)  # page-aligned, as unbuffered reads need
    if not buf:
        k32.CloseHandle(h)
        raise OSError(ctypes.get_last_error(), "VirtualAlloc failed")
    try:
        hh = hashlib.blake2b(digest_size=32)
        n = 0
        got = wintypes.DWORD(0)
        while True:
            if not k32.ReadFile(h, buf, size, ctypes.byref(got), None):
                raise OSError(ctypes.get_last_error(), "ReadFile failed", path)
            if got.value == 0:
                break
            hh.update(ctypes.string_at(buf, got.value))
            n += got.value
            if got.value < size:
                break
        return hh.hexdigest(), n
    finally:
        k32.VirtualFree(buf, 0, 0x8000)
        k32.CloseHandle(h)


CHECK_UNCACHED = "read back from the disk, past the Windows file cache"
CHECK_CACHED = "read back after flushing to disk (the system's file cache may have answered)"


def _choose_check(folder: str) -> str:
    """Use the uncached read-back only if it proves itself on a test file first (so a fault in it can never fail
    good copies); otherwise read back after fsync, and say so in the log (24.6 rule 3)."""
    if os.name != "nt" or os.environ.get("STUDIO_IMPORT_CACHED_CHECK"):
        return CHECK_CACHED
    probe = os.path.join(folder, "check.partial")
    try:
        os.makedirs(folder, exist_ok=True)
        data = os.urandom(3 * 1024 * 1024 + 12345)
        with open(probe, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        want = hashlib.blake2b(data, digest_size=32).hexdigest()
        got, n = _hash_uncached(probe)
        return CHECK_UNCACHED if (got == want and n == len(data)) else CHECK_CACHED
    except Exception:                                        # noqa: BLE001
        slog.log_exception("uncached read-back not available; checking after fsync", log)
        return CHECK_CACHED
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def _hash_back(path: str, check: str) -> tuple:
    if check == CHECK_UNCACHED:
        try:
            return _hash_uncached(path)
        except OSError:                                      # e.g. a second drive that refuses unbuffered reads
            log.warning("uncached read-back failed for %s; reading it back normally", path)
    return _hash_cached(path)


def _rename_no_overwrite(src: str, dst: str) -> None:
    """Rename, never replacing a file that is already there (rule 4). FileExistsError if ``dst`` exists."""
    if os.name == "nt":
        os.rename(src, dst)                                  # Windows refuses to replace an existing file
        return
    try:
        os.link(src, dst)                                    # atomic: fails if dst exists
        os.unlink(src)
        return
    except FileExistsError:
        raise
    except OSError:                                          # no hard links on this drive (exFAT, FAT32)
        if os.path.lexists(dst):
            raise FileExistsError(dst)
        os.rename(src, dst)


def clean_partials(folder: str) -> list:
    """Remove the ``.partial`` files a crashed import left in one destination folder (rule 2). Only photo and video
    partials are touched. Returns the names removed."""
    removed = []
    try:
        names = os.listdir(folder)
    except OSError:
        return removed
    for n in names:
        if n.lower().endswith(".partial") and kind_of(n[:-8]) != OTHER:
            try:
                os.remove(os.path.join(folder, n))
                removed.append(n)
            except OSError as e:
                log.warning("could not remove %s: %s", n, e)
    return removed


@dataclass
class ImportReport:
    source: str = ""
    copied: list = field(default_factory=list)          # (source rel, destination)
    skipped: list = field(default_factory=list)         # (source rel, why): imported before / already there
    renamed: list = field(default_factory=list)         # (source rel, destination): same name, different photo
    failed: list = field(default_factory=list)          # (source rel, why)
    second_failed: list = field(default_factory=list)   # (source rel, why)
    left_behind: dict = field(default_factory=dict)
    days_off: int = 0
    left_text: str = ""
    cleaned: list = field(default_factory=list)         # stale .partial files removed
    folders: list = field(default_factory=list)         # destination folders, in day order
    check: str = ""
    cancelled: bool = False
    source_gone: bool = False
    stopped: str = ""                                   # why the whole import stopped early ('' = it did not)
    bytes: int = 0
    seconds: float = 0.0
    log_path: str = ""

    @property
    def all_checked(self) -> bool:
        """Every file that was to be copied is copied and checked (the message that says the card can come out)."""
        return not self.failed and not self.cancelled and not self.stopped

    def summary_lines(self) -> list:
        out = [f"Copied and checked: {len(self.copied)}"]
        imported = sum(1 for _, why in self.skipped if why == "imported before")
        there = len(self.skipped) - imported
        if imported:
            out.append(f"Imported before, skipped: {imported}")
        if there:
            out.append(f"Already in the folder (same photo), skipped: {there}")
        if self.renamed:
            out.append(f"Same name as a different photo, kept under a new name: {len(self.renamed)}")
        if self.failed:
            out.append(f"Failed: {len(self.failed)}")
            out += [f"    {rel}: {why}" for rel, why in self.failed[:20]]
            if len(self.failed) > 20:
                out.append(f"    ... and {len(self.failed) - 20} more (see the log)")
        if self.second_failed:
            out.append(f"Second copy failed for {len(self.second_failed)} file(s); the first copies are fine")
        if self.left_text:
            out.append("Left on the card by choice: " + self.left_text)
        if self.stopped:
            out.append("Stopped early: " + self.stopped)
        elif self.cancelled:
            out.append("Cancelled. Import again to copy the rest: what was copied is not copied twice.")
        return out


def _log_file() -> tuple:
    """logs/import_<date>_<time>.log, never shared with another import (rule 8)."""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(str(paths.get("LOGS")), f"import_{stamp}")
    for i in range(1, 1000):
        path = base + (f"_{i}" if i > 1 else "") + ".log"
        try:
            return path, open(path, "x", encoding="utf-8")
        except FileExistsError:
            continue
    return base + ".log", open(base + ".log", "a", encoding="utf-8")


def run_import(plan: ImportPlan, on_progress=None, cancel=None) -> ImportReport:
    """Copy what the plan says, by the copy rules (24.6). ``on_progress(info)`` gets a dict: done, total (files),
    bytes_done, bytes_total, current (name), speed (bytes/s), eta (s). ``cancel``: an Event or a callable; the
    file being copied is finished or removed, never left half-written under its final name."""
    o = plan.options
    stop = _stopper(cancel)
    where = "the card" if os.path.isdir(os.path.join(plan.source, "DCIM")) else "the source folder"
    rep = ImportReport(source=plan.source, left_behind=dict(plan.left_behind), days_off=plan.days_off,
                       left_text=plan.left_text(where))
    rep.log_path, lf = _log_file()
    t0 = time.perf_counter()
    dest = os.path.abspath(o.dest)
    second = os.path.abspath(o.second) if o.second else ""
    rec = ImportRecord(o.record or record_path(dest))

    def wlog(text: str) -> None:
        lf.write(text.rstrip("\n") + "\n")
        lf.flush()

    wlog(f"Photographic Studio import, {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    wlog(f"from {plan.source}")
    wlog(f"to   {dest}" + (f"   second copy: {second}" if second else ""))
    wlog(f"options: raws={o.raws} jpegs={o.jpegs} videos={o.videos} by_year={o.by_year} day_start={o.day_start} "
         f"one_folder={o.one_folder} show_imported_before={o.show_imported_before}")
    problem = plan.problem
    if not problem and plan.to_copy and not plan.fits:
        problem = f"not enough free space on {drive_of(dest)} ({human_bytes(plan.shortfall)} short)"
    if not problem and plan.to_copy and not plan.second_fits:
        problem = f"not enough free space for the second copy ({human_bytes(plan.second_shortfall)} short)"
    if problem:
        rep.stopped = problem
        wlog("NOT STARTED: " + problem)
        lf.close()
        return rep
    for it in plan.items:
        if it.action == IMPORTED:
            rep.skipped.append((it.src.rel, "imported before"))
            wlog(f"SKIP     {it.src.rel}  (imported before)")
    todo = plan.to_copy
    folders = []                                   # in day order: "Open in Library" shows the first day
    for g in plan.groups:
        if g.folder and g.folder not in folders and any(it.folder == g.folder for it in todo):
            folders.append(g.folder)
    for it in todo:
        if it.folder and it.folder not in folders:
            folders.append(it.folder)
    rep.folders = folders
    total_bytes = sum(it.src.size for it in todo)
    prog = {"done": 0, "bytes": 0, "last": 0.0, "current": ""}

    def report(force=False):
        if not on_progress:
            return
        now = time.perf_counter()
        if not force and now - prog["last"] < 0.25:
            return
        prog["last"] = now
        el = now - t0
        speed = prog["bytes"] / el if el > 0.5 else 0.0
        on_progress({"done": prog["done"], "total": len(todo), "bytes_done": prog["bytes"],
                     "bytes_total": total_bytes, "current": prog["current"], "speed": speed,
                     "eta": (total_bytes - prog["bytes"]) / speed if speed > 0 else None})

    def tick(k):
        prog["bytes"] += k
        report()

    try:
        with _KeepAwake():
            try:
                for f in folders:
                    os.makedirs(f, exist_ok=True)
                    rep.cleaned += [os.path.join(f, n) for n in clean_partials(f)]
                    if second:
                        sf = os.path.join(second, os.path.relpath(f, dest))
                        os.makedirs(sf, exist_ok=True)
                        rep.cleaned += [os.path.join(sf, n) for n in clean_partials(sf)]
                for c in rep.cleaned:
                    wlog(f"CLEANED  {c}  (left by an earlier import that did not finish)")
                rep.check = _choose_check(os.path.join(dest, RECORD_DIR)) if todo else CHECK_CACHED
                wlog(f"check: {rep.check}")
            except OSError as e:
                rep.stopped = f"could not create the destination folder: {e.strerror or e}"
                wlog("STOPPED: " + rep.stopped)
                todo = []
            taken_by_folder: dict = {}
            for i, it in enumerate(todo):
                if stop():
                    rep.cancelled = True
                    wlog("CANCELLED by the photographer")
                    break
                start_bytes = prog["bytes"]
                prog["done"], prog["current"] = i, it.src.name
                report(force=True)
                try:
                    result = _import_one(it, dest, second, rec, rep, wlog, taken_by_folder, stop, tick)
                except ImportStopped as e:
                    rep.stopped = str(e)
                    wlog("STOPPED: " + rep.stopped)
                    break
                if result == "cancelled":
                    rep.cancelled = True
                    wlog("CANCELLED by the photographer")
                    break
                prog["bytes"] = start_bytes + it.src.size      # retries and compares read more than once
    except Exception as e:                                     # noqa: BLE001  (logged, reported, never silent)
        slog.log_exception("import failed", log)
        rep.stopped = rep.stopped or f"unexpected problem: {e}"
        wlog("STOPPED: " + rep.stopped)
    rep.seconds = time.perf_counter() - t0
    if not rep.cancelled and not rep.stopped:
        prog["done"], prog["bytes"], prog["current"] = len(todo), total_bytes, ""
        report(force=True)
    wlog("")
    wlog(f"{rep.bytes / 1e6:.1f} MB in {rep.seconds:.1f} s")
    for line in rep.summary_lines():
        wlog(line)
    if rep.all_checked:
        wlog("All photos are copied and checked. The Studio wrote nothing to the card.")
    lf.close()
    log.info("import %s -> %s: %d copied, %d skipped, %d renamed, %d failed%s (log %s)", plan.source, dest,
             len(rep.copied), len(rep.skipped), len(rep.renamed), len(rep.failed),
             f", stopped: {rep.stopped}" if rep.stopped else (", cancelled" if rep.cancelled else ""), rep.log_path)
    return rep


def _source_gone(plan_source: str) -> bool:
    with _quiet_errors():
        return not os.path.isdir(plan_source)


def _import_one(it: PlanItem, dest: str, second: str, rec: ImportRecord, rep: ImportReport, wlog, taken_by_folder,
                stop, tick=None) -> str:
    """Copy one file to its folder (and the second copy), check it, name it, record it. Returns 'copied',
    'skipped', 'failed' or 'cancelled'. Raises ImportStopped when the whole import must stop."""
    src = it.src
    t0 = time.perf_counter()
    folder = it.folder
    sfolder = os.path.join(second, os.path.relpath(folder, dest)) if second else ""
    taken = taken_by_folder.setdefault(os.path.normcase(folder), set(_listing(folder)))
    name = it.name                                   # the name the copy is written under (.partial while checked)
    existing_hash = None
    if it.action == EXISTS:                          # same name and size already there: compare contents first
        try:
            existing_hash, _ = _hash_cached(os.path.join(folder, it.name))
        except OSError:
            existing_hash = None
    reads = checks = 0
    while True:
        if stop():
            return "cancelled"
        part = os.path.join(folder, name + ".partial")
        spart = os.path.join(sfolder, name + ".partial") if sfolder and existing_hash is None else ""
        try:
            digest, second_err = _copy_blocks(src.path, src.size, part, spart, existing_hash is not None, stop, tick)
        except _Cancelled:
            return "cancelled"
        except _ReadError as e:
            if _source_gone(rep.source):
                raise ImportStopped("the card was taken out (or the folder disappeared). Put it back and import "
                                    "again: what was copied is not copied twice.")
            reads += 1
            if reads <= 2:                           # rule 6: retry twice, then fail this file and go on
                wlog(f"RETRY    {src.rel}  (read problem: {e.reason})")
                continue
            rep.failed.append((src.rel, f"could not be read from the card ({e.reason})"))
            wlog(f"FAILED   {src.rel}  could not be read: {e.reason}")
            return "failed"
        except _WriteError as e:
            if not drive_connected(dest):
                raise ImportStopped(f"the destination drive {drive_of(dest)} was disconnected.")
            if e.full:
                raise ImportStopped(f"the destination drive {drive_of(dest)} is full.")
            rep.failed.append((src.rel, f"could not be written ({e.reason})"))
            wlog(f"FAILED   {src.rel}  could not be written: {e.reason}")
            return "failed"
        if existing_hash is not None:
            if digest == existing_hash:
                rep.skipped.append((src.rel, "already in the folder"))
                wlog(f"SKIP     {src.rel}  (the same photo is already at {os.path.join(folder, name)})")
                return "skipped"
            name, existing_hash = it.alt, None       # same name and size, a different photo: keep both (rule 4)
            continue
        try:                                         # rule 3: read the copy back, compare hash and size
            back, nb = _hash_back(part, rep.check)
        except OSError:
            back, nb = "", -1
        if back != digest or nb != src.size:
            _remove(part)
            if spart:
                _remove(spart)
            checks += 1
            if checks <= 1:
                wlog(f"RETRY    {src.rel}  (the copy did not match the card; copying it again)")
                continue
            rep.failed.append((src.rel, "the copy did not match the card, twice (a faulty card, reader or drive?)"))
            wlog(f"FAILED   {src.rel}  the copy did not match the card twice")
            return "failed"
        break
    try:
        os.utime(part, (src.mtime, src.mtime))       # the copy keeps the card's modified time
    except OSError:
        pass
    final = _finalise(part, folder, name, taken)     # rule 2: named only after the check
    final_name = os.path.basename(final)
    renamed = final_name.lower() != src.name.lower()
    if sfolder:                                      # checked the same way; its failure never undoes the first copy
        if second_err:
            _remove(spart)
            rep.second_failed.append((src.rel, second_err))
            wlog(f"SECOND   {src.rel}  FAILED: {second_err}")
        else:
            try:
                back, nb = _hash_back(spart, rep.check)
                if back != digest or nb != src.size:
                    raise OSError("the second copy did not match the card")
                os.utime(spart, (src.mtime, src.mtime))
                staken = taken_by_folder.setdefault(os.path.normcase(sfolder), set(_listing(sfolder)))
                sfinal = _finalise(spart, sfolder, final_name, staken)
                wlog(f"SECOND   {src.rel}  -> {sfinal}")
            except (OSError, ImportStopped) as e:
                _remove(spart)
                rep.second_failed.append((src.rel, str(e)))
                wlog(f"SECOND   {src.rel}  FAILED: {e}")
    rep.bytes += src.size
    if renamed:
        rep.renamed.append((src.rel, final))
    rep.copied.append((src.rel, final))
    try:
        to = os.path.relpath(final, dest).replace(os.sep, "/")
    except ValueError:
        to = final
    try:
        rec.add({"name": src.name, "size": src.size, "taken": src.taken if src.taken_from == "photo" else "",
                 "hash": "blake2b:" + digest, "to": to, "when": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    except OSError as e:
        log.warning("could not add to the import record %s: %s", rec.path, e)
    wlog(f"{'RENAMED ' if renamed else 'COPIED  '} {src.rel}  -> {final}  {src.size} bytes  blake2b:{digest[:16]}  "
         f"{time.perf_counter() - t0:.2f} s")
    return "copied"


def _finalise(part: str, folder: str, name: str, taken: set) -> str:
    """Rename a checked ``.partial`` to its final name; if that name appeared meanwhile, the next free one."""
    final = os.path.join(folder, name)
    for _ in range(1000):
        try:
            _rename_no_overwrite(part, final)
            taken.add(os.path.basename(final).lower())
            return final
        except FileExistsError:
            taken.add(os.path.basename(final).lower())
            final = os.path.join(folder, _next_free(name, taken))
    raise ImportStopped(f"could not find a free name for {name}")


def _remove(p: str) -> None:
    if not p:
        return
    try:
        os.remove(p)
    except OSError:
        pass


class _ReadError(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class _WriteError(Exception):
    def __init__(self, reason, full=False):
        super().__init__(reason)
        self.reason, self.full = reason, full


class _Cancelled(Exception):
    pass


def _open_new(path: str):
    try:
        return open(path, "xb")
    except FileExistsError:
        os.remove(path)
        return open(path, "xb")


def _is_full(e: OSError) -> bool:
    import errno
    return getattr(e, "errno", 0) == errno.ENOSPC or getattr(e, "winerror", 0) in (39, 112)


def _copy_blocks(src: str, size: int, part: str, spart: str, hash_only: bool = False, stop=None, tick=None):
    """Read the source once in 8 MB blocks (one reader on the card, rule 7), hashing as it goes and writing the main
    copy and the second copy from the same blocks; flush and fsync both. ``hash_only`` writes nothing (to compare
    with a file already in the destination). On any failure the .partial files are removed. Returns
    (hash, second-copy error or '')."""
    h = hashlib.blake2b(digest_size=32)
    buf = bytearray(BLOCK)
    mv = memoryview(buf)
    out = sout = None
    second_err = ""
    n = 0
    try:
        try:
            fin = open(src, "rb", buffering=0)               # read-only: the card is never changed (rule 1)
        except OSError as e:
            raise _ReadError(e.strerror or str(e))
        with fin:
            if not hash_only:
                try:
                    out = _open_new(part)
                except OSError as e:
                    raise _WriteError(e.strerror or str(e), _is_full(e))
                if spart:
                    try:
                        sout = _open_new(spart)
                    except OSError as e:
                        second_err = e.strerror or str(e)
            while True:
                if stop is not None and stop():
                    raise _Cancelled()
                try:
                    k = fin.readinto(buf)
                except OSError as e:
                    raise _ReadError(e.strerror or str(e))
                if not k:
                    break
                block = mv[:k]
                h.update(block)
                n += k
                if out is not None:
                    try:
                        out.write(block)
                    except OSError as e:
                        raise _WriteError(e.strerror or str(e), _is_full(e))
                if sout is not None:
                    try:
                        sout.write(block)
                    except OSError as e:
                        second_err = e.strerror or str(e)
                        _close_quietly(sout)
                        sout = None
                if tick is not None:
                    tick(k)
        if n != size:
            raise _ReadError(f"read {n} bytes where the card listed {size}")
        if out is not None:
            try:
                out.flush()
                os.fsync(out.fileno())
            except OSError as e:
                raise _WriteError(e.strerror or str(e), _is_full(e))
        if sout is not None:
            try:
                sout.flush()
                os.fsync(sout.fileno())
            except OSError as e:
                second_err = e.strerror or str(e)
    except BaseException:
        _close_quietly(out)
        _close_quietly(sout)
        if not hash_only:
            _remove(part)
            _remove(spart)
        raise
    _close_quietly(out)
    _close_quietly(sout)
    return h.hexdigest(), second_err


def _close_quietly(fh) -> None:
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass


# ============================================================================================ small helpers
def human_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("bytes", "KB") else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} TB"


def summary_counts(scan: Scan, options: ImportOptions) -> dict:
    """For the card list and the card-found notice: photos on the card and how many are new."""
    plan = plan_import(scan, options)
    return {"photos": scan.photos, "files": len(scan.files), "new": len(plan.to_copy),
            "imported": plan.imported_before}


__all__ = ["CardWatcher", "find_cards", "list_drives", "scan_source", "confirm_capture_times", "plan_import", "run_import",
           "ImportOptions", "ImportPlan", "ImportReport", "ImportRecord", "Scan", "SourceFile", "Card", "Drive",
           "safe_folder_name", "day_key", "parse_time", "free_space", "drive_connected", "inside_onedrive",
           "record_path", "clean_partials", "human_bytes", "summary_counts", "default_photos_dir"]

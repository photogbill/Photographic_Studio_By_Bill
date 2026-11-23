"""Camera profiles (.dcp, the DNG camera profile format): read, apply, write.

A profile replaces stage 11's camera matrix (white-balanced camera RGB -> ProPhoto) and can add the profile's
hue/saturation tables (ProfileHueSatMap, ProfileLookTable). Implemented from the public DNG Specification
(Adobe, version 1.6, chapter 6 "Mapping Camera Color Space to CIE XYZ Space" and the profile tags); no SDK code.

* Two calibration illuminants are interpolated by inverse colour temperature of the photo's white balance, as the
  spec describes. ForwardMatrix is used when present (it maps white-balanced camera RGB straight to XYZ D50);
  otherwise ColorMatrix with Bradford adaptation to D50.
* HueSatMap and LookTable are applied in HSV of linear ProPhoto RGB, trilinear, hue wrapping.
* The profile's own tone curve is NOT used: the Studio's tone curve and Light panel stay in charge of brightness.

Profiles you add live in settings/camera_profiles (private: commercial profiles must not be shared). Shared
profiles we make ourselves go in profiles/camera_profiles.

Lightroom's own profiles (Adobe Standard, and the "Camera ..." profiles that match the camera's picture styles) are
found where Adobe installs them on this PC and offered as "lr:<profile name>" for the photo's camera: read in place,
never copied or shared. An edit that names one renders with the camera matrix on a PC without Lightroom. Their
ProfileToneCurve and BaselineExposureOffset are read for Tone > "Lightroom-like" (tone.py).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from ... import paths
from .spaces import D50_XY, PROPHOTO, XYZ_to_xy, bradford, xy_to_XYZ

log = slog.get("engine.dcp")

EXT = ".dcp"

# DNG LightSource codes -> correlated colour temperature (K), as used for profile interpolation
ILLUMINANT_K = {1: 5500, 2: 4150, 3: 2850, 4: 5500, 9: 5500, 10: 6500, 11: 7500, 12: 6430, 13: 5000, 14: 4150,
                15: 3450, 16: 2940, 17: 2856, 18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504, 23: 5003, 24: 3200}

T_MODEL, T_CM1, T_CM2, T_ILL1, T_ILL2 = 50708, 50721, 50722, 50778, 50779
T_FM1, T_FM2, T_NAME, T_HSD, T_HS1, T_HS2 = 50964, 50965, 50936, 50937, 50938, 50939
T_TONE, T_EMBED, T_COPY, T_LKD, T_LK, T_HSENC, T_LKENC, T_BEO = 50940, 50941, 50942, 50981, 50982, 51107, 51108, 51109


@dataclass
class Profile:
    name: str = ""
    camera: str = ""
    copyright: str = ""
    illuminants: list = field(default_factory=list)     # [code1, code2?]
    color_matrices: list = field(default_factory=list)  # 3x3 XYZ -> camera, one per illuminant
    forward_matrices: list = field(default_factory=list)  # 3x3 WB camera -> XYZ D50 (optional)
    hsm_dims: tuple | None = None                       # (hue, sat, val) divisions
    hsm: list = field(default_factory=list)             # one (val, hue, sat, 3) table per illuminant
    hsm_srgb: bool = False                              # value axis encoded with the sRGB curve
    look_dims: tuple | None = None
    look: np.ndarray | None = None
    look_srgb: bool = False
    tone: np.ndarray | None = None                      # ProfileToneCurve, (n, 2) linear in/out pairs
    baseline_offset: float | None = None                # BaselineExposureOffset (stops), when the profile has one
    path: str = ""


# ------------------------------------------------------------------------------------------ reading
def read(path: str) -> Profile:
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        raise ValueError(f"{os.path.basename(path)} is not a DNG camera profile")
    e = "<" if data[:2] == b"II" else ">"
    magic = struct.unpack(e + "H", data[2:4])[0]
    if magic not in (0x4352, 42):
        raise ValueError(f"{os.path.basename(path)} is not a DNG camera profile (magic {magic:#x})")
    off = struct.unpack(e + "I", data[4:8])[0]
    n = struct.unpack(e + "H", data[off:off + 2])[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8}
    tags = {}
    for k in range(n):
        p = off + 2 + 12 * k
        tag, typ, cnt = struct.unpack(e + "HHI", data[p:p + 8])
        size = sizes.get(typ, 1) * cnt
        voff = p + 8 if size <= 4 else struct.unpack(e + "I", data[p + 8:p + 12])[0]
        raw = data[voff:voff + size]
        if typ == 2:
            val = raw.split(b"\0")[0].decode("latin-1")
        elif typ in (5, 10):
            f = "I" if typ == 5 else "i"
            v = struct.unpack(e + f * (2 * cnt), raw)
            val = [v[2 * i] / v[2 * i + 1] if v[2 * i + 1] else 0.0 for i in range(cnt)]
        elif typ == 11:
            val = list(struct.unpack(e + "f" * cnt, raw))
        elif typ == 12:
            val = list(struct.unpack(e + "d" * cnt, raw))
        elif typ in (3, 4, 9):
            f = {3: "H", 4: "I", 9: "i"}[typ]
            val = list(struct.unpack(e + f * cnt, raw))
        else:
            val = raw
        tags[tag] = val
    pr = Profile(path=path)
    pr.camera = tags.get(T_MODEL, "")
    pr.name = tags.get(T_NAME, "") or os.path.splitext(os.path.basename(path))[0]
    pr.copyright = tags.get(T_COPY, "")
    if T_TONE in tags and len(tags[T_TONE]) >= 4:
        pr.tone = np.asarray(tags[T_TONE], np.float64).reshape(-1, 2)
    if T_BEO in tags and tags[T_BEO]:
        pr.baseline_offset = float(tags[T_BEO][0])
    for ti, tc, tf in ((T_ILL1, T_CM1, T_FM1), (T_ILL2, T_CM2, T_FM2)):
        if tc in tags:
            pr.illuminants.append(int(tags.get(ti, [21])[0]))
            pr.color_matrices.append(np.array(tags[tc], float).reshape(3, 3))
            if tf in tags:
                pr.forward_matrices.append(np.array(tags[tf], float).reshape(3, 3))
    if not pr.color_matrices:
        raise ValueError(f"{os.path.basename(path)} has no colour matrix")
    if len(pr.forward_matrices) not in (0, len(pr.color_matrices)):
        pr.forward_matrices = []
    if T_HSD in tags and T_HS1 in tags:
        d = tuple(int(v) for v in tags[T_HSD])
        pr.hsm_dims = d
        for t in (T_HS1, T_HS2):
            if t in tags:
                pr.hsm.append(_table(tags[t], d))
        pr.hsm_srgb = bool(tags.get(T_HSENC, [0])[0])
    if T_LKD in tags and T_LK in tags:
        d = tuple(int(v) for v in tags[T_LKD])
        pr.look_dims = d
        pr.look = _table(tags[T_LK], d)
        pr.look_srgb = bool(tags.get(T_LKENC, [0])[0])
    return pr


def _table(vals, dims) -> np.ndarray:
    h, s, v = dims
    a = np.asarray(vals, np.float32)
    if a.size != h * s * max(v, 1) * 3:
        raise ValueError("hue/saturation table size does not match its dimensions")
    return a.reshape(max(v, 1), h, s, 3)                # stored value-hue-saturation, as the spec says


# ------------------------------------------------------------------------------------------ writing (tests, our own profiles)
def write(pr: Profile, path: str) -> str:
    """Write a little-endian .dcp (the tags this module reads)."""
    entries = {T_MODEL: (2, pr.camera), T_NAME: (2, pr.name)}
    if pr.copyright:
        entries[T_COPY] = (2, pr.copyright)
    for k, (ti, tc, tf) in enumerate(((T_ILL1, T_CM1, T_FM1), (T_ILL2, T_CM2, T_FM2))):
        if k < len(pr.color_matrices):
            entries[ti] = (3, [int(pr.illuminants[k])])
            entries[tc] = (10, list(np.asarray(pr.color_matrices[k], float).ravel()))
            if k < len(pr.forward_matrices):
                entries[tf] = (10, list(np.asarray(pr.forward_matrices[k], float).ravel()))
    if pr.hsm_dims and pr.hsm:
        entries[T_HSD] = (4, list(pr.hsm_dims))
        for t, tab in zip((T_HS1, T_HS2), pr.hsm):
            entries[t] = (11, list(np.asarray(tab, float).ravel()))
        entries[T_HSENC] = (4, [int(pr.hsm_srgb)])
    if pr.tone is not None:
        entries[T_TONE] = (11, list(np.asarray(pr.tone, float).ravel()))
    if pr.baseline_offset is not None:
        entries[T_BEO] = (10, [float(pr.baseline_offset)])
    if pr.look_dims and pr.look is not None:
        entries[T_LKD] = (4, list(pr.look_dims))
        entries[T_LK] = (11, list(np.asarray(pr.look, float).ravel()))
        entries[T_LKENC] = (4, [int(pr.look_srgb)])

    def enc(typ, vals):
        if typ == 2:
            b = vals.encode("latin-1") + b"\0"
            return b, len(b)
        if typ == 3:
            return struct.pack("<" + "H" * len(vals), *vals), len(vals)
        if typ == 4:
            return struct.pack("<" + "I" * len(vals), *vals), len(vals)
        if typ == 10:
            out = b"".join(struct.pack("<ii", int(round(v * 1000000)), 1000000) for v in vals)
            return out, len(vals)
        return struct.pack("<" + "f" * len(vals), *vals), len(vals)

    n = len(entries)
    data_off = 8 + 2 + 12 * n + 4
    head, body = bytearray(), bytearray()
    for tag in sorted(entries):
        typ, vals = entries[tag]
        raw, cnt = enc(typ, vals)
        if len(raw) <= 4:
            head += struct.pack("<HHI", tag, typ, cnt) + raw.ljust(4, b"\0")
        else:
            head += struct.pack("<HHII", tag, typ, cnt, data_off + len(body))
            body += raw + (b"\0" if len(raw) % 2 else b"")
    blob = b"IIRC" + struct.pack("<I", 8) + struct.pack("<H", n) + bytes(head) + struct.pack("<I", 0) + bytes(body)
    with open(path, "wb") as fh:
        fh.write(blob)
    return path


# ------------------------------------------------------------------------------------------ colour maths
def _weight(pr: Profile, cct: float) -> float:
    """Weight of illuminant 1 (DNG spec: linear in inverse temperature between the two, clamped)."""
    if len(pr.color_matrices) < 2:
        return 1.0
    t1, t2 = ILLUMINANT_K.get(pr.illuminants[0], 2856), ILLUMINANT_K.get(pr.illuminants[1], 6504)
    if t1 == t2:
        return 1.0
    g = (1.0 / cct - 1.0 / t2) / (1.0 / t1 - 1.0 / t2)
    return float(min(max(g, 0.0), 1.0))


def _mix(mats, g):
    return mats[0] if len(mats) == 1 else g * mats[0] + (1.0 - g) * mats[1]


def _cct(xy) -> float:
    from .camera import xy_to_cct
    return float(min(max(xy_to_cct(*xy), 2000.0), 25000.0))


def white_and_weight(pr: Profile, neutral) -> tuple:
    """Iterate white xy <-> colour matrix (spec 6.2.4). neutral = camera RGB of white (1 / multipliers)."""
    n = np.asarray(neutral, float)
    xy = D50_XY
    g = 1.0
    for _ in range(30):
        g = _weight(pr, _cct(xy))
        cm = _mix(pr.color_matrices, g)
        XYZ = np.linalg.solve(cm, n)
        nxy = XYZ_to_xy(XYZ)
        if abs(nxy[0] - xy[0]) + abs(nxy[1] - xy[1]) < 1e-7:
            xy = nxy
            break
        xy = nxy
    return xy, g


def camera_to_working(pr: Profile, mult) -> tuple[np.ndarray, float]:
    """(3x3 white-balanced camera -> ProPhoto, weight of illuminant 1). ``mult`` = the edit's WB multipliers."""
    m = np.asarray(mult, float)
    neutral = 1.0 / m
    neutral = neutral / neutral.max()
    xy, g = white_and_weight(pr, neutral)
    to_pp = np.linalg.inv(PROPHOTO.to_xyz_d50)
    if pr.forward_matrices:
        fm = _mix(pr.forward_matrices, g)
        M = fm                                           # WB'd camera (neutral = 1,1,1) -> XYZ D50
    else:
        cm = _mix(pr.color_matrices, g)
        cam_to_xyz = np.linalg.inv(cm) @ np.diag(neutral)  # WB'd camera -> XYZ under the white
        M = bradford(xy_to_XYZ(xy), xy_to_XYZ(D50_XY)) @ cam_to_xyz
    w = M @ np.ones(3)
    M = M / w[1]                                         # neutral (1,1,1) -> D50 white with Y = 1
    return (to_pp @ M).astype(np.float32), g


def _rgb_to_hsv(rgb):
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-12), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dd = np.maximum(d, 1e-12)
    h = np.where(mx == r, (g - b) / dd, np.where(mx == g, 2.0 + (b - r) / dd, 4.0 + (r - g) / dd))
    h = np.where(d > 0, np.mod(h, 6.0), 0.0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    h = np.mod(h, 6.0)
    i = np.floor(h).astype(np.int32)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.empty(h.shape + (3,), np.float32)
    sel = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    for k, (a, b, c) in enumerate(sel):
        m = i == k
        out[..., 0][m], out[..., 1][m], out[..., 2][m] = a[m], b[m], c[m]
    return out


def apply_table(rgb: np.ndarray, tab: np.ndarray, srgb_val: bool) -> np.ndarray:
    """Apply a (val, hue, sat, 3) hue / saturation / value table to linear ProPhoto RGB (trilinear)."""
    nv, nh, ns, _ = tab.shape
    pos = np.maximum(rgb, 0.0)
    h, s, v = _rgb_to_hsv(pos)
    hf = h * (nh / 6.0)
    h0 = np.floor(hf).astype(np.int32) % nh
    h1 = (h0 + 1) % nh
    fh = (hf - np.floor(hf)).astype(np.float32)
    sf = np.clip(s, 0, 1) * (ns - 1)
    s0 = np.minimum(np.floor(sf).astype(np.int32), max(ns - 2, 0))
    s1 = np.minimum(s0 + 1, ns - 1)
    fs = (sf - s0).astype(np.float32)
    if nv > 1:
        vv = np.clip(v, 0, 1)
        if srgb_val:
            from .spaces import srgb_encode
            vv = srgb_encode(vv)
        vf = vv * (nv - 1)
        v0 = np.minimum(np.floor(vf).astype(np.int32), nv - 2)
        v1 = v0 + 1
        fv = (vf - v0).astype(np.float32)
    else:
        v0 = v1 = np.zeros(h.shape, np.int32)
        fv = np.zeros(h.shape, np.float32)

    def look(vi, hi, si):
        return tab[vi, hi, si]
    acc = np.zeros(h.shape + (3,), np.float32)
    for vi, wv in ((v0, 1 - fv), (v1, fv)):
        for hi, wh in ((h0, 1 - fh), (h1, fh)):
            for si, ws in ((s0, 1 - fs), (s1, fs)):
                acc += (wv * wh * ws)[..., None] * look(vi, hi, si)
    hue = h + acc[..., 0] / 60.0
    sat = np.clip(s * acc[..., 1], 0.0, 1.0)
    val = v * acc[..., 2]
    out = _hsv_to_rgb(hue, sat, val)
    neg = rgb.min(axis=-1) < 0                          # out-of-gamut colours pass through unchanged
    out[neg] = rgb[neg]
    return out


def apply_tables(rgb: np.ndarray, pr: Profile, g: float) -> np.ndarray:
    if pr.hsm:
        tab = pr.hsm[0] if len(pr.hsm) == 1 else (g * pr.hsm[0] + (1 - g) * pr.hsm[1]).astype(np.float32)
        rgb = apply_table(rgb, tab, pr.hsm_srgb)
    if pr.look is not None:
        rgb = apply_table(rgb, pr.look, pr.look_srgb)
    return rgb


# ------------------------------------------------------------------------------------------ finding profiles
def profile_dirs() -> list:
    return [("shared", str(paths.PROFILES / "camera_profiles")), ("yours", str(paths.get("SETTINGS") / "camera_profiles"))]


_CACHE: dict = {}


def list_profiles(camera: str | None = None) -> list:
    """[{"name", "file", "camera", "source"}] for this camera model (all if None)."""
    out = []
    for source, d in profile_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(EXT):
                continue
            try:
                pr = load(f)
            except Exception:
                slog.log_exception(f"camera profile {f} unreadable", log)
                continue
            if camera and pr.camera.strip().lower() != camera.strip().lower():
                continue
            out.append({"name": pr.name, "file": f, "camera": pr.camera, "source": source})
    return out


def find(name: str, camera=None) -> str | None:
    """A profile file by name. "lr:<name>" is Lightroom's profile of that name for ``camera`` ((make, model)) on
    this PC."""
    if name.startswith(LR_PREFIX):
        if not camera:
            return None
        want = name[len(LR_PREFIX):].strip().lower()
        for e in lightroom_profiles(*camera):
            if e["name"].strip().lower() == want:
                return e["path"]
        return None
    if os.path.isfile(name):
        return name
    for _, d in profile_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def load(name: str, camera=None) -> Profile:
    p = find(name, camera)
    if p is None:
        where = "Lightroom's profiles on this PC" if name.startswith(LR_PREFIX) else [d for _, d in profile_dirs()]
        raise FileNotFoundError(f"camera profile {name!r} not found in {where}")
    st = os.stat(p)
    key = (p, st.st_mtime, st.st_size)
    if key not in _CACHE:
        _CACHE[key] = read(p)
    return _CACHE[key]


def add_profile(src_path: str) -> str:
    """Copy a .dcp into settings/camera_profiles (checked first). Returns the file name to store in edits."""
    import shutil
    read(src_path)
    d = paths.get("SETTINGS") / "camera_profiles"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / os.path.basename(src_path)
    shutil.copy2(src_path, dst)
    return dst.name


# ------------------------------------------------------------------------------------------ Lightroom's profiles
LR_PREFIX = "lr:"
_LR_LOCK = __import__("threading").Lock()
_LR_INDEX: dict = {}                                    # dirs key -> [{"camera", "name", "path", "beo"}]
_MAKES = {"nikon corporation": "nikon", "nikon": "nikon", "sony": "sony", "canon": "canon", "pentax": "pentax",
          "pentax corporation": "pentax", "ricoh imaging company, ltd.": "pentax", "ricoh": "ricoh",
          "fujifilm": "fujifilm", "olympus imaging corp.": "olympus", "olympus corporation": "olympus",
          "om digital solutions": "om", "panasonic": "panasonic", "leica camera ag": "leica", "hasselblad": "hasselblad"}


def lightroom_dirs() -> list:
    """Where Adobe Camera Raw / Lightroom keep camera profiles on this PC (only those that exist). The environment
    variable STUDIO_LIGHTROOM_PROFILES (folders separated by os.pathsep) replaces the list (tests)."""
    env = os.environ.get("STUDIO_LIGHTROOM_PROFILES")
    if env is not None:
        cands = [d for d in env.split(os.pathsep) if d]
    else:
        cands = []
        for var in ("PROGRAMDATA", "APPDATA"):
            if os.environ.get(var):
                cands.append(os.path.join(os.environ[var], "Adobe", "CameraRaw", "CameraProfiles"))
        home = os.path.expanduser("~")
        cands += ["/Library/Application Support/Adobe/CameraRaw/CameraProfiles",
                  os.path.join(home, "Library", "Application Support", "Adobe", "CameraRaw", "CameraProfiles")]
    return [d for d in cands if os.path.isdir(d)]


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def camera_keys(make: str, model: str) -> set:
    """Ways Adobe may name this camera (Adobe writes "Sony ILCE-7M4" where the file says SONY / ILCE-7M4)."""
    make, model = str(make or "").strip(), str(model or "").strip()
    brand = _MAKES.get(make.lower(), make.split(" ")[0].lower() if make else "")
    keys = {_norm(model), _norm(make + model), _norm(brand + model)}
    return {k for k in keys if k}


def _header(path: str):
    """(camera, profile name, BaselineExposureOffset or None) read from a .dcp's first directory only, without
    reading its tables (Adobe's folders hold over a thousand profiles)."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8 or head[:2] not in (b"II", b"MM"):
            return None
        e = "<" if head[:2] == b"II" else ">"
        off = struct.unpack(e + "I", head[4:8])[0]
        fh.seek(off)
        n = struct.unpack(e + "H", fh.read(2))[0]
        ents = fh.read(12 * n)
        cam = name = None
        beo = None
        for k in range(n):
            tag, typ, cnt = struct.unpack(e + "HHI", ents[12 * k:12 * k + 8])
            val = ents[12 * k + 8:12 * k + 12]
            if tag in (T_MODEL, T_NAME) and typ == 2:
                if cnt <= 4:
                    raw = val[:cnt]
                else:
                    fh.seek(struct.unpack(e + "I", val)[0])
                    raw = fh.read(min(cnt, 256))
                txt = raw.split(b"\0")[0].decode("latin-1")
                if tag == T_MODEL:
                    cam = txt
                else:
                    name = txt
            elif tag == T_BEO and typ == 10:
                fh.seek(struct.unpack(e + "I", val)[0])
                a, b = struct.unpack(e + "ii", fh.read(8))
                beo = a / b if b else None
        if not cam:
            return None
        return cam, name or os.path.splitext(os.path.basename(path))[0], beo


def _dir_stamp(dirs) -> str:
    """Every folder under Adobe's profile folders with its modified time: changes when Adobe adds a camera."""
    parts = []
    for d in dirs:
        for root, sub, _ in os.walk(d):
            sub.sort()
            try:
                parts.append(f"{root}|{os.stat(root).st_mtime:.0f}")
            except OSError:
                pass
    import hashlib
    return hashlib.sha1("\n".join(parts).encode("utf-8", "replace")).hexdigest()


def _index_file():
    return paths.get("SETTINGS") / "lightroom_profiles.json"


def _lightroom_index() -> list:
    """Every Lightroom profile's camera, name and offset; kept in settings/lightroom_profiles.json beside the app so
    the scan of Adobe's folders (over a thousand files) happens once, and again only when Adobe adds profiles."""
    import json
    import time
    dirs = lightroom_dirs()
    with _LR_LOCK:
        if not dirs:
            return []
        stamp = _dir_stamp(dirs)
        if stamp in _LR_INDEX:
            return _LR_INDEX[stamp]
        try:
            saved = json.loads(_index_file().read_text(encoding="utf-8"))
            if saved.get("stamp") == stamp and isinstance(saved.get("profiles"), list):
                _LR_INDEX.clear()
                _LR_INDEX[stamp] = saved["profiles"]
                return saved["profiles"]
        except (OSError, ValueError):
            pass
        t0 = time.perf_counter()
        out = []
        for d in dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if not f.lower().endswith(EXT):
                        continue
                    p = os.path.join(root, f)
                    try:
                        h = _header(p)
                    except (OSError, struct.error, ValueError):
                        log.debug("Lightroom profile %s unreadable", p, exc_info=True)
                        continue
                    if h:
                        out.append({"camera": h[0], "name": h[1], "path": p, "beo": h[2], "key": _norm(h[0])})
        _LR_INDEX.clear()
        _LR_INDEX[stamp] = out
        log.info("Lightroom profiles: %d found in %s (%.2f s)", len(out), dirs, time.perf_counter() - t0)
        try:
            f = _index_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"stamp": stamp, "profiles": out}), encoding="utf-8")
        except OSError:
            log.debug("Lightroom profile index not saved", exc_info=True)
        return out


def warm_lightroom_index() -> None:
    """Build (or load) the index in the background at start, so the first photo opens without waiting for it."""
    import threading

    def run():
        try:
            _lightroom_index()
        except Exception:                               # noqa: BLE001  (a background nicety must never fail loudly)
            slog.log_exception("Lightroom profile index", log)
    threading.Thread(target=run, name="lr-profiles", daemon=True).start()


def lightroom_profiles(make: str, model: str) -> list:
    """Lightroom's profiles for this camera on this PC: [{"name", "file" ("lr:<name>"), "camera", "source",
    "path", "beo"}], Adobe Standard first, then the camera-matching ones. [] without Lightroom."""
    keys = camera_keys(make, model)
    seen, out = set(), []
    for e in _lightroom_index():
        if e["key"] in keys and e["name"].lower() not in seen:
            seen.add(e["name"].lower())
            out.append({"name": e["name"], "file": LR_PREFIX + e["name"], "camera": e["camera"],
                        "source": "lightroom", "path": e["path"], "beo": e["beo"]})
    out.sort(key=lambda e: (0 if e["name"].lower().startswith("adobe standard") else 1, e["name"].lower()))
    return out


def lightroom_baseline(make: str, model: str) -> float | None:
    """Lightroom's hidden brightness lift for this camera (stops), or None when it cannot be read on this PC.
    Adobe's camera-matching profiles carry a BaselineExposureOffset that cancels it exactly (seen on this project's
    PC: Nikon Z 8 -0.20 and Sony A7R V -0.35, against +0.20 and +0.35 in Adobe's DNGs of those cameras), so it is
    minus the commonest offset among them (HDR variants left out)."""
    vals = [e["beo"] for e in lightroom_profiles(make, model)
            if e["beo"] is not None and e["name"].lower().startswith("camera") and "hdr" not in e["name"].lower()]
    if not vals:
        return None
    vals = [round(v, 2) for v in vals]
    return -max(set(vals), key=vals.count)

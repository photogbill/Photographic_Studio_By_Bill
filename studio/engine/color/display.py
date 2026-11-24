"""The screen: what the Studio shows goes from its ProPhoto working space straight into the monitor's own colour
profile, as Lightroom and Photoshop do, never squeezed through sRGB on the way (Bill, 2026-11-23: what is seen must
be what is printed). A wide-gamut screen shows its whole range; a screen without a profile is what Windows itself
treats as sRGB, and the Studio says so (calibrate it) instead of pretending.

* Windows: the profile Windows assigns to the main display (GetICMProfileW), read where Windows keeps it; nothing is
  written anywhere. Elsewhere, or for tests, STUDIO_MONITOR_ICC names a profile file.
* Matrix/curve profiles (what calibrators make, and most display profiles) are applied exactly here: linear ProPhoto
  -> XYZ D50 -> the monitor's RGB -> its tone curves, inverted. Table-based profiles go through LittleCMS (Pillow's
  ImageCms) from an Adobe RGB intermediate.
* Soft proofing (soft_proof): the photo as a printer and paper would print it, shown on this screen, with what the
  paper cannot hold marked.
"""
from __future__ import annotations

import os
import struct
import threading

import numpy as np

from ... import log as slog

log = slog.get("engine.display")

_LOCK = threading.Lock()
_CURRENT: dict = {}


def monitor_profile_path() -> str | None:
    """The main display's colour profile file, or None when it has none (or this is not Windows)."""
    env = os.environ.get("STUDIO_MONITOR_ICC")
    if env is not None:
        return env if env and os.path.isfile(env) else None
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        gdi, user = ctypes.windll.gdi32, ctypes.windll.user32
        hdc = user.GetDC(0)
        try:
            n = wintypes.DWORD(0)
            gdi.GetICMProfileW(hdc, ctypes.byref(n), None)
            if not n.value:
                return None
            buf = ctypes.create_unicode_buffer(n.value)
            if not gdi.GetICMProfileW(hdc, ctypes.byref(n), buf):
                return None
            p = buf.value
        finally:
            user.ReleaseDC(0, hdc)
        return p if p and os.path.isfile(p) else None
    except Exception:                                   # noqa: BLE001  (no profile: the honest fallback below)
        log.debug("monitor profile not read", exc_info=True)
        return None


# ------------------------------------------------------------------------------------------ ICC matrix/curve profiles
def _tags(data: bytes) -> dict:
    n = struct.unpack(">I", data[128:132])[0]
    out = {}
    for k in range(n):
        sig, off, size = struct.unpack(">4sII", data[132 + 12 * k:144 + 12 * k])
        out[sig.decode("latin-1")] = data[off:off + size]
    return out


def _s15(b: bytes) -> float:
    return struct.unpack(">i", b)[0] / 65536.0


def _xyz(tag: bytes) -> np.ndarray:
    return np.array([_s15(tag[8 + 4 * i:12 + 4 * i]) for i in range(3)])


def _curve(tag: bytes):
    """The tone curve as a function (device value 0..1 -> linear 0..1) and its inverse."""
    kind = tag[:4]
    if kind == b"curv":
        n = struct.unpack(">I", tag[8:12])[0]
        if n == 0:
            return (lambda v: v), (lambda y: y)
        if n == 1:
            g = struct.unpack(">H", tag[12:14])[0] / 256.0
            return (lambda v: np.power(np.clip(v, 0, 1), g)), (lambda y: np.power(np.clip(y, 0, 1), 1.0 / g))
        t = np.array(struct.unpack(">" + "H" * n, tag[12:12 + 2 * n]), np.float64) / 65535.0
        x = np.linspace(0.0, 1.0, n)
        t = np.maximum.accumulate(t)
        return (lambda v: np.interp(v, x, t)), (lambda y: np.interp(y, t, x))
    if kind == b"para":
        ft = struct.unpack(">H", tag[8:10])[0]
        cnt = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ft]
        p = [_s15(tag[12 + 4 * i:16 + 4 * i]) for i in range(cnt)] + [0.0] * 7
        g, a, b, c, d, e, f = p[:7]
        xs = np.linspace(0.0, 1.0, 4097)
        if ft == 0:
            ys = xs ** g
        elif ft == 1:
            ys = np.where(xs >= -b / a, np.maximum(a * xs + b, 0) ** g, 0.0)
        elif ft == 2:
            ys = np.where(xs >= -b / a, np.maximum(a * xs + b, 0) ** g + c, c)
        elif ft == 3:
            ys = np.where(xs >= d, np.maximum(a * xs + b, 0) ** g, c * xs)
        else:
            ys = np.where(xs >= d, np.maximum(a * xs + b, 0) ** g + e, c * xs + f)
        ys = np.maximum.accumulate(np.clip(ys, 0, 1))
        return (lambda v: np.interp(v, xs, ys)), (lambda y: np.interp(y, ys, xs))
    raise ValueError("unsupported tone curve")


def describe(data: bytes) -> str:
    try:
        t = _tags(data).get("desc", b"")
        if t[:4] == b"desc":
            n = struct.unpack(">I", t[8:12])[0]
            return t[12:12 + n].split(b"\0")[0].decode("latin-1")
        if t[:4] == b"mluc":
            cnt = struct.unpack(">I", t[8:12])[0]
            if cnt:
                ln, off = struct.unpack(">II", t[20:28])
                return t[off:off + ln].decode("utf-16-be", "replace").strip("\0")
    except (struct.error, ValueError, IndexError):
        pass
    return "monitor profile"


class Display:
    """One way onto the screen: ``to8(linear ProPhoto)`` -> 8-bit RGB for this monitor."""

    def __init__(self, profile_path: str | None):
        from .spaces import PROPHOTO
        self.path, self.kind, self.name = profile_path, "srgb", "no monitor profile (Windows treats the screen as sRGB)"
        self._cms = None
        if not profile_path:
            return
        with open(profile_path, "rb") as fh:
            data = fh.read()
        self.name = describe(data)
        try:
            t = _tags(data)
            M = np.stack([_xyz(t["rXYZ"]), _xyz(t["gXYZ"]), _xyz(t["bXYZ"])], axis=1)   # monitor RGB -> XYZ D50
            self.M = (np.linalg.inv(M) @ PROPHOTO.to_xyz_d50).astype(np.float32)
            self.inv = [_curve(t[k])[1] for k in ("rTRC", "gTRC", "bTRC")]
            self.kind = "matrix"
        except (KeyError, ValueError, np.linalg.LinAlgError, struct.error):
            from PIL import ImageCms
            from . import icc
            src = ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc.profile_bytes("adobergb")))
            dst = ImageCms.ImageCmsProfile(profile_path)
            self._cms = ImageCms.buildTransform(src, dst, "RGB", "RGB",
                                                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
            self.kind = "table"
        log.info("screen: %s (%s profile, %s)", self.name, self.kind, profile_path)

    @property
    def managed(self) -> bool:
        return self.kind != "srgb"

    def to8(self, pp: np.ndarray) -> np.ndarray:
        from ..io.export import ExportSpec, to_output
        if _PROOF["path"]:
            try:
                return soft_proof(pp, self)
            except Exception:                           # noqa: BLE001  (a bad printer profile: say so, show normally)
                slog.log_exception("soft proof failed; showing the photo without it", log)
                set_proof(None)
        if self.kind == "srgb":
            return to_output(pp, ExportSpec(format="jpeg", space="srgb", bits=8))
        if self.kind == "table":
            from PIL import Image
            a = to_output(pp, ExportSpec(format="jpeg", space="adobergb", bits=8))
            return np.asarray(self._cms.apply(Image.fromarray(a)))
        from .spaces import apply_matrix
        from . import gamut
        rgb = apply_matrix(np.ascontiguousarray(pp, np.float32), self.M)
        rgb = np.clip(gamut.compress(rgb), 0.0, 1.0)
        out = np.empty(rgb.shape, np.uint8)
        for c in range(3):
            out[..., c] = np.clip(self.inv[c](rgb[..., c]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return out


def current() -> Display:
    """The screen the Studio is on (read once; again after reload())."""
    with _LOCK:
        if "d" not in _CURRENT:
            try:
                _CURRENT["d"] = Display(monitor_profile_path())
            except Exception as exc:                     # noqa: BLE001  (a broken profile: say so, show as Windows does)
                slog.log_exception("monitor profile unusable", log)
                d = Display(None)
                d.name = f"monitor profile unreadable ({exc}); shown as Windows treats it (sRGB)"
                _CURRENT["d"] = d
        return _CURRENT["d"]


def reload() -> Display:
    with _LOCK:
        _CURRENT.clear()
    return current()


# ------------------------------------------------------------------------------------------ soft proofing
_PROOF: dict = {"path": None, "intent": "perceptual", "gamut": False, "cache": None}
INTENTS = {"perceptual": 0, "relative": 1}


def set_proof(path: str | None, intent: str = "perceptual", gamut: bool = False) -> None:
    """Show every preview as ``path`` (a printer / paper ICC profile) would print it; None turns it off."""
    with _LOCK:
        _PROOF.update(path=path or None, intent=intent if intent in INTENTS else "perceptual", gamut=bool(gamut),
                      cache=None)
    log.info("soft proof: %s (%s%s)", path or "off", intent, ", gamut warning" if gamut else "")


def proof_state() -> dict:
    return {k: _PROOF[k] for k in ("path", "intent", "gamut")}


def _proof_transforms(d: "Display"):
    import io
    from PIL import ImageCms
    from . import icc
    key = (_PROOF["path"], _PROOF["intent"], d.path)
    c = _PROOF.get("cache")
    if c and c[0] == key:
        return c[1]
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc.profile_bytes("adobergb")))
    prn = ImageCms.ImageCmsProfile(_PROOF["path"])
    scr = ImageCms.ImageCmsProfile(d.path) if d.path else ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    intent = INTENTS[_PROOF["intent"]]
    flags = ImageCms.Flags.SOFTPROOFING | (ImageCms.Flags.BLACKPOINTCOMPENSATION if intent == 1 else 0)
    view = ImageCms.buildProofTransform(src, scr, prn, "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                        proofRenderingIntent=intent, flags=flags)
    mode = "CMYK" if prn.profile.xcolor_space.strip() == "CMYK" else "RGB"
    to_p = ImageCms.buildTransform(src, prn, "RGB", mode, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    back = ImageCms.buildTransform(prn, src, mode, "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    out = (view, to_p, back)
    _PROOF["cache"] = (key, out)
    return out


def soft_proof(pp: np.ndarray, d: "Display" | None = None) -> np.ndarray:
    """The photo as the chosen printer and paper print it, on this screen (8-bit RGB). With the gamut warning on,
    colours the paper cannot hold (more than 6 units off after printing, in OKLab x 100) show as flat magenta."""
    from PIL import Image
    from ..io.export import ExportSpec, to_output
    d = d or current()
    a = to_output(pp, ExportSpec(format="jpeg", space="adobergb", bits=8))
    view, to_p, back = _proof_transforms(d)
    shown = np.asarray(view.apply(Image.fromarray(a))).copy()
    if _PROOF["gamut"]:
        rt = np.asarray(back.apply(to_p.apply(Image.fromarray(a)))).astype(np.float32)
        from .grade import to_oklab
        from .spaces import SPACES, PROPHOTO, apply_matrix, conversion_matrix

        def lab(u):
            v = u.astype(np.float32) / 255.0
            v = np.power(v, 563.0 / 256.0)                # Adobe RGB's tone curve
            return to_oklab(apply_matrix(v, conversion_matrix(SPACES["adobergb"], PROPHOTO)))
        dE = np.linalg.norm(lab(a) - lab(rt), axis=-1) * 100.0
        shown[dE > 6.0] = (255, 0, 255)
    return shown

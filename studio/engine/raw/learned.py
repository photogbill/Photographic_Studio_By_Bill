"""Running a camera's own trained denoising network (BUILD_PLAN_NOISE F.1).

The network is trained on the user's PC (studio/lab) from clean photos plus that camera's real noise, and saved as
an ONNX file in calibration_local/<camera>/models/. It works on the four colour planes after the variance-
stabilising transform, where the noise has strength one at every ISO, so one network serves them all.

ONNX Runtime is optional (plan 23.2): on the graphics card through DirectML when that build is installed, else
CUDA, else the processor. With no runtime or no model, ``for_camera`` returns None and Clean uses Hybrid. The
result always passes the detail guard and the noise envelope afterwards (rule: nothing may be invented).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import threading

import numpy as np

from ... import log as slog, paths

log = slog.get("engine.learned")
TILE, OVERLAP, MULTIPLE = 384, 32, 8
_LOCK = threading.Lock()
_NETS: dict = {}


def _add_lab_packages() -> None:
    """The optional packages (ONNX Runtime, and PyTorch for training) live in <app>/lab_packages, apart from the
    Studio's own, so they can be removed by deleting one folder."""
    d = str(paths.APP_ROOT / "lab_packages")
    if os.path.isdir(d) and d not in sys.path:
        sys.path.append(d)


def runtime():
    """The onnxruntime module, or None when it is not installed."""
    _add_lab_packages()
    try:
        import onnxruntime as ort
        return ort
    except Exception:                                       # noqa: BLE001  (missing, or a broken install)
        return None


def model_dir(make: str, model: str) -> str:
    from ..calib.profile import local_data_dir
    d = os.path.join(local_data_dir(make, model), "models")
    os.makedirs(d, exist_ok=True)
    return d


def find_model(make: str, model: str) -> str | None:
    files = sorted(glob.glob(os.path.join(model_dir(make, model), "*.onnx")), key=os.path.getmtime)
    return files[-1] if files else None


def card(path: str) -> dict:
    try:
        return json.load(open(os.path.splitext(path)[0] + ".json", encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class Net:
    def __init__(self, path: str, session):
        self.path, self.session = path, session
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.input = session.get_inputs()[0].name
        self.providers = session.get_providers()

    def _one(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: x[None].astype(np.float32)})[0][0]

    def run(self, planes: np.ndarray) -> np.ndarray:
        """(4, h, w) unit-noise planes in, the same cleaned. Worked in overlapping tiles blended with a soft
        edge, so 45 MP fits in little memory and no tile border shows."""
        c, h, w = planes.shape
        out = np.zeros((c, h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        step = TILE - 2 * OVERLAP
        ramp = np.clip((np.arange(TILE, dtype=np.float32) + 0.5) / OVERLAP, 0.0, 1.0)
        win1 = np.minimum(ramp, ramp[::-1])
        for y0 in range(0, max(h - 2 * OVERLAP, 1), step):
            for x0 in range(0, max(w - 2 * OVERLAP, 1), step):
                y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                t = planes[:, y0:y1, x0:x1]
                ph, pw = (-t.shape[1]) % MULTIPLE, (-t.shape[2]) % MULTIPLE
                if ph or pw:
                    t = np.pad(t, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(t.shape[1:]) > MULTIPLE
                               else "edge")
                r = self._one(t)[:, :y1 - y0, :x1 - x0]
                wy, wx = win1[:y1 - y0].copy(), win1[:x1 - x0].copy()
                if y0 == 0:
                    wy[:OVERLAP] = 1.0
                if x0 == 0:
                    wx[:OVERLAP] = 1.0
                if y1 == h:
                    wy[-OVERLAP:] = 1.0
                if x1 == w:
                    wx[-OVERLAP:] = 1.0
                wgt = np.maximum(wy[:, None] * wx[None, :], 1e-4)
                out[:, y0:y1, x0:x1] += r * wgt
                wsum[y0:y1, x0:x1] += wgt
        return out / np.maximum(wsum, 1e-6)


def load(path: str) -> Net | None:
    ort = runtime()
    if ort is None:
        return None
    have = ort.get_available_providers()
    order = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in have]
    for prov in order:                                       # a graphics driver can refuse: fall through to the CPU
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            net = Net(path, ort.InferenceSession(path, sess_options=so, providers=[prov]))
            log.info("learned denoiser %s loaded on %s", net.name, prov)
            return net
        except Exception:                                    # noqa: BLE001
            slog.log_exception(f"learned denoiser could not start on {prov}", log)
    return None


def for_camera(camera: tuple | None) -> Net | None:
    """This camera's newest network, loaded once and kept; None when there is none or no runtime."""
    if not camera or not camera[1]:
        return None
    path = find_model(camera[0] or "", camera[1])
    if path is None:
        return None
    key = (path, os.path.getmtime(path))
    with _LOCK:
        if key not in _NETS:
            _NETS.clear()
            _NETS[key] = load(path)
        return _NETS[key]


def status(make: str, model: str) -> dict:
    """For the Cameras page: is there a runtime, is there a model, what does its card say."""
    path = find_model(make, model)
    ort = runtime()
    return dict(runtime=ort is not None, providers=ort.get_available_providers() if ort else [],
                model=path, card=card(path) if path else {})

"""Convert Microsoft's scratch detector (Bringing Old Photos Back to Life, MIT licence) for the Studio.

    python tools/convert_scratch_detector.py global_checkpoints.zip   (or FT_Epoch_latest.pt)

Reads checkpoints/detection/FT_Epoch_latest.pt without PyTorch (the legacy torch.save format is a pickle followed
by the raw tensor storages), keeps only the model's weights (not the optimiser's), folds batch normalisation into
each convolution and writes models/scratches/scratch_detector.npz (8-bit weights, one scale per filter). See studio/engine/local/scratchnet.py.
"""
from __future__ import annotations

import collections
import io
import os
import pickle
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from studio.engine.local import scratchnet  # noqa: E402

MEMBER = "checkpoints/detection/FT_Epoch_latest.pt"
EL = {"FloatStorage": ("<f4", 4), "LongStorage": ("<i8", 8), "HalfStorage": ("<f2", 2), "IntStorage": ("<i4", 4)}


class _Stor:
    def __init__(self, *a):
        self.a = a


class _T:
    def __init__(self, st, off, size, stride, *rest):
        self.st, self.off, self.size, self.stride = st, off, tuple(size), stride


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _T
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if "Storage" in name:
            return name
        return lambda *a, **k: (mod, name, a)

    def persistent_load(self, pid):
        return _Stor(*pid)


def state_dict(data: bytes) -> dict:
    s = io.BytesIO(data)
    pickle.load(s), pickle.load(s), pickle.load(s)          # magic number, protocol, system info
    obj = _U(s).load()
    keys = pickle.load(s)
    sizes = {}

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, _T):
            sizes[o.st.a[2]] = (o.st.a[1], o.st.a[4])
    walk(obj)
    offs, off = {}, s.tell()
    for k in keys:
        offs[k] = off + 8
        typ, n = sizes[k]
        off += 8 + n * EL[typ][1]
    out = {}
    for name, t in obj["model_state"].items():
        typ, n = sizes[t.st.a[2]]
        if typ != "FloatStorage":
            continue
        a = np.frombuffer(data, EL[typ][0], n, offs[t.st.a[2]])
        cnt = int(np.prod(t.size)) if t.size else 1
        out[name] = a[t.off:t.off + cnt].reshape(t.size).astype(np.float32)
    return out


def main(src: str) -> None:
    if src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            data = z.read(MEMBER)
    else:
        with open(src, "rb") as fh:
            data = fh.read()
    w = scratchnet.convert(state_dict(data))
    dst = scratchnet.model_path()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, **w)
    print(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB, {sum(v.size for v in w.values()) / 1e6:.1f} M weights)")


if __name__ == "__main__":
    main(sys.argv[1])

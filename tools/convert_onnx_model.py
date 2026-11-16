"""Convert an ONNX model for the Studio's NumPy runner (studio/engine/local/onnxnp.py).

    python tools/convert_onnx_model.py ddcolor-onnx-float.zip models/colour/ddcolor.npz

Needs the `onnx` package on the PC doing the conversion (pip install onnx); the Studio itself never does. Reads the
model (and its external weights file, from a zip or a folder), writes one .npz: the graph as JSON and the weights,
those of 4096 values or more as 8-bit integers with one scale per output channel (a quarter of the size), the rest
as float32. Each node lists the activations no later node reads, so the runner can free them as it goes.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile

import numpy as np

QUANT_MIN = 4096


def _attrs(node) -> dict:
    import onnx
    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, (list, tuple)):
            v = [x.decode("utf-8") if isinstance(x, bytes) else (float(x) if isinstance(x, float) else int(x))
                 for x in v]
        elif isinstance(v, float):
            v = float(v)
        elif isinstance(v, int):
            v = int(v)
        elif isinstance(v, onnx.TensorProto):            # e.g. ConstantOfShape's fill value
            arr = onnx.numpy_helper.to_array(v)
            out[a.name + "_arr"] = float(arr.ravel()[0]) if arr.size == 1 else arr.tolist()
            continue
        else:
            raise ValueError(f"{node.op_type}: attribute {a.name} of an unsupported kind")
        out[a.name] = v
    return out


def convert(onnx_path: str, out_path: str, f16: bool = False) -> dict:
    """``f16``: keep big weights as 16-bit floats (half the size, near-exact) instead of 8-bit integers (a quarter
    of the size); for models whose answers move too much with 8 bits (SAM's image encoder)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path, load_external_data=False)
    base = os.path.dirname(onnx_path)
    g = m.graph
    weights = {i.name: numpy_helper.to_array(i, base_dir=base) for i in g.initializer}
    for n in g.node:                                      # Constant nodes become weights too
        if n.op_type == "Constant":
            weights[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    nodes = [n for n in g.node if n.op_type != "Constant"]
    # where each weight is used decides its per-channel axis
    qaxis = {}
    for n in nodes:
        a = _attrs(n)
        if n.op_type == "Conv" and len(n.input) > 1:
            qaxis[n.input[1]] = 0
        elif n.op_type == "Gemm" and len(n.input) > 1:
            qaxis[n.input[1]] = 0 if a.get("transB", 0) else 1
        elif n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] in weights:
            qaxis[n.input[1]] = weights[n.input[1]].ndim - 1
    last = {}
    for i, n in enumerate(nodes):
        for name in n.input:
            if name and name not in weights:
                last[name] = i
    outs = {o.name for o in g.output}
    jnodes = []
    for i, n in enumerate(nodes):
        free = [name for name in set(n.input) if name and last.get(name) == i and name not in outs]
        jnodes.append({"op": n.op_type, "in": list(n.input), "out": list(n.output), "attr": _attrs(n), "free": free})
    inits = {i.name for i in g.initializer}
    ins = [i for i in g.input if i.name not in inits]
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in ins}   # 0 = any size
    graph = {"nodes": jnodes, "inputs": [i.name for i in ins], "outputs": [o.name for o in g.output],
             "input_shape": shapes[ins[0].name], "input_shapes": shapes, "qaxis": {}}
    arrays = {}
    for name, w in weights.items():
        w = np.asarray(w)
        if f16 and w.size >= QUANT_MIN and w.dtype == np.float32:
            arrays[name] = w.astype(np.float16)
        elif name in qaxis and w.size >= QUANT_MIN and w.dtype == np.float32:
            ax = qaxis[name]
            red = tuple(k for k in range(w.ndim) if k != ax)
            s = np.abs(w).max(axis=red) / 127.0 + 1e-12
            shp = [1] * w.ndim
            shp[ax] = -1
            arrays[name + "::q"] = np.clip(np.round(w / s.reshape(shp)), -127, 127).astype(np.int8)
            arrays[name + "::s"] = s.astype(np.float32)
            graph["qaxis"][name] = ax
        else:
            arrays[name] = w.astype(np.float32) if w.dtype == np.float64 else w
    arrays["__graph__"] = np.frombuffer(json.dumps(graph).encode("utf-8"), np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return graph


def fold(src: str, dst: str, shapes: dict) -> str:
    """Fold a model to fixed input shapes (onnxsim: shape bookkeeping, branches and constant parts become plain
    weights), so the runner sees only real layers. Needs `pip install onnxsim` on the converting PC."""
    import onnx
    try:
        import onnxsim
    except ImportError:                                   # onnxslim: pure Python, does the same folding
        import onnxslim
        onnx.save(onnxslim.slim(src, input_shapes=[f"{k}:{','.join(map(str, v))}" for k, v in shapes.items()]), dst)
        return dst
    m = onnx.load(src)
    ms, ok = onnxsim.simplify(m, overwrite_input_shapes=shapes)
    if not ok:
        raise RuntimeError("the folded model does not match the original")
    onnx.save(ms, dst)
    return dst


def main(src: str, dst: str, f16: bool = False) -> None:
    if src.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(src) as z:
                z.extractall(td)
            onnx_files = [os.path.join(r, f) for r, _, fs in os.walk(td) for f in fs if f.endswith(".onnx")]
            g = convert(onnx_files[0], dst, f16)
    else:
        g = convert(src, dst, f16)
    kind = "16-bit floats" if f16 else "8 bits"
    print(f"wrote {dst}: {len(g['nodes'])} layers, {len(g['qaxis']) if not f16 else 'big'} weights in {kind}, "
          f"{os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    # python tools/convert_onnx_model.py SRC.onnx DST.npz [--f16] [--fold name=1,3,518,518 [name=...]]
    args = sys.argv[1:]
    f16 = "--f16" in args
    args = [a for a in args if a != "--f16"]
    if len(args) > 2 and args[2] == "--fold":
        shapes = {kv.split("=")[0]: [int(v) for v in kv.split("=")[1].split(",")] for kv in args[3:]}
        with tempfile.TemporaryDirectory() as td:
            main(fold(args[0], os.path.join(td, "folded.onnx"), shapes), args[1], f16)
    else:
        main(args[0], args[1], f16)

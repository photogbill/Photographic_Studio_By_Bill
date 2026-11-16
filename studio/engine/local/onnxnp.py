"""A very small runner for converted ONNX graphs, in NumPy (no onnxruntime, nothing to install).

tools/convert_onnx_model.py turns an ONNX model (with the `onnx` package, on a developer's PC) into one .npz file:
the graph as JSON (nodes in order, their attributes, input and output names) and the weights, large ones as 8-bit
integers with one scale per output channel. This file runs it: the layer types of DDColor (Kang et al. 2023) as
Qualcomm exports it, in ONNX's own NCHW layout, float32. Anything else is refused with a clear message, so a
different model never runs half-understood.
"""
from __future__ import annotations

import json

import numpy as np

OPS = ("Add", "AveragePool", "BatchNormalization", "Concat", "Conv", "DepthToSpace", "Div", "Gelu", "Gemm",
       "LayerNormalization", "MatMul", "Mul", "Pad", "Relu", "Reshape", "Softmax", "Split", "Sub", "Transpose",
       "Sigmoid", "Tanh", "Sqrt", "Pow", "Erf", "Identity", "Flatten",
       # depth and click-to-select, after folding to a fixed input
       "ConvTranspose", "Resize", "MaxPool", "ReduceMean", "Gather", "Slice", "Squeeze", "Unsqueeze", "Expand",
       "Tile", "Cast", "Where", "Equal", "Greater", "Less", "Not", "Neg", "Sin", "Cos", "Exp", "Clip", "Mod",
       "ReduceSum", "ReduceMax", "Abs", "Floor", "Max", "Min", "ScatterND", "OneHot", "ConstantOfShape", "Range",
       "Shape")


class GraphError(ValueError):
    pass


class Graph:
    """A converted model: ``Graph.load(path)(x)`` with x float32 shaped as the model's input -> {name: array}."""

    def __init__(self, graph: dict, weights: dict):
        self.nodes = graph["nodes"]
        self.inputs = graph["inputs"]
        self.outputs = graph["outputs"]
        self.input_shape = tuple(graph["input_shape"])
        self.input_shapes = {k: tuple(v) for k, v in (graph.get("input_shapes") or
                                                     {self.inputs[0]: graph["input_shape"]}).items()}
        self.w = weights
        bad = sorted({n["op"] for n in self.nodes} - set(OPS))
        if bad:
            raise GraphError(f"layer types not supported by this runner: {', '.join(bad)}")

    @classmethod
    def load(cls, path: str) -> "Graph":
        with np.load(path) as z:
            graph = json.loads(bytes(z["__graph__"]).decode("utf-8"))
            w = {}
            for k in z.files:
                if k == "__graph__" or k.endswith("::s"):
                    continue
                if k.endswith("::q"):
                    name = k[:-3]
                    q, s = z[k], z[name + "::s"]
                    ax = graph["qaxis"][name]
                    shp = [1] * q.ndim
                    shp[ax] = -1
                    w[name] = q.astype(np.float32) * s.reshape(shp)
                else:
                    a = z[k]
                    w[k] = a.astype(np.float32) if a.dtype == np.float16 else a
        return cls(graph, w)

    def __call__(self, x: np.ndarray | None = None, **named) -> dict:
        """Run the model: ``g(x)`` for a one-input model, or ``g(name=array, ...)`` for every input (0 in a model's
        input shape means any size there)."""
        if x is not None:
            named[self.inputs[0]] = x
        v = dict(self.w)
        for name in self.inputs:
            if name not in named:
                raise GraphError(f"the model needs the input {name}")
            a = np.asarray(named[name], np.float32)
            want = self.input_shapes.get(name, ())
            if len(want) != a.ndim or any(wd and wd != ad for wd, ad in zip(want, a.shape)):
                raise GraphError(f"input {name} shape {a.shape}, the model wants {want}")
            v[name] = a
        for n in self.nodes:
            args = [v[i] if i else None for i in n["in"]]
            r = _run(n["op"], args, n.get("attr", {}))
            if isinstance(r, tuple):
                for name, val in zip(n["out"], r):
                    v[name] = val
            else:
                v[n["out"][0]] = r
            for name in n.get("free", ()):                # drop activations no later layer reads
                v.pop(name, None)
        return {k: v[k] for k in self.outputs}


# ------------------------------------------------------------------------------------------ layers (NCHW)
def _conv(x, w, b, a):
    """2-D convolution, NCHW: one matrix product per kernel tap (groups: depthwise directly, else per group)."""
    s = a.get("strides", [1, 1])
    p = a.get("pads", [0, 0, 0, 0])
    d = a.get("dilations", [1, 1])
    g = a.get("group", 1)
    if d != [1, 1]:
        raise GraphError("dilated convolution is not supported")
    n, c, h, wd = x.shape
    co, ci, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p[0], p[2]), (p[1], p[3])))
    oh = (h + p[0] + p[2] - kh) // s[0] + 1
    ow = (wd + p[1] + p[3] - kw) // s[1] + 1
    out = np.zeros((n, co, oh, ow), np.float32)
    if g == c and ci == 1 and co == c:                   # depthwise
        for i in range(kh):
            for j in range(kw):
                out += xp[:, :, i:i + s[0] * (oh - 1) + 1:s[0], j:j + s[1] * (ow - 1) + 1:s[1]] * \
                    w[None, :, 0, i, j, None, None]
    else:
        cg, og = c // g, co // g
        for gi in range(g):
            xs = xp[:, gi * cg:(gi + 1) * cg]
            ws = w[gi * og:(gi + 1) * og]
            # im2col over the taps: (n, cg*kh*kw, oh*ow), one matrix product
            cols = np.stack([xs[:, :, i:i + s[0] * (oh - 1) + 1:s[0], j:j + s[1] * (ow - 1) + 1:s[1]]
                             for i in range(kh) for j in range(kw)], axis=2)        # (n, cg, taps, oh, ow)
            cols = cols.reshape(n, cg * kh * kw, oh * ow)
            wm = ws.reshape(og, cg * kh * kw)
            out[:, gi * og:(gi + 1) * og] = (wm @ cols).reshape(n, og, oh, ow)
    if b is not None:
        out += b[None, :, None, None]
    return out


def _avgpool(x, a):
    k = a.get("kernel_shape")
    s = a.get("strides", [1, 1])
    p = a.get("pads", [0, 0, 0, 0])
    if any(p) or a.get("ceil_mode", 0):
        raise GraphError("padded average pooling is not supported")
    n, c, h, w = x.shape
    oh, ow = (h - k[0]) // s[0] + 1, (w - k[1]) // s[1] + 1
    acc = np.zeros((n, c, oh, ow), np.float32)
    for i in range(k[0]):
        for j in range(k[1]):
            acc += x[:, :, i:i + s[0] * (oh - 1) + 1:s[0], j:j + s[1] * (ow - 1) + 1:s[1]]
    return acc / float(k[0] * k[1])


def _layernorm(x, scale, bias, a):
    ax = a.get("axis", -1)
    ax = ax if ax >= 0 else x.ndim + ax
    axes = tuple(range(ax, x.ndim))
    mu = x.mean(axis=axes, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=axes, keepdims=True)
    y = (x - mu) / np.sqrt(var + a.get("epsilon", 1e-5))
    y = y * scale
    if bias is not None:
        y = y + bias
    return y.astype(np.float32)


def _d2s(x, a):
    b = a.get("blocksize")
    n, c, h, w = x.shape
    if a.get("mode", "DCR") == "CRD":
        y = x.reshape(n, c // (b * b), b, b, h, w).transpose(0, 1, 4, 2, 5, 3)
    else:
        y = x.reshape(n, b, b, c // (b * b), h, w).transpose(0, 3, 4, 1, 5, 2)
    return y.reshape(n, c // (b * b), h * b, w * b)


def _run(op, x, a):
    if op == "Conv":
        return _conv(x[0], x[1], x[2] if len(x) > 2 else None, a)
    if op == "Gemm":
        A, B = x[0], x[1]
        if a.get("transA", 0):
            A = A.T
        if a.get("transB", 0):
            B = B.T
        y = a.get("alpha", 1.0) * (A @ B)
        if len(x) > 2 and x[2] is not None:
            y = y + a.get("beta", 1.0) * x[2]
        return y.astype(np.float32)
    if op == "MatMul":
        return np.matmul(x[0], x[1]).astype(np.float32)
    if op == "Add":
        return (x[0] + x[1]).astype(np.float32)
    if op == "Sub":
        return (x[0] - x[1]).astype(np.float32)
    if op == "Mul":
        return (x[0] * x[1]).astype(np.float32)
    if op == "Div":
        return (x[0] / x[1]).astype(np.float32)
    if op == "Pow":
        return np.power(x[0], x[1]).astype(np.float32)
    if op == "Sqrt":
        return np.sqrt(x[0])
    if op == "Relu":
        return np.maximum(x[0], 0.0)
    if op == "Sigmoid":
        return (1.0 / (1.0 + np.exp(-x[0]))).astype(np.float32)
    if op == "Tanh":
        return np.tanh(x[0])
    if op in ("Gelu", "Erf"):
        from scipy.special import erf
        if op == "Erf":
            return erf(x[0]).astype(np.float32)
        if a.get("approximate", "none") == "tanh":
            t = x[0]
            return (0.5 * t * (1.0 + np.tanh(0.7978845608 * (t + 0.044715 * t ** 3)))).astype(np.float32)
        return (0.5 * x[0] * (1.0 + erf(x[0] / np.sqrt(2.0)))).astype(np.float32)
    if op == "Softmax":
        ax = a.get("axis", -1)
        z = np.exp(x[0] - x[0].max(axis=ax, keepdims=True))
        return (z / z.sum(axis=ax, keepdims=True)).astype(np.float32)
    if op == "LayerNormalization":
        return _layernorm(x[0], x[1], x[2] if len(x) > 2 else None, a)
    if op == "BatchNormalization":
        X, sc, b, m, var = x[:5]
        k = sc / np.sqrt(var + a.get("epsilon", 1e-5))
        return (X * k[None, :, None, None] + (b - m * k)[None, :, None, None]).astype(np.float32)
    if op == "Transpose":
        return np.ascontiguousarray(np.transpose(x[0], a.get("perm")))
    if op == "Reshape":
        shape = [int(v) for v in np.asarray(x[1]).ravel()]
        if not a.get("allowzero", 0):
            shape = [x[0].shape[i] if v == 0 else v for i, v in enumerate(shape)]
        return x[0].reshape(shape)
    if op == "Flatten":
        ax = a.get("axis", 1)
        return x[0].reshape(int(np.prod(x[0].shape[:ax])), -1)
    if op == "Concat":
        return np.concatenate(x, axis=a.get("axis", 0))
    if op == "Split":
        ax = a.get("axis", 0)
        if len(x) > 1 and x[1] is not None:
            sizes = [int(v) for v in np.asarray(x[1]).ravel()]
            return tuple(np.split(x[0], np.cumsum(sizes)[:-1], axis=ax))
        return tuple(np.array_split(x[0], a.get("num_outputs"), axis=ax))
    if op == "Pad":
        pads = [int(v) for v in np.asarray(x[1]).ravel()]
        half = len(pads) // 2
        pw = [(pads[i], pads[i + half]) for i in range(half)]
        mode = a.get("mode", "constant")
        if mode == "constant":
            cv = float(np.asarray(x[2]).ravel()[0]) if len(x) > 2 and x[2] is not None else 0.0
            return np.pad(x[0], pw, constant_values=cv)
        return np.pad(x[0], pw, mode={"edge": "edge", "reflect": "reflect"}[mode])
    if op == "AveragePool":
        return _avgpool(x[0], a)
    if op == "DepthToSpace":
        return _d2s(x[0], a)
    if op == "Identity":
        return x[0]
    r = _run2(op, x, a)
    if r is not None:
        return r
    raise GraphError(f"layer type {op} is not supported")


# ------------------------------------------------------------------------------------------ more layers
# Depth Anything V2 (Yang et al. 2024) and Segment Anything 2.1 (Ravi et al. 2024), after their graphs are folded
# to a fixed input size (tools/convert_onnx_model.py --fold): shape bookkeeping becomes constants, these remain.
def _axes(x, a, i):
    """Axes from the attribute (old opsets) or from input i (new opsets), or None."""
    if len(x) > i and x[i] is not None:
        return [int(v) for v in np.asarray(x[i]).ravel()]
    ax = a.get("axes")
    return list(ax) if ax is not None else None


def _convT(x, w, b, a):
    """Transposed 2-D convolution (groups 1), NCHW: each kernel tap adds its contribution at stride spacing."""
    s = a.get("strides", [1, 1])
    p = a.get("pads", [0, 0, 0, 0])
    op_ = a.get("output_padding", [0, 0])
    if a.get("group", 1) != 1 or a.get("dilations", [1, 1]) != [1, 1]:
        raise GraphError("grouped or dilated transposed convolution is not supported")
    n, ci, h, wd = x.shape
    _, co, kh, kw = w.shape
    fh, fw = (h - 1) * s[0] + kh + op_[0], (wd - 1) * s[1] + kw + op_[1]
    full = np.zeros((n, co, fh, fw), np.float32)
    xm = x.reshape(n, ci, h * wd)
    for i in range(kh):
        for j in range(kw):
            t = np.einsum("nci,co->noi", xm, w[:, :, i, j], optimize=True).reshape(n, co, h, wd)
            full[:, :, i:i + s[0] * (h - 1) + 1:s[0], j:j + s[1] * (wd - 1) + 1:s[1]] += t
    out = full[:, :, p[0]:fh - p[2], p[1]:fw - p[3]]
    if b is not None:
        out = out + b[None, :, None, None]
    return np.ascontiguousarray(out, np.float32)


def _src_coords(n_out, n_in, scale, mode):
    o = np.arange(n_out, dtype=np.float64)
    if mode == "align_corners":
        return o * (n_in - 1) / max(n_out - 1, 1)
    if mode == "asymmetric":
        return o / scale
    if mode == "pytorch_half_pixel" and n_out == 1:
        return np.zeros(1)
    return (o + 0.5) / scale - 0.5                      # half_pixel, pytorch_half_pixel


def _interp_axis(x, axis, n_out, scale, a):
    """Resample one axis: linear, cubic (Keys, coefficient a) or nearest, ONNX coordinate conventions."""
    n_in = x.shape[axis]
    mode = a.get("mode", "nearest")
    ctm = a.get("coordinate_transformation_mode", "half_pixel")
    src = _src_coords(n_out, n_in, scale, ctm)
    xm = np.moveaxis(x, axis, -1)
    if mode == "nearest":
        nm = a.get("nearest_mode", "round_prefer_floor")
        if nm == "floor":
            idx = np.floor(src)
        elif nm == "ceil":
            idx = np.ceil(src)
        elif nm == "round_prefer_ceil":
            idx = np.floor(src + 0.5)
        else:
            idx = np.ceil(src - 0.5)
        y = xm[..., np.clip(idx.astype(np.int64), 0, n_in - 1)]
    elif mode == "linear":
        src = np.clip(src, 0, n_in - 1)
        i0 = np.floor(src).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        t = (src - i0).astype(np.float32)
        y = xm[..., i0] * (1 - t) + xm[..., i1] * t
    elif mode == "cubic":
        ca = float(a.get("cubic_coeff_a", -0.75))
        i0 = np.floor(src).astype(np.int64)
        t = src - i0
        def k(d):
            d = np.abs(d)
            return np.where(d <= 1, (ca + 2) * d ** 3 - (ca + 3) * d ** 2 + 1,
                            np.where(d < 2, ca * d ** 3 - 5 * ca * d ** 2 + 8 * ca * d - 4 * ca, 0.0))
        y = 0
        for off in (-1, 0, 1, 2):
            wgt = k(t - off).astype(np.float32)
            y = y + xm[..., np.clip(i0 + off, 0, n_in - 1)] * wgt
    else:
        raise GraphError(f"resize mode {mode} is not supported")
    return np.moveaxis(np.asarray(y, np.float32), -1, axis)


def _resize(x, a):
    X = x[0]
    scales = np.asarray(x[2]).ravel() if len(x) > 2 and x[2] is not None and np.asarray(x[2]).size else None
    sizes = np.asarray(x[3]).ravel().astype(np.int64) if len(x) > 3 and x[3] is not None else None
    if sizes is None:
        sizes = np.floor(np.asarray(X.shape) * scales).astype(np.int64)
    if scales is None:
        scales = sizes / np.asarray(X.shape, np.float64)
    y = X
    for ax in range(X.ndim):
        if int(sizes[ax]) != X.shape[ax] or abs(float(scales[ax]) - 1.0) > 1e-9:
            y = _interp_axis(y, ax, int(sizes[ax]), float(scales[ax]), a)
    return np.ascontiguousarray(y, np.float32)


def _maxpool(x, a):
    k = a.get("kernel_shape")
    s = a.get("strides", [1, 1])
    p = a.get("pads", [0, 0, 0, 0])
    if a.get("ceil_mode", 0) or a.get("dilations", [1, 1]) != [1, 1]:
        raise GraphError("this max pooling variant is not supported")
    xp = np.pad(x, ((0, 0), (0, 0), (p[0], p[2]), (p[1], p[3])), constant_values=-np.inf)
    n, c, h, w = xp.shape
    oh, ow = (h - k[0]) // s[0] + 1, (w - k[1]) // s[1] + 1
    out = np.full((n, c, oh, ow), -np.inf, np.float32)
    for i in range(k[0]):
        for j in range(k[1]):
            out = np.maximum(out, xp[:, :, i:i + s[0] * (oh - 1) + 1:s[0], j:j + s[1] * (ow - 1) + 1:s[1]])
    return out


_CAST = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64, 2: np.uint8,
         3: np.int8}


def _run2(op, x, a):
    if op == "ConvTranspose":
        return _convT(x[0], x[1], x[2] if len(x) > 2 else None, a)
    if op == "Resize":
        return _resize(x, a)
    if op == "MaxPool":
        return _maxpool(x[0], a)
    if op == "ReduceMean":
        ax = _axes(x, a, 1)
        return np.mean(x[0], axis=None if ax is None else tuple(ax), keepdims=bool(a.get("keepdims", 1))
                       ).astype(np.float32)
    if op == "Gather":
        return np.take(x[0], np.asarray(x[1]).astype(np.int64), axis=a.get("axis", 0))
    if op == "Slice":
        d = x[0]
        st = [int(v) for v in np.asarray(x[1]).ravel()]
        en = [int(v) for v in np.asarray(x[2]).ravel()]
        ax = [int(v) for v in np.asarray(x[3]).ravel()] if len(x) > 3 and x[3] is not None else list(range(len(st)))
        sp = [int(v) for v in np.asarray(x[4]).ravel()] if len(x) > 4 and x[4] is not None else [1] * len(st)
        sl = [slice(None)] * d.ndim
        for s0, e0, a0, p0 in zip(st, en, ax, sp):
            sl[a0] = slice(s0, max(min(e0, 2 ** 62), -2 ** 62), p0)
        return d[tuple(sl)]
    if op == "Squeeze":
        ax = _axes(x, a, 1)
        return np.squeeze(x[0], axis=None if ax is None else tuple(ax))
    if op == "Unsqueeze":
        y = x[0]
        for v in sorted(_axes(x, a, 1)):
            y = np.expand_dims(y, v)
        return y
    if op == "Expand":
        shp = [int(v) for v in np.asarray(x[1]).ravel()]
        return np.broadcast_to(x[0], np.broadcast_shapes(x[0].shape, tuple(shp))).copy()
    if op == "Tile":
        return np.tile(x[0], [int(v) for v in np.asarray(x[1]).ravel()])
    if op == "Cast":
        return np.asarray(x[0]).astype(_CAST.get(a.get("to", 1), np.float32))
    if op == "Where":
        return np.where(x[0], x[1], x[2])
    if op == "Equal":
        return np.equal(x[0], x[1])
    if op == "Greater":
        return np.greater(x[0], x[1])
    if op == "Less":
        return np.less(x[0], x[1])
    if op == "Not":
        return np.logical_not(x[0])
    if op == "Neg":
        return -x[0]
    if op == "Sin":
        return np.sin(x[0]).astype(np.float32)
    if op == "Cos":
        return np.cos(x[0]).astype(np.float32)
    if op == "Exp":
        return np.exp(x[0]).astype(np.float32)
    if op == "Clip":
        lo = x[1] if len(x) > 1 and x[1] is not None else a.get("min", -np.inf)
        hi = x[2] if len(x) > 2 and x[2] is not None else a.get("max", np.inf)
        return np.clip(x[0], lo, hi)
    if op == "Mod":
        return np.fmod(x[0], x[1]) if a.get("fmod", 0) else np.mod(x[0], x[1])
    if op == "ReduceSum":
        ax = _axes(x, a, 1)
        return np.sum(x[0], axis=None if ax is None else tuple(ax), keepdims=bool(a.get("keepdims", 1)))
    if op == "ReduceMax":
        ax = _axes(x, a, 1)
        return np.max(x[0], axis=None if ax is None else tuple(ax), keepdims=bool(a.get("keepdims", 1)))
    if op == "Abs":
        return np.abs(x[0])
    if op == "Floor":
        return np.floor(x[0])
    if op == "Max":
        return np.maximum.reduce([np.asarray(v) for v in x])
    if op == "Min":
        return np.minimum.reduce([np.asarray(v) for v in x])
    if op == "ScatterND":
        out = np.array(x[0], copy=True)
        idx = np.asarray(x[1]).astype(np.int64)
        upd = np.asarray(x[2])
        flat = idx.reshape(-1, idx.shape[-1])
        uf = upd.reshape((flat.shape[0],) + upd.shape[idx.ndim - 1:])
        for k_, ii in enumerate(flat):
            out[tuple(ii)] = uf[k_]
        return out
    if op == "OneHot":
        ind = np.asarray(x[0]).astype(np.int64)
        depth = int(np.asarray(x[1]).ravel()[0])
        off, on = np.asarray(x[2]).ravel()[:2]
        ax = a.get("axis", -1)
        ind = np.where(ind < 0, ind + depth, ind)
        oh_ = (np.arange(depth) == ind[..., None]).astype(np.float32)
        oh_ = oh_ * (on - off) + off
        return np.moveaxis(oh_, -1, ax) if ax != -1 else oh_
    if op == "ConstantOfShape":
        shp = [int(v) for v in np.asarray(x[0]).ravel()]
        val = a.get("value_arr")
        return np.full(shp, 0.0 if val is None else val, np.float32)
    if op == "Range":
        return np.arange(np.asarray(x[0]).ravel()[0], np.asarray(x[1]).ravel()[0], np.asarray(x[2]).ravel()[0])
    if op == "Shape":
        return np.asarray(np.shape(x[0]), np.int64)
    return None

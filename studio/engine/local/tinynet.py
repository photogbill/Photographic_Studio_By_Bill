"""A very small runner for TensorFlow Lite models, in NumPy (no TensorFlow, no OpenCV, nothing to install).

It reads the .tflite file (a FlatBuffer, read field by field below) and runs the layer types that the face and
people models in models/ use: convolution, depthwise and transposed convolution, max and average pooling, padding,
addition, multiplication, ReLU, PReLU, hard swish, logistic, reshape, concatenation, depth-to-space, bilinear resize,
float16 constants and sparse (pruned) weights. Anything else is refused with a clear message, so a different model
never runs half-understood. Batch size 1, NHWC layout, float32 throughout.

The models are Google's MediaPipe face detector (BlazeFace, Bazarevsky et al. 2019), face mesh (Kartynnik et al.
2019), selfie segmentation and BlazePose (Bazarevsky et al. 2020), Apache License 2.0 (models/*/NOTICE). Running
them here keeps photos on this PC.
"""
from __future__ import annotations

import struct

import numpy as np

# builtin operator codes (TensorFlow Lite schema)
ADD, AVERAGE_POOL_2D, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, DEPTH_TO_SPACE, DEQUANTIZE = 0, 1, 2, 3, 4, 5, 6
LOGISTIC, MAX_POOL_2D, MUL, RELU, RESHAPE, RESIZE_BILINEAR = 14, 17, 18, 19, 22, 23
SOFTMAX, CUSTOM, PAD, TRANSPOSE, PRELU, TRANSPOSE_CONV, SUM = 25, 32, 34, 39, 54, 67, 74
RESIZE_NEAREST_NEIGHBOR, HARD_SWISH, DENSIFY = 97, 117, 124
_NAMES = {ADD: "ADD", AVERAGE_POOL_2D: "AVERAGE_POOL_2D", CONCATENATION: "CONCATENATION", CONV_2D: "CONV_2D",
          DEPTHWISE_CONV_2D: "DEPTHWISE_CONV_2D", DEPTH_TO_SPACE: "DEPTH_TO_SPACE", DEQUANTIZE: "DEQUANTIZE",
          LOGISTIC: "LOGISTIC", MAX_POOL_2D: "MAX_POOL_2D", MUL: "MUL", RELU: "RELU", RESHAPE: "RESHAPE",
          RESIZE_BILINEAR: "RESIZE_BILINEAR", CUSTOM: "CUSTOM", PAD: "PAD", PRELU: "PRELU", HARD_SWISH: "HARD_SWISH",
          DENSIFY: "DENSIFY", SOFTMAX: "SOFTMAX", TRANSPOSE: "TRANSPOSE", TRANSPOSE_CONV: "TRANSPOSE_CONV", SUM: "SUM",
          RESIZE_NEAREST_NEIGHBOR: "RESIZE_NEAREST_NEIGHBOR"}
CUSTOM_OPS = ("Convolution2DTransposeBias",)       # MediaPipe's transposed convolution (selfie segmentation)
_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 9: np.int8}


class ModelError(ValueError):
    pass


# ------------------------------------------------------------------------------------------ FlatBuffer reading
class _Table:
    """One FlatBuffer table: fields by index, with the defaults the schema gives."""

    def __init__(self, buf: bytes, pos: int):
        self.buf, self.pos = buf, pos
        vt = pos - struct.unpack_from("<i", buf, pos)[0]
        self.vt, self.vt_len = vt, struct.unpack_from("<H", buf, vt)[0]

    def _off(self, i: int) -> int:
        o = 4 + 2 * i
        return struct.unpack_from("<H", self.buf, self.vt + o)[0] if o < self.vt_len else 0

    def scalar(self, i: int, fmt: str, default=0):
        o = self._off(i)
        return struct.unpack_from("<" + fmt, self.buf, self.pos + o)[0] if o else default

    def _target(self, i: int):
        o = self._off(i)
        if not o:
            return None
        p = self.pos + o
        return p + struct.unpack_from("<I", self.buf, p)[0]

    def table(self, i: int):
        t = self._target(i)
        return None if t is None else _Table(self.buf, t)

    def vector(self, i: int, fmt: str | None = None):
        """A vector of scalars (fmt, as a NumPy array) or of tables (fmt None)."""
        t = self._target(i)
        if t is None:
            return np.zeros(0, fmt) if fmt else []
        n = struct.unpack_from("<I", self.buf, t)[0]
        if fmt:
            return np.frombuffer(self.buf, np.dtype("<" + fmt), n, t + 4).copy()
        out = []
        for k in range(n):
            p = t + 4 + 4 * k
            out.append(_Table(self.buf, p + struct.unpack_from("<I", self.buf, p)[0]))
        return out

    def string(self, i: int) -> str:
        t = self._target(i)
        if t is None:
            return ""
        n = struct.unpack_from("<I", self.buf, t)[0]
        return self.buf[t + 4:t + 4 + n].decode("utf-8", "replace")

    def raw_bytes(self, i: int) -> bytes:
        t = self._target(i)
        if t is None:
            return b""
        n = struct.unpack_from("<I", self.buf, t)[0]
        return self.buf[t + 4:t + 4 + n]


# ------------------------------------------------------------------------------------------ the model
class Model:
    """A TensorFlow Lite model ready to run: ``Model(path)(x)`` with x float32 (1, H, W, C) -> {output name: array}."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            buf = f.read()
        if len(buf) < 8 or buf[4:8] != b"TFL3":
            raise ModelError(f"{path} is not a TensorFlow Lite model")
        root = _Table(buf, struct.unpack_from("<I", buf, 0)[0])
        codes, customs = [], []
        for oc in root.vector(1):
            codes.append(max(oc.scalar(0, "b", 0), oc.scalar(3, "i", 0)))
            customs.append(oc.string(1))
        buffers = [b.raw_bytes(0) for b in root.vector(4)]
        sg = root.vector(2)
        if len(sg) != 1:
            raise ModelError("only single-graph models are supported")
        g = sg[0]
        self.tensors = []
        for t in g.vector(0):
            shape = tuple(int(v) for v in t.vector(0, "i"))
            dtype = _DTYPES.get(t.scalar(1, "b", 0))
            if dtype is None:
                raise ModelError(f"tensor type {t.scalar(1, 'b', 0)} is not supported")
            data = buffers[t.scalar(2, "I", 0)] if t.scalar(2, "I", 0) < len(buffers) else b""
            sp = t.table(6)
            if data and sp is not None:
                const = np.frombuffer(data, dtype)                  # the non-zero values of a pruned tensor
            else:
                const = np.frombuffer(data, dtype).reshape(shape) if data else None
            self.tensors.append({"name": t.string(3), "shape": shape, "dtype": dtype, "const": const, "sparsity": sp})
        self.inputs = [int(v) for v in g.vector(1, "i")]
        self.outputs = [int(v) for v in g.vector(2, "i")]
        self.ops = []
        for op in g.vector(3):
            k = op.scalar(0, "I", 0)
            code = codes[k]
            if code not in _NAMES:
                raise ModelError(f"layer type {code} is not supported by this runner")
            opt = op.table(4)
            if code == CUSTOM:
                if customs[k] not in CUSTOM_OPS:
                    raise ModelError(f"custom layer {customs[k]!r} is not supported by this runner")
                raw = op.raw_bytes(5)                               # padding, stride w, stride h (int32 each)
                opt = struct.unpack_from("<3i", raw, 0) if len(raw) >= 12 else (1, 2, 2)
            self.ops.append((code, [int(v) for v in op.vector(1, "i")], [int(v) for v in op.vector(2, "i")], opt))
        self.input_shape = self.tensors[self.inputs[0]]["shape"]
        self.output_names = [self.tensors[i]["name"] for i in self.outputs]
        # pruned weights are expanded and float16 weights turned into float32 once, here
        self._consts = {}
        for code, ins, outs, _ in self.ops:
            if code == DENSIFY:
                t = self.tensors[ins[0]]
                if t["const"] is not None and t["sparsity"] is not None:
                    self._consts[outs[0]] = densify(t["const"], t["shape"], t["sparsity"])
            elif code == DEQUANTIZE:
                src = self._consts.get(ins[0])
                if src is None:
                    src = self.tensors[ins[0]]["const"]
                if src is not None:
                    self._consts[outs[0]] = np.asarray(src).astype(np.float32)

    def _value(self, vals, i):
        if i in vals:
            return vals[i]
        if i in self._consts:
            return self._consts[i]
        c = self.tensors[i]["const"]
        if c is None:
            raise ModelError(f"tensor {self.tensors[i]['name']} has no value")
        return c.astype(np.float32) if c.dtype == np.float16 else c

    def __call__(self, x: np.ndarray) -> dict:
        x = np.asarray(x, np.float32)
        if tuple(x.shape) != tuple(self.input_shape):
            raise ModelError(f"input shape {x.shape}, the model wants {self.input_shape}")
        vals = {self.inputs[0]: x}
        for code, ins, outs, opt in self.ops:
            if code in (DEQUANTIZE, DENSIFY) and outs[0] in self._consts:
                continue
            a = [self._value(vals, i) if i >= 0 else None for i in ins]
            vals[outs[0]] = _run(code, a, opt, self.tensors[outs[0]]["shape"])
        return {self.tensors[i]["name"]: vals[i] for i in self.outputs}


# ------------------------------------------------------------------------------------------ sparse weights
def _index_vector(dm: _Table, type_field: int, value_field: int) -> np.ndarray:
    kind = dm.scalar(type_field, "B", 0)
    t = dm.table(value_field)
    if t is None or kind == 0:
        return np.zeros(0, int)
    return t.vector(0, {1: "i", 2: "H", 3: "B"}[kind]).astype(np.int64)


def densify(values: np.ndarray, shape, sp: _Table) -> np.ndarray:
    """A pruned tensor (TensorFlow Lite sparsity: dense and CSR dimensions in traversal order) as a dense array."""
    order = [int(v) for v in sp.vector(0, "i")]
    if sp.vector(1, "i").size:
        raise ModelError("block-sparse weights are not supported by this runner")
    dims = sp.vector(2)
    out = np.zeros(int(np.prod(shape)), values.dtype)
    meta = []
    for d in dims:
        if d.scalar(0, "b", 0) == 0:
            meta.append(("dense", d.scalar(1, "i", 0), None, None))
        else:
            meta.append(("csr", 0, _index_vector(d, 2, 3), _index_vector(d, 4, 5)))
    strides = np.cumprod([1] + list(shape[::-1]))[:-1][::-1]        # row-major strides of the dense tensor
    # walk the levels, keeping (position in the next level, flat offset in the dense tensor) for every path
    pos = np.zeros(1, np.int64)
    off = np.zeros(1, np.int64)
    for lvl, (kind, size, seg, idx) in enumerate(meta):
        st = strides[order[lvl]]
        if kind == "dense":
            i = np.arange(size, dtype=np.int64)
            pos = (pos[:, None] * size + i[None, :]).ravel()
            off = (off[:, None] + i[None, :] * st).ravel()
        else:
            counts = seg[pos + 1] - seg[pos]
            starts = np.repeat(seg[pos], counts)
            k = starts + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
            off = np.repeat(off, counts) + idx[k] * st
            pos = k
    out[off] = values[pos]
    return out.reshape(shape)


# ------------------------------------------------------------------------------------------ layers
def _act(x, kind: int):
    if kind == 1:
        return np.maximum(x, 0.0)
    if kind == 3:
        return np.clip(x, 0.0, 6.0)
    if kind == 2:
        return np.clip(x, -1.0, 1.0)
    if kind:
        raise ModelError(f"fused activation {kind} is not supported")
    return x


def _same_pad(n: int, k: int, s: int, d: int = 1):
    out = -(-n // s)
    total = max((out - 1) * s + (k - 1) * d + 1 - n, 0)
    return out, total // 2, total - total // 2


def _pads(x, kh, kw, sh, sw, dh, dw, padding, fill=0.0):
    H, W = x.shape[1:3]
    if padding == 0:                                   # SAME
        oh, t, b = _same_pad(H, kh, sh, dh)
        ow, l, r = _same_pad(W, kw, sw, dw)
        if t or b or l or r:
            x = np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)), constant_values=fill)
    else:                                              # VALID
        oh = (H - (kh - 1) * dh - 1) // sh + 1
        ow = (W - (kw - 1) * dw - 1) // sw + 1
    return x, oh, ow


def _conv(x, w, b, opt, depthwise: bool):
    if depthwise:
        padding, sw_, sh_, mult, act = (opt.scalar(0, "b"), opt.scalar(1, "i", 1), opt.scalar(2, "i", 1),
                                        opt.scalar(3, "i", 1), opt.scalar(4, "b"))
        dw_, dh_ = opt.scalar(5, "i", 1), opt.scalar(6, "i", 1)
        if mult != 1 and w.shape[3] != x.shape[3]:
            x = np.repeat(x, w.shape[3] // x.shape[3], axis=3)
    else:
        padding, sw_, sh_, act = opt.scalar(0, "b"), opt.scalar(1, "i", 1), opt.scalar(2, "i", 1), opt.scalar(3, "b")
        dw_, dh_ = opt.scalar(4, "i", 1), opt.scalar(5, "i", 1)
    kh, kw = w.shape[1], w.shape[2]
    xp, oh, ow = _pads(x, kh, kw, sh_, sw_, dh_, dw_, padding)
    xp = xp[0]
    out = np.zeros((oh, ow, w.shape[0] if not depthwise else w.shape[3]), np.float32)
    for i in range(kh):
        for j in range(kw):
            p = xp[i * dh_:i * dh_ + (oh - 1) * sh_ + 1:sh_, j * dw_:j * dw_ + (ow - 1) * sw_ + 1:sw_]
            if depthwise:
                out += p * w[0, i, j]
            else:
                out += p @ w[:, i, j, :].T
    if b is not None:
        out += b
    return _act(out, act)[None]


def _avgpool(x, opt):
    padding, sw_, sh_, fw, fh, act = (opt.scalar(0, "b"), opt.scalar(1, "i", 1), opt.scalar(2, "i", 1),
                                      opt.scalar(3, "i", 1), opt.scalar(4, "i", 1), opt.scalar(5, "b"))
    xp, oh, ow = _pads(x, fh, fw, sh_, sw_, 1, 1, padding, 0.0)
    ones, _, _ = _pads(np.ones_like(x[..., :1]), fh, fw, sh_, sw_, 1, 1, padding, 0.0)
    xp, ones = xp[0], ones[0]
    acc = np.zeros((oh, ow, x.shape[3]), np.float32)
    cnt = np.zeros((oh, ow, 1), np.float32)
    for i in range(fh):
        for j in range(fw):
            acc += xp[i:i + (oh - 1) * sh_ + 1:sh_, j:j + (ow - 1) * sw_ + 1:sw_]
            cnt += ones[i:i + (oh - 1) * sh_ + 1:sh_, j:j + (ow - 1) * sw_ + 1:sw_]
    return _act(acc / np.maximum(cnt, 1.0), act)[None]   # padding is not counted, as in TensorFlow Lite


def _transpose_conv(output_shape, w, x, b, opt):
    """TensorFlow Lite's TRANSPOSE_CONV: weights (out, kh, kw, in), the output shape given as an input."""
    padding, sw_, sh_ = opt.scalar(0, "b"), opt.scalar(1, "i", 1), opt.scalar(2, "i", 1)
    act = opt.scalar(3, "b") if opt is not None else 0
    oh, ow = int(output_shape[1]), int(output_shape[2])
    H, W = x.shape[1:3]
    kh, kw, oc = w.shape[1], w.shape[2], w.shape[0]
    fh, fw = (H - 1) * sh_ + kh, (W - 1) * sw_ + kw
    full = np.zeros((max(fh, oh), max(fw, ow), oc), np.float32)
    for i in range(kh):
        for j in range(kw):
            full[i:i + (H - 1) * sh_ + 1:sh_, j:j + (W - 1) * sw_ + 1:sw_] += x[0] @ w[:, i, j, :].T
    if padding == 0:                                   # SAME: the padding TensorFlow would have used going forward
        t = max((H - 1) * sh_ + kh - oh, 0) // 2
        l_ = max((W - 1) * sw_ + kw - ow, 0) // 2
    else:
        t = l_ = 0
    out = full[t:t + oh, l_:l_ + ow]
    if b is not None:
        out = out + b
    return _act(out, act)[None].astype(np.float32)


def _resize_nearest(x, size, opt):
    align, half = (bool(opt.scalar(0, "B", 0)), bool(opt.scalar(1, "B", 0))) if opt is not None else (False, False)
    H, W = x.shape[1:3]
    oh, ow = int(size[0]), int(size[1])

    def idx(n_out, n_in):
        o = np.arange(n_out, dtype=np.float64)
        scale = (n_in - 1) / (n_out - 1) if align and n_out > 1 else n_in / n_out
        s = (o + 0.5) * scale if half else o * scale
        i = np.round(s).astype(int) if align else np.floor(s).astype(int)
        return np.clip(i, 0, n_in - 1)
    return x[:, idx(oh, H)][:, :, idx(ow, W)]


def _transpose_conv_bias(x, w, b, opt):
    """MediaPipe's Convolution2DTransposeBias: weights (out, kh, kw, in), stride s, SAME padding."""
    padding, sw_, sh_ = opt
    H, W = x.shape[1:3]
    kh, kw, oc = w.shape[1], w.shape[2], w.shape[0]
    fh, fw = (H - 1) * sh_ + kh, (W - 1) * sw_ + kw
    full = np.zeros((fh, fw, oc), np.float32)
    for i in range(kh):
        for j in range(kw):
            full[i:i + (H - 1) * sh_ + 1:sh_, j:j + (W - 1) * sw_ + 1:sw_] += x[0] @ w[:, i, j, :].T
    oh, ow = (H * sh_, W * sw_) if padding == 1 else (fh, fw)
    t, l_ = (fh - oh) // 2, (fw - ow) // 2
    out = full[t:t + oh, l_:l_ + ow]
    if b is not None:
        out = out + b
    return out[None].astype(np.float32)


def _depth_to_space(x, bs: int):
    _, H, W, C = x.shape
    oc = C // (bs * bs)
    return x.reshape(1, H, W, bs, bs, oc).transpose(0, 1, 3, 2, 4, 5).reshape(1, H * bs, W * bs, oc)


def _maxpool(x, opt):
    padding, sw_, sh_, fw, fh, act = (opt.scalar(0, "b"), opt.scalar(1, "i", 1), opt.scalar(2, "i", 1),
                                      opt.scalar(3, "i", 1), opt.scalar(4, "i", 1), opt.scalar(5, "b"))
    xp, oh, ow = _pads(x, fh, fw, sh_, sw_, 1, 1, padding, -np.inf)
    xp = xp[0]
    out = None
    for i in range(fh):
        for j in range(fw):
            p = xp[i:i + (oh - 1) * sh_ + 1:sh_, j:j + (ow - 1) * sw_ + 1:sw_]
            out = p.copy() if out is None else np.maximum(out, p)
    return _act(out, act)[None]


def _resize_bilinear(x, size, opt):
    align, half = (bool(opt.scalar(2, "B", 0)), bool(opt.scalar(3, "B", 0))) if opt is not None else (False, False)
    H, W = x.shape[1:3]
    oh, ow = int(size[0]), int(size[1])

    def coords(n_out, n_in):
        o = np.arange(n_out, dtype=np.float64)
        if align and n_out > 1:
            s = o * (n_in - 1) / (n_out - 1)
        elif half:
            s = np.maximum((o + 0.5) * n_in / n_out - 0.5, 0.0)
        else:
            s = o * n_in / n_out
        i0 = np.minimum(np.floor(s).astype(int), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (s - i0).astype(np.float32)
    y0, y1, fy = coords(oh, H)
    x0, x1, fx = coords(ow, W)
    a = x[0]
    top = a[y0][:, x0] * (1 - fx)[None, :, None] + a[y0][:, x1] * fx[None, :, None]
    bot = a[y1][:, x0] * (1 - fx)[None, :, None] + a[y1][:, x1] * fx[None, :, None]
    return (top * (1 - fy)[:, None, None] + bot * fy[:, None, None])[None].astype(np.float32)


def _run(code, a, opt, out_shape):
    if code == CONV_2D:
        return _conv(a[0], a[1], a[2] if len(a) > 2 else None, opt, False)
    if code == DEPTHWISE_CONV_2D:
        return _conv(a[0], a[1], a[2] if len(a) > 2 else None, opt, True)
    if code == ADD:
        return _act(a[0] + a[1], opt.scalar(0, "b") if opt is not None else 0)
    if code == RELU:
        return np.maximum(a[0], 0.0)
    if code == MUL:
        return _act(a[0] * a[1], opt.scalar(0, "b") if opt is not None else 0)
    if code == LOGISTIC:
        return (1.0 / (1.0 + np.exp(-np.clip(a[0], -60.0, 60.0)))).astype(np.float32)
    if code == HARD_SWISH:
        return (a[0] * np.clip(a[0] + 3.0, 0.0, 6.0) / 6.0).astype(np.float32)
    if code == AVERAGE_POOL_2D:
        return _avgpool(a[0], opt)
    if code == DEPTH_TO_SPACE:
        return _depth_to_space(a[0], opt.scalar(0, "i", 2))
    if code == CUSTOM:
        return _transpose_conv_bias(a[0], a[1], a[2] if len(a) > 2 else None, opt)
    if code == TRANSPOSE_CONV:
        return _transpose_conv(np.asarray(a[0]).ravel(), a[1], a[2], a[3] if len(a) > 3 and a[3] is not None else None,
                               opt)
    if code == TRANSPOSE:
        return np.transpose(a[0], [int(v) for v in np.asarray(a[1]).ravel()])
    if code == SOFTMAX:
        beta = opt.scalar(0, "f", 1.0) if opt is not None else 1.0
        z = a[0] * beta
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        return (z / z.sum(axis=-1, keepdims=True)).astype(np.float32)
    if code == SUM:
        axes = tuple(int(v) for v in np.asarray(a[1]).ravel())
        keep = bool(opt.scalar(0, "B", 0)) if opt is not None else False
        return a[0].sum(axis=axes, keepdims=keep).astype(np.float32)
    if code == RESIZE_NEAREST_NEIGHBOR:
        return _resize_nearest(a[0], np.asarray(a[1]).ravel(), opt)
    if code == PRELU:
        return np.where(a[0] >= 0, a[0], a[0] * a[1]).astype(np.float32)
    if code == MAX_POOL_2D:
        return _maxpool(a[0], opt)
    if code == PAD:
        p = np.asarray(a[1], int).reshape(-1, 2)
        return np.pad(a[0], [tuple(r) for r in p])
    if code == RESHAPE:
        shape = out_shape if out_shape else tuple(int(v) for v in np.asarray(a[1]).ravel())
        return a[0].reshape(shape)
    if code == CONCATENATION:
        axis = opt.scalar(0, "i", 0)
        return _act(np.concatenate(a, axis=axis), opt.scalar(1, "b"))
    if code == RESIZE_BILINEAR:
        return _resize_bilinear(a[0], np.asarray(a[1]).ravel(), opt)
    if code == DEQUANTIZE:
        return np.asarray(a[0], np.float32)
    raise ModelError(f"layer type {code} is not supported")

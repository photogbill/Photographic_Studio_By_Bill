"""Finding faces and their features, on this PC (models/faces, run by tinynet.py).

Two Google MediaPipe models do the work (Apache License 2.0, models/faces/NOTICE):

* BlazeFace finds faces (Bazarevsky et al., "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs",
  2019). Two versions run: the short-range one (faces filling a good part of the frame) and the full-range one
  (smaller faces), on the upright photo and, for very dark or candle-lit photos, on a brightened and colour-
  balanced copy too, because the detector was trained on ordinary daylight pictures.
* The face mesh places 468 points on a face (Kartynnik et al., "Real-time Facial Surface Geometry from Monocular
  Video on Mobile GPUs", 2019): eyes, brows, lips, nose, the outline of the face. It needs only a rough box around
  the face, so when the detector misses a face (a half-lit candle portrait), two clicks on the eyes are enough.

Everything here works in plain image pixels of an upright image; the caller turns the points into fractions of the
S frame so they stay attached to the photo.
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np
from scipy.ndimage import map_coordinates

from ... import log as slog

log = slog.get("engine.faces")

MODEL_FILES = {"short": "face_detection_short_range.tflite", "full": "face_detection_full_range.tflite",
               "mesh": "face_landmark.tflite"}
_MODELS: dict = {}
_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------------ mesh point sets
# outlines (MediaPipe face mesh topology), in order around each shape
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]   # the person's right
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_BROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
LEFT_BROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10
MOUTH_R, MOUTH_L = 61, 291
NOSE_WING_R, NOSE_WING_L = 64, 294           # where the nose wing meets the cheek (smile lines start here)
NOSTRILS = [(79, 20), (309, 250)]            # each nostril: two points across it
JAW_R = [172, 136, 150, 149, 176, 148]       # the jaw below the cheek, each side (jowls)
JAW_L = [397, 365, 379, 378, 400, 377]


# ------------------------------------------------------------------------------------------ models
def model_dir() -> str:
    from ... import paths
    return os.path.join(str(paths.APP_ROOT), "models", "faces")


def available() -> bool:
    d = model_dir()
    return all(os.path.isfile(os.path.join(d, f)) for f in MODEL_FILES.values())


def _model(key: str):
    with _LOCK:
        if key not in _MODELS:
            from .tinynet import Model
            path = os.path.join(model_dir(), MODEL_FILES[key])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"face model missing: {path} (run setup again, or copy models/faces back)")
            _MODELS[key] = Model(path)
        return _MODELS[key]


# ------------------------------------------------------------------------------------------ detection
def _anchors(size: int, strides, per_layer: int) -> np.ndarray:
    """SSD anchor centres (fractions), MediaPipe's SsdAnchorsCalculator with fixed-size anchors."""
    out = []
    i = 0
    while i < len(strides):
        j = i
        while j < len(strides) and strides[j] == strides[i]:
            j += 1
        n = (j - i) * per_layer
        g = int(math.ceil(size / strides[i]))
        for y in range(g):
            for x in range(g):
                out += [((x + 0.5) / g, (y + 0.5) / g)] * n
        i = j
    return np.asarray(out, np.float32)


_DET = {"short": (128, [8, 16, 16, 16], 2, 0.5), "full": (192, [4], 1, 0.6)}
_ANCH: dict = {}


def _letterbox(img: np.ndarray, size: int):
    """Fit an image into a size x size square (black borders), keeping its shape. Returns (square, scale, ox, oy):
    image pixel x = (square pixel x - ox) / scale."""
    h, w = img.shape[:2]
    s = size / float(max(h, w))
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    ys = (np.arange(nh) + 0.5) / s - 0.5          # array indices (pixel centres at whole numbers)
    xs = (np.arange(nw) + 0.5) / s - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    small = np.stack([map_coordinates(img[..., c], [Y, X], order=1, mode="nearest") for c in range(3)], -1)
    if s < 0.5:                                    # shrinking a lot: average first so small details do not alias
        from scipy.ndimage import uniform_filter
        k = max(1, int(round(0.5 / s)))
        blur = np.stack([uniform_filter(img[..., c], k, mode="nearest") for c in range(3)], -1)
        small = np.stack([map_coordinates(blur[..., c], [Y, X], order=1, mode="nearest") for c in range(3)], -1)
    sq = np.zeros((size, size, 3), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    sq[oy:oy + nh, ox:ox + nw] = small
    return sq, s, ox, oy


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _weighted_nms(dets: list, thr: float = 0.3) -> list:
    """MediaPipe's weighted non-maximum suppression: overlapping finds are averaged, weighted by their scores."""
    dets = sorted(dets, key=lambda d: -d["score"])
    out = []
    while dets:
        top = dets[0]
        group = [d for d in dets if _iou(top["box"], d["box"]) > thr]
        dets = [d for d in dets if _iou(top["box"], d["box"]) <= thr]
        w = np.array([d["score"] for d in group])
        box = np.average([d["box"] for d in group], axis=0, weights=w)
        kp = np.average([d["kp"] for d in group], axis=0, weights=w)
        out.append({"box": box, "kp": kp, "score": float(top["score"]), "model": top["model"],
                    "variant": top.get("variant", "")})
    return out


def detect(img: np.ndarray, which=("short", "full")) -> list:
    """Faces in an upright image (float 0..1, display-encoded RGB). Returns [{box: [x0, y0, x1, y1], kp: (6, 2),
    score}] in image coordinates (pixel i covers i..i+1, so its centre is i + 0.5); keypoints: the person's right eye, left eye, nose tip, mouth, right ear, left ear."""
    found = []
    for key in which:
        size, strides, per, thr = _DET[key]
        if key not in _ANCH:
            _ANCH[key] = _anchors(size, strides, per)
        anc = _ANCH[key]
        sq, s, ox, oy = _letterbox(img, size)
        out = _model(key)((sq * 2.0 - 1.0)[None])
        reg = next(v for k, v in out.items() if "regress" in k)[0]
        cls = next(v for k, v in out.items() if "class" in k)[0, :, 0]
        score = 1.0 / (1.0 + np.exp(-np.clip(cls.astype(np.float64), -60, 60)))
        for i in np.nonzero(score >= thr)[0]:
            r = reg[i] / size
            cx, cy = r[0] + anc[i, 0], r[1] + anc[i, 1]
            w, h = r[2], r[3]
            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * size
            kp = (r[4:16].reshape(6, 2) + anc[i]) * size
            box = np.array([(box[0] - ox) / s, (box[1] - oy) / s, (box[2] - ox) / s, (box[3] - oy) / s])
            kp = np.stack([(kp[:, 0] - ox) / s, (kp[:, 1] - oy) / s], -1)
            found.append({"box": box, "kp": kp, "score": float(score[i]), "model": key})
    return _weighted_nms(found)


def variants(img: np.ndarray) -> list:
    """The photo as given, plus (when it is dark or strongly tinted, as candle light is) a brightened copy with the
    colour cast taken halfway out: what the models were trained on."""
    out = [("as shown", img)]
    Y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    med = float(np.median(Y))
    hi = float(np.percentile(Y, 99.5))
    mean = img.reshape(-1, 3).mean(axis=0) + 1e-4
    cast = float(mean.max() / mean.min())
    if med < 0.25 or hi < 0.7 or cast > 1.35:
        g = img / max(hi, 0.05)
        gain = np.sqrt(mean.mean() / mean)              # half the cast
        g = np.clip(g * gain, 0.0, 1.0)
        m2 = float(np.median(g @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
        if 0.01 < m2 < 0.4:
            g = np.power(g, math.log(0.4) / math.log(m2))
        out.append(("brightened", g.astype(np.float32)))
    return out


# ------------------------------------------------------------------------------------------ the face mesh
def roi_from_eyes(right_eye, left_eye, box=None, scale: float = 1.5):
    """The square, rotated crop the face mesh wants: (cx, cy, size, angle). From the detector's box and eyes, or
    (box None) from the two eyes alone (clicked by hand): a face is about 2.6 eye-spacings across."""
    rx, ry = map(float, right_eye)
    lx, ly = map(float, left_eye)
    ang = math.atan2(ly - ry, lx - rx)
    if box is not None:
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        size = max(box[2] - box[0], box[3] - box[1]) * scale
    else:
        d = math.hypot(lx - rx, ly - ry)
        mx, my = 0.5 * (rx + lx), 0.5 * (ry + ly)
        # the face's centre sits about 0.45 eye-spacings below the eyes (towards the nose), along the face
        cx, cy = mx - 0.45 * d * math.sin(ang), my + 0.45 * d * math.cos(ang)
        size = 3.2 * d * scale / 1.5
    return cx, cy, size, ang


def _roi_points(roi, u, v):
    """Crop coordinates (0..1 across the crop) -> image pixels."""
    cx, cy, size, ang = roi
    a, b = np.asarray(u) - 0.5, np.asarray(v) - 0.5
    c, s = math.cos(ang), math.sin(ang)
    return cx + size * (c * a - s * b), cy + size * (s * a + c * b)


def crop(img: np.ndarray, roi, n: int = 192) -> np.ndarray:
    g = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(g, g)
    X, Y = _roi_points(roi, U, V)
    return np.stack([map_coordinates(img[..., c], [Y - 0.5, X - 0.5], order=1, mode="constant",
                                     cval=0.0) for c in range(3)], -1).astype(np.float32)


def _normalise_crop(c: np.ndarray) -> np.ndarray:
    """Even out a dark or tinted face crop (centre-weighted levels, half the colour cast out)."""
    n = c.shape[0]
    q = c[n // 4:3 * n // 4, n // 4:3 * n // 4].reshape(-1, 3)
    mean = q.mean(axis=0) + 1e-4
    g = c * np.sqrt(mean.mean() / mean)
    Y = q @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = float(np.percentile(Y, 99)) + 1e-4
    g = np.clip(g / hi, 0.0, 1.0)
    med = float(np.median(np.clip(Y / hi, 1e-4, 1.0)))
    if 0.01 < med < 0.45:
        g = np.power(g, math.log(0.45) / math.log(med))
    return g.astype(np.float32)


def mesh(img: np.ndarray, roi, refine: int = 1):
    """468 face points (image pixels, (468, 2)) and the model's confidence that it saw a face (0..1). One
    refinement: the crop is re-centred on the first answer, as MediaPipe does when it tracks a face."""
    best = None
    for it in range(refine + 1):
        trials = []
        c = crop(img, roi)
        for cc in (c, _normalise_crop(c)):
            out = _model("mesh")(cc[None])
            pts = next(v for k, v in out.items() if v.size == 1404).reshape(468, 3)
            logit = float(next(v for k, v in out.items() if v.size == 1).ravel()[0])
            conf = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
            X, Y = _roi_points(roi, pts[:, 0] / 192.0, pts[:, 1] / 192.0)
            trials.append((conf, np.stack([X, Y], -1)))
        conf, P = max(trials, key=lambda t: t[0])
        if best is None or conf >= best[0] - 0.05:
            best = (conf, P)
        if it < refine:
            roi = roi_from_points(best[1])
    return best[1], best[0]


def roi_from_points(P: np.ndarray, scale: float = 1.5):
    """The crop around a face mesh (rotated with the eyes, square, 1.5 x the face's extent)."""
    r, l_ = P[33], P[263]
    ang = math.atan2(l_[1] - r[1], l_[0] - r[0])
    c, s = math.cos(ang), math.sin(ang)
    u = P[:, 0] * c + P[:, 1] * s
    v = -P[:, 0] * s + P[:, 1] * c
    uc, vc = 0.5 * (u.min() + u.max()), 0.5 * (v.min() + v.max())
    size = max(u.max() - u.min(), v.max() - v.min()) * scale
    return uc * c - vc * s, uc * s + vc * c, size, ang


def eye_centres(P: np.ndarray):
    return P[RIGHT_EYE].mean(axis=0), P[LEFT_EYE].mean(axis=0)


def find(img: np.ndarray, min_conf: float = 0.5) -> list:
    """Every face in an upright image: [{"pts": (468, 2) image pixels, "conf": mesh confidence, "score": detector
    score, "variant": which copy found it}], largest first."""
    found = []
    for name, v in variants(img):
        for d in detect(v):
            d["variant"] = name
            found.append(d)
    log.debug("face detector: %d find(s) before merging", len(found))
    dets = _weighted_nms(found, 0.3) if found else []
    faces = []
    for d in dets:
        roi = roi_from_eyes(d["kp"][0], d["kp"][1], d["box"])
        P, conf = mesh(img, roi)
        if conf < min_conf:
            log.info("face at %s: mesh not sure (%.2f), skipped", np.round(d["box"]).astype(int).tolist(), conf)
            continue
        faces.append({"pts": P, "conf": conf, "score": d["score"], "variant": d.get("variant", "")})
    # one face, one entry: two finds of the same face (detector variants) keep the surer mesh
    kept = []
    for f in sorted(faces, key=lambda f: -f["conf"]):
        e = np.mean(eye_centres(f["pts"]), axis=0)
        span = np.ptp(f["pts"][:, 0])
        if all(np.hypot(*(e - np.mean(eye_centres(k["pts"]), axis=0))) > 0.3 * span for k in kept):
            kept.append(f)
    kept.sort(key=lambda f: -np.ptp(f["pts"][:, 0]))
    return kept


def from_eyes(img: np.ndarray, right_eye, left_eye) -> dict:
    """A face from two clicked eyes (image pixels): the mesh run on the crop the eyes imply."""
    P, conf = mesh(img, roi_from_eyes(right_eye, left_eye), refine=1)
    return {"pts": P, "conf": conf, "score": 0.0, "variant": "eyes"}


def size_of(P: np.ndarray) -> float:
    """A face's width (pixels): the distance across the face outline at the cheeks."""
    return float(np.hypot(*(P[234] - P[454])))

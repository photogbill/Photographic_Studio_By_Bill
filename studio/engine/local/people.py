"""Finding people and their parts, on this PC (models/people, run by tinynet.py), for the Person mask.

Three Google MediaPipe models (Apache License 2.0, models/people/NOTICE):

* Selfie segmentation marks every pixel as person or not, on the whole photo at 256 x 256.
* Optional: the multiclass selfie segmentation model (models/people/selfie_multiclass_256x256.tflite, downloaded by
  Bill from Google's model page) labels every pixel as background, hair, body skin, face skin, clothes or other
  (accessories). When it is there, hair, skin, face and clothes come from it; otherwise they are worked out below.
* BlazePose (Bazarevsky et al., "BlazePose: On-device Real-time Body Pose tracking", 2020) finds each person with
  a detector that looks at the face and the hips, then places 33 body points on them (nose, eyes, ears, mouth,
  shoulders, elbows, wrists, hands, hips, knees, ankles, feet), each with how sure it is that the point is in view,
  and draws its own, closer person mask around the body.

The person mask and the body points are stored on the photo; the parts (skin, face, hair, clothes, arms, legs,
torso, background) are worked out at render time from them, the face map (if Find faces was used) and the photo's
own colours, then snapped to the photo's edges. Everything works in pixels of an upright image here; the caller turns
it into fractions of the S frame.
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ... import log as slog

log = slog.get("engine.people")

MODEL_FILES = {"seg": "selfie_segmentation.tflite", "detect": "pose_detection.tflite",
               "pose": "pose_landmark_full.tflite"}
MULTICLASS = "selfie_multiclass_256x256.tflite"   # optional: labels hair, body skin, face skin, clothes directly
HAIR = "hair_segmenter.tflite"                     # optional: hair alone, at 512 x 512 on a crop around the head
CLASSES = ("background", "hair", "body_skin", "face_skin", "clothes", "others")
_MODELS: dict = {}
_LOCK = threading.Lock()
PARTS = ("person", "background", "skin", "face", "hair", "clothes", "arms", "legs", "torso")
PART_NAMES = {"person": "Whole person", "background": "Background", "skin": "Skin", "face": "Face", "hair": "Hair",
              "clothes": "Clothes", "arms": "Arms", "legs": "Legs", "torso": "Torso"}

# BlazePose's 33 points
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, MOUTH_L, MOUTH_R = 0, 2, 5, 7, 8, 9, 10
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_IDX, R_IDX = 11, 12, 13, 14, 15, 16, 19, 20
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_FT, R_FT = 23, 24, 25, 26, 27, 28, 31, 32
ARMS = ((L_SH, L_EL, L_WR, L_IDX), (R_SH, R_EL, R_WR, R_IDX))
LEGS = ((L_HIP, L_KN, L_AN, L_FT), (R_HIP, R_KN, R_AN, R_FT))
VISIBLE = 0.5


def model_dir() -> str:
    from ... import paths
    return os.path.join(str(paths.APP_ROOT), "models", "people")


def available() -> bool:
    d = model_dir()
    return all(os.path.isfile(os.path.join(d, f)) for f in MODEL_FILES.values())


def multiclass_available() -> bool:
    return os.path.isfile(os.path.join(model_dir(), MULTICLASS))


def hair_available() -> bool:
    return os.path.isfile(os.path.join(model_dir(), HAIR))


def _model(key: str):
    with _LOCK:
        if key not in _MODELS:
            from .tinynet import Model
            path = os.path.join(model_dir(), MODEL_FILES.get(key, {"multi": MULTICLASS, "hair": HAIR}.get(key, "")))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"people model missing: {path} (run setup again, or copy models/people back)")
            _MODELS[key] = Model(path)
        return _MODELS[key]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -60, 60)))


# ------------------------------------------------------------------------------------------ the whole-photo mask
def selfie_mask(img: np.ndarray) -> np.ndarray:
    """Person probability (0..1) for every pixel of an upright image (display-encoded RGB 0..1)."""
    from .faces import _letterbox
    sq, s, ox, oy = _letterbox(img, 256)
    m = _model("seg")(sq[None])
    prob = next(iter(m.values()))[0, :, :, 0]
    h, w = img.shape[:2]
    ys = (np.arange(h) + 0.5) * s + oy - 0.5
    xs = (np.arange(w) + 0.5) * s + ox - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(prob, [Y, X], order=1, mode="nearest").astype(np.float32)


# ------------------------------------------------------------------------------------------ the parts model
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def classes(img: np.ndarray, roi=None) -> np.ndarray:
    """Per-pixel probabilities (h, w, 6) of background, hair, body skin, face skin, clothes, other, for an upright
    image: on a crop around one person (``roi``, as the pose model uses) or on the whole photo. Input scaled to
    -1..1, as the model's own metadata says (mean 127.5, spread 127.5)."""
    from .faces import _letterbox, crop
    h, w = img.shape[:2]
    if roi is not None:
        c = crop(img, roi, 256)
        prob = _softmax(next(iter(_model("multi")((c * 2.0 - 1.0)[None]).values()))[0])
        cx, cy, size, ang = roi
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
        dx, dy = xs - cx, ys - cy
        co, si = math.cos(ang), math.sin(ang)
        a = (co * dx + si * dy) / size + 0.5
        b = (-si * dx + co * dy) / size + 0.5
        inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        out = np.zeros((h, w, 6), np.float32)
        out[..., 0] = 1.0
        for k in range(6):
            v = map_coordinates(prob[..., k], [b * 256 - 0.5, a * 256 - 0.5], order=1, mode="nearest")
            out[..., k] = np.where(inside, v, out[..., k])
        return out
    sq, s, ox, oy = _letterbox(img, 256)
    prob = _softmax(next(iter(_model("multi")((sq * 2.0 - 1.0)[None]).values()))[0])
    ys = (np.arange(h) + 0.5) * s + oy - 0.5
    xs = (np.arange(w) + 0.5) * s + ox - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([map_coordinates(prob[..., k], [Y, X], order=1, mode="nearest") for k in range(6)],
                    -1).astype(np.float32)


def hair_prob(img: np.ndarray) -> np.ndarray:
    """Hair probability (0..1) for every pixel of a crop around one head (upright, display-encoded RGB 0..1), from
    the 512 x 512 hair model. Its fourth input is last frame's mask in video; a still photo has none (zeros)."""
    from .faces import _letterbox
    h, w = img.shape[:2]
    sq, s, ox, oy = _letterbox(img, 512)
    x = np.concatenate([sq, np.zeros((512, 512, 1), np.float32)], -1)[None]
    z = next(iter(_model("hair")(x).values()))[0]
    prob = _softmax(z)[..., 1]
    ys = (np.arange(h) + 0.5) * s + oy - 0.5
    xs = (np.arange(w) + 0.5) * s + ox - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(prob, [Y, X], order=1, mode="nearest").astype(np.float32)


def head_box(P: np.ndarray, w: int, h: int):
    """A square around one person's head and hair in image pixels (x0, y0, x1, y1), from the body points."""
    up, shw, head, hw = _person_frame(np.c_[P[:, 0] / w, P[:, 1] / h, P[:, 2]], w, h)
    c = head - up * 0.35 * hw
    half = 1.7 * hw
    return (max(0, int(c[0] - half)), max(0, int(c[1] - half)), min(w, int(c[0] + half) + 1),
            min(h, int(c[1] + half) + 1))


# ------------------------------------------------------------------------------------------ BlazePose
_ANCH = None


def detect(img: np.ndarray, thr: float = 0.5) -> list:
    """People in an upright image: [{"roi": (cx, cy, size, angle), "score"}] (image coordinates, the crop the pose
    model wants: centred on the hips, square, upright along the body, 1.25 x the body)."""
    global _ANCH
    from .faces import _anchors, _letterbox, _weighted_nms
    if _ANCH is None:
        _ANCH = _anchors(224, [8, 16, 32, 32, 32], 2)
    sq, s, ox, oy = _letterbox(img, 224)
    out = _model("detect")((sq * 2.0 - 1.0)[None])
    reg = next(v for v in out.values() if v.shape[-1] == 12)[0]
    cls = next(v for v in out.values() if v.shape[-1] == 1)[0, :, 0]
    score = _sigmoid(cls)
    found = []
    for i in np.nonzero(score >= thr)[0]:
        r = reg[i] / 224.0
        cx, cy = r[0] + _ANCH[i, 0], r[1] + _ANCH[i, 1]
        box = np.array([cx - r[2] / 2, cy - r[3] / 2, cx + r[2] / 2, cy + r[3] / 2]) * 224
        kp = (r[4:12].reshape(4, 2) + _ANCH[i]) * 224
        box = np.array([(box[0] - ox) / s, (box[1] - oy) / s, (box[2] - ox) / s, (box[3] - oy) / s])
        kp = np.stack([(kp[:, 0] - ox) / s, (kp[:, 1] - oy) / s], -1)
        found.append({"box": box, "kp": kp, "score": float(score[i]), "model": "pose"})
    out = []
    for d in _weighted_nms(found, 0.3):
        out.append({"roi": roi_from_points(d["kp"][0], d["kp"][1]), "score": d["score"]})
    return out


def roi_from_points(centre, scale_pt, scale: float = 1.25):
    """MediaPipe's alignment rect: centred on the hips, size twice the distance to the scale point, rotated so the
    body is upright in the crop."""
    cx, cy = map(float, centre)
    sx, sy = map(float, scale_pt)
    size = 2.0 * math.hypot(sx - cx, sy - cy) * scale
    ang = math.pi / 2 - math.atan2(-(sy - cy), sx - cx)
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    return cx, cy, size, ang


def pose(img: np.ndarray, roi, refine: int = 1):
    """33 body points ((33, 3): x, y in image coordinates, how sure the point is in view), the model's confidence that
    a person is there, and its person mask (256 x 256 probabilities in the crop) with the crop it belongs to."""
    from .faces import _normalise_crop, _roi_points, crop
    best = None
    for it in range(refine + 1):
        trials = []
        c = crop(img, roi, 256)
        for cc in (c, _normalise_crop(c)):
            out = _model("pose")(cc[None])
            lm = next(v for v in out.values() if v.size == 195).reshape(39, 5)
            conf = float(next(v for v in out.values() if v.size == 1).ravel()[0])
            seg = next(v for v in out.values() if v.shape[1:3] == (256, 256))[0, :, :, 0]
            X, Y = _roi_points(roi, lm[:, 0] / 256.0, lm[:, 1] / 256.0)
            pts = np.stack([X, Y, _sigmoid(lm[:, 3])], -1)
            trials.append((conf, pts, _sigmoid(seg).astype(np.float32), roi))
        cand = max(trials, key=lambda t: t[0])
        if best is None or cand[0] >= best[0] - 0.05:
            best = cand
        if it < refine:
            P = best[1]
            roi = roi_from_points(P[33, :2], P[34, :2])      # the model's own alignment points
    conf, P, seg, roi = best
    return P[:33], conf, seg, roi


def seg_to_image(seg: np.ndarray, roi, h: int, w: int) -> np.ndarray:
    """A crop's person mask laid back onto the image (zero outside the crop)."""
    cx, cy, size, ang = roi
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    dx, dy = xs - cx, ys - cy
    c, s = math.cos(ang), math.sin(ang)
    a = (c * dx + s * dy) / size + 0.5
    b = (-s * dx + c * dy) / size + 0.5
    return map_coordinates(seg, [b * 256 - 0.5, a * 256 - 0.5], order=1, mode="constant", cval=0.0).astype(np.float32)


def find(img: np.ndarray, faces: list | None = None, min_conf: float = 0.5) -> dict:
    """Everyone in an upright image: {"mask": person probability (h, w), "people": [{"pts": (33, 3), "conf"}]}.
    ``faces``: face meshes already found (image coordinates), used to place the pose crop when the body detector
    misses a person."""
    from .faces import variants
    h, w = img.shape[:2]
    vs = variants(img)
    rois = []
    for name, v in vs:
        for d in detect(v):
            d["variant"] = name
            rois.append(d)
    if not rois:                                      # no body found: start from the faces, or the person mask
        for P in faces or []:
            P = np.asarray(P, float)
            top, chin = P[10], P[152]
            hgt = float(np.hypot(*(chin - top)))
            down = (chin - top) / max(hgt, 1e-6)
            centre = chin + down * 2.2 * hgt                 # the hips, roughly, below a face
            rois.append({"roi": roi_from_points(centre, centre - down * 3.0 * hgt), "score": 0.0, "variant": "face"})
    mask = np.maximum(selfie_mask(vs[-1][1]), selfie_mask(vs[0][1])) if len(vs) > 1 else selfie_mask(img)
    if not rois and mask.max() > 0.5:
        ys, xs = np.nonzero(mask > 0.5)
        cx, cy = float(xs.mean()), float(ys.mean())
        size = 1.25 * max(np.ptp(xs), np.ptp(ys))
        rois.append({"roi": (cx, cy, size, 0.0), "score": 0.0, "variant": "mask"})
    people, masks, rois_used = [], [], []
    for d in sorted(rois, key=lambda d: -d["score"]):
        src = dict(vs).get(d.get("variant"), img) if d.get("variant") in dict(vs) else img
        P, conf, seg, roi = pose(src, d["roi"])
        if conf < min_conf:
            continue
        centre = P[[L_SH, R_SH, L_HIP, R_HIP], :2].mean(axis=0)
        if any(np.hypot(*(centre - q["pts"][[L_SH, R_SH, L_HIP, R_HIP], :2].mean(axis=0))) <
               0.3 * max(1.0, float(np.hypot(*(q["pts"][L_SH, :2] - q["pts"][R_SH, :2])))) for q in people):
            continue                                  # the same person found twice
        people.append({"pts": P, "conf": conf, "variant": d.get("variant", "")})
        masks.append(seg_to_image(seg, roi, h, w))
        rois_used.append((src, roi))
    if masks:
        close = np.maximum.reduce(masks)
        inside = gaussian_filter((close > 0).astype(np.float32), 2.0)
        mask = close * inside + mask * (1.0 - inside)   # the closer mask where a body crop covers, else the frame's
    out = {"mask": mask.astype(np.float32), "people": people}
    if multiclass_available():
        with slog.stage("people.parts_model", log, people=len(rois_used)):
            cl = classes(vs[-1][1] if len(vs) > 1 else img)          # the whole photo, for anyone not in a crop
            for src, roi in rois_used:                                # and a closer look at each person
                near = classes(src, roi)
                w_near = 1.0 - near[..., :1]                          # where the crop sees a person, it decides
                cl = near * w_near + cl * (1.0 - w_near)
            cl /= np.maximum(cl.sum(axis=-1, keepdims=True), 1e-6)
        out["classes"] = cl
        out["mask"] = np.maximum(out["mask"], 1.0 - cl[..., 0]).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------ parts at render time
def _seg_dist(xs, ys, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    if n < 1e-9:
        return np.hypot(xs - ax, ys - ay), np.zeros_like(xs)
    t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / n, 0.0, 1.0)
    return np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)), t


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _limb(xs, ys, chain, P, r0: float, r1: float):
    """A soft tube along a limb's points (only the stretches whose ends are in view), radius r0 at the root to r1."""
    out = np.zeros(xs.shape, np.float32)
    n = len(chain) - 1
    for k in range(n):
        a, b = chain[k], chain[k + 1]
        if P[a, 2] < VISIBLE or P[b, 2] < VISIBLE:
            break
        d, t = _seg_dist(xs, ys, P[a, 0], P[a, 1], P[b, 0], P[b, 1])
        r = r0 + (r1 - r0) * (k + t) / n
        out = np.maximum(out, (1.0 - _smooth((d / r - 0.85) / 0.3)).astype(np.float32))
    return out


def _ellipse(xs, ys, c, ax_u, ax_v, ru, rv, feather=0.25):
    dx, dy = xs - c[0], ys - c[1]
    u = (dx * ax_u[0] + dy * ax_u[1]) / ru
    v = (dx * ax_v[0] + dy * ax_v[1]) / rv
    d = np.sqrt(u * u + v * v)
    return (1.0 - _smooth((d - (1.0 - feather)) / feather)).astype(np.float32)


def _person_frame(P: np.ndarray, sw: float, sh: float):
    """(up unit vector, shoulder width, head centre, head width) for one person, in S pixels."""
    pts = P[:, :2] * np.array([sw, sh])
    vis = P[:, 2]
    sh_c = pts[[L_SH, R_SH]].mean(axis=0)
    hip_ok = vis[L_HIP] > VISIBLE and vis[R_HIP] > VISIBLE
    head = pts[[L_EAR, R_EAR]].mean(axis=0) if min(vis[L_EAR], vis[R_EAR]) > 0.2 else pts[[L_EYE, R_EYE]].mean(axis=0)
    if hip_ok:
        up = sh_c - pts[[L_HIP, R_HIP]].mean(axis=0)
    else:
        up = head - sh_c
    up = up / (np.hypot(*up) + 1e-9)
    shw = float(np.hypot(*(pts[L_SH] - pts[R_SH])))
    hw = float(np.hypot(*(pts[L_EAR] - pts[R_EAR])))
    if min(vis[L_EAR], vis[R_EAR]) < 0.2:
        hw = 2.4 * float(np.hypot(*(pts[L_EYE] - pts[R_EYE])))
    shw = max(shw, 1.6 * hw, 4.0)
    return up, shw, head, max(hw, 0.35 * shw, 2.0)


def part_alpha(sh: dict, c) -> np.ndarray:
    """The Person mask shape's alpha for its part, on a render's pixels (masks.Ctx)."""
    from .masks import decode_bitmap, refine
    bm = decode_bitmap(sh["data"], int(sh["w"]), int(sh["h"])).astype(np.float32) / 255.0
    fy = c.ys / c.sh * bm.shape[0] - 0.5
    fx = c.xs / c.sw * bm.shape[1] - 0.5
    P0 = map_coordinates(bm, [fy, fx], order=1, mode="nearest").astype(np.float32)
    person = refine(P0, c, max(6.0, 0.003 * c.long))
    part = sh.get("part", "person")
    if part == "person":
        return person
    if part == "background":
        return (1.0 - person).astype(np.float32)
    if part == "hair" and sh.get("hair_maps"):        # the hair model's own maps, one per head, finest of all
        hair = np.zeros(c.xs.shape, np.float32)
        for hm in sh["hair_maps"]:
            x0, y0, x1, y1 = (float(v) for v in hm["box"])
            b = decode_bitmap(hm["data"], int(hm["w"]), int(hm["h"])).astype(np.float32) / 255.0
            fy3 = (c.ys / c.sh - y0) / max(y1 - y0, 1e-9) * b.shape[0] - 0.5
            fx3 = (c.xs / c.sw - x0) / max(x1 - x0, 1e-9) * b.shape[1] - 0.5
            hair = np.maximum(hair, map_coordinates(b, [fy3, fx3], order=1, mode="constant", cval=0.0))
        return refine(hair * np.clip(person * 1.5, 0.0, 1.0), c, 2.0)
    cls = None
    if sh.get("classes") and part in ("skin", "face", "hair", "clothes"):
        cw, ch = int(sh["cw"]), int(sh["ch"])
        fy2 = c.ys / c.sh * ch - 0.5
        fx2 = c.xs / c.sw * cw - 0.5

        def chan(name):
            d = sh["classes"].get(name)
            if not d:
                return np.zeros(c.xs.shape, np.float32)
            b = decode_bitmap(d, cw, ch).astype(np.float32) / 255.0
            return map_coordinates(b, [fy2, fx2], order=1, mode="nearest").astype(np.float32)
        cls = {k: chan(k) for k in ("hair", "body_skin", "face_skin", "clothes")}
    if cls is not None:                                # the parts model labelled these directly
        if part == "hair":
            return refine(person * cls["hair"], c, 3.0)
        if part == "face":
            return refine(person * cls["face_skin"], c, 3.0)
        if part == "skin":
            return refine(person * np.clip(cls["body_skin"] + cls["face_skin"], 0.0, 1.0), c, 3.0)
        return refine(person * cls["clothes"], c, 3.0)
    people = [np.asarray(p, float).reshape(33, 3) for p in sh.get("people") or []]
    faces = [np.asarray(f, float).reshape(-1, 2) * np.array([c.sw, c.sh]) for f in sh.get("faces") or []]
    xs, ys = c.xs, c.ys
    face = np.zeros(xs.shape, np.float32)
    hair_zone = np.zeros(xs.shape, np.float32)
    hair_top = np.zeros(xs.shape, np.float32)          # above the face: where hair is (nearly) sure to be hair
    hair_long = np.zeros(xs.shape, np.float32)         # below the face, down to the shoulders
    arms = np.zeros(xs.shape, np.float32)
    legs = np.zeros(xs.shape, np.float32)
    torso = np.zeros(xs.shape, np.float32)
    for P in people:
        up, shw, head, hw = _person_frame(P, c.sw, c.sh)
        side = np.array([up[1], -up[0]])
        pts = P[:, :2] * np.array([c.sw, c.sh])
        # the face: the face map's outline if one sits on this head, else an oval from the eyes and mouth
        fo = [f for f in faces if np.hypot(*(f.mean(axis=0) - head)) < 1.2 * hw]
        if fo:
            f = fo[0]
            fc = f.mean(axis=0)
            ru = float(np.max(np.abs((f - fc) @ side))) * 1.05
            rv = float(np.max(np.abs((f - fc) @ up))) * 1.05
        else:
            mouth = pts[[MOUTH_L, MOUTH_R]].mean(axis=0)
            eyes = pts[[L_EYE, R_EYE]].mean(axis=0)
            fc = 0.5 * (mouth + eyes) + up * 0.1 * hw
            ru, rv = 0.52 * hw, 0.72 * hw
        face = np.maximum(face, _ellipse(xs, ys, fc, side, up, ru, rv, 0.2))
        # where hair can be: around the top and sides of the head, and down to the shoulders for long hair
        hc = fc + up * 0.25 * rv
        hz = _ellipse(xs, ys, hc, side, up, 1.9 * ru, 1.75 * rv, 0.35)
        hair_zone = np.maximum(hair_zone, hz)
        above = _smooth(((xs - fc[0]) * up[0] + (ys - fc[1]) * up[1] - 0.8 * rv) / (0.3 * rv))
        hair_top = np.maximum(hair_top, (hz * above).astype(np.float32))
        sh_c = pts[[L_SH, R_SH]].mean(axis=0)
        dx, dy = xs - fc[0], ys - fc[1]
        along = -(dx * up[0] + dy * up[1])                  # downwards from the face centre
        across = np.abs(dx * side[0] + dy * side[1])
        drop = float(np.dot(fc - sh_c, up)) + 0.35 * hw
        long_hair = (1.0 - _smooth((along - drop) / (0.3 * hw))) * _smooth(along / (0.3 * hw)) * \
            (1.0 - _smooth((across - 0.9 * ru * 1.6) / (0.4 * ru)))
        hair_zone = np.maximum(hair_zone, long_hair.astype(np.float32))
        hair_long = np.maximum(hair_long, long_hair.astype(np.float32))
        for chain in ARMS:
            arms = np.maximum(arms, _limb(xs, ys, chain, np.c_[pts, P[:, 2]], 0.24 * shw, 0.14 * shw))
        for chain in LEGS:
            legs = np.maximum(legs, _limb(xs, ys, chain, np.c_[pts, P[:, 2]], 0.33 * shw, 0.18 * shw))
        # the torso: between the shoulders and the hips (or down to the frame's edge when the hips are out of view)
        hips_ok = P[L_HIP, 2] > VISIBLE and P[R_HIP, 2] > VISIBLE
        bot = pts[[L_HIP, R_HIP]].mean(axis=0) if hips_ok else sh_c - up * 3.0 * shw
        half = 0.5 * shw * 1.15
        d_up = (xs - bot[0]) * up[0] + (ys - bot[1]) * up[1]
        length = float(np.dot(sh_c - bot, up))
        d_side = np.abs((xs - 0.5 * (sh_c[0] + bot[0])) * side[0] + (ys - 0.5 * (sh_c[1] + bot[1])) * side[1])
        t = (_smooth((d_up + 0.15 * shw) / (0.2 * shw)) * (1.0 - _smooth((d_up - length - 0.05 * shw) / (0.2 * shw)))
             * (1.0 - _smooth((d_side - half) / (0.2 * shw))))
        torso = np.maximum(torso, t.astype(np.float32))
    face = face * person
    # skin: this person's own skin colour (from the face, else the arms), in any light
    lab = c.lab
    L = np.maximum(lab[..., 0], 1e-3)
    ra, rb = lab[..., 1] / L, lab[..., 2] / L
    ref = (face > 0.6) if (face > 0.6).sum() > 30 else ((arms * person) > 0.6)
    lg = np.log2(np.maximum(c.rgb @ np.array([0.288, 0.712, 0.0001], np.float32), 1e-6))
    if ref.sum() > 30:
        ma, mb, ml = float(np.median(ra[ref])), float(np.median(rb[ref])), float(np.median(lg[ref]))
        dcol = ((ra - ma) ** 2 + (rb - mb) ** 2) / 0.06 ** 2
        skin_like = np.exp(-0.5 * (dcol + ((lg - ml) / 2.5) ** 2))       # lenient: a candle-lit side is stops darker
        skin_strict = np.exp(-0.5 * (dcol + ((lg - ml) / 1.0) ** 2))     # for telling dark brown hair from skin
    else:
        from .masks import skin_weight
        skin_like = skin_strict = skin_weight(lab)
    skin_like, skin_strict = skin_like.astype(np.float32), skin_strict.astype(np.float32)
    if part == "face":
        return refine(face, c, 4.0)
    # hair: above the face it is hair unless it is plainly skin (a bald head, a high forehead); at the sides and
    # below, only what is not skin-coloured, and below the face only what looks like the hair on top
    side_zone = np.clip(hair_zone - hair_top, 0.0, 1.0)
    hair = person * (1.0 - face) * np.clip(hair_top * (1.0 - 0.5 * skin_strict) + side_zone * (1.0 - skin_strict),
                                           0.0, 1.0)
    top = (hair * hair_top) > 0.5
    if top.sum() > 30:
        ha, hb, hl = float(np.median(ra[top])), float(np.median(rb[top])), float(np.median(lg[top]))
        like = np.exp(-0.5 * (((ra - ha) ** 2 + (rb - hb) ** 2) / 0.08 ** 2 + ((lg - hl) / 1.2) ** 2))
        below = np.clip(hair_long - hair_top, 0.0, 1.0)
        hair = hair * (1.0 - below * (1.0 - like.astype(np.float32)))
    skin = person * np.maximum(skin_like * (1.0 - hair), face * skin_like ** 0.5)
    if part == "skin":
        return refine(skin, c, 3.0)
    if part == "hair":
        return refine(hair, c, 3.0)
    if part == "clothes":
        return refine(person * (1.0 - np.maximum(skin_like, hair)) * (1.0 - face), c, 3.0)
    if part == "arms":
        return refine(person * arms * (1.0 - face), c, 3.0)
    if part == "legs":
        return refine(person * legs, c, 3.0)
    if part == "torso":
        return refine(person * torso * (1.0 - arms) * (1.0 - legs) * (1.0 - face), c, 3.0)
    raise ValueError(f"unknown person part {part!r}")

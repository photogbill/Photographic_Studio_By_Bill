"""The MyPaint brush engine, ported to Python (BUILD_PLAN_PAINT P2), and MyPaint brushes (.myb) read as they are.

Ported from libmypaint (Martin Renold and contributors, ISC licence; settings table in mypaint_settings.json, copied
unchanged from libmypaint's brushsettings.json). What is ported, with libmypaint's own formulas:
- every setting as a base value plus the sum of its input curves (mypaint-mapping.c), all 65 settings, 18 inputs;
- the stroke: slow position tracking, tracking noise, dabs counted by actual radius, basic radius and time
  (count_dabs_to), linear stepping between pen events, the low-pass filters for speed 1 and 2, direction and offset
  by speed, the stroke input (threshold, duration, hold time), custom input (update_states_and_setting_values);
- each dab (prepare_and_draw_dab): opacity with "opaque linearize", offset and radius by random with its alpha
  correction, anti-aliasing widening, elliptical dabs, HSV and HSL colour changes, smudge with smudge length and
  smudge radius, eraser, lock alpha; the dab's profile from mypaint-tiled-surface (two straight segments in r^2).
Not ported yet (said here so nobody assumes it): libmypaint's spectral "paint mode" (WGM) for dabs; smudge mixes in
Classic when paint mode is 0 and through our Kubelka-Munk (colour.pigment_mix) otherwise, which is close but not
WGM; smudge buckets; gridmap; the directional offsets (offset_x / offset_y / offset_angle*); colorize; posterize;
snap to pixel. Brushes that lean on those look different from MyPaint.
"""
from __future__ import annotations

import colorsys
import json
import math
import random as _random
from pathlib import Path

import numpy as np

from .smoothing import Sample
from .tiles import TILE, Layer, LayerLocked

_TABLE = json.loads((Path(__file__).with_name("mypaint_settings.json")).read_text(encoding="utf-8"))
SETTINGS = [s["internal_name"] for s in _TABLE["settings"]]
DEFAULTS = {s["internal_name"]: float(s["default"]) for s in _TABLE["settings"]}
INPUTS = [i["id"] for i in _TABLE["inputs"]]
RADIUS_MIN, RADIUS_MAX = 0.2, 1000.0


class MyPaintBrush:
    """A MyPaint brush: base values and input curves per setting."""

    def __init__(self, name: str = "MyPaint brush"):
        self.name = name
        self.base = dict(DEFAULTS)
        # setting -> {input: [(x, y), ...]}; libmypaint's defaults (mypaint_brush_from_defaults): opacity follows
        # pressure through "opaque multiply"
        self.curves: dict = {"opaque_multiply": {"pressure": [(0.0, 0.0), (1.0, 1.0)]}}
        self.comment = ""

    @classmethod
    def from_myb(cls, text: str, name: str = "MyPaint brush") -> "MyPaintBrush":
        d = json.loads(text)
        if int(d.get("version", 0)) != 3:
            raise ValueError("only MyPaint brushes of version 3 (MyPaint 1.0 and later, .myb as JSON) can be read")
        b = cls(name)
        b.comment = str(d.get("comment", "") or "")
        for sname, sv in d.get("settings", {}).items():
            if sname not in b.base:
                continue                            # a setting from a newer MyPaint: ignored
            b.base[sname] = float(sv.get("base_value", b.base[sname]))
            b.curves.pop(sname, None)               # a setting in the file replaces its curves
            ins = {}
            for iname, pts in (sv.get("inputs") or {}).items():
                if iname in INPUTS and len(pts) >= 2:
                    ins[iname] = [(float(x), float(y)) for x, y in pts]
            if ins:
                b.curves[sname] = ins
        return b

    def to_myb(self) -> str:
        out = {"version": 3, "comment": self.comment, "settings": {}}
        for s in SETTINGS:
            out["settings"][s] = {"base_value": self.base[s], "inputs": {k: [list(p) for p in v]
                                                                          for k, v in self.curves.get(s, {}).items()}}
        return json.dumps(out, indent=1)

    def value(self, setting: str, inputs: dict) -> float:
        """Base value plus each input curve's value (mypaint_mapping_calculate)."""
        r = self.base[setting]
        for iname, pts in self.curves.get(setting, {}).items():
            x = inputs.get(iname, 0.0)
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            i = 2
            while i < len(pts) and x > x1:
                x0, y0 = x1, y1
                x1, y1 = pts[i]
                i += 1
            r += y0 if (x0 == x1 or y0 == y1) else (y1 * (x - x0) + y0 * (x1 - x)) / (x1 - x0)
        return r

    def is_constant(self, setting: str) -> bool:
        return not self.curves.get(setting)


def load_myb(path: str) -> MyPaintBrush:
    p = Path(path)
    return MyPaintBrush.from_myb(p.read_text(encoding="utf-8"), p.stem.replace("_", " "))


def _exp_decay(T: float, t: float) -> float:
    return math.exp(-t / T) if T > 0.001 else 0.0


def _mod(a, n):
    return a - n * math.floor(a / n)


def _ang_diff(a, b):
    d = _mod(b - a + 180.0, 360.0) - 180.0
    return d


class MyPaintStroke:
    """One stroke with a MyPaint brush on one layer; the same ``add`` / ``finish`` as brush.Stroke."""

    def __init__(self, layer: Layer, brush: MyPaintBrush, colour=(0, 0, 0), undo=None, eraser: bool | None = None,
                 seed: int | None = None, sample=None):
        if layer.is_group:
            raise LayerLocked(f"\"{layer.name}\" is a group. Pick a layer inside it to paint.")
        if layer.locked:
            raise LayerLocked(f"The layer \"{layer.name}\" is locked. Unlock it in the Layers panel to paint on it.")
        self.layer, self.b, self.undo = layer, brush, undo
        h, s, v = colorsys.rgb_to_hsv(*[float(c) for c in colour[:3]])
        self.base = dict(brush.base)
        self.base.update(color_h=h, color_s=s, color_v=v)
        if eraser:
            self.base["eraser"] = 1.0
        self.rng = _random.Random(seed)
        self.random_input = self.rng.random()
        self.sample = sample if sample is not None else layer.read
        self.orig: dict = {}
        self.dabs = 0
        self._dirty = None
        self.s = None                                # the state, made at the first sample
        g = []
        for k in ("speed1_gamma", "speed2_gamma"):
            gamma = math.exp(self.base[k])
            c1 = math.log(45.0 + gamma)
            m = 0.015 * (45.0 + gamma)
            g.append((gamma, m, 0.5 - m * c1))
        self.speed_map = g
        self.smudge = [0.0, 0.0, 0.0, 0.0]          # carried paint (premultiplied, as libmypaint's legacy bucket)
        self.set: dict = {}

    def _setting_values(self, inputs):
        b = self.b
        out = {}
        for name in SETTINGS:
            r = self.base[name]
            for iname, pts in b.curves.get(name, {}).items():
                x = inputs.get(iname, 0.0)
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                i = 2
                while i < len(pts) and x > x1:
                    x0, y0 = x1, y1
                    x1, y1 = pts[i]
                    i += 1
                r += y0 if (x0 == x1 or y0 == y1) else (y1 * (x - x0) + y0 * (x1 - x)) / (x1 - x0)
            out[name] = r
        return out

    # ---- pen samples in
    def add(self, smp: Sample):
        self._dirty = None
        decl = 90.0 - min(90.0, math.hypot(smp.tilt_x, smp.tilt_y))
        asc = math.degrees(math.atan2(smp.tilt_y, smp.tilt_x)) if (smp.tilt_x or smp.tilt_y) else 0.0
        self._stroke_to(smp.x, smp.y, max(0.0, smp.pressure), decl, asc, smp.t)
        return self._dirty

    def finish(self):
        self._dirty = None
        self.layer.prune()
        return self._dirty

    def _reset(self, x, y, p, t):
        self.s = {"x": x, "y": y, "pressure": p, "ax": x, "ay": y, "decl": 90.0, "asc": 0.0, "t": t,
                  "speed1": 0.0, "speed2": 0.0, "dx_slow": 0.0, "dy_slow": 0.0, "dir_dx": 0.0, "dir_dy": 0.0,
                  "stroke": 1.0, "stroke_started": False, "custom": 0.0, "partial": 0.0,
                  "radius": math.exp(self.base["radius_logarithmic"]), "ratio": 1.0, "angle": 0.0,
                  "dpbr": self.base["dabs_per_basic_radius"], "dpar": self.base["dabs_per_actual_radius"],
                  "dps": self.base["dabs_per_second"]}
        self.set = self._setting_values({})

    def _stroke_to(self, x, y, pressure, decl, asc, t):
        if self.s is None:
            self._reset(x, y, pressure, t)
            return
        S = self.s
        dtime = max(1e-4, t - S["t"])
        S["t"] = t
        base_r = math.exp(self.base["radius_logarithmic"])
        if self.base["tracking_noise"]:
            noise = base_r * self.base["tracking_noise"]
            if noise > 0.001:
                x += noise * self.rng.gauss(0, 1)
                y += noise * self.rng.gauss(0, 1)
        fac = 1.0 - _exp_decay(self.base["slow_tracking"], 100.0 * dtime)
        x = S["x"] + (x - S["x"]) * fac
        y = S["y"] + (y - S["y"]) * fac
        moved = S["partial"]
        left = dtime
        todo = self._count(x, y, left)
        n = 0
        while moved + todo >= 1.0 and n < 2000:
            if moved > 0:
                ddab, moved = 1.0 - moved, 0.0
            else:
                ddab = 1.0
            f = ddab / todo
            self._update(ddab, f * (x - S["x"]), f * (y - S["y"]), f * (pressure - S["pressure"]),
                         f * (decl - S["decl"]), f * _ang_diff(S["asc"], asc), f * left)
            self._dab()
            self.random_input = self.rng.random()
            left -= f * left
            todo = self._count(x, y, left)
            n += 1
        self._update(todo, x - S["x"], y - S["y"], pressure - S["pressure"], decl - S["decl"],
                     _ang_diff(S["asc"], asc), left)
        S["partial"] = moved + todo

    def _count(self, x, y, dt):
        S = self.s
        base_r = min(max(math.exp(self.base["radius_logarithmic"]), RADIUS_MIN), RADIUS_MAX)
        dx, dy = x - S["x"], y - S["y"]
        if S["ratio"] > 1.0:
            a = math.radians(S["angle"])
            cs, sn = math.cos(a), math.sin(a)
            yyr, xxr = (dy * cs - dx * sn) * S["ratio"], dy * sn + dx * cs
            dist = math.hypot(yyr, xxr)
        else:
            dist = math.hypot(dx, dy)
        return dist / S["radius"] * S["dpar"] + dist / base_r * S["dpbr"] + dt * S["dps"]

    def _update(self, ddab, dx, dy, dp, ddecl, dasc, dtime):
        S = self.s
        dtime = dtime if dtime > 0 else 0.001
        S["x"] += dx
        S["y"] += dy
        S["pressure"] = max(0.0, S["pressure"] + dp)
        S["decl"] += ddecl
        S["asc"] += dasc
        base_r = math.exp(self.base["radius_logarithmic"])
        p = S["pressure"]
        lim, thr = 0.0001, self.base["stroke_threshold"]
        if not S["stroke_started"] and p > thr + lim:
            S["stroke_started"], S["stroke"] = True, 0.0
        elif S["stroke_started"] and p <= thr * 0.9 + lim:
            S["stroke_started"] = False
        ndx, ndy = dx / dtime, dy / dtime
        nspeed = math.hypot(ndx, ndy)
        ndist = math.hypot(ndx / base_r, ndy / base_r) * dtime
        (g0, m0, q0), (g1, m1, q1) = self.speed_map
        inputs = {
            "pressure": p * math.exp(self.base["pressure_gain_log"]),
            "speed1": math.log(g0 + S["speed1"]) * m0 + q0,
            "speed2": math.log(g1 + S["speed2"]) * m1 + q1,
            "random": self.random_input,
            "stroke": min(S["stroke"], 1.0),
            "direction": _mod(math.degrees(math.atan2(S["dir_dy"], S["dir_dx"])) + 180.0, 180.0),
            "direction_angle": _mod(math.degrees(math.atan2(S["dir_dy"], S["dir_dx"])) + 360.0, 360.0),
            "tilt_declination": S["decl"],
            "tilt_ascension": _mod(S["asc"] + 180.0, 360.0) - 180.0,
            "viewzoom": 0.0, "brush_radius": self.base["radius_logarithmic"], "custom": S["custom"],
        }
        st = self.set = self._setting_values(inputs)
        S["dpbr"], S["dpar"], S["dps"] = st["dabs_per_basic_radius"], st["dabs_per_actual_radius"], \
            st["dabs_per_second"]
        f = 1.0 - _exp_decay(st["slow_tracking_per_dab"], ddab)
        S["ax"] += (S["x"] - S["ax"]) * f
        S["ay"] += (S["y"] - S["ay"]) * f
        S["speed1"] += (nspeed - S["speed1"]) * (1.0 - _exp_decay(st["speed1_slowness"], dtime))
        S["speed2"] += (nspeed - S["speed2"]) * (1.0 - _exp_decay(st["speed2_slowness"], dtime))
        tc = max(0.002, math.exp(st["offset_by_speed_slowness"] * 0.01) - 1.0)
        f = 1.0 - _exp_decay(tc, dtime)
        S["dx_slow"] += (ndx - S["dx_slow"]) * f
        S["dy_slow"] += (ndy - S["dy_slow"]) * f
        step = math.hypot(dx, dy)
        f = 1.0 - _exp_decay(math.exp(st["direction_filter"] * 0.5) - 1.0, step)
        ox, oy, ddx, ddy = S["dir_dx"], S["dir_dy"], dx, dy
        if (ox - ddx) ** 2 + (oy - ddy) ** 2 > (ox + ddx) ** 2 + (oy + ddy) ** 2:
            ddx, ddy = -ddx, -ddy
        S["dir_dx"] += (ddx - ox) * f
        S["dir_dy"] += (ddy - oy) * f
        S["custom"] += (st["custom_input"] - S["custom"]) * (1.0 - _exp_decay(st["custom_input_slowness"], 0.1))
        freq = math.exp(-st["stroke_duration_logarithmic"])
        stroke = max(0.0, S["stroke"] + ndist * freq)
        wrap = 1.0 + max(0.0, st["stroke_holdtime"])
        S["stroke"] = 1.0 if (stroke >= wrap and wrap > 10.9) else (math.fmod(stroke, wrap) if stroke >= wrap
                                                                     else stroke)
        S["radius"] = min(max(math.exp(st["radius_logarithmic"]), RADIUS_MIN), RADIUS_MAX)
        S["ratio"] = st["elliptical_dab_ratio"]
        S["angle"] = _mod(st["elliptical_dab_angle"] + 180.0, 180.0) - 180.0

    # ---- one dab
    def _dab(self):
        S, st, B = self.s, self.set, self.base
        opaque = min(max(max(0.0, st["opaque"]) * st["opaque_multiply"], 0.0), 1.0)
        lin = B["opaque_linearize"]
        if lin:
            dpp = max(1.0, (S["dpar"] + S["dpbr"]) * 2.0)
            dpp = 1.0 + lin * (dpp - 1.0)
            opaque = 1.0 - (1.0 - opaque) ** (1.0 / dpp)
        x, y = S["ax"], S["ay"]
        base_r = math.exp(B["radius_logarithmic"])
        if st["offset_by_speed"]:
            x += S["dx_slow"] * st["offset_by_speed"] * 0.1 / 1.0
            y += S["dy_slow"] * st["offset_by_speed"] * 0.1 / 1.0
        if st["offset_by_random"]:
            amp = max(0.0, st["offset_by_random"])
            x += self.rng.gauss(0, 1) * amp * base_r
            y += self.rng.gauss(0, 1) * amp * base_r
        radius = S["radius"]
        if st["radius_by_random"]:
            radius = min(max(math.exp(st["radius_logarithmic"] + self.rng.gauss(0, 1) * st["radius_by_random"]),
                             RADIUS_MIN), RADIUS_MAX)
            corr = (S["radius"] / radius) ** 2
            if corr <= 1.0:
                opaque *= corr
        r, g, b = colorsys.hsv_to_rgb(_mod(B["color_h"], 1.0), min(max(B["color_s"], 0), 1),
                                      min(max(B["color_v"], 0), 1))
        sm_len = st["smudge_length"]
        if sm_len < 1.0 and (st["smudge"] != 0.0 or not self.b.is_constant("smudge")):
            self._pick_up(x, y, radius, sm_len)
        target_a = 1.0
        sv = st["smudge"]
        if sv > 0.0:
            sf = min(1.0, sv)
            target_a = min(max((1.0 - sf) + sf * self.smudge[3], 0.0), 1.0)
            if target_a > 0:
                if st["paint_mode"] > 0 and self.smudge[3] > 1e-6:
                    from .colour import pigment_mix
                    sc = np.array(self.smudge[:3], np.float32) / self.smudge[3]
                    r, g, b = (float(v) for v in pigment_mix(np.array([r, g, b], np.float32), sc, sf))
                else:
                    r = (sf * self.smudge[0] + (1 - sf) * r) / target_a
                    g = (sf * self.smudge[1] + (1 - sf) * g) / target_a
                    b = (sf * self.smudge[2] + (1 - sf) * b) / target_a
            else:
                r, g, b = 1.0, 0.0, 0.0
        if st["eraser"]:
            target_a *= (1.0 - st["eraser"])
        if st["change_color_h"] or st["change_color_hsv_s"] or st["change_color_v"]:
            h, s_, v = colorsys.rgb_to_hsv(r, g, b)
            h += st["change_color_h"]
            s_ += s_ * v * st["change_color_hsv_s"]
            v += st["change_color_v"]
            r, g, b = colorsys.hsv_to_rgb(_mod(h, 1.0), min(max(s_, 0), 1), min(max(v, 0), 1))
        if st["change_color_l"] or st["change_color_hsl_s"]:
            h, l, s_ = colorsys.rgb_to_hls(r, g, b)
            l += st["change_color_l"]
            s_ += s_ * min(abs(1.0 - l), abs(l)) * 2.0 * st["change_color_hsl_s"]
            r, g, b = colorsys.hls_to_rgb(_mod(h, 1.0), min(max(l, 0), 1), min(max(s_, 0), 1))
        hardness = min(max(st["hardness"], 0.0), 1.0)
        fade = radius * (1.0 - hardness)
        aa = st["anti_aliasing"]
        if fade < aa:
            optical = radius - (1.0 - hardness) * radius / 2.0
            hardness = (optical - aa / 2.0) / (optical + aa / 2.0)
            radius = aa / (1.0 - hardness)
        if opaque <= 0 or radius <= 0:
            return
        self.dabs += 1
        self._draw(x, y, radius, (r, g, b), opaque, hardness, target_a, S["ratio"], S["angle"],
                   st["lock_alpha"])

    def _mask(self, x, y, radius, hardness, ratio, angle):
        r_ext = radius * 1.0 + 1.5
        x0, y0 = int(math.floor(x - r_ext)), int(math.floor(y - r_ext))
        n = int(math.ceil(2 * r_ext)) + 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        dx, dy = xx + x0 + 0.5 - x, yy + y0 + 0.5 - y
        a = math.radians(angle)
        cs, sn = math.cos(a), math.sin(a)
        yyr = (dy * cs - dx * sn) * max(ratio, 1.0)
        xxr = dy * sn + dx * cs
        rr = (yyr * yyr + xxr * xxr) / (radius * radius)
        h = min(max(hardness, 1e-4), 0.9999)
        opa = np.where(rr <= h, 1.0 + rr * -(1.0 / h - 1.0), h / (1.0 - h) + rr * -(h / (1.0 - h)))
        opa = np.where(rr <= 1.0, np.clip(opa, 0, 1), 0.0).astype(np.float32)
        return opa, x0, y0

    def _pick_up(self, x, y, radius, sm_len):
        """libmypaint's legacy smudge: the colour under the smudge radius, weighted by the dab, joins the carried
        paint by (1 - smudge length)."""
        sr = min(max(radius * math.exp(self.set["smudge_radius_log"]), RADIUS_MIN), RADIUS_MAX)
        m, x0, y0 = self._mask(x, y, sr, 0.5, 1.0, 0.0)
        px = self.sample(x0, y0, m.shape[1], m.shape[0])
        tot = float(m.sum())
        if tot <= 0:
            return
        c = (px * m[..., None]).sum(axis=(0, 1)) / tot
        a = float(c[3])
        lim = self.set["smudge_transparency"]
        if (lim > 0.0 and a < lim) or (lim < 0.0 and a > -lim):
            return
        rgb = c[:3] / a if a > 1e-6 else np.zeros(3)
        uf = max(0.01, sm_len)
        fac_new = (1.0 - uf) * a
        for i in range(3):
            self.smudge[i] = uf * self.smudge[i] + fac_new * float(rgb[i])
        self.smudge[3] = min(max(uf * self.smudge[3] + fac_new, 0.0), 1.0)

    def _draw(self, x, y, radius, rgb, opaque, hardness, target_a, ratio, angle, lock_alpha):
        m, x0, y0 = self._mask(x, y, radius, hardness, ratio, angle)
        h, w = m.shape
        lay = self.layer
        col = np.array(rgb, np.float32)
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                key = (tx, ty)
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
                mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                if not mm.any():
                    continue
                tile = lay.tiles.get(key)
                if tile is None and (target_a < 1.0 or lay.alpha_lock or lock_alpha >= 1.0):
                    continue
                if key not in self.orig:
                    if self.undo is not None:
                        self.undo.touch(lay, key)
                    self.orig[key] = None if tile is None else tile.copy()
                if tile is None:
                    tile = lay.tile(key, True)
                sl = (slice(ay0 - oy, ay1 - oy), slice(ax0 - ox, ax1 - ox))
                dst = tile[sl]
                opa_a = (mm * opaque)[..., None]                 # libmypaint: normal and eraser in one
                opa_b = 1.0 - opa_a
                out = np.empty_like(dst)
                out[..., :3] = opa_a * target_a * col + opa_b * dst[..., :3]
                out[..., 3:4] = opa_a * target_a + opa_b * dst[..., 3:4]
                la = max(lock_alpha, 1.0 if lay.alpha_lock else 0.0)
                if la > 0:                                       # keep the layer's alpha, recolour only
                    da = dst[..., 3:4]
                    straight = out[..., :3] / np.maximum(out[..., 3:4], 1e-8)
                    locked = np.concatenate([straight * da, da], -1)
                    out = out * (1 - la) + locked * la
                tile[sl] = out.astype(np.float32)
                d = self._dirty
                self._dirty = (ax0, ay0, ax1, ay1) if d is None else (min(d[0], ax0), min(d[1], ay0),
                                                                      max(d[2], ax1), max(d[3], ay1))

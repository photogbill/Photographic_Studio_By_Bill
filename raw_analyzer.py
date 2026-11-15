#!/usr/bin/env python3
"""
Photographic Studio By Bill - Raw Analyzer (step 1 of the denoiser project)

Measures a raw file's noise BEFORE anything touches it:
  * read noise and black level from the masked "optical black" border (per channel)
  * row (horizontal) banding: size, slow-drift share, periodic lines
  * column (vertical) pattern: size and periodic lines (cyclostationary signature)
  * single-image noise model (photon transfer) from the two green channels
  * shadow clipping risk: pixels below black, and where a clipped black would sit
  * hot-pixel candidates (isolated vs point-like), green imbalance

Usage:
  python raw_analyzer.py                 -> every raw file in this folder (and subfolders)
  python raw_analyzer.py FILE_OR_FOLDER  -> just those
  add --no-open to skip opening the report

Writes (beside this script, never on C:):
  reports\\raw_report_YYYYMMDD_HHMMSS.html   the visual report
  reports\\<rawname>.json                    machine-readable measurements (for the denoiser)
  logs\\raw_analyzer.log                     full debug log
"""
import io
import os
import sys
import json
import math
import time
import html
import struct
import logging
import datetime
import traceback

import numpy as np

VERSION = "0.2"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
REPORT_DIR = os.path.join(HERE, "reports")
RAW_EXT = (".cr2", ".cr3", ".dng", ".arw", ".pef", ".nef", ".raf", ".orf", ".rw2")
SKIP_DIRS = {".venv", "reports", "logs", "__pycache__"}

log = logging.getLogger("raw_analyzer")

# Provisional traffic-light thresholds. To be tuned against William's eye.
THRESH = {
    "row_pct": (5.0, 10.0),        # row banding, % of random noise
    "col_pct": (5.0, 10.0),        # column pattern, % of random noise
    "below_black_pct": (1.0, 5.0), # worst channel, % of pixels below black
    "floor_255": (4.0, 10.0),      # clipped-black lift, worst channel, 0-255 display units
    "hot_per_mp": (5.0, 50.0),     # isolated hot-pixel candidates per megapixel
}


# ----------------------------------------------------------------------------- logging
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "raw_analyzer.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)


# ----------------------------------------------------------------------------- small stats helpers
def clipped_stats(x, k=4.0, iters=6, max_n=2_000_000):
    """Sigma-clipped mean and std (astronomy style). Robust to outliers, not quantized like MAD."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size > max_n:
        x = x[:: int(math.ceil(x.size / max_n))]
    m, s = float(np.median(x)), float(x.std())
    for _ in range(iters):
        if s <= 0:
            break
        sel = np.abs(x - m) < k * s
        if sel.sum() < 10:
            break
        m2, s2 = float(x[sel].mean()), float(x[sel].std())
        done = abs(s2 - s) <= 1e-6 * max(s, 1e-12)
        m, s = m2, s2
        if done:
            break
    return m, s


def finite(v):
    return v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))


_CHI2_Q25 = None


def chi2_q25_63():
    """25th percentile of chi2(63)/63, used to un-bias the low-percentile block variance."""
    global _CHI2_Q25
    if _CHI2_Q25 is None:
        rng = np.random.default_rng(12345)
        _CHI2_Q25 = float(np.percentile(rng.chisquare(63, 400_000) / 63.0, 25))
    return _CHI2_Q25


def spectrum(series, white_var, thresh_db=13.0, max_lines=8):
    """Periodogram of a 1-D series. Returns power in dB above pure white noise, plus detected lines.
    A 'line' is a bin standing thresh_db above its local median: a repeating (cyclostationary) pattern."""
    y = np.asarray(series, np.float64)
    y = y - y.mean()
    n = len(y)
    if n < 64 or not finite(white_var) or white_var <= 0:
        return None
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(n)
    ref_white = n * white_var
    k = 31
    pad = np.pad(P, (k // 2, k // 2), mode="reflect")
    base = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    base = np.maximum(base, 1e-30)
    rel_local = 10 * np.log10(np.maximum(P, 1e-30) / base)
    lines = []
    for i in np.argsort(rel_local[3:])[::-1] + 3:      # ignore the 2 lowest bins (whole-frame tilt)
        if rel_local[i] < thresh_db:
            break
        if any(abs(i - j) <= 2 for j, _, _ in lines):
            continue
        lines.append((int(i), float(1.0 / f[i]), float(rel_local[i])))
        if len(lines) >= max_lines:
            break
    excess = np.maximum(P[1:] - ref_white, 0.0)
    slow = f[1:] < 1.0 / 64.0
    slow_share = float(excess[slow].sum() / excess.sum()) if excess.sum() > 0 else float("nan")
    # display copy: ~600 log-spaced segments, keeping each segment's peak so spikes survive
    db = 10 * np.log10(np.maximum(P[1:], 1e-30) / ref_white)
    per = 1.0 / f[1:]
    edges = np.unique(np.geomspace(1, len(db), 600).astype(int) - 1)
    dper, ddb = [], []
    for e0, e1 in zip(edges, np.append(edges[1:], len(db))):
        if e1 <= e0:
            continue
        k = e0 + int(np.argmax(db[e0:e1]))
        dper.append(float(per[k]))
        ddb.append(float(db[k]))
    return dict(period=dper, db_white=ddb, lines=[(p, d) for _, p, d in lines], slow_share=slow_share)


# ----------------------------------------------------------------------------- metadata
def _boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > end:
                break
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr:
            break
        yield typ, i + hdr, min(i + size, end)
        i += size


def _cr3_tiff_blobs(fh):
    """CR3 is an ISO-BMFF container; Canon keeps TIFF-format metadata in CMT1 (IFD0) and CMT2 (EXIF)."""
    data = fh.read(16 * 1024 * 1024)
    for typ, s, e in _boxes(data, 0, len(data)):
        if typ == b"moov":
            for t2, s2, e2 in _boxes(data, s, e):
                if t2 == b"uuid":
                    for t3, s3, e3 in _boxes(data, s2 + 16, e2):
                        if t3 in (b"CMT1", b"CMT2"):
                            yield data[s3:e3]


def read_exif(path):
    info = {}
    try:
        import exifread
    except ImportError:
        log.warning("exifread not installed - camera metadata will be missing")
        return info
    try:
        tags = {}
        with open(path, "rb") as fh:
            if path.lower().endswith(".cr3"):
                for blob in _cr3_tiff_blobs(fh):
                    tags.update(exifread.process_file(io.BytesIO(blob), details=False))
            else:
                tags = exifread.process_file(fh, details=False)

        def get(name):
            for k, v in tags.items():
                if k.split(" ")[-1] == name:
                    return str(v)
            return None

        for key, name in [("make", "Make"), ("model", "Model"), ("iso", "ISOSpeedRatings"),
                          ("exposure", "ExposureTime"), ("fnumber", "FNumber"), ("focal", "FocalLength"),
                          ("date", "DateTimeOriginal"), ("lens", "LensModel")]:
            v = get(name)
            if v is not None:
                info[key] = v.strip()
        if "iso" not in info:
            v = get("RecommendedExposureIndex") or get("PhotographicSensitivity")
            if v:
                info["iso"] = v
        # True ISO (BUILD_PLAN.md section 4): Canon writes 65535 in the standard EXIF ISO field above
        # ISO 51200. The Studio metadata reader checks the 32-bit EXIF fields and Canon's maker notes.
        try:
            from studio.engine.io.metadata import read_metadata
            md = read_metadata(path)
            if md.iso:
                if info.get("iso") and str(info["iso"]).strip() != str(int(round(md.iso))):
                    log.info("  true ISO %d (EXIF field said %s; from %s)", round(md.iso), info["iso"], md.iso_source)
                info["iso"] = str(int(round(md.iso)))
                info["iso_source"] = md.iso_source
            if md.camera_temperature_c is not None:
                info["camera_temp_c"] = md.camera_temperature_c
        except Exception as e:
            log.debug("true-ISO reader unavailable or failed: %s", e)
        try:
            fn = info.get("fnumber")
            if fn is not None:
                a, _, b = fn.partition("/")
                fval = float(a) / float(b) if b else float(a)
                info["fnumber_val"] = fval
                if fval == 0:
                    info["manual_lens"] = True
        except Exception:
            pass
    except Exception as e:
        log.warning("metadata read failed for %s: %s", path, e)
        log.debug(traceback.format_exc())
    return info


# ----------------------------------------------------------------------------- the analysis
def analyze(path):
    import rawpy
    t0 = time.time()
    res = {"file": path, "name": os.path.basename(path), "notes": [], "ok": True}
    res["exif"] = read_exif(path)
    log.info("  decoding ...")
    with rawpy.imread(path) as r:
        s = r.sizes
        raw = np.array(r.raw_image, dtype=np.float32)
        try:
            col = np.array(r.raw_colors, dtype=np.uint8)
        except Exception:
            col = None
        pattern = np.array(r.raw_pattern)
        desc = r.color_desc.decode("ascii", "replace") if isinstance(r.color_desc, bytes) else str(r.color_desc)
        black = np.array(r.black_level_per_channel, dtype=np.float64)
        white = float(r.white_level)
        wb = [float(x) for x in r.camera_whitebalance]
    tm, lm, h, w = int(s.top_margin), int(s.left_margin), int(s.height), int(s.width)
    res["sizes"] = dict(raw_h=int(s.raw_height), raw_w=int(s.raw_width), h=h, w=w, top=tm, left=lm)
    res["cfa"] = dict(pattern=pattern.tolist(), desc=desc)
    res["libraw"] = dict(black=black.tolist(), white=white, wb=wb)
    log.debug("sizes %s pattern %s desc %s black %s white %s wb %s", res["sizes"], pattern.tolist(), desc, black, white, wb)

    bayer = pattern.shape == (2, 2) and sorted(pattern.ravel().tolist()) == [0, 1, 2, 3] and len(desc) >= 4
    if not bayer:
        res["notes"].append("Not a 2x2 Bayer layout (pattern %s): plane-based measurements skipped." % pattern.tolist())
    if col is None or col.shape != raw.shape:
        log.debug("building raw_colors from pattern")
        P = pattern.shape[0]
        yy = (np.arange(raw.shape[0]) - tm) % P
        xx = (np.arange(raw.shape[1]) - lm) % pattern.shape[1]
        col = pattern[yy[:, None], xx[None, :]].astype(np.uint8)

    nch = int(col.max()) + 1
    names = [desc[c] if c < len(desc) else "?" for c in range(nch)]
    gidx = [c for c in range(nch) if names[c] == "G"]
    if len(gidx) == 2:
        names[gidx[0]], names[gidx[1]] = "G1", "G2"
    res["channels"] = names

    raw -= black[col].astype(np.float32)            # raw now = black-subtracted (LibRaw black)
    sub = raw
    rng_c = [white - black[c] for c in range(nch)]  # usable range per channel

    # ---------------- left optical black (masked columns)
    ob = {"ok": False}
    offs = np.zeros(nch)
    sig = np.full(nch, np.nan)
    if lm >= 16:
        region = sub[tm:tm + h, :lm]
        creg = col[tm:tm + h, :lm]
        cmed = np.median(region[::4], axis=0)
        _, s0 = clipped_stats(region[::8, lm // 4: 3 * lm // 4])
        g = float(np.median(cmed))
        good = np.where(np.abs(cmed - g) < max(3.0, 0.5 * s0))[0]
        good = good[(good >= 4) & (good <= lm - 5)]
        if len(good) >= 8:
            R = region[:, good]
            C = creg[:, good]
            for c in range(nch):
                offs[c], sig[c] = clipped_stats(R[C == c])
            Rc = R - offs[C].astype(np.float32)
            _, sig_all = clipped_stats(Rc)
            # clipped row means: a hot pixel inside the masked border must not look like a banded row
            keep_px = np.abs(Rc) < 5.0 * sig_all
            nkeep = np.maximum(keep_px.sum(axis=1), 1)
            rows = np.where(keep_px, Rc, 0.0).sum(axis=1) / nkeep
            exp_var_rows = (np.where(keep_px, sig[C] ** 2, 0.0).sum(axis=1) / nkeep) / nkeep
            white_rows = float(exp_var_rows.mean())
            row_true = math.sqrt(max(float(rows.var()) - white_rows, 0.0))
            spec_r = spectrum(rows, white_rows)
            ob.update(ok=True, cols=[int(good.min()), int(good.max())], ncols=int(len(good)),
                      offset=offs.tolist(), sigma=sig.tolist(), sigma_all=sig_all,
                      row_std=float(rows.std()), row_expected=math.sqrt(white_rows), row_true=row_true,
                      row_pct=100.0 * row_true / sig_all if sig_all > 0 else float("nan"),
                      row_spec=spec_r)
            n = len(rows)
            step = max(1, n // 1500)
            trace = rows[: (n // step) * step].reshape(-1, step).mean(axis=1)
            k = 64
            smooth = np.convolve(rows, np.ones(k) / k, mode="valid")
            ob["row_trace"] = dict(x=(np.arange(len(trace)) * step + step / 2).tolist(), y=trace.tolist(),
                                   sx=(np.arange(len(smooth)) + k / 2)[::step].tolist(), sy=smooth[::step].tolist())
            log.info("  masked border: %d columns, read noise %.2f DN, row banding %.1f%% of random noise",
                     len(good), sig_all, ob["row_pct"])
        else:
            res["notes"].append("Left masked border found but no usable dark columns.")
    else:
        res["notes"].append("No left masked border (left margin %d px): read noise comes from the image only." % lm)
    res["ob"] = ob

    # ---------------- top optical black (masked rows) -> column pattern
    top = {"ok": False}
    if tm >= 10:
        T = sub[:tm, lm:lm + w]
        CT = col[:tm, lm:lm + w]
        idx = np.arange(2, tm - 1)
        st = [clipped_stats(T[i, ::3]) for i in idx]
        rm = np.array([a for a, _ in st])
        rs = np.array([b for _, b in st])
        ref = ob["sigma_all"] if ob.get("ok") else float(np.median(rs))
        keep = (rs > 0.7 * ref) & (rs < 1.3 * ref)
        if keep.sum() >= 4:
            mm = float(np.median(rm[keep]))
            mad = 1.4826 * float(np.median(np.abs(rm[keep] - mm))) + 1e-9
            keep &= np.abs(rm - mm) < max(5 * mad, 3 * ref / math.sqrt(max(w / 3, 1)))
        rows_used = idx[keep]
        top["rows_rejected"] = [int(i) for i in idx[~keep]]
        if len(rows_used) >= 4:
            TT = T[rows_used].copy()
            CC = CT[rows_used]
            csig = np.zeros(nch)
            for c in range(nch):
                mc, csig[c] = clipped_stats(TT[CC == c])
                TT[CC == c] -= mc
            cm = TT.mean(axis=0)
            exp_var_cols = (csig[CC] ** 2).mean(axis=0) / len(rows_used)
            white_cols = float(exp_var_cols.mean())
            col_true = math.sqrt(max(float(cm.var()) - white_cols, 0.0))
            denom = ob["sigma_all"] if ob.get("ok") else float(np.sqrt((csig ** 2).mean()))
            top.update(ok=True, rows=len(rows_used), col_std=float(cm.std()), col_expected=math.sqrt(white_cols),
                       col_true=col_true, col_pct=100.0 * col_true / denom if denom > 0 else float("nan"),
                       col_spec=spectrum(cm, white_cols))
            log.info("  masked top rows: %d used, column pattern %.1f%% of random noise", len(rows_used), top["col_pct"])
        else:
            res["notes"].append("Top masked rows did not pass the consistency check; column pattern not measured.")
    else:
        res["notes"].append("No top masked rows (top margin %d px): column pattern not measured." % tm)
    res["top"] = top

    if not bayer:
        res["elapsed"] = time.time() - t0
        return res

    # ---------------- visible image, per channel plane (black corrected by the masked border)
    V = sub[tm:tm + h, lm:lm + w]
    CV = col[tm:tm + h, lm:lm + w]
    pos = {int(CV[i, j]): (i, j) for i in (0, 1) for j in (0, 1)}
    planes = {c: V[i::2, j::2] - np.float32(offs[c]) for c, (i, j) in pos.items()}
    vis = {}
    hist = {}
    for c in range(nch):
        p = planes[c]
        sc = sig[c] if finite(sig[c]) else clipped_stats(p[::7, ::7])[1]
        below = float((p < 0).mean() * 100)
        clipped = float((p >= 0.995 * (rng_c[c] - offs[c])).mean() * 100)
        vis[names[c]] = dict(median=float(np.median(p[::3, ::3])), below_black_pct=below, clipped_pct=clipped)
        lo, hi = -5 * sc, 10 * sc
        hh, edges = np.histogram(p[::2, ::2], bins=120, range=(lo, hi))
        hist[names[c]] = dict(x=((edges[:-1] + edges[1:]) / 2 / sc).tolist(), y=(hh / max(hh.max(), 1)).tolist())
    res["visible"] = vis
    res["near_black_hist"] = hist

    # where a clipped black would sit (zero-signal area clipped at black: lift = sigma/sqrt(2*pi))
    gref = np.mean([wb[c] for c in gidx]) if gidx and all(wb[c] > 0 for c in gidx) else (wb[1] if wb[1] > 0 else 1.0)
    floor = {}
    for letter in ("R", "G", "B"):
        cs = [c for c in range(nch) if names[c].startswith(letter)]
        if not cs or not all(finite(sig[c]) for c in cs):
            continue
        lift = float(np.mean([sig[c] for c in cs])) / math.sqrt(2 * math.pi)
        wbn = float(np.mean([wb[c] for c in cs])) / gref if gref > 0 else 1.0
        lin = lift / float(np.mean([rng_c[c] for c in cs])) * wbn
        srgb = 12.92 * lin if lin <= 0.0031308 else 1.055 * lin ** (1 / 2.4) - 0.055
        floor[letter] = dict(lift_dn=lift, display_255=255 * srgb)
    res["black_floor"] = floor

    # ---------------- single-image noise model from G1 - G2 (photon transfer)
    ptc = {"ok": False}
    if len(gidx) == 2:
        a, b = planes[gidx[0]], planes[gidx[1]]
        hh_ = (min(a.shape[0], b.shape[0]) // 8) * 8
        ww_ = (min(a.shape[1], b.shape[1]) // 8) * 8
        a, b = a[:hh_, :ww_], b[:hh_, :ww_]
        m = ((a + b) / 2).reshape(hh_ // 8, 8, ww_ // 8, 8).mean(axis=(1, 3)).ravel()
        d = ((a - b) / np.float32(math.sqrt(2))).reshape(hh_ // 8, 8, ww_ // 8, 8)
        dv = d.var(axis=(1, 3), ddof=1).ravel()
        dm = d.mean(axis=(1, 3)).ravel()
        grng = float(np.mean([rng_c[c] for c in gidx]))
        lo = max(1.0, float(np.percentile(m, 1)))
        hi = float(np.percentile(m, 99.5))
        corr = chi2_q25_63()
        pts = []
        if hi > lo * 1.5:
            edges = np.concatenate([[-np.inf], np.geomspace(lo, hi, 16)])
            for e0, e1 in zip(edges[:-1], edges[1:]):
                sel = (m >= e0) & (m < e1)
                if sel.sum() >= 150:
                    pts.append((float(np.median(m[sel])), float(np.percentile(dv[sel], 25) / corr), int(sel.sum())))
        ptc["points"] = pts
        fitpts = [p for p in pts if p[0] > 0 and p[0] <= 0.4 * grng]
        g_ob = float(np.sqrt(np.mean([sig[c] ** 2 for c in gidx]))) if all(finite(sig[c]) for c in gidx) else float("nan")
        if len(fitpts) >= 4:
            X = np.array(fitpts)
            mm_, vv, nn = X[:, 0], X[:, 1], X[:, 2]
            wgt = np.sqrt(nn) / np.maximum(vv, 1e-9)

            def wfit(cols, y):
                A = np.vstack(cols).T
                return np.linalg.lstsq(A * wgt[:, None], y * wgt, rcond=None)[0]

            # 1) image-only fit (free intercept): an independent cross-check of read noise
            coef = wfit([np.ones_like(mm_), mm_, mm_ ** 2], vv)
            if coef[2] < 0:
                coef = np.append(wfit([np.ones_like(mm_), mm_], vv), 0.0)
            b_free, a_free, c_free = [float(x) for x in coef]
            # 2) model fit: intercept pinned to the masked-border read noise (two methods, one answer)
            if finite(g_ob) and g_ob > 0:
                b_ob = g_ob ** 2
                coef2 = wfit([mm_, mm_ ** 2], vv - b_ob)
                if coef2[1] < 0:
                    coef2 = np.array([wfit([mm_], vv - b_ob)[0], 0.0])
                a1, c2, b0 = float(coef2[0]), float(coef2[1]), b_ob
                anchored = True
            else:
                a1, c2, b0, anchored = a_free, c_free, b_free, False
            span = mm_.max() / max(mm_.min(), 1e-9)
            # the image can only vouch for read noise if its darkest flat areas are read-noise dominated
            reach = a1 > 0 and finite(g_ob) and a1 * mm_.min() <= 1.5 * g_ob ** 2
            if a1 > 0:
                ptc.update(ok=True, a=a1, b=b0, c=c2, anchored=anchored, gain_e_per_dn=1.0 / a1,
                           gain_free=(1.0 / a_free) if a_free > 0 else float("nan"),
                           read_dn_image=math.sqrt(b_free) if (b_free > 0 and (reach or not finite(g_ob))) else float("nan"),
                           read_dn_ob=g_ob, read_e=(g_ob / a1) if finite(g_ob) else float("nan"),
                           prnu_pct=100 * math.sqrt(c2) if c2 > 0 else 0.0,
                           full_well_e=grng / a1, span=span,
                           confidence="good" if span >= 8 else ("fair" if span >= 3 else "low"))
                if finite(g_ob) and g_ob > 0:
                    ptc["dr_stops"] = math.log2(grng / g_ob)
                if anchored and (not finite(ptc["gain_free"]) or abs(ptc["gain_free"] / ptc["gain_e_per_dn"] - 1) > 0.25):
                    ptc["confidence"] = "low"
                    res["notes"].append("Noise model: the image-only fit disagrees with the anchored fit; treat e-/DN as rough "
                                        "(too few flat areas, or strong texture). The calibration session will settle it.")
                log.info("  noise model: %.4f e-/DN (anchored=%s; image-only fit %.4f), image read noise %s vs masked border %.2f DN",
                         1 / a1, anchored, ptc["gain_free"],
                         ("%.1f" % ptc["read_dn_image"]) if finite(ptc["read_dn_image"]) else "not resolvable", g_ob)
            else:
                res["notes"].append("Noise model fit failed (slope not positive) - image has too little flat area.")
        else:
            res["notes"].append("Too few brightness levels with flat areas to fit the noise model.")
        # green imbalance in midtones
        sel = (m > 0.05 * grng) & (m < 0.5 * grng)
        if sel.sum() > 100:
            res["green_imbalance_pct"] = float(np.median(dm[sel] * math.sqrt(2) / m[sel]) * 100)
    res["ptc"] = ptc

    # ---------------- hot pixel candidates
    hot = {}
    try:
        Z = np.zeros(V.shape, dtype=np.float32)
        for c, (i, j) in pos.items():
            p = planes[c]
            nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * np.float32(0.25)
            rr = p[1:-1, 1:-1] - nb
            if ptc.get("ok") and finite(sig[c]):
                var = sig[c] ** 2 + ptc["a"] * np.maximum(nb, 0) + ptc["c"] * nb * nb
                sr = np.sqrt(var * 1.25).astype(np.float32)
            else:
                sr = np.float32(clipped_stats(rr[::5, ::5])[1] or 1.0)
            Z[i::2, j::2][1:-1, 1:-1] = rr / sr
        ys, xs = np.nonzero(Z > 8.0)
        inside = (ys > 1) & (ys < Z.shape[0] - 2) & (xs > 1) & (xs < Z.shape[1] - 2)
        ys, xs = ys[inside], xs[inside]
        if len(ys) > 300_000:
            res["notes"].append("Very many hot-pixel candidates; counted from a sample.")
            pick = np.random.default_rng(0).choice(len(ys), 300_000, replace=False)
            scale = len(ys) / 300_000
            ys, xs = ys[pick], xs[pick]
        else:
            scale = 1.0
        nmax = np.full(len(ys), -np.inf, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nmax = np.maximum(nmax, Z[ys + dy, xs + dx])
        iso = nmax < 3.0
        mp = V.size / 1e6
        zs = Z[ys, xs]
        order = np.argsort(zs[iso])[::-1][:12]
        hot = dict(isolated=int(iso.sum() * scale), point_like=int((~iso).sum() * scale),
                   isolated_per_mp=float(iso.sum() * scale / mp),
                   top=[dict(y=int(ys[iso][k] + tm), x=int(xs[iso][k] + lm), z=float(zs[iso][k])) for k in order])
        del Z
        log.info("  hot-pixel candidates: %d isolated, %d point-like (real lights also trigger)",
                 hot["isolated"], hot["point_like"])
    except Exception as e:
        log.warning("hot pixel scan failed: %s", e)
        log.debug(traceback.format_exc())
    res["hot"] = hot
    res["elapsed"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------- SVG charts
def _nice_ticks(lo, hi, n=5):
    if not (finite(lo) and finite(hi)) or hi <= lo:
        return [lo]
    span = hi - lo
    raw_step = span / n
    mag = 10 ** math.floor(math.log10(raw_step))
    step = mag
    for mlt in (1, 2, 2.5, 5, 10):
        step = mlt * mag
        if span / step <= n:
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9 * span:
        out.append(round(t, 10))
        t += step
    return out


def _log_ticks(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    major = [10 ** k for k in range(a, b + 1) if lo <= 10 ** k <= hi]
    if len(major) < 3:
        major = sorted({m * 10 ** k for k in range(a, b + 1) for m in (1, 2, 5) if lo <= m * 10 ** k <= hi})
    return major


def _fmt(v):
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 10000 or av < 0.01:
        return "%.0e" % v
    if av >= 100:
        return "%.0f" % v
    if av >= 1:
        return ("%.1f" % v).rstrip("0").rstrip(".")
    return ("%.2f" % v).rstrip("0").rstrip(".")


def svg_plot(series, title="", xlabel="", ylabel="", xlog=False, ylog=False, xlim=None, ylim=None,
             width=600, height=270, hlines=(), vlines=(), marks=()):
    L, Rm, T, B = 64, 18, 34, 44
    pw, ph = width - L - Rm, height - T - B
    xs_all, ys_all = [], []
    for s in series:
        x = np.asarray(s["x"], float)
        y = np.asarray(s["y"], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if xlog:
            ok &= x > 0
        if ylog:
            ok &= y > 0
        s["_x"], s["_y"] = x[ok], y[ok]
        xs_all.append(x[ok])
        ys_all.append(y[ok])
    xs_all = np.concatenate(xs_all) if xs_all else np.array([0.0, 1.0])
    ys_all = np.concatenate(ys_all) if ys_all else np.array([0.0, 1.0])
    if xs_all.size == 0:
        xs_all = np.array([1.0, 10.0])
    if ys_all.size == 0:
        ys_all = np.array([1.0, 10.0])
    x0, x1 = xlim if xlim else (float(xs_all.min()), float(xs_all.max()))
    y0, y1 = ylim if ylim else (float(ys_all.min()), float(ys_all.max()))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    if not ylim and not ylog:
        pad = 0.06 * (y1 - y0)
        y0, y1 = y0 - pad, y1 + pad
    fx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    fy = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X0, X1, Y0, Y1 = fx(x0), fx(x1), fy(y0), fy(y1)

    def px(v):
        return L + (fx(v) - X0) / (X1 - X0) * pw

    def py(v):
        return T + ph - (fy(v) - Y0) / (Y1 - Y0) * ph

    o = ['<svg class="chart" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (width, height)]
    o.append('<text x="%d" y="20" class="ct">%s</text>' % (L, html.escape(title)))
    o.append('<rect x="%d" y="%d" width="%d" height="%d" class="plotbg"/>' % (L, T, pw, ph))
    xt = _log_ticks(x0, x1) if xlog else _nice_ticks(x0, x1)
    yt = _log_ticks(y0, y1) if ylog else _nice_ticks(y0, y1)
    for t in xt:
        X = px(t)
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>' % (X, T, X, T + ph))
        o.append('<text x="%.1f" y="%d" class="tk" text-anchor="middle">%s</text>' % (X, T + ph + 15, _fmt(t)))
    for t in yt:
        Y = py(t)
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (L, Y, L + pw, Y))
        o.append('<text x="%d" y="%.1f" class="tk" text-anchor="end">%s</text>' % (L - 6, Y + 4, _fmt(t)))
    o.append('<text x="%.1f" y="%d" class="al" text-anchor="middle">%s</text>' % (L + pw / 2, height - 6, html.escape(xlabel)))
    o.append('<text x="14" y="%.1f" class="al" text-anchor="middle" transform="rotate(-90 14 %.1f)">%s</text>'
             % (T + ph / 2, T + ph / 2, html.escape(ylabel)))
    o.append('<clipPath id="c%d"><rect x="%d" y="%d" width="%d" height="%d"/></clipPath>' % (id(series) % 10**8, L, T, pw, ph))
    clip = 'clip-path="url(#c%d)"' % (id(series) % 10**8)
    for (yv, color, label, dash) in hlines:
        if y0 <= yv <= y1:
            Y = py(yv)
            o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1.4" stroke-dasharray="%s"/>'
                     % (L, Y, L + pw, Y, color, dash))
            o.append('<text x="%d" y="%.1f" class="lbl" fill="%s" text-anchor="end">%s</text>' % (L + pw - 4, Y - 4, color, html.escape(label)))
    for (xv, color, label) in vlines:
        if x0 <= xv <= x1:
            X = px(xv)
            o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.2" stroke-dasharray="3 3"/>' % (X, T, X, T + ph, color))
            o.append('<text x="%.1f" y="%d" class="lbl" fill="%s">%s</text>' % (X + 3, T + 12, color, html.escape(label)))
    legend = []
    for s in series:
        x, y = s["_x"], s["_y"]
        if x.size == 0:
            continue
        color = s.get("color", "#e6ecff")
        if s.get("kind") == "dots":
            for xv, yv in zip(x, y):
                o.append('<circle cx="%.1f" cy="%.1f" r="%s" fill="%s" %s/>' % (px(xv), py(yv), s.get("r", 3.5), color, clip))
        else:
            pts = " ".join("%.1f,%.1f" % (px(xv), py(yv)) for xv, yv in zip(x, y))
            o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" stroke-opacity="%s" %s %s/>'
                     % (pts, color, s.get("width", 1.4), s.get("opacity", 1), ('stroke-dasharray="%s"' % s["dash"]) if s.get("dash") else "", clip))
        if s.get("label"):
            legend.append((s["label"], color))
    for (xv, yv, text, color) in marks:
        if x0 <= xv <= x1 and y0 <= yv <= y1:
            o.append('<circle cx="%.1f" cy="%.1f" r="4" fill="none" stroke="%s" stroke-width="1.6"/>' % (px(xv), py(yv), color))
            o.append('<text x="%.1f" y="%.1f" class="lbl" fill="%s">%s</text>' % (px(xv) + 6, py(yv) - 6, color, html.escape(text)))
    lx = L + pw - 6
    for k, (label, color) in enumerate(legend):
        yy = T + 14 + 15 * k
        o.append('<rect x="%d" y="%d" width="10" height="3" fill="%s"/>' % (lx - 150, yy - 4, color))
        o.append('<text x="%d" y="%d" class="lg">%s</text>' % (lx - 135, yy, html.escape(label)))
    o.append("</svg>")
    return "".join(o)


# ----------------------------------------------------------------------------- HTML report
CHAN_COLOR = {"R": "#f87171", "G1": "#4ade80", "G2": "#a3e635", "G": "#4ade80", "B": "#60a5fa"}

CSS = """
:root{--bg:#070b18;--card:#101833;--card2:#0c1328;--ink:#e6ecff;--dim:#93a1c9;--ob:#22d3ee;--img:#c084fc;--meta:#94a3b8;
--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444;--na:#475569}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 10% -10%,#1b2a6b55,transparent),var(--bg);
color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:1320px;margin:0 auto;padding:28px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:.3px}h1 span{background:linear-gradient(90deg,#22d3ee,#c084fc);-webkit-background-clip:text;color:transparent}
h2{font-size:22px;margin:40px 0 6px}h3{font-size:15px;margin:0 0 8px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.8px}
.sub{color:var(--dim)}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0 4px}
.src{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.4px;border:1px solid}
.src.ob{color:var(--ob);border-color:var(--ob)}.src.img{color:var(--img);border-color:var(--img)}.src.meta{color:var(--meta);border-color:var(--meta)}
table{border-collapse:collapse;width:100%;margin-top:10px;background:var(--card2);border-radius:12px;overflow:hidden}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #1e2a52;font-size:13px}th{color:var(--dim);font-weight:600}
.chip{display:inline-block;min-width:10px;height:10px;border-radius:99px;margin-right:6px;vertical-align:middle}
.ok{background:var(--ok);box-shadow:0 0 10px var(--ok)}.warn{background:var(--warn);box-shadow:0 0 10px var(--warn)}
.bad{background:var(--bad);box-shadow:0 0 10px var(--bad)}.na{background:var(--na)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin-top:12px}
.card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid #1f2b57;border-radius:14px;padding:14px 14px 12px;position:relative}
.card .k{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.7px;display:flex;justify-content:space-between;align-items:center}
.card .v{font-size:24px;font-weight:700;margin:6px 0 2px}.card .d{font-size:12.5px;color:var(--dim)}
.card.st-ok{border-left:4px solid var(--ok)}.card.st-warn{border-left:4px solid var(--warn)}.card.st-bad{border-left:4px solid var(--bad)}
.swatch{display:inline-block;width:46px;height:22px;border-radius:6px;border:1px solid #334;vertical-align:middle;margin-right:6px}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px;margin-top:14px}
.panel{background:var(--card2);border:1px solid #1f2b57;border-radius:14px;padding:10px 12px}
.panel p{margin:4px 2px 2px;color:var(--dim);font-size:12.5px}
svg.chart{width:100%;height:auto}.plotbg{fill:#0a1024}.grid{stroke:#1c2850;stroke-width:1}
.tk{fill:#7f8db8;font-size:10.5px}.al{fill:#9fb0dd;font-size:11.5px}.ct{fill:#e6ecff;font-size:13px;font-weight:600}
.lbl{font-size:10.5px;font-weight:600}.lg{fill:#c8d3f5;font-size:10.5px}
.notes{margin-top:10px;color:#fbbf24;font-size:12.5px}.foot{margin-top:40px;color:var(--dim);font-size:12px}
.fail{background:#2a0f16;border:1px solid #7f1d1d;border-radius:12px;padding:12px;margin-top:10px}
"""


def st(value, key):
    if not finite(value):
        return "na"
    a, b = THRESH[key]
    return "ok" if value < a else ("warn" if value < b else "bad")


def lines_text(spec, skip_cfa=True):
    if not spec or not spec.get("lines"):
        return "none above the noise"
    out = []
    for p, d in spec["lines"]:
        if skip_cfa and abs(p - 2.0) < 0.02:
            out.append("2 px (colour-filter rhythm)")
        else:
            out.append("%s px (+%.0f dB)" % (_fmt(p), d))
    return ", ".join(out[:5])


def card(key, value, detail, src, status=None):
    cls = (" st-" + status) if status in ("ok", "warn", "bad") else ""
    srcmap = {"ob": ("ob", "MASKED PIXELS"), "img": ("img", "IMAGE"), "meta": ("meta", "METADATA")}
    sc, sl = srcmap[src]
    return ('<div class="card%s"><div class="k">%s <span class="src %s">%s</span></div><div class="v">%s</div><div class="d">%s</div></div>'
            % (cls, html.escape(key), sc, sl, value, detail))


def file_section(r, n):
    e = r.get("exif", {})
    title = "%s - %s" % (r["name"], e.get("model", "unknown camera"))
    o = ['<h2 id="f%d">%s</h2>' % (n, html.escape(title))]
    bits = []
    if e.get("iso"):
        bits.append("ISO " + e["iso"])
    if e.get("exposure"):
        bits.append(e["exposure"] + " s")
    if e.get("manual_lens"):
        bits.append("manual lens (no electronic contacts)")
    elif e.get("fnumber_val"):
        bits.append("f/%s" % _fmt(e["fnumber_val"]))
    if e.get("focal") and not e.get("manual_lens"):
        bits.append(e["focal"] + " mm")
    if e.get("date"):
        bits.append(e["date"])
    sz = r.get("sizes", {})
    if sz:
        bits.append("%d x %d visible, masked border %d cols / %d rows" % (sz["w"], sz["h"], sz["left"], sz["top"]))
    o.append('<div class="sub">%s</div>' % html.escape("  ·  ".join(bits)))
    if not r.get("ok"):
        o.append('<div class="fail">Analysis failed: %s</div>' % html.escape(r.get("error", "unknown error")))
        return "".join(o)

    ob, top, ptc, vis, hot = r.get("ob", {}), r.get("top", {}), r.get("ptc", {}), r.get("visible", {}), r.get("hot", {})
    cards = []
    if ob.get("ok"):
        det = "per channel: " + ", ".join("%s %.1f" % (nm, s) for nm, s in zip(r["channels"], ob["sigma"]))
        if ptc.get("ok") and finite(ptc.get("read_e")):
            val = "%.1f DN · %.2f e&#8315;" % (ob["sigma_all"], ptc["read_e"])
        else:
            val = "%.1f DN" % ob["sigma_all"]
        if finite(ptc.get("dr_stops")):
            det += "<br>engineering dynamic range %.1f stops" % ptc["dr_stops"]
        cards.append(card("Read noise", val, det, "ob"))
        rs = ob.get("row_spec") or {}
        det = "true row noise %.2f DN" % ob["row_true"]
        if finite(rs.get("slow_share")):
            det += " · slow-drift share %.0f%%" % (100 * rs["slow_share"])
        det += "<br>repeating lines: " + html.escape(lines_text(rs))
        cards.append(card("Row banding (horizontal)", "%.1f%% of noise" % ob["row_pct"], det, "ob", st(ob["row_pct"], "row_pct")))
    if top.get("ok"):
        det = "true column pattern %.2f DN from %d masked rows" % (top["col_true"], top["rows"])
        det += "<br>repeating lines: " + html.escape(lines_text(top.get("col_spec")))
        cards.append(card("Column pattern (vertical)", "%.1f%% of noise" % top["col_pct"], det, "ob", st(top["col_pct"], "col_pct")))
    if vis:
        worst = max(vis.items(), key=lambda kv: kv[1]["below_black_pct"])
        det = ", ".join("%s %.1f%%" % (k, v["below_black_pct"]) for k, v in vis.items())
        cards.append(card("Pixels below black", "%.1f%% (%s)" % (worst[1]["below_black_pct"], worst[0]), det, "img",
                          st(worst[1]["below_black_pct"], "below_black_pct")))
    fl = r.get("black_floor", {})
    if all(k in fl for k in ("R", "G", "B")):
        rgb = [int(round(min(255, max(0, fl[k]["display_255"])))) for k in ("R", "G", "B")]
        worst = max(fl[k]["display_255"] for k in fl)
        sw = '<span class="swatch" style="background:rgb(%d,%d,%d)"></span><span class="swatch" style="background:#000"></span>' % tuple(rgb)
        det = "if a converter clips at black before denoising, true black lands here (left) instead of black (right). Approximate: white balance only, no colour matrix or tone curve."
        cards.append(card("Clipped-black floor", sw + "RGB %d,%d,%d" % tuple(rgb), det, "ob", st(worst, "floor_255")))
    if ptc.get("ok"):
        det = "fit from %d brightness levels (confidence %s)%s" % (
            len(ptc.get("points", [])), ptc["confidence"], ", anchored to masked-border read noise" if ptc.get("anchored") else "")
        if finite(ptc.get("gain_free")) and ptc.get("anchored"):
            det += "<br>image-only fit: %.3f e&#8315;/DN (%+.0f%%)" % (ptc["gain_free"], 100 * (ptc["gain_free"] / ptc["gain_e_per_dn"] - 1))
        if finite(ptc.get("read_dn_image")) and finite(ptc.get("read_dn_ob")) and ptc["read_dn_ob"] > 0:
            diff = 100 * (ptc["read_dn_image"] / ptc["read_dn_ob"] - 1)
            det += "<br>read noise from image %.1f DN vs masked border %.1f DN (%+.0f%%)" % (ptc["read_dn_image"], ptc["read_dn_ob"], diff)
        else:
            det += "<br>image has no flat areas dark enough to check read noise independently"
        if ptc.get("prnu_pct"):
            det += "<br>pixel sensitivity spread (PRNU) ~%.2f%%" % ptc["prnu_pct"]
        cards.append(card("Noise model (photon transfer)", "%.3f e&#8315;/DN" % ptc["gain_e_per_dn"], det, "img"))
    if hot:
        det = "%d point-like (real point lights or stars also land here)" % hot["point_like"]
        cards.append(card("Hot-pixel candidates", "%d isolated" % hot["isolated"], det + "<br>%.1f per megapixel" % hot["isolated_per_mp"],
                          "img", st(hot["isolated_per_mp"], "hot_per_mp")))
    if finite(r.get("green_imbalance_pct")):
        cards.append(card("Green balance (G1 vs G2)", "%+.2f%%" % r["green_imbalance_pct"],
                          "midtone difference between the two green channels; causes maze artifacts when large", "img"))
    lib = r.get("libraw", {})
    if ob.get("ok"):
        det = "LibRaw black %s · masked-border offset %s DN" % (
            "/".join("%g" % b for b in lib.get("black", [])), "/".join("%+.2f" % x for x in ob["offset"]))
        cards.append(card("Black level check", "%+.2f DN" % float(np.mean(ob["offset"])), det, "ob"))
    o.append('<div class="cards">%s</div>' % "".join(cards))

    charts = []
    if ptc.get("points"):
        pts = [p for p in ptc["points"] if p[0] > 0]
        ser = [dict(x=[p[0] for p in pts], y=[p[1] for p in pts], kind="dots", color="#c084fc", label="measured (G1-G2)")]
        hl = []
        if ptc.get("ok"):
            xs = np.geomspace(max(min(p[0] for p in pts), 0.5), max(p[0] for p in pts), 200)
            ys = ptc["b"] + ptc["a"] * xs + ptc["c"] * xs ** 2
            ser.append(dict(x=xs.tolist(), y=ys.tolist(), color="#f0abfc", width=2, label="noise model"))
            if finite(ptc.get("read_dn_ob")):
                hl.append((ptc["read_dn_ob"] ** 2, "#22d3ee", "read noise from masked border", "5 4"))
        yv = [p[1] for p in pts] + [h_[0] for h_ in hl]
        ylim = (min(yv) * 0.7, max(yv) * 1.4) if yv else None
        charts.append('<div class="panel">%s<p>Each dot is one brightness level. The flat left part is the sensor\'s read noise; the rising part is photon (shot) noise. The pink curve is the noise model the denoiser will use, pinned to the read noise measured in the masked border (cyan line). On bright pictures the dots never reach the flat part, so only the masked border can tell us the read noise.</p></div>'
                      % svg_plot(ser, "Photon transfer: noise vs brightness", "signal (DN above black)", "noise variance (DN²)",
                                 xlog=True, ylog=True, hlines=hl, ylim=ylim))
    if ob.get("row_trace"):
        tr = ob["row_trace"]
        ser = [dict(x=tr["x"], y=tr["y"], color="#22d3ee", width=0.9, opacity=0.55, label="row average"),
               dict(x=tr["sx"], y=tr["sy"], color="#fde047", width=2.2, label="64-row smoothing")]
        charts.append('<div class="panel">%s<p>The average of the masked pixels in every row, top to bottom. With no banding this would be flat fuzz. Slow waves (yellow) are drift in the readout electronics printed row by row.</p></div>'
                      % svg_plot(ser, "Row banding seen in the masked border", "row (top to bottom)", "row offset (DN)"))
    for spec, name, what in ((ob.get("row_spec"), "Row banding spectrum", "rows"), (top.get("col_spec"), "Column pattern spectrum", "columns")):
        if not spec:
            continue
        per = np.array(spec["period"])
        db = np.array(spec["db_white"])
        keep = per <= len(per)
        ser = [dict(x=per[keep].tolist(), y=db[keep].tolist(), color="#38bdf8" if what == "rows" else "#a78bfa", width=1.1)]
        marks = []
        for p, d in spec["lines"][:6]:
            k = int(np.argmin(np.abs(per - p)))
            marks.append((p, db[k], "%s px" % _fmt(p), "#fde047"))
        lo_y = float(np.percentile(db, 1)) - 2
        hi_y = float(db.max()) + 4
        chart = svg_plot(ser, name, "repeat length (pixels)", "dB above random noise", xlog=True,
                         ylim=(lo_y, hi_y), hlines=[(0.0, "#64748b", "pure random noise", "4 3")], marks=marks)
        charts.append('<div class="panel">%s<p>Power of the %s pattern by repeat length, in dB above pure pixel noise (0 dB). A floor raised evenly across all lengths is line-to-line jitter; spikes are repeating patterns (cyclostationary fingerprints of the readout, such as groups of columns sharing an amplifier); a hump at long lengths is slow drift.</p></div>'
                      % (chart, "row" if what == "rows" else "column"))
    hist = r.get("near_black_hist")
    if hist:
        ser = [dict(x=v["x"], y=v["y"], color=CHAN_COLOR.get(k, "#fff"), width=1.6, label=k,
                    dash="5 3" if k == "G2" else None) for k, v in hist.items()]
        charts.append('<div class="panel">%s<p>Pixel values near black, in units of each channel\'s read noise. Everything left of the dashed line is below black: real signal plus negative noise that many converters clip away, lifting and tinting the shadows. The denoiser keeps it.</p></div>'
                      % svg_plot(ser, "Near-black histogram per channel", "value (x read noise)", "relative count",
                                 vlines=[(0.0, "#f43f5e", "black")]))
    o.append('<div class="charts">%s</div>' % "".join(charts))
    if r.get("notes"):
        o.append('<div class="notes">%s</div>' % "<br>".join("&#9888; " + html.escape(nn) for nn in r["notes"]))
    return "".join(o)


def build_report(results, out_path):
    import rawpy
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    o = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Raw Analyzer report</title><style>%s</style></head><body><div class='wrap'>" % CSS]
    o.append("<h1><span>Raw Analyzer</span> · Photographic Studio By Bill</h1>")
    o.append("<div class='sub'>%d file(s) · %s · what each raw's noise looks like before anything touches it</div>" % (len(results), now))
    o.append("<div class='legend'><span class='src ob'>MASKED PIXELS</span> measured in the sensor's covered border (no light: pure electronics)"
             "<span class='src img'>IMAGE</span> estimated from the picture itself"
             "<span class='src meta'>METADATA</span> from the file</div>")
    o.append("<table><tr><th>File</th><th>Camera</th><th>ISO</th><th>Read noise</th><th>Row banding</th><th>Column pattern</th><th>Column lines</th><th>Below black</th><th>Black floor</th><th>Hot px</th><th>e&#8315;/DN</th></tr>")
    for n, r in enumerate(results):
        e = r.get("exif", {})
        if not r.get("ok"):
            o.append("<tr><td><a href='#f%d' style='color:#93c5fd'>%s</a></td><td colspan='10'><span class='chip bad'></span>failed: %s</td></tr>"
                     % (n, html.escape(r["name"]), html.escape(r.get("error", ""))))
            continue
        ob, top, ptc, vis, hot = r.get("ob", {}), r.get("top", {}), r.get("ptc", {}), r.get("visible", {}), r.get("hot", {})
        rowp = ob.get("row_pct") if ob.get("ok") else None
        colp = top.get("col_pct") if top.get("ok") else None
        bb = max((v["below_black_pct"] for v in vis.values()), default=None) if vis else None
        fl = r.get("black_floor", {})
        flw = max((v["display_255"] for v in fl.values()), default=None) if fl else None
        cells = [
            "<a href='#f%d' style='color:#93c5fd'>%s</a>" % (n, html.escape(r["name"])),
            html.escape(e.get("model", "?")), html.escape(e.get("iso", "?")),
            ("%.1f DN" % ob["sigma_all"]) if ob.get("ok") else "-",
            ("<span class='chip %s'></span>%.1f%%" % (st(rowp, "row_pct"), rowp)) if finite(rowp) else "-",
            ("<span class='chip %s'></span>%.1f%%" % (st(colp, "col_pct"), colp)) if finite(colp) else "-",
            html.escape(", ".join(_fmt(p) for p, _ in (top.get("col_spec") or {}).get("lines", [])[:4] if abs(p - 2) > 0.02)) or "-",
            ("<span class='chip %s'></span>%.1f%%" % (st(bb, "below_black_pct"), bb)) if finite(bb) else "-",
            ("<span class='chip %s'></span>%.0f/255" % (st(flw, "floor_255"), flw)) if finite(flw) else "-",
            ("<span class='chip %s'></span>%d" % (st(hot.get("isolated_per_mp"), "hot_per_mp"), hot["isolated"])) if hot else "-",
            ("%.3f" % ptc["gain_e_per_dn"]) if ptc.get("ok") else "-",
        ]
        o.append("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in cells))
    o.append("</table>")
    o.append("<div class='sub' style='margin-top:6px'>Traffic lights are provisional thresholds, to be tuned against your eye. Row/column percentages compare the pattern to the random noise.</div>")
    # bodies at a glance: median per camera and ISO
    groups = {}
    for r in results:
        if not r.get("ok"):
            continue
        e = r.get("exif", {})
        try:
            iso = int(float(e.get("iso", "0").split()[0]))
        except Exception:
            iso = 0
        groups.setdefault((e.get("model", "?"), iso), []).append(r)
    if len(groups) > 1:
        o.append("<h2>Bodies at a glance</h2><div class='sub'>Median per camera and ISO. Read noise and patterns come from the masked border, so they describe the sensor, not the scene.</div>")
        o.append("<table><tr><th>Camera</th><th>ISO</th><th>Files</th><th>Read noise</th><th>Read noise (e&#8315;)</th><th>e&#8315;/DN</th><th>Row banding</th><th>Column pattern</th><th>Column lines seen</th></tr>")

        def med(vals):
            vals = [v for v in vals if finite(v)]
            return float(np.median(vals)) if vals else float("nan")

        for (model, iso) in sorted(groups):
            rs = groups[(model, iso)]
            rd = med([r["ob"].get("sigma_all") for r in rs if r.get("ob", {}).get("ok")])
            re_ = med([r["ptc"].get("read_e") for r in rs if r.get("ptc", {}).get("ok")])
            gn = med([r["ptc"].get("gain_e_per_dn") for r in rs if r.get("ptc", {}).get("ok")])
            rw = med([r["ob"].get("row_pct") for r in rs if r.get("ob", {}).get("ok")])
            cl = med([r["top"].get("col_pct") for r in rs if r.get("top", {}).get("ok")])
            seen = {}
            for r in rs:
                for p, _ in (r.get("top", {}).get("col_spec") or {}).get("lines", []):
                    if abs(p - 2) > 0.02:
                        seen[_fmt(p)] = seen.get(_fmt(p), 0) + 1
            seen_s = ", ".join("%s px (%d)" % (k, v) for k, v in sorted(seen.items(), key=lambda kv: -kv[1])[:4]) or "-"
            o.append("<tr><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                html.escape(model), iso or "?", len(rs),
                ("%.1f DN" % rd) if finite(rd) else "-", ("%.2f" % re_) if finite(re_) else "-", ("%.3f" % gn) if finite(gn) else "-",
                ("<span class='chip %s'></span>%.1f%%" % (st(rw, "row_pct"), rw)) if finite(rw) else "-",
                ("<span class='chip %s'></span>%.1f%%" % (st(cl, "col_pct"), cl)) if finite(cl) else "-",
                html.escape(seen_s)))
        o.append("</table>")
    for n, r in enumerate(results):
        o.append(file_section(r, n))
    o.append("<div class='foot'>Raw Analyzer v%s · rawpy %s / LibRaw %s · numpy %s · nothing in the raw files was changed.</div>"
             % (VERSION, rawpy.__version__, ".".join(str(x) for x in rawpy.libraw_version), np.__version__))
    o.append("</div></body></html>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(o))


# ----------------------------------------------------------------------------- main
def collect(args):
    paths = []
    for a in args or [HERE]:
        a = a.strip().strip('"')
        if os.path.isdir(a):
            for dp, dn, fn in os.walk(a):
                dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
                for f in sorted(fn):
                    if f.lower().endswith(RAW_EXT):
                        paths.append(os.path.join(dp, f))
        elif os.path.isfile(a) and a.lower().endswith(RAW_EXT):
            paths.append(a)
        else:
            log.warning("Skipping (not a raw file or folder): %s", a)
    return paths


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k not in ("near_black_hist", "row_trace", "period", "db_white")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv):
    setup_logging()
    no_open = "--no-open" in argv
    args = [a for a in argv if a != "--no-open"]
    log.info("Raw Analyzer v%s - Photographic Studio By Bill", VERSION)
    log.debug("python %s, args %s", sys.version.replace("\n", " "), argv)
    try:
        import rawpy
        log.debug("rawpy %s, LibRaw %s, numpy %s", rawpy.__version__, rawpy.libraw_version, np.__version__)
    except Exception as e:
        log.error("rawpy is not installed (%s). Run 'Analyze Raws.bat' so it can set things up.", e)
        return 2
    paths = collect(args)
    if not paths:
        log.error("No raw files found. Drop CR2/CR3 files (or a folder) onto 'Analyze Raws.bat'.")
        return 1
    os.makedirs(REPORT_DIR, exist_ok=True)
    results = []
    for i, p in enumerate(paths, 1):
        log.info("[%d/%d] %s", i, len(paths), p)
        try:
            r = analyze(p)
            log.info("  done in %.1f s", r.get("elapsed", 0))
        except Exception as e:
            log.error("  FAILED: %s", e)
            log.debug(traceback.format_exc())
            r = {"file": p, "name": os.path.basename(p), "ok": False, "error": "%s: %s" % (type(e).__name__, e), "exif": read_exif(p)}
        results.append(r)
        try:
            jp = os.path.join(REPORT_DIR, os.path.splitext(os.path.basename(p))[0] + ".json")
            with open(jp, "w", encoding="utf-8") as fh:
                json.dump(_jsonable(r), fh, indent=1)
        except Exception as e:
            log.warning("  could not write JSON: %s", e)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPORT_DIR, "raw_report_%s.html" % stamp)
    build_report(results, out)
    log.info("Report: %s", out)
    if not no_open and os.name == "nt":
        try:
            os.startfile(out)
        except Exception as e:
            log.warning("could not open the report automatically: %s", e)
    return 0 if any(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

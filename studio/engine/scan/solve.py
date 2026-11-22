"""Every view placed: the Studio's own camera position solver (BUILD_PLAN_DP3D E.2-E.5).

Built for Bill's shooting: hand-held, unordered, few wide views, each a focus stack with its own measured depth.

1. **Features** in every view (inside its subject mask): plain SIFT first; when some view cannot be joined to at least
   two others that way, the whole set is described again with affine simulation (ASIFT), which matches views up to
   about 80 degrees apart.
2. **Pairs**: every pair matched and verified (``twoview.pair``: the pose from the matches and from the depth). The
   pairs that hold form a graph; its connected parts are the objects. A view joined to nothing is named with the
   reason.
3. **Incremental placement**, order never assumed: the strongest pair with a real angle first; then the view with
   the most matches to points already placed, by PnP (RANSAC, refined) and by the depth (a similarity between the
   view's own depth cloud and the placed points, over its one shape number kappa), compared: both agree (cyan) or one
   alone (amber). A point seen by one placed view only is placed from that view's depth, so a new view far round the
   object still finds placed points to hold on to.
4. **Bundle adjustment with depth** (``scipy.optimize.least_squares``, Huber, sparse Jacobian pattern): each view's
   pose, focal length (2 % prior round its EXIF value: focus breathing), its depth shape (m, kappa); one radial
   distortion k1 shared by the lens; the points. Residuals: reprojection in pixels and the depth of each point against
   the view's fused depth at that pixel (weighted by its confidence). Outliers (over 3 px, or depth over 3 sigma) go
   in rounds. ``use_depth=False`` is plain structure from motion (no depth anywhere), kept for comparison.

Units: the first placed view's mean distance to what it measured is 1 (its depth scale m = 1). Real units come from
the scale step (E.7) afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ... import log as slog
from . import geom as G
from . import sift
from . import twoview as TV

log = slog.get("engine.scan.solve")

REPROJ_MAX = 3.0            # px: observations further off are dropped in the clean-up rounds
TRI_ANGLE = 1.5             # degrees: a point triangulated from rays closer than this is not trusted
DEPTH_SIGMA = 0.01          # relative inverse-distance noise assumed for the depth terms until it is measured
MIN_DEPTH_SIGMA = 0.002     # the depth terms never weigh more than this noise allows
PX_SIGMA = 1.0              # pixels: the matches' spread assumed until it is measured
MIN_PX_SIGMA = 0.15         # ... never below this
FOCAL_PRIOR = 0.02
BREATHING = 0.03            # focus breathing: stacks of one lens keep within about 3 % of each other's focal length
FEW_VIEWS = 8               # below this many placed views, the lens distortion is held and the focal prior tightened
STRONG = 2                  # a view joined to fewer views than this by plain SIFT calls for affine simulation
REG_MIN = 6

HOW_TEXT = {"both": "both agree (the matches and the measured depth)", "matches": "from the matches alone",
            "depth": "from the measured depth alone", "first": "the first view (the frame of reference)"}
HOW_COLOUR = {"both": "cyan", "first": "cyan", "matches": "amber", "depth": "amber", None: "red"}


@dataclass
class Solution:
    names: list
    placed: list                       # bool per view
    R: list                            # world-to-camera rotation per view (None when not placed)
    t: list
    f: list                            # solved focal length, pixels of the working size
    m: list                            # inverse-distance scale (depth units -> world)
    kappa: list                        # depth shape
    how: list                          # both | matches | depth | first | None
    reason: list                       # why a view was not placed ("" when placed)
    group: list                        # the object (connected part) each view belongs to, -1 alone
    k1: float = 0.0
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    point_src: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))   # 1 triangulated, 2 from depth
    obs_view: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_point: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    quality: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    use_depth: bool = True
    affine: bool = False
    notes: list = field(default_factory=list)
    tried: list = field(default_factory=list)     # every pair: candidate matches and how many held

    def centre(self, i):
        return None if not self.placed[i] else G.centre(self.R[i], self.t[i])

    def summary(self) -> dict:
        n = len(self.names)
        placed = [i for i in range(n) if self.placed[i]]
        q = [x for x in self.quality if x]
        return {"views": n, "placed": len(placed),
                "not_placed": [{"view": self.names[i], "reason": self.reason[i]} for i in range(n)
                               if not self.placed[i]],
                "reprojection_px": round(float(np.mean([x["reprojection_px"] for x in q])), 3) if q else None,
                "depth_agreement": round(float(np.mean([x["depth_rms"] for x in q if x["depth_rms"] is not None])), 4)
                if any(x["depth_rms"] is not None for x in q) else None,
                "points": int(len(self.points)), "affine": self.affine, "use_depth": self.use_depth}


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _components(n, edges):
    uf = _UF()
    for i in range(n):
        uf.find(i)
    for i, j in edges:
        uf.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# ------------------------------------------------------------------------------------------ features and pairs
def match_all(views, feats, use_depth, pairs_only=None, progress=None, cancel=None, base=0.0, span=1.0, tried=None):
    """Verified pairs {(i, j): PairResult}. ``tried`` (a dict) receives every pair's candidate matches and how many
    held (for the plain-words reason when nothing can be placed)."""
    n = len(views)
    todo = [(i, j) for i in range(n) for j in range(i + 1, n) if pairs_only is None or (i, j) in pairs_only]
    out = {}
    for k, (i, j) in enumerate(todo):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        affine = hasattr(feats[i], "sim") and hasattr(feats[j], "sim")
        m = sift.match_affine(feats[i], feats[j]) if affine else sift.match(feats[i], feats[j])
        held = 0
        if len(m) >= TV.MIN_INLIERS:
            pr = TV.pair(views[i], views[j], feats[i], feats[j], m, i, j, use_depth)
            if pr is not None:
                out[(i, j)] = pr
                held = pr.n_inliers
        if tried is not None:
            tried[(i, j)] = (int(len(m)), int(held))
        if progress:
            progress(base + span * (k + 1) / max(len(todo), 1), f"matching views {views[i].name} and {views[j].name}")
    return out


# ------------------------------------------------------------------------------------------ the solver
class _State:
    def __init__(self, views, feats, pairs, comp, use_depth):
        self.views, self.feats, self.use_depth = views, feats, use_depth
        n = len(views)
        self.placed = [False] * n
        self.R = [None] * n
        self.t = [None] * n
        self.f = [float(v.f) for v in views]
        self.f0 = list(self.f)
        self.m = [None] * n
        self.kappa = [None] * n
        self.how = [None] * n
        self.order = []
        self.k1 = 0.0
        self.depth_sigma = DEPTH_SIGMA       # measured from the depth residuals once there are enough
        self.px_sigma = PX_SIGMA             # the matches' own measured spread, pixels (likewise)
        # tracks over (view, feature) by union-find of every verified inlier match inside this object
        uf = _UF()
        for (i, j), pr in pairs.items():
            if i not in comp or j not in comp:
                continue
            for a, b in pr.matches[pr.inliers]:
                uf.union((i, int(a)), (j, int(b)))
        tracks = {}
        for node in list(uf.p.keys()):
            tracks.setdefault(uf.find(node), []).append(node)
        ov, ot, of = [], [], []
        T = 0
        for nodes in tracks.values():
            vs = [v for v, _ in nodes]
            if len(nodes) < 2 or len(set(vs)) != len(vs):     # one place in a view seen twice: ambiguous, dropped
                continue
            for v, fi in nodes:
                ov.append(v)
                ot.append(T)
                of.append(fi)
            T += 1
        self.ov = np.array(ov, int)
        self.ot = np.array(ot, int)
        self.of = np.array(of, int)
        self.xy = np.zeros((len(ov), 2))
        self.tau = np.zeros(len(ov))
        self.conf = np.zeros(len(ov))
        for v in set(ov):
            s = self.ov == v
            xy = feats[v].xy[self.of[s]].astype(np.float64)
            self.xy[s] = xy
            self.tau[s] = TV.sample(views[v].tau, xy)
            self.conf[s] = TV.sample(views[v].conf, xy)
        self.alive = np.ones(len(ov), bool)
        self.X = np.full((T, 3), np.nan)
        self.src = np.zeros(T, np.int8)
        self.T = T

    # -------------------------------------------------------------- helpers
    def norm_xy(self, idx):
        v = self.ov[idx]
        cx = np.array([self.views[i].cx for i in v])
        cy = np.array([self.views[i].cy for i in v])
        f = np.array([self.f[i] for i in v])
        x = (self.xy[idx, 0] - cx) / f
        y = (self.xy[idx, 1] - cy) / f
        if self.k1:                                   # undo the radial distortion (two fixed-point rounds)
            xu, yu = x.copy(), y.copy()
            for _ in range(3):
                d = 1 + self.k1 * (xu * xu + yu * yu)
                xu, yu = x / d, y / d
            x, y = xu, yu
        return np.stack([x, y], -1)

    def depth_w(self, idx):
        v = self.ov[idx]
        m = np.array([self.m[i] if self.m[i] is not None else np.nan for i in v])
        k = np.array([self.kappa[i] if self.kappa[i] is not None else np.nan for i in v])
        return m * (1 + k * self.tau[idx])

    def place(self, i, R, t, how, m=None, kappa=None):
        self.placed[i] = True
        self.R[i], self.t[i] = np.asarray(R, float), np.asarray(t, float)
        self.how[i] = how
        if m is not None:
            self.m[i] = float(m)
        if kappa is not None:
            self.kappa[i] = float(kappa)
        self.order.append(i)

    def placed_obs(self):
        return self.alive & np.array([self.placed[v] for v in self.ov]) if len(self.ov) else np.zeros(0, bool)

    # -------------------------------------------------------------- points
    def triangulate(self):
        po = self.placed_obs()
        cnt = np.bincount(self.ot[po], minlength=self.T)
        tri = np.nonzero((cnt >= 2) & (self.src != 1))[0]
        if tri.size:
            sel = po & np.isin(self.ot, tri)
            idx = np.nonzero(sel)[0]
            order = np.argsort(self.ot[idx], kind="stable")
            idx = idx[order]
            tr = self.ot[idx]
            u, start, counts = np.unique(tr, return_index=True, return_counts=True)
            K = int(counts.max())
            Cs = np.full((len(u), K, 3), np.nan)
            Ds = np.full((len(u), K, 3), np.nan)
            nx = self.norm_xy(idx)
            pos = np.arange(len(idx)) - np.repeat(start, counts)
            row = np.repeat(np.arange(len(u)), counts)
            for k_, i in enumerate(idx):
                v = self.ov[i]
                Cs[row[k_], pos[k_]] = G.centre(self.R[v], self.t[v])
                Ds[row[k_], pos[k_]] = self.R[v].T @ np.array([nx[k_, 0], nx[k_, 1], 1.0])
            X = G.midpoint(Cs, Ds)
            # checks: in front of every view, small reprojection, enough angle
            ok = np.isfinite(X).all(1)
            maxang = np.zeros(len(u))
            for k_, i in enumerate(idx):
                r = row[k_]
                if not ok[r]:
                    continue
                v = self.ov[i]
                xc = self.R[v] @ X[r] + self.t[v]
                if xc[2] <= 0:
                    ok[r] = False
                    continue
                e = np.hypot(xc[0] / xc[2] - nx[k_, 0], xc[1] / xc[2] - nx[k_, 1]) * self.f[v]
                if e > 2 * REPROJ_MAX:
                    ok[r] = False
            for r in range(len(u)):
                if not ok[r]:
                    continue
                c = Cs[r][np.isfinite(Cs[r]).all(1)]
                if len(c) >= 2:
                    a = G.ray_angle_deg(c[:, None, :], c[None, :, :], X[r][None, None, :])
                    maxang[r] = np.nanmax(a)
            ok &= maxang >= TRI_ANGLE
            self.X[u[ok]] = X[ok]
            self.src[u[ok]] = 1
        if self.use_depth:
            self.depth_points()

    def depth_points(self):
        """Points seen by one placed view only, placed by that view's depth."""
        po = self.placed_obs()
        cnt = np.bincount(self.ot[po], minlength=self.T)
        one = (cnt == 1) & (self.src != 1)
        sel = po & one[self.ot] & (self.conf > TV.CONF_MIN)
        idx = np.nonzero(sel)[0]
        if not idx.size:
            return
        w = self.depth_w(idx)
        good = np.isfinite(w) & (w > 1e-6)
        idx, w = idx[good], w[good]
        nx = self.norm_xy(idx)
        for k_, i in enumerate(idx):
            v = self.ov[i]
            Xc = np.array([nx[k_, 0], nx[k_, 1], 1.0]) / w[k_]
            self.X[self.ot[i]] = self.R[v].T @ (Xc - self.t[v])
            self.src[self.ot[i]] = 2

    def fit_depth(self, v):
        """The view's depth shape (m, kappa) from its triangulated points, robust least squares."""
        s = self.alive & (self.ov == v) & (self.src[self.ot] == 1) & (self.conf > TV.CONF_MIN)
        idx = np.nonzero(s)[0]
        if idx.size < 8:
            return False
        Xc = self.X[self.ot[idx]] @ self.R[v].T + self.t[v]
        w = 1.0 / np.maximum(Xc[:, 2], 1e-9)
        tau = self.tau[idx]
        wt = self.conf[idx]
        keep = np.ones(len(idx), bool)
        for _ in range(3):
            A = np.stack([np.ones(keep.sum()), tau[keep]], 1) * np.sqrt(wt[keep])[:, None]
            b = w[keep] * np.sqrt(wt[keep])
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            r = (sol[0] + sol[1] * tau) / w - 1
            mad = np.median(np.abs(r[keep])) * 1.4826 + 1e-9
            keep = np.abs(r) < max(3 * mad, 0.01)
            if keep.sum() < 8:
                return False
        m = float(sol[0])
        if m <= 0:
            return False
        self.m[v], self.kappa[v] = m, float(sol[1] / m)
        return True

    # -------------------------------------------------------------- registration
    def candidates(self, comp):
        has = np.isfinite(self.X[:, 0])
        out = []
        for v in comp:
            if self.placed[v]:
                continue
            s = self.alive & (self.ov == v) & has[self.ot]
            out.append((int(s.sum()), v))
        return sorted(out, reverse=True)

    def register(self, v):
        has = np.isfinite(self.X[:, 0])
        s = self.alive & (self.ov == v) & has[self.ot]
        idx = np.nonzero(s)[0]
        X = self.X[self.ot[idx]]
        nx = self.norm_xy(idx)
        thr = TV.THR_PX / self.f[v]
        rng = np.random.default_rng(v + 17)
        pp = None
        if len(idx) >= 8:
            R, t, inl = G.pnp_ransac(X, nx, thr, rng=rng)
            if R is not None and inl.sum() >= REG_MIN:
                pp = (R, t, inl)
        pd = None
        if self.use_depth:
            pd = self._depth_register(v, idx, X, nx, thr, rng)
        scene = np.nanmedian(np.linalg.norm(X - np.nanmedian(X, 0), axis=1)) + 1e-9
        if pp and pd:
            rot = G.angle_deg(pp[0], pd[0])
            dc = np.linalg.norm(G.centre(pp[0], pp[1]) - G.centre(pd[0], pd[1])) / scene
            if rot < 3.0 and dc < 0.05:
                self.place(v, pp[0], pp[1], "both", pd[3], pd[4])
                self._mark(idx, pp[2] | pd[2])
                return True
        n_p = int(pp[2].sum()) if pp else 0
        n_d = int(pd[2].sum()) if pd else 0
        if pd and n_d > n_p:
            self.place(v, pd[0], pd[1], "depth", pd[3], pd[4])
            self._mark(idx, pd[2])
            return True
        if pp:
            self.place(v, pp[0], pp[1], "matches")
            self._mark(idx, pp[2])
            if self.use_depth:
                self.fit_depth(v) or self._fit_depth_any(v)
            return True
        return False

    def _mark(self, idx, inl):
        self.alive[idx[~inl]] = False

    def _fit_depth_any(self, v):
        """Depth shape from any placed points (also those placed by depth) when too few are triangulated."""
        s = self.alive & (self.ov == v) & np.isfinite(self.X[self.ot, 0]) & (self.conf > TV.CONF_MIN)
        idx = np.nonzero(s)[0]
        if idx.size < 4:
            return False
        Xc = self.X[self.ot[idx]] @ self.R[v].T + self.t[v]
        w = 1 / np.maximum(Xc[:, 2], 1e-9)
        A = np.stack([np.ones(len(idx)), self.tau[idx]], 1)
        sol, *_ = np.linalg.lstsq(A, w, rcond=None)
        if sol[0] <= 0:
            return False
        self.m[v], self.kappa[v] = float(sol[0]), float(sol[1] / sol[0])
        return True

    def _depth_register(self, v, idx, X, nx, thr, rng, samples=300):
        tau, conf = self.tau[idx], self.conf[idx]
        ok = conf > TV.CONF_MIN
        cand = np.nonzero(ok)[0]
        if cand.size < 4:
            return None
        rays = np.concatenate([nx, np.ones((len(nx), 1))], 1)
        smp = np.array([rng.choice(cand, 3, replace=False) for _ in range(samples)])
        best = (-1, None)
        for k in TV.KGRID:
            w = 1 + k * tau
            Q = rays / np.maximum(w, 1e-3)[:, None]
            s, R, t = G.umeyama(X[smp], Q[smp])
            # camera: X_cam(world units) = R^T (X - t); its own depth units: / s
            Xc = np.einsum("sji,smj->smi", R, X[None] - t[:, None, :])        # (S, n, 3)
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.linalg.norm(Xc[..., :2] / Xc[..., 2:3] - nx[None], axis=-1)
                d = np.abs(Xc[..., 2] / s[:, None] * w[None] - 1)
            inl = (e < thr) & (d < TV.DEPTH_TOL) & (Xc[..., 2] > 0) & ok[None] & (w > 0.15)[None]
            cnt = np.where(np.isfinite(s) & (s > 0), inl.sum(1), -1)
            b = int(np.argmax(cnt))
            if cnt[b] > best[0]:
                best = (int(cnt[b]), (k, s[b], R[b], t[b]))
        if best[1] is None or best[0] < REG_MIN:
            return None
        k, s, R, t = best[1]
        Rc, tc = R.T, -R.T @ t

        def errs(p):
            Rm = G.rodrigues(p[:3])
            Xc_ = X @ Rm.T + p[3:6]
            ww = np.exp(p[6]) * (1 + p[7] * tau)
            e_ = Xc_[:, :2] / np.where(np.abs(Xc_[:, 2:3]) > 1e-9, Xc_[:, 2:3], 1e-9) - nx
            d_ = Xc_[:, 2] * ww - 1
            return e_, d_, Xc_, ww
        par = np.concatenate([G.log_rotation(Rc), tc, [np.log(1.0 / s), k]])

        def inl_of(p):
            e_, d_, Xc_, ww = errs(p)
            return (np.linalg.norm(e_, axis=1) < thr) & (np.abs(d_) < TV.DEPTH_TOL) & (Xc_[:, 2] > 0) & ok & (ww > 0)
        inl = inl_of(par)
        for _ in range(2):
            ii = np.nonzero(inl)[0]
            if ii.size < REG_MIN:
                break

            def res(p):
                e_, d_ = errs(p)[:2]
                return np.concatenate([(e_[ii] / thr).ravel(), d_[ii] / 0.01])
            par = least_squares(res, par, loss="huber", max_nfev=150).x
            inl = inl_of(par)
        if inl.sum() < REG_MIN:
            return None
        return (G.rodrigues(par[:3]), par[3:6], inl, float(np.exp(par[6])), float(par[7]))

    # -------------------------------------------------------------- bundle adjustment
    def adjust(self, max_nfev=60, fixed=None):
        """Bundle adjustment (``ba``): Levenberg-Marquardt with analytic derivatives and the Schur complement.
        ``max_nfev``: the most iterations (kept as the name the callers use)."""
        from . import ba as BA
        vs = [v for v in self.order if self.placed[v]]
        if len(vs) < 2:
            return
        fixed = vs[0] if fixed is None else fixed
        po = self.placed_obs() & (self.src[self.ot] == 1)
        cnt = np.bincount(self.ot[po], minlength=self.T)
        po &= cnt[self.ot] >= 2
        idx = np.nonzero(po)[0]
        if idx.size < 10:
            return
        pts = np.unique(self.ot[idx])
        pmap = -np.ones(self.T, int)
        pmap[pts] = np.arange(len(pts))
        vmap = {v: k for k, v in enumerate(vs)}
        use_d = self.use_depth and all(self.m[v] is not None and self.kappa[v] is not None for v in vs)
        nv = len(vs)
        d0 = np.linalg.norm(G.centre(self.R[vs[1]], self.t[vs[1]]) - G.centre(self.R[fixed], self.t[fixed]))
        prob = BA.Problem(
            R=[self.R[v] for v in vs], t=[self.t[v] for v in vs], logf=[np.log(self.f[v]) for v in vs],
            logm=[np.log(self.m[v]) if use_d else 0.0 for v in vs],
            kappa=[self.kappa[v] if use_d else 0.0 for v in vs], k1=self.k1, X=self.X[pts],
            ov=[vmap[v] for v in self.ov[idx]], op=pmap[self.ot[idx]], uv=self.xy[idx], tau=self.tau[idx],
            conf=np.clip(self.conf[idx], 0, 1), use_d=(self.conf[idx] > TV.CONF_MIN) if use_d else
            np.zeros(len(idx), bool), cx=[self.views[v].cx for v in vs], cy=[self.views[v].cy for v in vs],
            logf0=[np.log(self.f0[v]) for v in vs], px_sigma=self.px_sigma, d_sigma=self.depth_sigma,
            f_prior=[max(getattr(self.views[v], "f_sigma", FOCAL_PRIOR), FOCAL_PRIOR / 2) for v in vs],
            f_links=_lens_links(vs, self.views, vmap), link_sigma=BREATHING, fixed=vmap[fixed],
            solve_k1=nv >= FEW_VIEWS, solve_depth=use_d,
            scale_pair=None if use_d else (vmap[fixed], vmap[vs[1]], d0))
        try:
            BA.solve(prob, iters=int(max_nfev))
        except Exception as exc:                     # noqa: BLE001 - a failed adjustment keeps the poses it had
            log.warning("bundle adjustment failed: %s", exc)
            return
        for k, v in enumerate(vs):
            self.R[v] = prob.R[k]
            self.t[v] = prob.t[k]
            self.f[v] = float(np.exp(prob.logf[k]))
            if use_d:
                self.m[v] = float(np.exp(prob.logm[k]))
                self.kappa[v] = float(prob.kappa[k])
        self.k1 = float(prob.k1)
        self.X[pts] = prob.X

    def clean(self):
        """Drop observations over REPROJ_MAX px (or whose depth disagrees by 3 sigma); points left with fewer than
        two are placed again from depth (or forgotten)."""
        po = self.placed_obs()
        idx = np.nonzero(po & np.isfinite(self.X[self.ot, 0]) & (self.src[self.ot] == 1))[0]
        if not idx.size:
            return 0
        e, dr = self.errors(idx)
        bad = e > REPROJ_MAX
        fin = np.isfinite(e) & (e < REPROJ_MAX)
        if fin.sum() > 20:
            # each component of a 2-D error with spread sigma: |e| has median sigma * sqrt(2 ln 2)
            self.px_sigma = float(np.clip(np.median(e[fin]) / 1.1774, MIN_PX_SIGMA, 2.0))
        if self.use_depth:
            ok = np.isfinite(dr)
            if ok.sum() > 20:
                sd = 1.4826 * np.median(np.abs(dr[ok])) + 1e-6
                bad |= ok & (np.abs(dr) > max(3 * sd, 3 * MIN_DEPTH_SIGMA))
                # the depth terms weigh what the depth is worth: its own measured spread (never below the floor)
                self.depth_sigma = float(np.clip(sd, MIN_DEPTH_SIGMA, 0.05))
        self.alive[idx[bad]] = False
        po = self.placed_obs()
        cnt = np.bincount(self.ot[po], minlength=self.T)
        lost = (cnt < 2) & (self.src == 1)
        self.X[lost] = np.nan
        self.src[lost] = 0
        if self.use_depth:
            self.depth_points()
        return int(bad.sum())

    def errors(self, idx):
        v = self.ov[idx]
        X = self.X[self.ot[idx]]
        e = np.full(len(idx), np.inf)
        dr = np.full(len(idx), np.nan)
        for vv in np.unique(v):
            s = v == vv
            Xc = X[s] @ self.R[vv].T + self.t[vv]
            z = np.where(Xc[:, 2] > 1e-9, Xc[:, 2], 1e-9)
            x, y = Xc[:, 0] / z, Xc[:, 1] / z
            d = 1 + self.k1 * (x * x + y * y)
            u = self.f[vv] * x * d + self.views[vv].cx
            w = self.f[vv] * y * d + self.views[vv].cy
            e[s] = np.hypot(u - self.xy[idx[s], 0], w - self.xy[idx[s], 1])
            if self.use_depth and self.m[vv] is not None:
                wm = self.m[vv] * (1 + self.kappa[vv] * self.tau[idx[s]])
                dd = 1 / z / np.where(np.abs(wm) > 1e-9, wm, 1e-9) - 1
                dr[s] = np.where(self.conf[idx[s]] > TV.CONF_MIN, dd, np.nan)
        return e, dr

    def quality(self, v):
        if not self.placed[v]:
            return None
        idx = np.nonzero(self.placed_obs() & (self.ov == v) & np.isfinite(self.X[self.ot, 0]) &
                         (self.src[self.ot] == 1))[0]
        if not idx.size:
            return {"reprojection_px": None, "depth_rms": None, "points": 0}
        e, dr = self.errors(idx)
        dd = dr[np.isfinite(dr)]
        return {"reprojection_px": round(float(np.sqrt(np.mean(e ** 2))), 3),
                "depth_rms": round(float(np.sqrt(np.mean(dd ** 2))), 4) if dd.size else None,
                "points": int(idx.size)}


def _lens_links(vs, views, vmap):
    """Pairs of placed views shot with the same lens, chained (each to the next), for the breathing links."""
    by = {}
    for v in vs:
        by.setdefault(getattr(views[v], "lens", "") or "?", []).append(vmap[v])
    out = []
    for g in by.values():
        out.extend(zip(g[:-1], g[1:]))
    return out


def find_focal(views, feats, raw_matches: dict, factors=None) -> float | None:
    """When the focal length is not known closely (macro: the lens moves out to focus, up to 2x its marked focal
    length), the factor on every view's focal length that lets the essential matrix explain the most matches at a
    tight threshold, over the pairs with the most matches. None when it cannot be told."""
    if factors is None:
        factors = np.geomspace(0.7, 2.4, 15)
    top = sorted(raw_matches.items(), key=lambda kv: -len(kv[1]))[:3]
    if not top:
        return None
    score = np.zeros(len(factors))
    for (i, j), m in top:
        if len(m) < 30:
            continue
        xi = feats[i].xy[m[:, 0]].astype(np.float64)
        xj = feats[j].xy[m[:, 1]].astype(np.float64)
        for k, fa in enumerate(factors):
            fi, fj = views[i].f * fa, views[j].f * fa
            ni = np.stack([(xi[:, 0] - views[i].cx) / fi, (xi[:, 1] - views[i].cy) / fi], -1)
            nj = np.stack([(xj[:, 0] - views[j].cx) / fj, (xj[:, 1] - views[j].cy) / fj], -1)
            thr = 1.0 / (0.5 * (fi + fj))
            R, t, inl = G.essential_pose_ransac(ni, nj, thr, iters=256, rng=np.random.default_rng(k))
            if R is None:
                continue
            d = np.sqrt(G.sampson(G.essential_of(R, t), ni[inl], nj[inl])) * 0.5 * (fi + fj)
            score[k] += inl.sum() - d.sum() / 1.0      # more inliers, tighter fit
    if score.max() <= 0:
        return None
    k = int(np.argmax(score))
    if 0 < k < len(factors) - 1:                     # a parabola through the best and its neighbours (log scale)
        a, b, c = score[k - 1], score[k], score[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        lf = np.log(factors)
        return float(np.exp(lf[k] + off * (lf[1] - lf[0])))
    return float(factors[k])


def _init_pair(pairs, comp):
    best = None
    for (i, j), pr in pairs.items():
        if i not in comp or j not in comp:
            continue
        score = pr.n_inliers * float(np.clip(pr.angle / 15.0, 0.2, 1.0)) * (1.2 if pr.how == "both" else 1.0)
        if best is None or score > best[0]:
            best = (score, pr)
    return best[1] if best else None


def solve(views, use_depth: bool = True, affine: str = "auto", progress=None, cancel=None,
          n_features: int = sift.N_MAX) -> Solution:
    """Place every view (see the module's text). ``affine``: auto | on | off. Returns a Solution."""
    import copy
    views = [copy.copy(v) for v in views]           # the starting focal may be changed: never the caller's views
    n = len(views)

    def prog(f, msg):
        if progress:
            progress(f, msg)
    feats = []
    for k, v in enumerate(views):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        prog(0.15 * k / max(n, 1), f"finding features in {v.name}")
        feats.append(sift.detect(v.image, v.mask, n_max=n_features))
    notes_f = []
    wide = [i for i, v in enumerate(views) if getattr(v, "f_sigma", 0.03) > 0.1]
    if wide and n >= 2:
        raw = {}
        for i in range(n):
            for j in range(i + 1, n):
                m = sift.match(feats[i], feats[j])
                if len(m) >= 30:
                    raw[(i, j)] = m
        fac = find_focal(views, feats, raw)
        if fac is not None and abs(np.log(fac)) > 0.03:
            for i in wide:
                views[i].f *= fac
            notes_f.append(f"The focal length was not known closely (a macro lens moves out to focus): the matches "
                           f"put it at {fac:.2f} x the marked one to start, then each stack's own is solved.")
    tried = {}
    pairs = match_all(views, feats, use_depth, progress=progress, cancel=cancel, base=0.15, span=0.15, tried=tried)
    used_affine = False
    deg = np.zeros(n, int)
    for (i, j) in pairs:
        deg[i] += 1
        deg[j] += 1
    comps = _components(n, list(pairs))
    weak = set(np.nonzero(deg < min(STRONG, n - 1))[0].tolist())
    comp_of = {v: ci for ci, c in enumerate(comps) for v in c}
    if affine == "on":
        weak = set(range(n))
    if (affine in ("auto", "on")) and weak and n >= 2:
        # the views plain matching could not join to two others are matched again with affine simulation, against
        # every view (both sides need their tilted versions); views joined well keep their plain matches
        used_affine = True
        for k, v in enumerate(views):
            if cancel is not None and cancel():
                raise InterruptedError("cancelled")
            prog(0.3 + 0.2 * k / max(n, 1), f"describing {v.name} as if seen tilted (wide angles)")
            feats[k] = sift.extend_affine(v.image, feats[k], v.mask, n_max=n_features)
        redo = {(i, j) for i in range(n) for j in range(i + 1, n)
                if i in weak or j in weak or comp_of[i] != comp_of[j]}
        new = match_all(views, feats, use_depth, pairs_only=redo, progress=progress, cancel=cancel, base=0.5,
                        span=0.2, tried=tried)
        for key in redo:
            pairs.pop(key, None)
        pairs.update(new)
        comps = _components(n, list(pairs))
    sol = Solution(names=[v.name for v in views], placed=[False] * n, R=[None] * n, t=[None] * n,
                   f=[float(v.f) for v in views], m=[None] * n, kappa=[None] * n, how=[None] * n,
                   reason=[""] * n, group=[-1] * n, use_depth=use_depth, affine=used_affine)
    sol.tried = [{"views": [views[i].name, views[j].name], "candidates": c, "held": h}
                 for (i, j), (c, h) in sorted(tried.items())]
    sol.pairs = [{"views": [views[i].name, views[j].name], "matches": int(len(pr.matches)),
                  "inliers": pr.n_inliers, "how": pr.how, "model": pr.model, "angle_deg": round(pr.angle, 1)}
                 for (i, j), pr in sorted(pairs.items())]
    all_pts, all_src, all_ov, all_op, all_xy = [], [], [], [], []
    base_pt = 0
    gi = 0
    for comp in comps:
        if len(comp) == 1:
            sol.reason[comp[0]] = ("shares no detail with any other view (no pair of matches held): another object, "
                                   "or too far round from the others")
            continue
        for v in comp:
            sol.group[v] = gi
        st = _State(views, feats, pairs, set(comp), use_depth)
        pr = _init_pair(pairs, set(comp))
        i, j = pr.i, pr.j
        st.place(i, np.eye(3), np.zeros(3), "first", 1.0 if use_depth else None,
                 pr.kappa_i if use_depth else None)
        st.place(j, pr.R, pr.t, pr.how, pr.m_j if use_depth else None, pr.kappa_j if use_depth else None)
        st.triangulate()
        if use_depth:
            for v in (i, j):
                if st.kappa[v] is None or st.m[v] is None:
                    if not st.fit_depth(v):
                        st._fit_depth_any(v)
            st.depth_points()
        st.adjust(max_nfev=40)
        st.clean()
        st.triangulate()
        placed_since = 0
        while True:
            if cancel is not None and cancel():
                raise InterruptedError("cancelled")
            cands = st.candidates(comp)
            if not cands or cands[0][0] < REG_MIN:
                for c, v in cands:
                    sol.reason[v] = (f"too few points shared with the placed views ({c}): needs more overlap with "
                                     "its neighbours")
                break
            placed_any = False
            for c, v in cands:
                if c < REG_MIN:
                    sol.reason[v] = f"too few points shared with the placed views ({c})"
                    continue
                if st.register(v):
                    placed_any = True
                    prog(0.7 + 0.25 * len(st.order) / max(len(comp), 1), f"placed {views[v].name}")
                    break
                sol.reason[v] = "its shared points disagree: no pose explains them (moved, or repeated detail)"
            if not placed_any:
                break
            st.triangulate()
            placed_since += 1
            if placed_since >= 2:
                st.adjust(max_nfev=30)
                st.clean()
                st.triangulate()
                placed_since = 0
        for rnd in range(3):
            st.adjust(max_nfev=80)
            if st.clean() == 0 and rnd:
                break
            st.triangulate()
        st.adjust(max_nfev=80)
        for v in comp:
            if st.placed[v]:
                sol.placed[v] = True
                sol.reason[v] = ""
                sol.R[v], sol.t[v], sol.f[v] = st.R[v], st.t[v], st.f[v]
                sol.m[v], sol.kappa[v], sol.how[v] = st.m[v], st.kappa[v], st.how[v]
        sol.k1 = st.k1
        good = np.isfinite(st.X[:, 0])
        remap = -np.ones(st.T, int)
        remap[good] = base_pt + np.arange(good.sum())
        all_pts.append(st.X[good])
        all_src.append(st.src[good])
        po = st.placed_obs() & good[st.ot]
        all_ov.append(st.ov[po])
        all_op.append(remap[st.ot[po]])
        all_xy.append(st.xy[po])
        base_pt += int(good.sum())
        for v in comp:
            q = st.quality(v)
            while len(sol.quality) < n:
                sol.quality.append(None)
            sol.quality[v] = q
        gi += 1
    while len(sol.quality) < n:
        sol.quality.append(None)
    if all_pts:
        sol.points = np.concatenate(all_pts)
        sol.point_src = np.concatenate(all_src)
        sol.obs_view = np.concatenate(all_ov)
        sol.obs_point = np.concatenate(all_op)
        sol.obs_xy = np.concatenate(all_xy)
    sol.notes.extend(notes_f)
    if used_affine:
        sol.notes.append("Some views were too far round for plain matching: every view was also described as if "
                         "seen tilted (affine simulation), which matches views up to about 80 degrees apart.")
    prog(1.0, "views placed")
    return sol


def why_nothing_placed(sol: Solution) -> str:
    """Plain words for a set where no two views could be placed together: the closest pair and what it means."""
    if not sol.tried:
        return "no two views share any detail the Studio could find"
    best = max(sol.tried, key=lambda t: (t["held"], t["candidates"]))
    return (f"no two views share enough detail to be placed together. The closest pair, {best['views'][0]} and "
            f"{best['views'][1]}, had {best['candidates']} candidate matches, of which {best['held']} agreed on one "
            f"camera position ({TV.MIN_INLIERS} are needed). This happens when neighbouring stacks are far apart "
            "round the object (more than about 60 degrees), on smooth or plain subjects, or on detail that repeats "
            "(leaf litter, gravel): shoot neighbours about 20-30 degrees apart so each shares about half of what the "
            "last one saw, keeping the subject large in the frame")


def compare(sol: Solution, R_true: list, C_true: list) -> dict:
    """Errors against known poses after the best similarity (for tests and the RealityScan cross-check): rotation
    (degrees) and position (share of the cameras' spread) per placed view."""
    idx = [i for i in range(len(sol.names)) if sol.placed[i]]
    if len(idx) < 3:
        return {"placed": len(idx), "rot_deg": [], "pos": []}
    Cs = np.array([sol.centre(i) for i in idx])
    Ct = np.array([C_true[i] for i in idx])
    s, R, t = G.umeyama(Ct, Cs)
    spread = np.sqrt(((Ct - Ct.mean(0)) ** 2).sum(1)).mean()
    pos = np.linalg.norm((s * Cs @ R.T + t) - Ct, axis=1) / spread
    # rotations: the frames aligned by the views' own rotations (chordal mean), so a position error does not show
    # as a rotation error through the alignment
    Rg = G.nearest_rotation(sum(np.asarray(R_true[i]).T @ sol.R[i] for i in idx))
    rot = [G.angle_deg(sol.R[i] @ Rg.T, R_true[i]) for i in idx]
    return {"placed": len(idx), "rot_deg": rot, "pos": list(pos), "scale": float(s)}

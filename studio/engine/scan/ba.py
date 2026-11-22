"""Bundle adjustment with Dual Pixel depth (BUILD_PLAN_DP3D E.5): Levenberg-Marquardt with analytic derivatives and
the Schur complement (Triggs, McLauchlan, Hartley and Fitzgibbon, "Bundle adjustment: a modern synthesis", 1999).

Per view: a rotation update (3, applied on the left: R <- exp(d) R), the translation (3), log focal length, log depth
scale m and the depth shape kappa (9). Shared: one radial distortion k1. Per point: its position (3).

Residuals (each divided by its measured spread, then Huber-weighted by iteratively reweighting):
* reprojection, u and v, pixels;
* depth: 1 / (z m (1 + kappa tau)) - 1, the point's inverse distance against the view's fused depth at that pixel,
  weighted by the depth's confidence;
* priors: log focal length round its EXIF value; k1 round 0.

The points are eliminated per point (3 x 3 blocks), the cameras solved densely (a few hundred numbers), then the points
back-substituted. scipy's general least squares with numerical derivatives stalls on this problem (it was tried
first: it stopped a few degrees from the truth on five views 72 degrees apart); this converges in a few iterations.
"""
from __future__ import annotations

import numpy as np

from . import geom as G

NP = 9          # parameters per view


def _scatter(nc, va, vb, blk):
    """A (nc, nc) matrix with each (NP x NP) block of ``blk`` added at block (va, vb)."""
    i = (va[:, None, None] * NP + np.arange(NP)[None, :, None])
    j = (vb[:, None, None] * NP + np.arange(NP)[None, None, :])
    return np.bincount((i * nc + j).ravel(), blk.ravel(), minlength=nc * nc).reshape(nc, nc)


def _skew(v):
    z = np.zeros(v.shape[:-1] + (3, 3))
    z[..., 0, 1], z[..., 0, 2], z[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    z[..., 1, 0], z[..., 2, 0], z[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return z


class Problem:
    """``R`` (V, 3, 3), ``t`` (V, 3), ``logf``, ``logm``, ``kappa`` (V,), ``k1``; points ``X`` (P, 3); observations
    ``ov`` (view), ``op`` (point), ``uv`` (pixels), ``tau``, ``conf``, ``use_d`` (bool per observation)."""

    def __init__(self, R, t, logf, logm, kappa, k1, X, ov, op, uv, tau, conf, use_d, cx, cy, logf0,
                 px_sigma=1.0, d_sigma=0.01, f_prior=0.02, k1_prior=0.03, fixed=0, solve_k1=True,
                 solve_depth=True, scale_pair=None, f_links=None, link_sigma=0.03):
        self.R = np.array(R, float)
        self.t = np.array(t, float)
        self.logf = np.array(logf, float)
        self.logm = np.array(logm, float)
        self.kappa = np.array(kappa, float)
        self.k1 = float(k1)
        self.X = np.array(X, float)
        self.ov, self.op = np.asarray(ov, int), np.asarray(op, int)
        self.uv = np.asarray(uv, float)
        self.tau, self.conf = np.asarray(tau, float), np.asarray(conf, float)
        self.use_d = np.asarray(use_d, bool) & solve_depth
        self.cx, self.cy = np.asarray(cx, float), np.asarray(cy, float)
        self.logf0 = np.asarray(logf0, float)
        self.px, self.ds = float(px_sigma), float(d_sigma)
        self.fp = np.broadcast_to(np.asarray(f_prior, float), (len(R),)).copy()
        self.kp = float(k1_prior)
        self.links = [] if f_links is None else [(int(a), int(b)) for a, b in f_links]
        self.ls = float(link_sigma)
        self.fixed = int(fixed)
        self.solve_k1 = bool(solve_k1)
        self.solve_depth = bool(solve_depth)
        self.scale_pair = scale_pair          # (a, b, d0): without depth, hold |C_b - C_a| = d0
        self.V, self.P = len(self.R), len(self.X)

    # ---------------------------------------------------------------- residuals and Jacobians
    def evaluate(self, jac=True):
        ov, op = self.ov, self.op
        R, t = self.R[ov], self.t[ov]
        RX = np.einsum("nij,nj->ni", R, self.X[op])
        Xc = RX + t
        z = np.where(Xc[:, 2] > 1e-9, Xc[:, 2], 1e-9)
        x, y = Xc[:, 0] / z, Xc[:, 1] / z
        r2 = x * x + y * y
        d = 1 + self.k1 * r2
        f = np.exp(self.logf[ov])
        u = f * x * d + self.cx[ov]
        v = f * y * d + self.cy[ov]
        ru = (u - self.uv[:, 0]) / self.px
        rv = (v - self.uv[:, 1]) / self.px
        wm = np.exp(self.logm[ov]) * (1 + self.kappa[ov] * self.tau)
        wm = np.where(np.abs(wm) > 1e-9, wm, 1e-9)
        sq = np.sqrt(np.clip(self.conf, 0, 1))
        rd = np.where(self.use_d, (1.0 / (z * wm) - 1.0) / self.ds * sq, 0.0)
        out = {"ru": ru, "rv": rv, "rd": rd}
        if not jac:
            return out
        # d(x, y)/d(Xc)
        iz = 1.0 / z
        dx = np.stack([iz, np.zeros_like(z), -x * iz], -1)
        dy = np.stack([np.zeros_like(z), iz, -y * iz], -1)
        dudx = f * (d + 2 * self.k1 * x * x)
        dudy = f * (2 * self.k1 * x * y)
        dvdx = dudy
        dvdy = f * (d + 2 * self.k1 * y * y)
        du = (dudx[:, None] * dx + dudy[:, None] * dy) / self.px          # (n, 3) d u / d Xc
        dv = (dvdx[:, None] * dx + dvdy[:, None] * dy) / self.px
        # d Xc / d (rotation update) = -[R X]x ; / d t = I ; / d X = R
        dR = -_skew(RX)
        Jc_u = np.zeros((len(ov), NP))
        Jc_v = np.zeros((len(ov), NP))
        Jc_u[:, 0:3] = np.einsum("ni,nij->nj", du, dR)
        Jc_v[:, 0:3] = np.einsum("ni,nij->nj", dv, dR)
        Jc_u[:, 3:6] = du
        Jc_v[:, 3:6] = dv
        Jc_u[:, 6] = f * x * d / self.px
        Jc_v[:, 6] = f * y * d / self.px
        Jp_u = np.einsum("ni,nij->nj", du, R)
        Jp_v = np.einsum("ni,nij->nj", dv, R)
        Jk_u = f * x * r2 / self.px
        Jk_v = f * y * r2 / self.px
        # depth: rd = (1 / (z wm) - 1) / ds * sq
        g = np.where(self.use_d, sq / self.ds, 0.0)
        drdz = -g / (z * z * wm)
        dzc = np.zeros((len(ov), 3))
        dzc[:, 2] = 1.0
        Jc_d = np.zeros((len(ov), NP))
        Jc_d[:, 0:3] = drdz[:, None] * np.einsum("ni,nij->nj", dzc, dR)
        Jc_d[:, 3:6] = drdz[:, None] * dzc
        Jc_d[:, 7] = -g / (z * wm)
        Jc_d[:, 8] = -g / (z * wm) * self.tau / np.where(np.abs(1 + self.kappa[ov] * self.tau) > 1e-9,
                                                            1 + self.kappa[ov] * self.tau, 1e-9)
        Jp_d = drdz[:, None] * R[:, 2, :]
        out.update(Jc=(Jc_u, Jc_v, Jc_d), Jp=(Jp_u, Jp_v, Jp_d), Jk=(Jk_u, Jk_v))
        return out

    def cost(self, ev=None, huber=1.5):
        ev = ev or self.evaluate(jac=False)
        r = np.concatenate([ev["ru"], ev["rv"], ev["rd"]])
        a = np.abs(r)
        c = np.where(a <= huber, 0.5 * r * r, huber * (a - 0.5 * huber)).sum()
        c += 0.5 * np.sum(((self.logf - self.logf0) / self.fp) ** 2) + 0.5 * (self.k1 / self.kp) ** 2
        for a_, b_ in self.links:
            c += 0.5 * ((self.logf[a_] - self.logf[b_]) / self.ls) ** 2
        if self.scale_pair is not None:
            a_, b_, d0 = self.scale_pair
            c += 0.5 * self._scale_res() ** 2
        return float(c)

    def _centre(self, i):
        return -self.R[i].T @ self.t[i]

    def _scale_res(self):
        a, b, d0 = self.scale_pair
        return (np.linalg.norm(self._centre(b) - self._centre(a)) - d0) / (1e-3 * max(d0, 1e-12))

    # ---------------------------------------------------------------- one damped step
    def step(self, lam, huber=1.5):
        ev = self.evaluate(jac=True)
        V, P = self.V, self.P
        nc = V * NP + 1
        Hcc = np.zeros((nc, nc))
        gc = np.zeros(nc)
        Hpp = np.zeros((P, 3, 3))
        gp = np.zeros((P, 3))
        Hcp_blocks = []           # per observation: (view, point, 9 x 3 block, k1 row 1 x 3)
        ov, op = self.ov, self.op
        Bvp = np.zeros((len(ov), NP, 3))
        Bkp = np.zeros((len(ov), 3))
        for key, (Jc, Jp, Jk, r) in {
                "u": (ev["Jc"][0], ev["Jp"][0], ev["Jk"][0], ev["ru"]),
                "v": (ev["Jc"][1], ev["Jp"][1], ev["Jk"][1], ev["rv"]),
                "d": (ev["Jc"][2], ev["Jp"][2], np.zeros(len(ov)), ev["rd"])}.items():
            a = np.abs(r)
            w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))      # IRLS weights for Huber
            if key == "d":
                w = np.where(self.use_d, w, 0.0)
            # camera-camera: same view
            blk = np.einsum("n,ni,nj->nij", w, Jc, Jc)
            Hcc += _scatter(nc, ov, ov, blk)
            gc += np.bincount((ov[:, None] * NP + np.arange(NP)[None]).ravel(),
                              ((w * r)[:, None] * Jc).ravel(), minlength=nc)
            # k1
            kc = V * NP
            Hcc[kc, kc] += np.sum(w * Jk * Jk)
            gc[kc] += np.sum(w * r * Jk)
            kx = np.bincount((ov[:, None] * NP + np.arange(NP)[None]).ravel(), ((w * Jk)[:, None] * Jc).ravel(),
                             minlength=nc)
            kx[kc] = 0.0
            Hcc[kc] += kx
            Hcc[:, kc] += kx
            # point-point
            np.add.at(Hpp, op, np.einsum("n,ni,nj->nij", w, Jp, Jp))
            np.add.at(gp, op, (w * r)[:, None] * Jp)
            # camera-point
            Bvp += np.einsum("n,ni,nj->nij", w, Jc, Jp)
            Bkp += (w * Jk)[:, None] * Jp
        # priors
        for i in range(V):
            Hcc[i * NP + 6, i * NP + 6] += 1.0 / self.fp[i] ** 2
            gc[i * NP + 6] += (self.logf[i] - self.logf0[i]) / self.fp[i] ** 2
        for a_, b_ in self.links:                   # focus breathing: stacks of one lens change their focal smoothly
            ia, ib = a_ * NP + 6, b_ * NP + 6
            w_ = 1.0 / self.ls ** 2
            r_ = self.logf[a_] - self.logf[b_]
            Hcc[ia, ia] += w_
            Hcc[ib, ib] += w_
            Hcc[ia, ib] -= w_
            Hcc[ib, ia] -= w_
            gc[ia] += w_ * r_
            gc[ib] -= w_ * r_
        Hcc[V * NP, V * NP] += 1.0 / self.kp ** 2
        gc[V * NP] += self.k1 / self.kp ** 2
        if self.scale_pair is not None:
            # numerical derivative of the one scale residual w.r.t. the two views' translations
            a_, b_, d0 = self.scale_pair
            r0 = self._scale_res()
            J = np.zeros(nc)
            for vv in (a_, b_):
                for c in range(3):
                    old = self.t[vv, c]
                    self.t[vv, c] = old + 1e-6
                    J[vv * NP + 3 + c] = (self._scale_res() - r0) / 1e-6
                    self.t[vv, c] = old
            Hcc += np.outer(J, J)
            gc += J * r0
        # fixed and unused parameters
        fixed = np.zeros(nc, bool)
        f0 = self.fixed * NP
        fixed[f0:f0 + 6] = True
        if self.solve_depth:
            fixed[f0 + 7] = True
        else:
            fixed[np.arange(V) * NP + 7] = True
            fixed[np.arange(V) * NP + 8] = True
        if not self.solve_k1:
            fixed[V * NP] = True
        # damping
        dc = np.diag(Hcc).copy()
        Hcc_d = Hcc + lam * np.diag(np.maximum(dc, 1e-9))
        Hpp_d = Hpp + lam * np.einsum("pii->pi", Hpp)[:, :, None] * np.eye(3)[None] + 1e-12 * np.eye(3)[None]
        Hpp_inv = np.linalg.inv(Hpp_d)
        # Schur complement over the points
        Y = np.einsum("nij,njk->nik", Bvp, Hpp_inv[op])             # (n, 9, 3)
        Yk = np.einsum("nj,njk->nk", Bkp, Hpp_inv[op])               # (n, 3)
        S = Hcc_d.copy()
        rhs = gc.copy()
        # rhs -= B Hpp^-1 gp
        rhs -= np.bincount((ov[:, None] * NP + np.arange(NP)[None]).ravel(),
                           np.einsum("nik,nk->ni", Y, gp[op]).ravel(), minlength=nc)
        rhs[V * NP] -= np.sum(np.einsum("nk,nk->n", Yk, gp[op]))
        # S -= sum over pairs of observations of the same point: Y_a B_b^T
        order = np.argsort(op, kind="stable")
        ops = op[order]
        starts = np.r_[0, np.nonzero(np.diff(ops))[0] + 1]
        ends = np.r_[starts[1:], len(ops)]
        lens = ends - starts
        ia, ib = [], []
        for L in np.unique(lens):
            st = starts[lens == L]
            for a in range(L):
                for b in range(L):
                    ia.append(order[st + a])
                    ib.append(order[st + b])
        ia = np.concatenate(ia)
        ib = np.concatenate(ib)
        blk = np.einsum("nik,njk->nij", Y[ia], Bvp[ib])               # (pairs, 9, 9)
        va, vb = ov[ia], ov[ib]
        S -= _scatter(nc, va, vb, blk)
        kc = V * NP
        kb = np.einsum("nk,njk->nj", Yk[ia], Bvp[ib])                # k1 row with views
        kbv = np.bincount((vb[:, None] * NP + np.arange(NP)[None]).ravel(), kb.ravel(), minlength=nc)
        kbv[kc] = 0.0
        S[kc] -= kbv
        S[:, kc] -= kbv
        S[kc, kc] -= np.sum(np.einsum("nk,nk->n", Yk[ia], Bkp[ib]))
        # solve with the fixed ones held
        free = ~fixed
        dc_ = np.zeros(nc)
        try:
            dc_[free] = -np.linalg.solve(S[np.ix_(free, free)], rhs[free])
        except np.linalg.LinAlgError:
            dc_[free] = -np.linalg.lstsq(S[np.ix_(free, free)], rhs[free], rcond=None)[0]
        # back-substitute the points: dp = -Hpp^-1 (gp + B^T dc)
        Bt_dc = np.zeros((P, 3))
        np.add.at(Bt_dc, op, np.einsum("nij,ni->nj", Bvp, dc_[ov[:, None] * NP + np.arange(NP)[None]]) +
                  Bkp * dc_[kc])
        dp = -np.einsum("pij,pj->pi", Hpp_inv, gp + Bt_dc)
        return dc_, dp

    def apply(self, dc, dp):
        V = self.V
        d = dc[:V * NP].reshape(V, NP)
        self.R = G.rodrigues(d[:, 0:3]) @ self.R
        self.t = self.t + d[:, 3:6]
        self.logf = self.logf + d[:, 6]
        self.logm = self.logm + d[:, 7]
        self.kappa = self.kappa + d[:, 8]
        self.k1 = self.k1 + dc[V * NP]
        self.X = self.X + dp

    def snapshot(self):
        return (self.R.copy(), self.t.copy(), self.logf.copy(), self.logm.copy(), self.kappa.copy(), self.k1,
                self.X.copy())

    def restore(self, s):
        self.R, self.t, self.logf, self.logm, self.kappa, self.k1, self.X = (s[0].copy(), s[1].copy(), s[2].copy(),
                                                                            s[3].copy(), s[4].copy(), s[5],
                                                                            s[6].copy())


def solve(prob: Problem, iters: int = 30, lam: float = 1e-3, tol: float = 1e-7) -> dict:
    """Levenberg-Marquardt: a step is kept when it lowers the cost (the damping halves), else undone (it grows)."""
    cost = prob.cost()
    start = cost
    it = 0
    for it in range(iters):
        snap = prob.snapshot()
        try:
            dc, dp = prob.step(lam)
        except Exception:                            # noqa: BLE001 - a singular system: more damping
            lam *= 10
            continue
        prob.apply(dc, dp)
        new = prob.cost()
        if np.isfinite(new) and new < cost:
            rel = (cost - new) / max(cost, 1e-12)
            cost = new
            lam = max(lam / 3.0, 1e-9)
            if rel < tol:
                break
        else:
            prob.restore(snap)
            lam *= 8.0
            if lam > 1e8:
                break
    return {"cost": cost, "start": start, "iterations": it + 1}

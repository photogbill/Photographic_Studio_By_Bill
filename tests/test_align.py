"""Registration for the stacking tools (studio/engine/stack/align.py), on synthetic scenes with known motion."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter, map_coordinates

from conftest import scene
from studio.engine.stack import align as A


def _lum(h, w, seed=0, blur=1.0):
    """A band-limited luminance scene (a lens never delivers infinitely hard edges)."""
    s = A.luma(scene(h, w, seed))
    return gaussian_filter(s, blur).astype(np.float32)


def _shifted(img, dy, dx):
    return np.fft.ifft2(fourier_shift(np.fft.fft2(img), (dy, dx))).real.astype(np.float32)


def _noisy(img, sigma, seed):
    return (img + np.random.default_rng(seed).normal(0, sigma, img.shape)).astype(np.float32)


def _with_transform(T: A.Transform, h=512, w=768, m=48, seed=0):
    """(reference, image) where the content at p in the reference lies at T(p) in the image (crop coordinates)."""
    big = _lum(h + 2 * m, w + 2 * m, seed)
    O = np.array([[1, 0, m], [0, 1, m], [0, 0, 1.0]])
    Tb = A.Transform(O @ T.matrix @ np.linalg.inv(O), T.kind)
    img_big = A.warp(big, Tb.inverse(), order=3)
    return big[m:m + h, m:m + w], img_big[m:m + h, m:m + w]


def _similarity(scale, angle_deg, dy, dx, h, w):
    """A similarity about the frame centre, then a shift."""
    a = np.radians(angle_deg)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    R = np.array([[scale * np.cos(a), -scale * np.sin(a), 0], [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1]])
    C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    S = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return A.Transform(S @ C @ R @ np.linalg.inv(C), "similarity")


def test_known_subpixel_shift_is_recovered():
    ref = _lum(560, 816)
    img = _shifted(ref, 3.37, -1.62)
    ref, img = ref[24:-24, 24:-24], img[24:-24, 24:-24]
    ref, img = _noisy(ref, 0.004, 1), _noisy(img, 0.004, 2)
    T = A.register(ref, img, kind="shift")
    dy, dx = T.shift
    assert abs(dy - 3.37) < 0.05 and abs(dx + 1.62) < 0.05, (dy, dx)


def test_similarity_scale_rotation_and_shift():
    h, w = 512, 768
    truth = _similarity(1.012, 0.25, 5.0, -3.0, h, w)
    ref, img = _with_transform(truth, h, w)
    ref, img = _noisy(ref, 0.004, 3), _noisy(img, 0.004, 4)
    T = A.register(ref, img, kind="similarity")
    assert abs(T.scale() - 1.012) < 1e-4, T.scale()
    assert abs(T.angle_deg() - 0.25) < 0.01, T.angle_deg()
    ys = np.array([0, 0, h - 1, h - 1.0])
    xs = np.array([0, w - 1, 0, w - 1.0])
    Y, X = T.apply(ys, xs)
    Yt, Xt = truth.apply(ys, xs)
    assert np.max(np.hypot(Y - Yt, X - Xt)) < 0.2


def test_a_moving_object_does_not_pull_the_fit():
    ref = _lum(560, 816, seed=5)
    img = _shifted(ref, 2.3, 1.1)
    moved = _shifted(ref, 2.3 + 4.0, 1.1)
    H, W = ref.shape
    img[: H // 2, : W // 2] = moved[: H // 2, : W // 2]       # a quarter of the frame moved 4 px more
    ref, img = ref[24:-24, 24:-24], img[24:-24, 24:-24]
    T = A.register(_noisy(ref, 0.003, 6), _noisy(img, 0.003, 7), kind="similarity")
    h, w = ref.shape
    Y, X = T.apply(np.array([h / 2.0]), np.array([w / 2.0]))
    assert abs(Y[0] - h / 2.0 - 2.3) < 0.1 and abs(X[0] - w / 2.0 - 1.1) < 0.1, (Y - h / 2, X - w / 2)
    assert abs(T.scale() - 1.0) < 5e-4


def test_flow_recovers_a_smooth_warp():
    h, w, m = 512, 768, 32
    big = _lum(h + 2 * m, w + 2 * m, seed=8)
    Y, X = np.meshgrid(np.arange(h + 2 * m, dtype=float), np.arange(w + 2 * m, dtype=float), indexing="ij")
    ddy = 1.5 * np.sin(2 * np.pi * X / 300.0)
    ddx = 1.5 * np.cos(2 * np.pi * Y / 350.0)
    img = map_coordinates(big, [Y - ddy, X - ddx], order=3, mode="mirror").astype(np.float32)
    ref, img = big[m:-m, m:-m], img[m:-m, m:-m]
    ddy, ddx = ddy[m:-m, m:-m], ddx[m:-m, m:-m]
    f = A.flow(_noisy(ref, 0.003, 9), _noisy(img, 0.003, 10), tile=64, step=32, max_shift=6)
    fy, fx, q = f.fields(ref.shape)
    inner = (slice(32, -32), slice(32, -32))
    err = np.hypot(fy[inner] - ddy[inner], fx[inner] - ddx[inner])
    assert float(err.mean()) < 0.15, err.mean()


def test_warp_then_inverse_returns_the_image():
    h, w = 400, 600
    img = _lum(h, w, seed=11, blur=1.2)
    T = _similarity(1.008, 0.4, 2.2, -1.7, h, w)
    back = A.warp(A.warp(img, T), T.inverse())
    m = 16
    d = back[m:-m, m:-m] - img[m:-m, m:-m]
    psnr = 10 * np.log10(1.0 / np.mean(d * d))
    assert psnr > 40, psnr


def test_warp_colour_and_flow_shapes():
    img = np.random.default_rng(0).random((64, 80, 3)).astype(np.float32)
    out = A.warp(img, A.Transform.identity(), order=3)
    assert out.shape == img.shape and np.allclose(out, img, atol=1e-4)
    f = A.Flow(np.zeros((3, 4), np.float32), np.ones((3, 4), np.float32), np.ones((3, 4), np.float32), 8, 8, 20)
    moved = A.warp(img, A.Transform.identity(), order=1, flw=f)
    assert np.allclose(moved[:, 10:70], img[:, 11:71], atol=1e-5)


@pytest.mark.parametrize("kind", ["shift", "similarity", "affine", "homography"])
def test_fit_recovers_each_model_exactly(kind):
    rng = np.random.default_rng(3)
    cen = np.stack([rng.uniform(0, 1000, 60), rng.uniform(0, 1500, 60)], 1)
    M = {"shift": [[1, 0, 2.5], [0, 1, -1.25], [0, 0, 1]],
         "similarity": [[1.01, -0.004, 3.0], [0.004, 1.01, -2.0], [0, 0, 1]],
         "affine": [[1.01, 0.003, 3.0], [-0.002, 0.995, -2.0], [0, 0, 1]],
         "homography": [[1.01, 0.003, 3.0], [-0.002, 0.995, -2.0], [2e-6, -1e-6, 1]]}[kind]
    T = A.Transform(np.asarray(M, float), kind)
    Y, X = T.apply(cen[:, 0], cen[:, 1])
    sh = np.stack([Y, X], 1) - cen
    sh[:6] += 7.0                                         # six tiles fooled: dropped by the robust rounds
    got = A.fit(cen, sh, np.ones(60), kind)
    Y2, X2 = got.apply(cen[6:, 0], cen[6:, 1])
    assert np.max(np.hypot(Y2 - Y[6:], X2 - X[6:])) < 1e-6
    assert got.tiles_used == 54


def test_rescaled_transform_matches_binning():
    T = _similarity(1.01, 0.3, 4.0, -2.0, 200, 300)
    k = 4
    full = T.rescaled(k)
    # a point in the binned frame and its full-resolution centre must map consistently
    yb, xb = np.array([50.0]), np.array([70.0])
    Yb, Xb = T.apply(yb, xb)
    Yf, Xf = full.apply(yb * k + (k - 1) / 2.0, xb * k + (k - 1) / 2.0)
    assert np.allclose(Yf, Yb * k + (k - 1) / 2.0) and np.allclose(Xf, Xb * k + (k - 1) / 2.0)


def test_lens_auto_uses_the_one_phase_shift():
    from studio.engine.lens import auto
    assert auto._phase_shift is A.phase_shift


def test_flat_frame_gives_no_tiles_and_identity():
    # Two frames of a featureless grey card: nothing but independent noise in each, so no tile has anything
    # to line up on. (Adding new noise to a copy of the first frame would not do: the copy keeps the first
    # frame's noise pattern, which is real, shared texture and lines up correctly at zero shift.)
    ref = np.full((256, 256), 0.3, np.float32) + np.random.default_rng(0).normal(0, 0.01, (256, 256)).astype(
        np.float32)
    img = np.full((256, 256), 0.3, np.float32) + np.random.default_rng(1).normal(0, 0.01, (256, 256)).astype(
        np.float32)
    cen, sh, q = A.tile_shifts(ref, img, 64, 48)
    assert cen.shape[0] == 0
    shared = ref + np.random.default_rng(2).normal(0, 0.01, (256, 256)).astype(np.float32)
    cen, sh, q = A.tile_shifts(ref, shared, 64, 48)
    assert cen.shape[0] == 0 or np.abs(sh).max() < 0.25      # shared noise pattern: found, and at zero shift
    T = A.register(ref, img, "similarity")
    assert T.displacement(ref.shape) < 1.5

"""Property tests for pure functions this branch changed or tests.

HYPOTHESIS_PROFILE=stress uses 2000 examples. The default and CI profiles use 200.
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

from util.augmentation import (
    RandomBrightness,
    RandomCrop,
    RandomCropOut,
    RandomFlip,
    RandomNoise,
)
from util.MF_dataset import MF_dataset
from util.util import getScores, should_report_fps

_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "default")
if _PROFILE not in ("default", "ci", "stress"):
    _PROFILE = "default"
_MAX_EXAMPLES = 2000 if _PROFILE == "stress" else 200

settings.register_profile(
    "default",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "stress",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(_PROFILE)

PROP = settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _same(got, expected, rel=1e-8, abs_=1e-8):
    gf = float(got)
    ef = float(expected)
    if math.isnan(gf) or math.isnan(ef):
        assert math.isnan(gf) and math.isnan(ef)
        return
    if math.isinf(gf) or math.isinf(ef):
        assert gf == ef
        return
    assert gf == pytest.approx(ef, rel=rel, abs=abs_)


def reference_get_scores(conf_matrix):
    """Independent TN/FP/FN/TP implementation of the binary segmentation scores."""
    conf = np.asarray(conf_matrix)
    if conf.sum() == 0:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    conf = conf.astype(np.float64)
    tn, fp = conf[0, 0], conf[0, 1]
    fn, tp = conf[1, 0], conf[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        total = tn + fp + fn + tp
        globalacc = (tn + tp) / total
        pre_p = tp / (tp + fp)
        rec_p = tp / (tp + fn)
        iou_p = tp / (tp + fp + fn)
        pre_b = tn / (tn + fn)
        rec_b = tn / (tn + fp)
        iou_b = tn / (tn + fp + fn)
        f_p = 2 * (rec_p * pre_p) / (rec_p + pre_p)
        f_b = 2 * (rec_b * pre_b) / (rec_b + pre_b)
    return (globalacc, pre_p, rec_p, f_p, iou_p, pre_b, rec_b, f_b, iou_b)


def _finite_nonnegative(conf):
    arr = np.asarray(conf)
    return (
        arr.shape == (2, 2)
        and np.issubdtype(arr.dtype, np.number)
        and np.isfinite(arr).all()
        and (arr >= 0).all()
        and np.isfinite(arr.sum())
    )


def _assert_ranges(conf, got):
    tn, fp = float(conf[0, 0]), float(conf[0, 1])
    fn, tp = float(conf[1, 0]), float(conf[1, 1])
    if conf.sum() == 0:
        assert got == (0, 0, 0, 0, 0, 0, 0, 0, 0)
        return
    slack = 1e-5 if np.issubdtype(conf.dtype, np.floating) else 0.0
    assert -slack <= float(got[0]) <= 1.0 + slack
    specs = (
        (1, tp + fp),
        (2, tp + fn),
        (4, tp + fp + fn),
        (5, tn + fn),
        (6, tn + fp),
        (8, tn + fp + fn),
    )
    for idx, den in specs:
        val = float(got[idx])
        if den == 0:
            assert math.isnan(val)
        else:
            assert -slack <= val <= 1.0 + slack
    for fidx, pidx, ridx in ((3, 1, 2), (7, 5, 6)):
        fv, pv, rv = float(got[fidx]), float(got[pidx]), float(got[ridx])
        if math.isnan(pv) or math.isnan(rv) or (pv + rv) == 0.0:
            assert math.isnan(fv)
        else:
            assert -slack <= fv <= 1.0 + slack


def _compare_sklearn(conf, got):
    arr = np.asarray(conf)
    if arr.shape != (2, 2) or not np.issubdtype(arr.dtype, np.integer):
        return
    if (arr < 0).any():
        return
    total = int(arr.sum())
    if total <= 0 or total > 400:
        return
    y_true = np.repeat(np.array([0, 0, 1, 1]), arr.ravel())
    y_pred = np.repeat(np.array([0, 1, 0, 1]), arr.ravel())
    acc = accuracy_score(y_true, y_pred)
    _same(got[0], acc, rel=1e-9, abs_=1e-9)
    kwargs = {"labels": [0, 1], "average": None, "zero_division": np.nan}
    prec = precision_score(y_true, y_pred, **kwargs)
    rec = recall_score(y_true, y_pred, **kwargs)
    f1 = f1_score(y_true, y_pred, **kwargs)
    # class 0 is background, class 1 is pothole
    pairs = (
        (got[5], prec[0]),
        (got[1], prec[1]),
        (got[6], rec[0]),
        (got[2], rec[1]),
    )
    for left, right in pairs:
        _same(left, right, rel=1e-9, abs_=1e-9)
    # sklearn defines F1 as 0 when precision and recall are both 0.
    # getScores follows the harmonic mean and returns NaN for 0/0.
    for f_got, pre, rec_v, f_sk in (
        (got[7], prec[0], rec[0], f1[0]),
        (got[3], prec[1], rec[1], f1[1]),
    ):
        if math.isfinite(float(pre)) and math.isfinite(float(rec_v)) and float(pre) + float(rec_v) != 0.0:
            _same(f_got, f_sk, rel=1e-9, abs_=1e-9)
    tn, fp = int(arr[0, 0]), int(arr[0, 1])
    fn, tp = int(arr[1, 0]), int(arr[1, 1])
    try:
        iou = jaccard_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    except TypeError:
        iou = jaccard_score(y_true, y_pred, labels=[0, 1], average=None)
    if tn + fp + fn > 0:
        _same(got[8], iou[0], rel=1e-9, abs_=1e-9)
    if tp + fp + fn > 0:
        _same(got[4], iou[1], rel=1e-9, abs_=1e-9)


def _scores_comparable(arr):
    """False when a float sum cancels to a tiny residual.

    Class-swap symmetry and the reference formula agree in exact arithmetic.
    When positive and negative entries nearly cancel, NumPy's sum residual
    depends on addition order, and the quotient is not a stable invariant.
    """
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.floating):
        return True
    total = np.sum(arr, dtype=np.float64)
    # A cancelled sum is exactly 0 on one addition order and a tiny residual
    # on the class-swapped order. Only the all-zero matrix is stable then.
    if total == 0:
        return bool(np.all(arr == 0))
    if not np.isfinite(total):
        return True
    mag = np.max(np.abs(arr))
    if not np.isfinite(mag) or mag == 0:
        return True
    return abs(float(total)) > 1e-8 * float(mag)


def check_get_scores(conf):
    got = getScores(conf)
    assert len(got) == 9
    arr = np.asarray(conf)
    if (
        arr.size
        and np.issubdtype(arr.dtype, np.number)
        and not np.isnan(arr).any()
        and arr.sum() == 0
    ):
        assert got == (0, 0, 0, 0, 0, 0, 0, 0, 0)
    if not _scores_comparable(arr):
        return got
    expected = reference_get_scores(conf)
    # float32 sums are wider after getScores casts to float64; allow that rounding.
    tol = 1e-5 if arr.dtype == np.float32 else 1e-8
    for left, right in zip(got, expected):
        _same(left, right, rel=tol, abs_=tol)
    if _finite_nonnegative(arr):
        _assert_ranges(arr, got)
    if arr.ndim == 2 and arr.shape == (2, 2):
        swapped = getScores(arr[::-1, ::-1])
        _same(got[0], swapped[0], rel=tol, abs_=tol)
        for i, j in ((1, 5), (2, 6), (3, 7), (4, 8)):
            _same(got[i], swapped[j], rel=tol, abs_=tol)
            _same(got[j], swapped[i], rel=tol, abs_=tol)
    _compare_sklearn(arr, got)
    return got


@PROP
@given(
    tn=st.integers(min_value=0, max_value=2**40),
    fp=st.integers(min_value=0, max_value=2**40),
    fn=st.integers(min_value=0, max_value=2**40),
    tp=st.integers(min_value=0, max_value=2**40),
)
def test_get_scores_nonnegative_counts(tn, fp, fn, tp):
    check_get_scores(np.array([[tn, fp], [fn, tp]], dtype=np.int64))


@PROP
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False, width=64),
        min_size=4,
        max_size=4,
    )
)
def test_get_scores_finite_floats(values):
    # Nonnegative counts only. Signed values can cancel inside a class union
    # and the two algebraically equal IoU formulas then disagree by many ulps.
    check_get_scores(np.array(values, dtype=np.float64).reshape(2, 2))


@PROP
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
        min_size=4,
        max_size=4,
    )
)
def test_get_scores_float32_unit_interval(values):
    check_get_scores(np.array(values, dtype=np.float32).reshape(2, 2))


@pytest.mark.parametrize(
    "conf",
    [
        np.zeros((2, 2), dtype=np.int64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((0, 0)),
        np.array([]),
        np.zeros((2, 0)),
        np.zeros((0, 2)),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[0, 5], [0, 0]]),
        np.array([[0, 0], [4, 0]]),
        np.array([[8, 0], [0, 0]]),
        np.array([[2, 3], [4, 0]]),
        np.array([[0, 0], [0, 9]]),
        np.array([[5, 0], [0, 7]]),
        np.array([[16777217, 1], [1, 3]], dtype=np.int64),
        np.array([[0.5, 0.0], [0.0, 0.0]]),
        np.array([[1e300, 1.0], [0.0, 1e300]]),
        np.array([[1e300, 1e300], [1e300, 1e300]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[0.0, np.nan], [1.0, 2.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, np.inf], [0.0, 0.0]]),
        np.array([[1.0, -3.0], [0.0, 1.0]]),
        np.array([[5.0, -5.0], [0.0, 0.0]]),
        np.array([[-1.0, -1.0], [-1.0, -1.0]]),
    ],
)
def test_get_scores_edges(conf):
    check_get_scores(conf)


def test_get_scores_cancelled_float_sum_returns_nine_values():
    # Signed entries are outside the confusion-count domain. These matrices
    # are the NumPy 1.23 Hypothesis failures: the sum cancels, and the two
    # IoU formulas no longer match. The function must still return 9 numbers.
    samples = (
        np.array([[0.0, 1.0], [1.887379141862766e-15, -1.0]]),
        np.array([[1.0, 262145.0], [1.4551915228366852e-10, -262145.0]]),
    )
    for conf in samples:
        got = getScores(conf)
        assert len(got) == 9


def test_get_scores_large_integer_not_float32():
    conf = np.array([[16777217, 1], [1, 3]], dtype=np.int64)
    got = getScores(conf)
    _same(got[5], 16777217 / 16777218, rel=1e-12, abs_=0.0)
    assert float(got[5]) != pytest.approx(1.0, abs=1e-12)


def test_get_scores_single_class_support():
    background = getScores(np.array([[6, 0], [0, 0]]))
    assert float(background[0]) == pytest.approx(1.0)
    assert all(math.isnan(float(v)) for v in background[1:5])
    assert all(float(v) == pytest.approx(1.0) for v in background[5:])

    pothole = getScores(np.array([[0, 0], [0, 6]]))
    assert float(pothole[0]) == pytest.approx(1.0)
    assert all(float(v) == pytest.approx(1.0) for v in pothole[1:5])
    assert all(math.isnan(float(v)) for v in pothole[5:])


def test_get_scores_zero_true_positives_f1_is_nan():
    got = getScores(np.array([[2, 3], [4, 0]]))
    assert float(got[1]) == pytest.approx(0.0)
    assert float(got[2]) == pytest.approx(0.0)
    assert math.isnan(float(got[3]))
    assert float(got[4]) == pytest.approx(0.0)


@PROP
@given(
    n=st.integers(min_value=-20, max_value=10_000),
    warmup=st.integers(min_value=0, max_value=50),
)
def test_should_report_fps_matches_cutoff(n, warmup):
    assert should_report_fps(n, warmup) is (n > warmup)


@pytest.mark.parametrize(
    "n,expected",
    [(0, False), (5, False), (6, True), (-1, False), (10**9, True)],
)
def test_should_report_fps_edges(n, expected):
    assert should_report_fps(n) is expected


def _images(h, w, seed, low=0, high=256, label_low=0, label_high=2):
    rng = np.random.RandomState(seed)
    rgb = rng.randint(low, high, size=(h, w, 3)).astype(np.uint8)
    tdisp = rng.randint(low, high, size=(h, w, 3)).astype(np.uint8)
    label = rng.randint(label_low, label_high, size=(h, w)).astype(np.uint8)
    return rgb, tdisp, label


@PROP
@given(
    h=st.integers(min_value=1, max_value=24),
    w=st.integers(min_value=1, max_value=24),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_random_flip_identity_and_involution(h, w, seed):
    rgb, tdisp, label = _images(h, w, seed)
    same = RandomFlip(prob=0.0)(rgb.copy(), tdisp.copy(), label.copy())
    assert np.array_equal(same[0], rgb)
    assert np.array_equal(same[1], tdisp)
    assert np.array_equal(same[2], label)

    flipped = RandomFlip(prob=1.0)(rgb.copy(), tdisp.copy(), label.copy())
    assert np.array_equal(flipped[0], rgb[:, ::-1])
    assert np.array_equal(flipped[1], tdisp[:, ::-1])
    assert np.array_equal(flipped[2], label[:, ::-1])
    restored = RandomFlip(prob=1.0)(flipped[0].copy(), flipped[1].copy(), flipped[2].copy())
    assert np.array_equal(restored[0], rgb)
    assert np.array_equal(restored[1], tdisp)
    assert np.array_equal(restored[2], label)


def test_random_flip_empty():
    rgb = np.zeros((0, 0, 3), dtype=np.uint8)
    tdisp = np.zeros((0, 5, 3), dtype=np.uint8)
    label = np.zeros((0, 5), dtype=np.uint8)
    out = RandomFlip(prob=1.0)(rgb, tdisp[:0], label[:0])
    assert out[0].shape == (0, 0, 3)
    out = RandomFlip(prob=1.0)(np.zeros((0, 5, 3), dtype=np.uint8), tdisp, label)
    assert out[0].shape == (0, 5, 3)
    assert np.array_equal(out[2], label[:, ::-1])


@PROP
@given(
    h=st.integers(min_value=16, max_value=28),
    w=st.integers(min_value=16, max_value=28),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_random_crop_is_shared_window(h, w, seed):
    rng = np.random.RandomState(seed)
    rgb = rng.randint(0, 1000, size=(h, w, 3))
    tdisp = rng.randint(0, 1000, size=(h, w, 3))
    label = rng.randint(0, 1000, size=(h, w))
    np.random.seed(seed)
    same = RandomCrop(crop_rate=0.1, prob=0.0)(rgb.copy(), tdisp.copy(), label.copy())
    assert np.array_equal(same[0], rgb)
    np.random.seed(seed)
    out_rgb, out_td, out_lb = RandomCrop(crop_rate=0.1, prob=1.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    oh, ow = out_lb.shape
    assert out_rgb.shape == (oh, ow, 3)
    assert out_td.shape == (oh, ow, 3)
    assert 0 < oh <= h and 0 < ow <= w
    found = False
    for y in range(h - oh + 1):
        for x in range(w - ow + 1):
            if (
                np.array_equal(label[y:y + oh, x:x + ow], out_lb)
                and np.array_equal(rgb[y:y + oh, x:x + ow], out_rgb)
                and np.array_equal(tdisp[y:y + oh, x:x + ow], out_td)
            ):
                found = True
                break
        if found:
            break
    assert found


@PROP
@given(
    h=st.integers(min_value=16, max_value=28),
    w=st.integers(min_value=16, max_value=28),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_random_crop_out_zeros_one_rectangle(h, w, seed):
    rgb, tdisp, label = _images(h, w, seed, low=1, high=255, label_low=1, label_high=5)
    np.random.seed(seed)
    same = RandomCropOut(crop_rate=0.2, prob=0.0)(rgb.copy(), tdisp.copy(), label.copy())
    assert np.array_equal(same[2], label)
    np.random.seed(seed)
    out_rgb, out_td, out_lb = RandomCropOut(crop_rate=0.2, prob=1.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    assert out_rgb.shape == rgb.shape
    assert out_lb.shape == label.shape
    mask = out_lb != label
    assert np.array_equal(mask, np.any(out_rgb != rgb, axis=-1))
    assert np.array_equal(mask, np.any(out_td != tdisp, axis=-1))
    assert np.all(out_rgb[mask] == 0)
    assert np.all(out_td[mask] == 0)
    assert np.all(out_lb[mask] == 0)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        assert np.array_equal(out_lb, label)
        return
    rect = np.zeros(mask.shape, dtype=bool)
    rect[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    assert np.array_equal(rect, mask)


@PROP
@given(
    h=st.integers(min_value=2, max_value=16),
    w=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    bright_range=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_random_brightness_shared_factor(h, w, seed, bright_range):
    rng = np.random.RandomState(seed)
    rgb = rng.uniform(1.0, 10.0, size=(h, w, 3))
    tdisp = rng.uniform(1.0, 10.0, size=(h, w, 3))
    label = rng.randint(0, 2, size=(h, w))
    np.random.seed(seed)
    same = RandomBrightness(bright_range=bright_range, prob=0.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    assert np.array_equal(same[2], label)
    assert np.allclose(same[0], rgb)
    np.random.seed(seed)
    out_rgb, out_td, out_lb = RandomBrightness(bright_range=bright_range, prob=1.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    assert np.array_equal(out_lb, label)
    factor = float((out_rgb / rgb).ravel()[0])
    assert np.allclose(out_rgb, rgb * factor)
    assert np.allclose(out_td, tdisp * factor)
    assert (1.0 - bright_range) - 1e-12 <= factor <= (1.0 + bright_range) + 1e-12


def test_random_brightness_uint8_stays_in_range_when_headroom():
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    tdisp = np.full((8, 8, 3), 180, dtype=np.uint8)
    label = np.zeros((8, 8), dtype=np.uint8)
    np.random.seed(1)
    out_rgb, out_td, out_lb = RandomBrightness(bright_range=0.15, prob=1.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    assert np.array_equal(out_lb, label)
    assert out_rgb.dtype == np.uint8
    assert int(out_rgb.min()) >= math.floor(200 * 0.85)
    assert int(out_rgb.max()) <= math.floor(200 * 1.15)
    assert int(out_td.min()) >= math.floor(180 * 0.85)
    assert int(out_td.max()) <= math.floor(180 * 1.15)


@PROP
@given(
    h=st.integers(min_value=2, max_value=16),
    w=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    noise_range=st.integers(min_value=1, max_value=8),
)
def test_random_noise_same_offset_inside_margin(h, w, seed, noise_range):
    low = noise_range
    high = 256 - noise_range
    rgb, tdisp, label = _images(h, w, seed, low=low, high=high, label_high=3)
    np.random.seed(seed)
    same = RandomNoise(noise_range=noise_range, prob=0.0)(rgb.copy(), tdisp.copy(), label.copy())
    assert np.array_equal(same[0], rgb)
    assert np.array_equal(same[2], label)
    np.random.seed(seed)
    out_rgb, out_td, out_lb = RandomNoise(noise_range=noise_range, prob=1.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    assert np.array_equal(out_lb, label)
    assert out_rgb.shape == rgb.shape
    diff_rgb = out_rgb.astype(np.int16) - rgb.astype(np.int16)
    diff_td = out_td.astype(np.int16) - tdisp.astype(np.int16)
    assert np.array_equal(diff_rgb, diff_td)
    assert int(diff_rgb.min()) >= -noise_range
    assert int(diff_rgb.max()) < noise_range
    assert int(out_rgb.min()) >= 0 and int(out_rgb.max()) <= 255


def test_random_noise_clips_extremes():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    tdisp = np.full((4, 4, 3), 255, dtype=np.uint8)
    label = np.ones((4, 4), dtype=np.uint8)
    np.random.seed(0)
    out_rgb, out_td, out_lb = RandomNoise(noise_range=5, prob=1.0)(
        rgb.copy(), tdisp.copy(), label.copy()
    )
    assert np.array_equal(out_lb, label)
    assert int(out_rgb.min()) >= 0 and int(out_rgb.max()) <= 255
    assert int(out_td.min()) >= 0 and int(out_td.max()) <= 255


@PROP
@given(
    h=st.integers(min_value=2, max_value=10),
    w=st.integers(min_value=2, max_value=10),
    oh=st.integers(min_value=2, max_value=8),
    ow=st.integers(min_value=2, max_value=8),
    value=st.integers(min_value=0, max_value=255),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_mf_dataset_resize_normalizes_to_unit_interval(h, w, oh, ow, value, seed):
    rng = np.random.RandomState(seed)
    rgb = np.full((h, w, 3), value, dtype=np.uint8)
    tdisp = rng.randint(0, 256, size=(h, w, 3)).astype(np.uint8)
    label = rng.randint(0, 4, size=(h, w)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for folder in ("rgb", "tdisp", "label"):
            (root / "train" / folder).mkdir(parents=True)
        Image.fromarray(rgb).save(root / "train" / "rgb" / "a.png")
        Image.fromarray(tdisp).save(root / "train" / "tdisp" / "a.png")
        Image.fromarray(label).save(root / "train" / "label" / "a.png")
        (root / "train.txt").write_text("a\n")
        dataset = MF_dataset(str(root), "train", input_h=oh, input_w=ow)
        rgb_t, tdisp_t, label_t, name = dataset[0]
    assert name == "a"
    assert tuple(rgb_t.shape) == (3, oh, ow)
    assert tuple(tdisp_t.shape) == (3, oh, ow)
    assert tuple(label_t.shape) == (oh, ow)
    assert rgb_t.dtype == torch.float32
    assert tdisp_t.dtype == torch.float32
    assert label_t.dtype == torch.int64
    assert torch.all((rgb_t >= 0) & (rgb_t <= 1))
    assert torch.all((tdisp_t >= 0) & (tdisp_t <= 1))
    assert torch.allclose(rgb_t, torch.full_like(rgb_t, float(value) / 255.0))
    assert {int(v) for v in label_t.unique().tolist()} <= {int(v) for v in np.unique(label)}

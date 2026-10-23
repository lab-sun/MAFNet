import numpy as np
import pytest

from util.util import getScores


def test_get_scores_known_matrix():
    # rows are ground truth, columns are predictions: [[tn, fp], [fn, tp]]
    conf = np.array([[40, 10], [20, 30]])
    globalacc, pre_p, recall_p, F_score_p, iou_p, pre_b, recall_b, F_score_b, iou_b = getScores(conf)
    expected = (
        0.7,    # (40 + 30) / 100
        0.75,   # 30 / 40
        0.6,    # 30 / 50
        2 / 3,  # 2 * 0.6 * 0.75 / (0.6 + 0.75)
        0.5,    # 30 / (50 + 40 - 30)
        40 / 60,
        0.8,    # 40 / 50
        8 / 11, # 2 * (40/60) * 0.8 / ((40/60) + 0.8)
        40 / 70,
    )
    got = (globalacc, pre_p, recall_p, F_score_p, iou_p, pre_b, recall_b, F_score_b, iou_b)
    assert len(got) == 9
    for value, target in zip(got, expected):
        assert float(value) == pytest.approx(target)


def test_get_scores_perfect_matrix():
    conf = np.array([[5, 0], [0, 7]])
    got = getScores(conf)
    assert len(got) == 9
    for value in got:
        assert float(value) == pytest.approx(1.0)


def test_get_scores_all_zero():
    got = getScores(np.zeros((2, 2)))
    assert got == (0, 0, 0, 0, 0, 0, 0, 0, 0)

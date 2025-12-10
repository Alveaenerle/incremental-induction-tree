import pytest
import numpy as np
from src.utils.split_utils import get_best_split, _calculate_entropy
from src.core.stats import Stats


def create_mock_stats(total_count, label_dist, feature_dists):
    stats = Stats()
    stats.total_count = total_count
    stats.label_distribution = label_dist
    stats.distribution = feature_dists
    return stats


def test_entropy_zero():
    counts = [10, 0]
    total = 10
    assert _calculate_entropy(counts, total) == 0.0


def test_entropy_max():
    counts = [5, 5]
    total = 10
    assert np.isclose(_calculate_entropy(counts, total), 1.0)


def test_entropy_mixed():
    assert _calculate_entropy([4], 4) == 0


def test_entropy_empty():
    assert _calculate_entropy([], 0) == 0


def test_perfect_split():
    label_dist = {'A': 10, 'B': 10}
    total = 20

    feat0_dist = {
        0: {'A': 10},
        1: {'B': 10}
    }

    feat1_dist = {
        0: {'A': 5, 'B': 5},
        1: {'A': 5, 'B': 5}
    }

    stats = create_mock_stats(total, label_dist, [feat0_dist, feat1_dist])
    best_feat, gain = get_best_split(stats)

    assert best_feat == 0
    assert np.isclose(gain, 1.0)


def test_no_information_gain():
    label_dist = {'A': 10}
    total = 10
    feat0_dist = {1: {'A': 10}}

    stats = create_mock_stats(total, label_dist, [feat0_dist])
    best_feat, gain = get_best_split(stats)

    assert np.isclose(gain, 0.0)


def test_ignore_empty_branches():
    label_dist = {'A': 5, 'B': 5}
    total = 10

    feat0_dist = {
        0: {'A': 5, 'B': 0},
        1: {'A': 0, 'B': 5},
        2: {}
    }

    stats = create_mock_stats(total, label_dist, [feat0_dist])
    best_feat, gain = get_best_split(stats)

    assert best_feat == 0
    assert gain > 0


def test_better_partial_split():
    label_dist = {'A': 20, 'B': 20}
    total = 40

    feat0 = {
        1: {'A': 10, 'B': 10},
        2: {'A': 10, 'B': 10}
    }

    feat1 = {
        1: {'A': 18, 'B': 2},
        2: {'A': 2, 'B': 18}
    }

    stats = create_mock_stats(total, label_dist, [feat0, feat1])
    best_feat, gain = get_best_split(stats)

    assert best_feat == 1
    assert gain > 0

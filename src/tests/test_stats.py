from src.core.stats import (
    Stats,
)
import numpy as np


def test_stats_initialization():
    data = np.array([
        [1, 2, 'A'],
        [1, 3, 'A'],
        [2, 2, 'B'],
        [2, 3, 'B'],
        [3, 2, 'A'],
        [3, 3, 'A']
    ], dtype=object)

    stats = Stats(data)

    assert len(stats.distribution) == 2  # Two features
    assert stats.total_count == 6
    assert stats.label_distribution['A'] == 4
    assert stats.label_distribution['B'] == 2
    assert stats.distribution[0].feature_distribution[1].label_distribution['A'] == 2
    assert stats.distribution[0].feature_distribution[2].label_distribution['B'] == 2
    assert stats.distribution[0].feature_distribution[3].label_distribution['A'] == 2
    assert stats.distribution[1].feature_distribution[2].label_distribution['A'] == 2
    assert stats.distribution[1].feature_distribution[3].label_distribution['B'] == 2


def test_feature_stats():
    feature_stats = FeatureStats()
    label_stats_a = LabelStats()
    label_stats_a.add_label('A', 3)
    label_stats_b = LabelStats()
    label_stats_b.add_label('B', 2)

    feature_stats.add_value(1, label_stats_a)
    feature_stats.add_value(2, label_stats_b)

    assert feature_stats.feature_distribution[1].label_distribution['A'] == 3
    assert feature_stats.feature_distribution[2].label_distribution['B'] == 2


def test_label_stats():
    label_stats = LabelStats()
    label_stats.add_label('A', 4)
    label_stats.add_label('B', 1)

    assert label_stats.label_distribution['A'] == 4
    assert label_stats.label_distribution['B'] == 1
    assert label_stats.total_count == 5

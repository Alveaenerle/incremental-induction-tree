import numpy as np
from src.core.stats import (
    Stats,
    FeatureStats,
    LabelStats
)


def choose_best_split(stats):
    best_split = (float('-inf'), None, None)  # (information_gain, feature, value)
    for idx, feature_stats in enumerate(stats.distribution):
        for label_stats in feature_stats.feature_distribution:
            left_size = label_stats.total_count
            right_size = stats.total_count - left_size
            if left_size == 0 or right_size == 0:
                continue

            inf_gain = _calculate_info_gain(stats, left_size, right_size)
            if inf_gain > best_split[0]:
                best_split = (inf_gain, idx, value)

    return best_split


def split_data(stats, feature, value):
    feature_stats = stats.distribution[feature]
    left_stats = Stats()
    right_stats = Stats()
    


def _calculate_info_gain(stats, label_stats, left_size, right_size):
    total_size = left_size + right_size
    weight_left = left_size / total_size
    weight_right = right_size / total_size
    parent_entropy = _calculate_entropy(list(stats.label_distribution.values()), stats.total_count)
    left_entropy = _calculate_entropy(list(label_stats.label_distribution.values()), left_size)
    right_entropy = _calculate_entropy(
        [stats.label_distribution[label] - label_stats.label_distribution.get(label, 0) for label in stats.label_distribution],
        right_size
    )
    gain = parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)
    return gain


def _calculate_entropy(labels_count, total_count):
    probs = labels_count / total_count
    return -np.sum(probs * np.log2(probs))

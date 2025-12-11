import numpy as np


def get_best_split(stats):
    best_gain = float('-inf')
    best_feature_idx = None
    parent_entropy = _calculate_entropy(stats.label_distribution.values(), stats.total_count)

    for feat_idx, value_counts in enumerate(stats.distribution):
        weighted_child_entropy = 0

        for val, label_counts in value_counts.items():
            child_total = sum(label_counts.values())
            if child_total == 0:
                continue
            child_entropy = _calculate_entropy(label_counts.values(), child_total)
            weighted_child_entropy += (child_total / stats.total_count) * child_entropy

        gain = parent_entropy - weighted_child_entropy
        if gain > best_gain:
            best_gain = gain
            best_feature_idx = feat_idx

    return best_feature_idx, best_gain


def _calculate_entropy(counts, total_count):
    if total_count == 0:
        return 0

    counts = np.array(list(counts))
    probs = counts / total_count
    probs = probs[probs > 0]

    return -np.sum(probs * np.log2(probs))

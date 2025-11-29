import numpy as np


def choose_best_split(data):
    best_split = (float('-inf'), None, None)  # (information_gain, feature, value)
    for idx, feature_col in enumerate(data.T[:-1]):
        unique_values = np.unique(feature_col)
        for value in unique_values:
            left_split, right_split = split_data(data, idx, value)

            if len(left_split) == 0 or len(right_split) == 0:
                continue

            inf_gain = _calculate_entropy(data) - (
                (len(left_split) / len(data)) * _calculate_entropy(np.array(left_split)) +
                (len(right_split) / len(data)) * _calculate_entropy(np.array(right_split))
            )
            if inf_gain > best_split[0]:
                best_split = (inf_gain, idx, value)

    return best_split[1], best_split[2]


def split_data(data, feature, value):
    left_split = [row for row in data if row[feature] == value]
    right_split = [row for row in data if row[feature] != value]
    return np.array(left_split), np.array(right_split)


def _calculate_entropy(data):
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

from src.utils.split_utils import (
    choose_best_split,
    split_data
)
import numpy as np


def test_choose_best_split():
    data = np.array([
        [1, 2, 'A'],
        [1, 3, 'A'],
        [2, 2, 'B'],
        [2, 3, 'B'],
        [3, 2, 'A'],
        [3, 3, 'A']
    ], dtype=object)
    feature, value = choose_best_split(data)
    assert feature == 0
    assert value == 2


def test_split_data():
    data = np.array([
        [1, 2, 'A'],
        [1, 3, 'A'],
        [2, 2, 'B'],
        [2, 3, 'B'],
        [3, 2, 'A'],
        [3, 3, 'A']
    ])
    feature = 0  # First column
    value = 2
    left_split, right_split = split_data(data, feature, value)
    assert all(row[feature] == value for row in left_split)
    assert all(row[feature] != value for row in right_split)


def test_split_data_empty_split():
    data = np.array([
        [1, 2, 'A'],
        [1, 3, 'A']
    ])
    feature = 0  # First column
    value = 3
    left_split, right_split = split_data(data, feature, value)
    assert len(left_split) == 0
    assert len(right_split) == len(data)


def test_choose_best_split_no_variation():
    data = np.array([
        [1, 2, 'A'],
        [1, 2, 'A'],
        [1, 2, 'A']
    ])
    feature, value = choose_best_split(data)
    assert feature is None
    assert value is None

from src.utils.data_utils import discretize_dataset
import pandas as pd
import numpy as np
import pytest


def test_discretize_dataset():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['X', 'Y', 'X', 'Y', 'X']
    }
    df = pd.DataFrame(data)
    continuous_columns = ['A', 'B']

    discretized_array = discretize_dataset(df, continuous_columns, bins=3)

    assert isinstance(discretized_array, np.ndarray)

    col_a = discretized_array[:, 0]
    unique_a = np.unique(col_a)
    assert set(unique_a).issubset({0, 1, 2})

    col_b = discretized_array[:, 1]
    unique_b = np.unique(col_b)
    assert set(unique_b).issubset({0, 1, 2})

    col_c = discretized_array[:, 2]
    assert 'X' in col_c
    assert 'Y' in col_c


def test_discretize_dataset_with_non_unique_values():
    data = {
        'A': [1, 1, 1, 1, 1],
        'B': [10, 20, 30, 40, 50],
        'C': ['X', 'Y', 'X', 'Y', 'X']
    }
    df = pd.DataFrame(data)
    continuous_columns = ['A', 'B']

    discretized_array = discretize_dataset(df, continuous_columns, bins=3)

    col_a = discretized_array[:, 0]
    unique_a = np.unique(col_a)
    assert set(unique_a) == {0}

    col_b = discretized_array[:, 1]
    unique_b = np.unique(col_b)
    assert set(unique_b).issubset({0, 1, 2})

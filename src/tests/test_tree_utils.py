from src.utils.tree_utils import (
    is_pure,
    majority_class
)


def test_is_pure():
    pure_data = [
        [1, 2, 'A'],
        [3, 4, 'A'],
        [5, 6, 'A']
    ]
    impure_data = [
        [1, 2, 'A'],
        [3, 4, 'B'],
        [5, 6, 'A']
    ]
    assert is_pure(pure_data) is True
    assert is_pure(impure_data) is False


def test_majority_class():
    data = [
        [1, 2, 'A'],
        [3, 4, 'B'],
        [5, 6, 'A'],
        [7, 8, 'C'],
        [9, 10, 'A']
    ]
    assert majority_class(data) == 'A'


def test_majority_class_tie():
    data = [
        [1, 2, 'A'],
        [3, 4, 'B'],
        [5, 6, 'A'],
        [7, 8, 'B']
    ]
    result = majority_class(data)
    assert result in ('A', 'B')  # Accept either class in case of a tie

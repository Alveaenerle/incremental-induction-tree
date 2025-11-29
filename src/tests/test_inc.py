from src.algorithms.incremental_tree import IncrementalTree
import numpy as np


def test_incremental_tree_fit_and_predict():
    data = np.array([
        [1, 2, 'A'],
        [1, 3, 'A'],
        [2, 1, 'A'],
        [2, 3, 'B'],
        [3, 2, 'A'],
        [3, 3, 'A']
    ], dtype=object)

    tree = IncrementalTree()
    tree.fit(data)

    test_row_1 = [1, 1]
    prediction_1 = tree.predict(test_row_1)
    assert prediction_1 == 'A'

    test_row_2 = [2, 2]
    prediction_2 = tree.predict(test_row_2)
    assert prediction_2 == 'B'

    test_row_2 = [3, 1]
    prediction_2 = tree.predict(test_row_2)
    assert prediction_2 == 'A'

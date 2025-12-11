import pytest
import numpy as np
from src.algorithms.incremental_tree import IncrementalTree


class TestIncrementalTree:
    def test_initialization(self):
        tree = IncrementalTree()
        assert tree.root is None

    def test_fit_simple_data(self):
        data = np.array([
            [1, 10, 'A'],
            [1, 20, 'A'],
            [2, 10, 'B'],
            [2, 20, 'B']
        ], dtype=object)

        tree = IncrementalTree()
        tree.fit(data)

        assert tree.root is not None
        assert tree.root.feature == 0
        assert tree.predict([1, 10]) == 'A'
        assert tree.predict([2, 10]) == 'B'

    def test_update_from_scratch(self):
        tree = IncrementalTree()

        tree.update(np.array([1, 10, 'A'], dtype=object))
        assert tree.root.is_leaf()
        assert tree.root.output == 'A'

        tree.update(np.array([2, 20, 'B'], dtype=object))
        assert not tree.root.is_leaf()
        assert tree.predict([1, 10]) == 'A'
        assert tree.predict([2, 20]) == 'B'

    def test_split_leaf_logic(self):
        tree = IncrementalTree()
        tree.update(np.array([1, 1, 'A'], dtype=object))
        tree.update(np.array([1, 2, 'A'], dtype=object))

        assert tree.root.is_leaf()

        tree.update(np.array([1, 3, 'B'], dtype=object))

        assert not tree.root.is_leaf()
        assert tree.root.feature == 1

    def test_pull_up_restructuring(self):
        tree = IncrementalTree()

        data_phase_1 = [
            [0, 1, 'A'], [0, 2, 'A'],
            [1, 1, 'B'], [1, 2, 'B']
        ]

        for row in data_phase_1:
            tree.update(np.array(row, dtype=object))

        assert tree.root.feature == 0

        data_phase_2 = [
            [0, 1, 'A'], [1, 1, 'A'],
            [0, 2, 'B'], [1, 2, 'B']
        ]

        for _ in range(20):
            for row in data_phase_2:
                tree.update(np.array(row, dtype=object))

        assert tree.root.feature == 1
        assert tree.predict([0, 1]) == 'A'
        assert tree.predict([1, 2]) == 'B'

    def test_predict_unseen_value(self):
        tree = IncrementalTree()
        data = np.array([
            [1, 'A'],
            [2, 'B']
        ], dtype=object)
        tree.fit(data)

        pred = tree.predict([3])
        assert pred in ['A', 'B']

    def test_update_existing_leaf(self):
        tree = IncrementalTree()
        tree.fit(np.array([[1, 'A']], dtype=object))

        tree.update(np.array([1, 'A'], dtype=object))
        assert tree.root.statistics.total_count == 2
        assert len(tree.root.samples) == 2

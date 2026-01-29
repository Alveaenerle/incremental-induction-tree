"""Unit tests for IncrementalForest."""

import pytest
import numpy as np

from src.algorithms.incremental_forest import IncrementalForest
from src.algorithms.incremental_tree import IncrementalTree


class TestIncrementalForest:
    """Tests for IncrementalForest class."""

    def test_initialization_default(self):
        """Test default initialization."""
        forest = IncrementalForest()

        assert forest.n_estimators == 10
        assert forest.min_gain == 0.01
        assert len(forest.trees) == 10
        assert all(isinstance(t, IncrementalTree) for t in forest.trees)

    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        forest = IncrementalForest(
            n_estimators=5, min_gain=0.05, random_state=42
        )

        assert forest.n_estimators == 5
        assert forest.min_gain == 0.05
        assert len(forest.trees) == 5

    def test_fit_runs_without_errors(self):
        """Test that fit runs successfully on sample data."""
        data = np.array([
            [1, 10, 0],
            [1, 20, 0],
            [2, 10, 1],
            [2, 20, 1],
            [1, 30, 0],
            [2, 30, 1],
        ], dtype=object)

        forest = IncrementalForest(n_estimators=3, random_state=42)
        forest.fit(data)

        # All trees should have roots after fitting
        assert all(tree.root is not None for tree in forest.trees)

    def test_update_runs_without_errors(self):
        """Test that update runs successfully."""
        forest = IncrementalForest(n_estimators=3, random_state=42)

        samples = [
            np.array([1, 10, 'A'], dtype=object),
            np.array([2, 20, 'B'], dtype=object),
            np.array([1, 30, 'A'], dtype=object),
        ]

        for sample in samples:
            forest.update(sample)

        # All trees should have roots after updates
        assert all(tree.root is not None for tree in forest.trees)

    def test_predict_majority_voting(self):
        """Test that predict uses majority voting."""
        data = np.array([
            [1, 10, 'A'],
            [1, 20, 'A'],
            [1, 30, 'A'],
            [2, 10, 'B'],
            [2, 20, 'B'],
            [2, 30, 'B'],
        ], dtype=object)

        forest = IncrementalForest(n_estimators=5, random_state=42)
        forest.fit(data)

        # Should predict based on feature 0
        pred_a = forest.predict([1, 15])
        pred_b = forest.predict([2, 15])

        assert pred_a == 'A'
        assert pred_b == 'B'

    def test_predict_empty_forest(self):
        """Test prediction on empty forest returns None."""
        forest = IncrementalForest(n_estimators=3)
        pred = forest.predict([1, 2, 3])

        assert pred is None

    def test_get_node_count(self):
        """Test that node count sums across all trees."""
        data = np.array([
            [1, 10, 0],
            [2, 20, 1],
        ], dtype=object)

        forest = IncrementalForest(
            n_estimators=3, min_gain=0.0, random_state=42
        )
        forest.fit(data)

        total_nodes = forest.get_node_count()
        individual_counts = forest.get_tree_node_counts()

        assert total_nodes == sum(individual_counts)
        assert total_nodes > 0

    def test_get_tree_node_counts(self):
        """Test individual tree node counts."""
        data = np.array([
            [1, 10, 0],
            [2, 20, 1],
        ], dtype=object)

        forest = IncrementalForest(
            n_estimators=3, min_gain=0.0, random_state=42
        )
        forest.fit(data)

        counts = forest.get_tree_node_counts()

        assert len(counts) == 3
        assert all(c >= 0 for c in counts)


class TestPruning:
    """Tests for pre-pruning functionality."""

    def test_pruning_reduces_node_count(self):
        """Test that higher min_gain results in fewer nodes."""
        # Generate data that allows for multiple splits
        np.random.seed(42)
        n_samples = 100

        # Create data with some noise
        data = []
        for _ in range(n_samples):
            f1 = np.random.randint(0, 5)
            f2 = np.random.randint(0, 5)
            # Label with some noise
            label = 1 if (f1 + f2) > 4 else 0
            if np.random.random() < 0.1:  # 10% noise
                label = 1 - label
            data.append([f1, f2, label])

        data = np.array(data, dtype=object)

        # Tree with no pruning
        tree_no_prune = IncrementalTree(min_gain=0.0)
        tree_no_prune.fit(data)
        nodes_no_prune = tree_no_prune.get_node_count()

        # Tree with aggressive pruning
        tree_pruned = IncrementalTree(min_gain=0.5)
        tree_pruned.fit(data)
        nodes_pruned = tree_pruned.get_node_count()

        # Pruned tree should have fewer or equal nodes
        assert nodes_pruned <= nodes_no_prune, (
            f"Pruned tree ({nodes_pruned} nodes) should have <= nodes "
            f"than unpruned tree ({nodes_no_prune} nodes)"
        )

    def test_high_min_gain_creates_leaf_only(self):
        """Test that very high min_gain results in a single leaf."""
        data = np.array([
            [1, 10, 0],
            [2, 20, 1],
            [3, 30, 0],
            [4, 40, 1],
        ], dtype=object)

        # Very high min_gain should prevent any splits
        tree = IncrementalTree(min_gain=10.0)
        tree.fit(data)

        # With very high min_gain, tree should remain a leaf
        assert tree.root.is_leaf()
        assert tree.get_node_count() == 1

    def test_forest_pruning(self):
        """Test that forest respects min_gain parameter."""
        np.random.seed(42)
        data = np.array([
            [i % 5, (i * 7) % 5, i % 2]
            for i in range(50)
        ], dtype=object)

        forest_no_prune = IncrementalForest(
            n_estimators=5, min_gain=0.0, random_state=42
        )
        forest_no_prune.fit(data)

        forest_pruned = IncrementalForest(
            n_estimators=5, min_gain=0.5, random_state=42
        )
        forest_pruned.fit(data)

        nodes_no_prune = forest_no_prune.get_node_count()
        nodes_pruned = forest_pruned.get_node_count()

        # Pruned forest should have fewer or equal nodes
        assert nodes_pruned <= nodes_no_prune


class TestPredictProba:
    """Tests for probability prediction."""

    def test_predict_proba_returns_dict(self):
        """Test that predict_proba returns a dictionary."""
        data = np.array([
            [1, 'A'],
            [2, 'B'],
        ], dtype=object)

        forest = IncrementalForest(n_estimators=5, random_state=42)
        forest.fit(data)

        proba = forest.predict_proba([1])

        assert isinstance(proba, dict)
        assert sum(proba.values()) == pytest.approx(1.0)

    def test_predict_proba_empty_forest(self):
        """Test predict_proba on empty forest."""
        forest = IncrementalForest(n_estimators=3)
        proba = forest.predict_proba([1, 2])

        assert proba == {}


class TestOzaBagging:
    """Tests for Oza Bagging update mechanism."""

    def test_oza_bagging_updates_trees(self):
        """Test that Oza Bagging updates trees with varying frequency."""
        forest = IncrementalForest(n_estimators=3, random_state=42)

        # Update with multiple samples
        for i in range(10):
            sample = np.array([i % 3, i % 2, i % 2], dtype=object)
            forest.update(sample)

        # All trees should have been updated
        total_counts = sum(
            tree.root.statistics.total_count if tree.root else 0
            for tree in forest.trees
        )

        # Due to Poisson(1), average updates per sample per tree is 1
        # With 10 samples and 3 trees, expected ~30 total updates
        # (but with variance)
        assert total_counts > 0

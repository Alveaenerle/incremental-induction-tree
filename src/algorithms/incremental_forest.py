"""Incremental Random Forest implementation with Oza Bagging."""

from collections import Counter
from typing import Any, List, Optional, Union

import numpy as np

from src.algorithms.incremental_tree import IncrementalTree


class IncrementalForest:
    """
    Incremental Random Forest using Bootstrap Aggregation (Bagging) for batch
    training and Oza Bagging for online updates.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        min_gain: float = 0.01,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the Incremental Random Forest.

        Args:
            n_estimators: Number of trees in the forest.
            min_gain: Minimum information gain for splits (pre-pruning).
            random_state: Random seed for reproducibility.
        """
        self.n_estimators: int = n_estimators
        self.min_gain: float = min_gain
        self.random_state: Optional[int] = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.trees: List[IncrementalTree] = [
            IncrementalTree(min_gain=min_gain) for _ in range(n_estimators)
        ]

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the forest using Bootstrap Aggregation (Bagging).

        Each tree is trained on a random subset of data with replacement.

        Args:
            data: Training data as numpy array where last column is the label.
        """
        n_samples = len(data)

        for tree in self.trees:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = data[indices]
            tree.fit(bootstrap_sample)

    def update(self, sample: Union[np.ndarray, List[Any]]) -> None:
        """
        Update all trees with a single sample using Oza Bagging.

        For each tree, draw k from Poisson(1) distribution and update
        the tree k times with the sample.

        Args:
            sample: Single data sample where last element is the label.
        """
        for tree in self.trees:
            # Oza Bagging: sample k from Poisson(1)
            k = np.random.poisson(1)
            for _ in range(k):
                tree.update(sample)

    def predict(self, row: Union[np.ndarray, List[Any]]) -> Optional[Any]:
        """
        Predict the class using majority voting across all trees.

        Args:
            row: Feature values for prediction.

        Returns:
            Predicted class label (majority vote) or None if
            all trees are empty.
        """
        predictions = []

        for tree in self.trees:
            pred = tree.predict(row)
            if pred is not None:
                predictions.append(pred)

        if not predictions:
            return None

        vote_counts = Counter(predictions)
        return vote_counts.most_common(1)[0][0]

    def predict_proba(self, row: Union[np.ndarray, List[Any]]) -> dict:
        """
        Get prediction probabilities based on tree votes.

        Args:
            row: Feature values for prediction.

        Returns:
            Dictionary mapping class labels to vote proportions.
        """
        predictions = []

        for tree in self.trees:
            pred = tree.predict(row)
            if pred is not None:
                predictions.append(pred)

        if not predictions:
            return {}

        vote_counts = Counter(predictions)
        total = len(predictions)

        return {label: count / total for label, count in vote_counts.items()}

    def get_node_count(self) -> int:
        """
        Get the total number of nodes across all trees.

        Returns:
            Sum of node counts from all trees.
        """
        return sum(tree.get_node_count() for tree in self.trees)

    def get_tree_node_counts(self) -> List[int]:
        """
        Get individual node counts for each tree.

        Returns:
            List of node counts for each tree.
        """
        return [tree.get_node_count() for tree in self.trees]

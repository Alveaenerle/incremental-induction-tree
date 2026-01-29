from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.core.node import Node
from src.core.stats import Stats
from src.utils.split_utils import get_best_split


class IncrementalTree:
    """Incremental Decision Tree with pre-pruning support."""

    def __init__(self, min_gain: float = 0.01) -> None:
        """
        Initialize the Incremental Decision Tree.

        Args:
            min_gain: Minimum information gain required to split
                      a node (pre-pruning).
        """
        self.root: Optional[Node] = None
        self.min_gain: float = min_gain

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the tree on batch data.

        Args:
            data: Training data as numpy array where last column is the label.
        """
        self.root = Node(statistics=Stats(data=data))
        self.root.samples = list(data)
        self.root.output = self._get_majority_class(self.root)

        if not self._is_pure(self.root):
            best_feat, gain = get_best_split(self.root.statistics)
            if best_feat is not None and gain > self.min_gain:
                self._split_leaf(self.root, best_feat)
                for child in self.root.children.values():
                    self._recursive_build(child)

    def predict(self, row: Union[np.ndarray, List[Any]]) -> Optional[Any]:
        """
        Predict the class for a single sample.

        Args:
            row: Feature values for prediction.

        Returns:
            Predicted class label or None if tree is empty.
        """
        if self.root is None:
            return None
        node = self.root
        while not node.is_leaf():
            val = row[node.feature]
            if val in node.children:
                node = node.children[val]
            else:
                return node.output
        return node.output

    def update(self, sample: Union[np.ndarray, List[Any]]) -> None:
        """
        Update the tree with a single sample (online learning).

        Args:
            sample: Single data sample where last element is the label.
        """
        if self.root is None:
            self.root = Node(statistics=Stats(num_features=len(sample) - 1))
            self.root.output = sample[-1]
            self.root.add_sample(sample)
            self.root.statistics.add_sample(sample)
            return

        self._update_node(self.root, sample)

    def get_node_count(self) -> int:
        """
        Get the total number of nodes in the tree.

        Returns:
            Total node count (internal nodes + leaves).
        """
        if self.root is None:
            return 0
        return self._count_nodes(self.root)

    def _count_nodes(self, node: Node) -> int:
        """
        Recursively count nodes starting from a given node.

        Args:
            node: Starting node for counting.

        Returns:
            Number of nodes in the subtree rooted at node.
        """
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _update_node(
        self, node: Node, sample: Union[np.ndarray, List[Any]]
    ) -> None:
        """
        Recursively update a node with a new sample.

        Args:
            node: Node to update.
            sample: Data sample to add.
        """
        node.statistics.add_sample(sample)

        if node.is_leaf():
            node.add_sample(sample)
            node.output = self._get_majority_class(node)

            if not self._is_pure(node):
                best_feat, gain = get_best_split(node.statistics)
                if best_feat is not None and gain > self.min_gain:
                    self._split_leaf(node, best_feat)
            return

        best_feat, gain = get_best_split(node.statistics)

        should_pull_up = (
            best_feat != node.feature
            and best_feat is not None
            and gain > self.min_gain
        )
        if should_pull_up:
            self._pull_up(node, best_feat)

        val = sample[node.feature]
        if val not in node.children:
            num_feats = len(node.statistics.distribution)
            new_child = Node(statistics=Stats(num_features=num_feats))
            new_child.output = sample[-1]
            new_child.add_sample(sample)
            new_child.statistics.add_sample(sample)
            node.children[val] = new_child
        else:
            self._update_node(node.children[val], sample)

    def _pull_up(self, node: Node, new_feature: int) -> None:
        """
        Pull up a new feature to become the splitting feature at node.

        Args:
            node: Node to restructure.
            new_feature: New feature index to use for splitting.
        """
        if node.feature == new_feature:
            return

        for child in list(node.children.values()):
            if child.is_leaf():
                self._split_leaf(child, new_feature)
            elif child.feature != new_feature:
                self._pull_up(child, new_feature)

        old_feature = node.feature
        new_children_map: Dict[Any, Node] = {}
        all_new_vals: set = set()
        for child in node.children.values():
            all_new_vals.update(child.children.keys())

        for new_val in all_new_vals:
            num_feats = len(node.statistics.distribution)
            sub_node = Node(
                statistics=Stats(num_features=num_feats),
                feature=old_feature
            )
            sub_node.samples = None

            for old_val, old_child in node.children.items():
                if new_val in old_child.children:
                    grand_child = old_child.children[new_val]
                    sub_node.children[old_val] = grand_child
                    sub_node.statistics.merge(grand_child.statistics)

            sub_node.output = self._get_majority_class(sub_node)
            new_children_map[new_val] = sub_node

        node.feature = new_feature
        node.children = new_children_map

    def _split_leaf(self, node: Node, feature_idx: int) -> None:
        """
        Split a leaf node on a given feature.

        Args:
            node: Leaf node to split.
            feature_idx: Feature index to split on.
        """
        node.feature = feature_idx
        node.children = {}

        if not node.samples:
            return

        data = np.array(node.samples, dtype=object)
        known_values = node.statistics.distribution[feature_idx].keys()

        for val in known_values:
            mask = (data[:, feature_idx] == val)

            if not np.any(mask):
                continue

            subset = data[mask]
            num_feats = len(node.statistics.distribution)
            child = Node(statistics=Stats(num_features=num_feats))
            child.add_batch_samples(subset)
            child.statistics.add_batch(subset)
            child.output = self._get_majority_class(child)
            node.children[val] = child

        node.clear_samples()

    def _recursive_build(self, node: Node) -> None:
        """
        Recursively build the tree from a node.

        Args:
            node: Node to continue building from.
        """
        if node.is_leaf() and not self._is_pure(node):
            best_feat, gain = get_best_split(node.statistics)
            if best_feat is not None and gain > self.min_gain:
                self._split_leaf(node, best_feat)
                for child in node.children.values():
                    self._recursive_build(child)

    def _get_majority_class(self, node: Node) -> Optional[Any]:
        """
        Get the majority class from a node's label distribution.

        Args:
            node: Node to get majority class from.

        Returns:
            The most frequent class label or None if no labels.
        """
        if not node.statistics.label_distribution:
            return None
        return max(
            node.statistics.label_distribution,
            key=node.statistics.label_distribution.get
        )

    def _is_pure(self, node: Node) -> bool:
        """
        Check if a node contains only one class.

        Args:
            node: Node to check.

        Returns:
            True if node has 0 or 1 unique classes.
        """
        return len(node.statistics.label_distribution) <= 1

from src.core.node import Node
from src.core.stats import Stats
from src.utils.split_utils import get_best_split
import numpy as np


class IncrementalTree:
    def __init__(self):
        self.root = None

    def fit(self, data):
        self.root = Node(statistics=Stats(data=data))
        self.root.samples = list(data)
        self.root.output = self._get_majority_class(self.root)
        
        if not self._is_pure(self.root):
            best_feat, gain = get_best_split(self.root.statistics)
            if best_feat is not None and gain > 0:
                self._split_leaf(self.root, best_feat)
                for child in self.root.children.values():
                    self._recursive_build(child)

    def predict(self, row):
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

    def update(self, sample):
        if self.root is None:
            self.root = Node(statistics=Stats(num_features=len(sample)-1))
            self.root.output = sample[-1]
            self.root.add_sample(sample)
            self.root.statistics.add_sample(sample)
            return

        self._update_node(self.root, sample)

    def _update_node(self, node, sample):
        node.statistics.add_sample(sample)

        if node.is_leaf():
            node.add_sample(sample)
            node.output = self._get_majority_class(node)

            if not self._is_pure(node):
                best_feat, gain = get_best_split(node.statistics)
                if best_feat is not None and gain > 0:
                    self._split_leaf(node, best_feat)
            return

        # Pull-up logic
        best_feat, _ = get_best_split(node.statistics)
        if best_feat != node.feature and best_feat is not None:
            self._pull_up(node, best_feat)

        val = sample[node.feature]
        if val not in node.children:
            new_child = Node(statistics=Stats(num_features=len(node.statistics.distribution)))
            new_child.output = sample[-1]
            new_child.add_sample(sample)
            new_child.statistics.add_sample(sample)
            node.children[val] = new_child
        else:
            self._update_node(node.children[val], sample)

    def _pull_up(self, node, new_feature):
        if node.feature == new_feature:
            return

        for child in list(node.children.values()):
            if child.is_leaf():
                self._split_leaf(child, new_feature)
            elif child.feature != new_feature:
                self._pull_up(child, new_feature)

        old_feature = node.feature
        new_children_map = {}
        all_new_vals = set()
        for child in node.children.values():
            all_new_vals.update(child.children.keys())

        for new_val in all_new_vals:
            sub_node = Node(
                statistics=Stats(num_features=len(node.statistics.distribution)),
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

    def _split_leaf(self, node, feature_idx):
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
            child = Node(statistics=Stats(num_features=len(node.statistics.distribution)))
            child.add_batch_samples(subset)
            child.statistics.add_batch(subset)
            child.output = self._get_majority_class(child)
            node.children[val] = child
        node.clear_samples()

    def _recursive_build(self, node):
        if node.is_leaf() and not self._is_pure(node):
            best_feat, gain = get_best_split(node.statistics)
            if best_feat is not None and gain > 0:
                self._split_leaf(node, best_feat)
                for child in node.children.values():
                    self._recursive_build(child)

    def _get_majority_class(self, node):
        if not node.statistics.label_distribution:
            return None
        return max(node.statistics.label_distribution, key=node.statistics.label_distribution.get)

    def _is_pure(self, node):
        return len(node.statistics.label_distribution) <= 1

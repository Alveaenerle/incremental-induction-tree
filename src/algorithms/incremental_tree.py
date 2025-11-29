from src.core.node import Node
from src.utils.tree_utils import (
    is_pure,
    majority_class
)
from src.utils.split_utils import (
    choose_best_split,
    split_data
)


class IncrementalTree:
    def __init__(self):
        self.root = None

    def fit(self, data):
        self.root = self._fit(data)

    def predict(self, row):
        node = self.root
        while not node.is_leaf():
            if row[node.feature] == node.value:
                node = node.left
            else:
                node = node.right
        return node.output

    def _fit(self, data):
        if is_pure(data):
            return Node(output=majority_class(data))
        chosen_feature, chosen_value = choose_best_split(data)
        node = Node(feature=chosen_feature, value=chosen_value)
        left_data, right_data = split_data(data, chosen_feature, chosen_value)
        node.left = self._fit(left_data)
        node.right = self._fit(right_data)
        return node

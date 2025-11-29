class Node:
    def __init__(
        self,
        feature=None,
        value=None,
        left=None,
        right=None,
        output=None
    ):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.output = output

    def is_leaf(self):
        return self.output is not None

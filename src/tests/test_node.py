from src.core.node import Node


def test_node_initialization():
    node = Node(feature=0, value=5)
    assert node.feature == 0
    assert node.value == 5
    assert node.left is None
    assert node.right is None
    assert node.output is None


def test_node_is_leaf():
    leaf_node = Node(output='A')
    assert leaf_node.is_leaf() is True

    internal_node = Node(feature=1, value=10)
    assert internal_node.is_leaf() is False


def test_node_children_assignment():
    parent_node = Node(feature=2, value=15)
    left_child = Node(output='B')
    right_child = Node(output='C')

    parent_node.left = left_child
    parent_node.right = right_child

    assert parent_node.left == left_child
    assert parent_node.right == right_child

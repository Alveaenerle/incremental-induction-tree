import pytest
import numpy as np
from src.core.node import Node


class TestNode:
    def test_initialization_default(self):
        node = Node()
        assert node.feature is None
        assert node.output is None
        assert node.statistics is None
        assert node.children == {}
        assert node.samples == []

    def test_initialization_with_params(self):
        node = Node(feature=1, output='A')
        assert node.feature == 1
        assert node.output == 'A'
        assert not node.is_leaf()

    def test_is_leaf(self):
        leaf = Node()
        assert leaf.is_leaf() is True

        internal = Node(feature=0)
        assert internal.is_leaf() is False

    def test_add_sample(self):
        node = Node()
        sample = [1, 5, 'A']
        node.add_sample(sample)

        assert len(node.samples) == 1
        assert node.samples[0] == sample

    def test_add_batch_samples_list(self):
        node = Node()
        batch = [[1, 'A'], [2, 'B']]
        node.add_batch_samples(batch)

        assert len(node.samples) == 2
        assert node.samples == batch

    def test_add_batch_samples_numpy(self):
        node = Node()
        batch = np.array([
            [1, 10, 'A'],
            [2, 20, 'B']
        ], dtype=object)

        node.add_batch_samples(batch)

        assert len(node.samples) == 2
        assert isinstance(node.samples, list)
        assert node.samples[0][0] == 1
        assert node.samples[1][2] == 'B'

    def test_clear_samples(self):
        node = Node()
        node.add_sample([1, 'A'])
        assert node.samples is not None

        node.clear_samples()
        assert node.samples is None

    def test_children_management(self):
        node = Node(feature=0)
        child_left = Node(output='Left')
        child_right = Node(output='Right')

        node.children[0] = child_left
        node.children[1] = child_right

        assert len(node.children) == 2
        assert node.children[0] == child_left
        assert node.children[1] == child_right

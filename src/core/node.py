import numpy as np


class Node:
    def __init__(
        self,
        statistics=None,
        feature=None,
        output=None
    ):
        self.statistics = statistics
        self.feature = feature
        self.children = {}
        self.output = output
        self.samples = []

    def is_leaf(self):
        return self.feature is None

    def add_sample(self, sample):
        self.samples.append(sample)

    def add_batch_samples(self, samples_array):
        if isinstance(samples_array, np.ndarray):
            self.samples.extend(samples_array.tolist())
        else:
            self.samples.extend(samples_array)

    def clear_samples(self):
        self.samples = None

import numpy as np

class Node:
    def __init__(
        self,
        statistics=None,
        feature=None,
        output=None
    ):
        self.statistics = statistics
        self.feature = feature # Jeśli None -> Liść
        self.children = {}     # Mapa: wartość -> Dziecko
        self.output = output   # Klasa dominująca
        self.samples = []      # Bufor danych (tylko dla liści)

    def is_leaf(self):
        return self.feature is None

    def add_sample(self, sample):
        # Opcjonalnie: assert self.samples is not None, "Internal nodes cannot store samples!"
        self.samples.append(sample)

    def add_batch_samples(self, samples_array):
        if isinstance(samples_array, np.ndarray):
            self.samples.extend(samples_array.tolist())
        else:
            self.samples.extend(samples_array)

    def clear_samples(self):
        self.samples = None

    def __repr__(self):
        # Pomocne przy print(node)
        type_str = f"Leaf(output={self.output}, samples={len(self.samples) if self.samples else 0})" \
                   if self.is_leaf() else \
                   f"Node(feature={self.feature}, children={len(self.children)})"
        return type_str

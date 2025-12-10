class Stats:
    def __init__(self, num_features=None, data=None):
        self.total_count = 0
        self.label_distribution = {}
        self.distribution = []

        if data is not None and len(data) > 0:
            num_features = data.shape[1] - 1
            self._init_structure(num_features)
            self.add_batch(data)
        elif num_features is not None:
            self._init_structure(num_features)

    def _init_structure(self, num_features):
        self.distribution = [{} for _ in range(num_features)]

    def add_batch(self, data):
        for row in data:
            self.add_sample(row)

    def add_sample(self, row):
        label = row[-1]
        self.total_count += 1
        self.label_distribution[label] = self.label_distribution.get(label, 0) + 1

        if not self.distribution:
            self._init_structure(len(row) - 1)

        for feat_idx, value in enumerate(row[:-1]):
            if value not in self.distribution[feat_idx]:
                self.distribution[feat_idx][value] = {}

            label_counts = self.distribution[feat_idx][value]
            label_counts[label] = label_counts.get(label, 0) + 1

    def merge(self, other_stats):
        self.total_count += other_stats.total_count
        for label, count in other_stats.label_distribution.items():
            self.label_distribution[label] = self.label_distribution.get(label, 0) + count

        for i, feat_dict in enumerate(other_stats.distribution):
            if i >= len(self.distribution):
                self.distribution.append({})
            for val, labels in feat_dict.items():
                if val not in self.distribution[i]:
                    self.distribution[i][val] = {}
                for l, c in labels.items():
                    self.distribution[i][val][l] = self.distribution[i][val].get(l, 0) + c

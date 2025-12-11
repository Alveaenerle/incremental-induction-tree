import pytest
import numpy as np
from src.core.stats import Stats


class TestStatsInitialization:
    def test_empty_init(self):
        stats = Stats()
        assert stats.total_count == 0
        assert stats.label_distribution == {}
        assert stats.distribution == []

    def test_init_with_num_features(self):
        stats = Stats(num_features=3)
        assert stats.total_count == 0
        assert len(stats.distribution) == 3
        assert all(d == {} for d in stats.distribution)

    def test_init_with_data(self):
        data = np.array([
            [1, 10, 'A'],
            [1, 20, 'B']
        ], dtype=object)

        stats = Stats(data=data)

        assert stats.total_count == 2
        assert stats.label_distribution == {'A': 1, 'B': 1}
        assert len(stats.distribution) == 2

        assert stats.distribution[0][1]['A'] == 1
        assert stats.distribution[0][1]['B'] == 1

        assert stats.distribution[1][10]['A'] == 1
        assert stats.distribution[1][20]['B'] == 1


class TestStatsUpdates:
    def test_add_single_sample(self):
        stats = Stats(num_features=2)
        sample = np.array([1, 5, 'A'], dtype=object)

        stats.add_sample(sample)

        assert stats.total_count == 1
        assert stats.label_distribution['A'] == 1
        assert stats.distribution[0][1]['A'] == 1
        assert stats.distribution[1][5]['A'] == 1

    def test_add_batch(self):
        stats = Stats(num_features=2)
        batch = np.array([
            [1, 10, 'A'],
            [2, 10, 'A'],
            [1, 20, 'B']
        ], dtype=object)

        stats.add_batch(batch)

        assert stats.total_count == 3
        assert stats.label_distribution['A'] == 2
        assert stats.label_distribution['B'] == 1

        assert stats.distribution[1][10]['A'] == 2

    def test_dynamic_structure_init(self):
        stats = Stats()
        sample = np.array([100, 'X'], dtype=object)

        stats.add_sample(sample)

        assert stats.total_count == 1
        assert len(stats.distribution) == 1
        assert stats.distribution[0][100]['X'] == 1


class TestStatsMerge:
    def test_merge_disjoint_data(self):
        stats1 = Stats(data=np.array([[1, 'A']], dtype=object))
        stats2 = Stats(data=np.array([[2, 'B']], dtype=object))

        stats1.merge(stats2)

        assert stats1.total_count == 2
        assert stats1.label_distribution == {'A': 1, 'B': 1}

        assert stats1.distribution[0][1]['A'] == 1
        assert stats1.distribution[0][2]['B'] == 1

    def test_merge_overlapping_data(self):
        stats1 = Stats(data=np.array([[1, 'A']], dtype=object))
        stats2 = Stats(data=np.array([[1, 'A']], dtype=object))

        stats1.merge(stats2)

        assert stats1.total_count == 2
        assert stats1.label_distribution['A'] == 2
        assert stats1.distribution[0][1]['A'] == 2

    def test_merge_complex_structure(self):
        data1 = np.array([
            [0, 10, 'Y'], 
            [0, 20, 'N']
        ], dtype=object)
        stats1 = Stats(data=data1)

        data2 = np.array([
            [1, 10, 'Y'],
            [0, 10, 'N']
        ], dtype=object)
        stats2 = Stats(data=data2)

        stats1.merge(stats2)

        assert stats1.total_count == 4
        assert stats1.label_distribution['Y'] == 2
        assert stats1.label_distribution['N'] == 2

        assert stats1.distribution[1][10]['Y'] == 2
        assert stats1.distribution[1][10]['N'] == 1

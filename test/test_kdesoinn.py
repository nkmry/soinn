import unittest
from numpy.testing import assert_array_equal
from test_soinn import TestSoinn
import numpy as np
from scipy.sparse import dok_matrix
from kdesoinn import KdeSoinn


class TestKdeSoinn(TestSoinn):
    def setUp(self):
        self.soinn = KdeSoinn()
        self.soinn.nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]
                                    , dtype=np.float64)
        self.soinn.dim = 2
        self.soinn.adjacent_mat = dok_matrix((4, 4))
        self.soinn.winning_times = [1] * 4
        for i in range(4):
            self.soinn.network_sigmas.append(np.eye(2))

    def test_add_node(self):
        super().test_add_node()
        self.assertEqual(len(self.soinn.network_sigmas), 4)
        d = self.soinn.dim
        for i in range(4):
            assert_array_equal(self.soinn.network_sigmas[i], np.zeros((d, d)))

    def test_delete_old_edges(self):
        super().test_delete_old_edges()
        # TODO tset after implementing _update_network_sigma()

    def test_delete_nodes(self):
        # marking
        for i in range(4):
            self.soinn.network_sigmas[i][0, 0] = i
        super().test_delete_nodes()
        self.assertEqual(len(self.soinn.network_sigmas), 3)
        for i, v in zip(range(3), [0, 2, 3]):
            assert_array_equal(self.soinn.network_sigmas[i],
                               np.array([[v, 0.], [0., 1.]]))

    def test_delete_nodes_with_deleting_several_nodes(self):
        # marking
        for i in range(4):
            self.soinn.network_sigmas[i][0, 0] = i
        super().test_delete_nodes_with_deleting_several_nodes()
        self.assertEqual(len(self.soinn.network_sigmas), 2)
        for i, v in zip(range(2), [0, 2]):
            assert_array_equal(self.soinn.network_sigmas[i],
                               np.array([[v, 0.], [0., 1.]]))


if __name__ == '__main__':
    unittest.main()

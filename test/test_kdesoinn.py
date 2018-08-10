import unittest
from numpy.testing import assert_array_equal
from test_soinn import TestSoinn
import numpy as np
from numpy import ndarray
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

    def test_calculate_mahalanobis_distance(self):
        target = np.array([1, 0])
        source = np.array([0, 0])
        cov_mat = np.eye(2)
        d = self.soinn._calculate_mahalanobis_distance(target, source, cov_mat)
        self.assertEqual(d, 1.0)

        cov_mat = np.array([[2, 0], [0, 1]])
        d = self.soinn._calculate_mahalanobis_distance(target, source, cov_mat)
        self.assertEqual(d, 0.5)

        source = np.array([1, 1])
        d = self.soinn._calculate_mahalanobis_distance(target, source, cov_mat)
        self.assertEqual(d, 1.0)

    def test_check_valid_matrix(self):
        M = np.eye(3)
        self.assertEqual(KdeSoinn._check_valid_matrix(M), ' ')
        M[2, 1] = np.nan
        self.assertEqual(KdeSoinn._check_valid_matrix(M), 'n')
        M[2, 1] = np.inf
        self.assertEqual(KdeSoinn._check_valid_matrix(M), 'i')
        M = np.zeros(3)
        self.assertEqual(KdeSoinn._check_valid_matrix(M), 'z')
        M = np.ones((3, 3))
        self.assertEqual(KdeSoinn._check_valid_matrix(M), 'r')
        M = np.eye(3)
        M[2, 2] = -1
        self.assertEqual(KdeSoinn._check_valid_matrix(M), 'p')

    def test_get_pals(self):
        self.soinn.adjacent_mat[1, :] = np.array([0, 0, 1, 1])
        self.soinn.adjacent_mat[:, 1] = np.array([[0, 0, 1, 1]]).transpose()
        self.assertEqual([2, 3], sorted(self.soinn._get_pals(1)))
        self.assertEqual([1], self.soinn._get_pals(3))

    def test_get_connected_nodes(self):
        self.soinn.adjacent_mat[0, :] = np.array([0, 0, 1, 1])
        self.soinn.adjacent_mat[:, 0] = np.array([[0, 0, 1, 1]]).transpose()
        self.soinn.adjacent_mat[1, 2] = 1
        self.soinn.adjacent_mat[2, 1] = 1
        actual = self.soinn._get_connected_nodes(begin_node_index=1, depth=1)
        self.assertEqual([1, 2], actual)
        actual = self.soinn._get_connected_nodes(begin_node_index=1, depth=2)
        self.assertEqual([0, 1, 2], actual)

    def test_update_network_sigma(self):
        self.soinn.adjacent_mat[0, :] = np.array([0, 0, 1, 1])
        self.soinn.adjacent_mat[:, 0] = np.array([[0, 0, 1, 1]]).transpose()
        self.soinn._update_network_sigma(0)
        assert_array_equal(np.array([[.5, .5], [.5, 1]]),
                           self.soinn.network_sigmas[0])
        self.soinn._update_network_sigma(1)
        assert_array_equal(np.zeros((2, 2)), self.soinn.network_sigmas[1])

    def test_update_network_sigmas(self):
        self.soinn.adjacent_mat[0, :] = np.array([0, 0, 1, 1])
        self.soinn.adjacent_mat[:, 0] = np.array([[0, 0, 1, 1]]).transpose()
        self.soinn.adjacent_mat[1, :] = np.array([0, 0, 1, 0])
        self.soinn.adjacent_mat[:, 1] = np.array([[0, 0, 1, 0]]).transpose()
        self.soinn._update_network_sigmas(winner_index=0, connection_depth=2)
        assert_array_equal(np.array([[.5, .5], [.5, 1]]),
                           self.soinn.network_sigmas[0])
        assert_array_equal(np.array([[0, 0], [0, 1]]),
                           self.soinn.network_sigmas[1])


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2016 Yoshihiro Nakamura
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import numpy as np
from numpy import ndarray
from numpy.linalg import det, eigvals, inv
from scipy.sparse import dok_matrix
from soinn import Soinn


class KdeSoinn(Soinn):
    """ Kernel Density Estimator with SOINN (KDESOINN)
        Ver. 0.0.1
    """
    def __init__(self, delete_node_period=300, max_edge_age=50,
                 init_node_num=3, coeff=0.1):
        """
        :param delete_node_period:
            A period deleting nodes. The nodes that doesn't satisfy some
            condition are deleted every this period.
        :param max_edge_age:
            The maximum of edges' ages. If an edge's age is more than this,
            the edge is deleted.
        :param init_node_num:
            The number of nodes used for initialization
        :param coeff:
            Threshold coefficient
        """
        self.threshold_coefficient = coeff
        self.k = -1
        super().__init__(delete_node_period, max_edge_age, init_node_num)

    def _reset_state(self):
        super()._reset_state()
        self.network_sigmas = []

    def input_signal(self, signal: ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the
        state. So the function does batch training.
        :param signal: A new input signal
        :return:
        """
        signal = self._check_signal(signal)
        self.num_signal += 1

        if self.nodes.shape[0] < self.init_node_num:
            self._add_node(signal)
            return

        winners, dists = self._find_nearest_nodes(2, signal)
        sim_thresholds =\
            self._calculate_mahalanobis_sim_thresholds(winners, signal)
        if dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]:
            self._add_node(signal)
        else:
            self._add_edge(winners)
            self._increment_edge_ages(winners[1])
            winners[1] = self._delete_old_edges(winners[1])
            self._update_winner(winners[1], signal)
            self._update_network_sigmas(winners[1], 1)

        if self.num_signal % self.delete_node_period == 0:
            self._delete_noise_nodes()
            self._adjust_network()

    def _add_node(self, signal):
        super()._add_node(signal)
        d = len(signal)
        self.network_sigmas.append(np.zeros((d, d)))

    def _delete_old_edges(self, winner_index):
        """
        :return: winner_index after deletion
        """
        delete_node_candidates = []
        for k, v in self.adjacent_mat[winner_index, :].items():
            if v > self.max_edge_age + 1:
                delete_node_candidates.append(k[1])
                self._set_edge_weight((winner_index, k[1]), 0)
        delete_node_indexes = []
        for i in delete_node_candidates:
            if len(self.adjacent_mat[i, :]) == 0:
                delete_node_indexes.append(i)
            #else:
            #    self._update_network_sigma(i)
            # TODO implement _update_network_sigma()
        self._delete_nodes(delete_node_indexes)
        delete_count = sum(
            [1 if i < winner_index else 0 for i in delete_node_indexes])
        return winner_index - delete_count

    def _delete_nodes(self, indexes):
        super()._delete_nodes(indexes)
        for i in reversed(np.sort(indexes)):
            del self.network_sigmas[i]

    def _calculate_mahalanobis_sim_thresholds(self, node_indexes: list, signal):
        sq_dists = np.zeros(2,1)
        for i in range(2):
            C = self._calculate_node_covariance_matrix(node_indexes[i])
            sq_dists[i] = self._calculate_mahalanobis_distance(
                signal, self.nodes[node_indexes[i], :], C)
        return sq_dists

    def _calculate_node_covariance_matrix(self, node_index):
        n = self.nodes.shape[0]
        connections = self.adjacent_mat[node_index, :].toarray()
        pal_indexes = [i for i, c in zip(range(n),connections) if c > 0]
        if len(pal_indexes) > 0:
            pals = self.nodes[pal_indexes, :]
            repeats = [0] * n
            repeats[node_index] = len(pal_indexes)
            w = np.sqrt(min(np.sum(((pals - np.repeat(
                self.nodes, repeats, axis=0)) ** 2), axis=1)))
            M = self.network_sigmas[node_index]\
                + self.threshold_coefficient * w * np.eye(self.dim)
        else:
            _, sq_dists = self._find_nearest_nodes(2, self.nodes[node_index, :])
            w = np.sqrt(sq_dists(2))
            M = self.threshold_coefficient * w * np.eye(self.dim)
        return M

    @staticmethod
    def _calculate_mahalanobis_distance(
            target: ndarray, source: ndarray, cov_mat: ndarray) -> ndarray:
        e = KdeSoinn._check_valid_matrix(cov_mat)
        if e != ' ':
            print(e)
            cov_mat = KdeSoinn._modify_irregular_matrix(cov_mat)
        dif = target - source
        sq_dist = np.dot(np.dot(dif, inv(cov_mat)), dif)
        return sq_dist

    @staticmethod
    def _modify_irregular_matrix(M: ndarray) -> ndarray:
        tol = 1.0e-5
        c_ = 0
        while KdeSoinn._check_valid_matrix(M) != ' ':
            M = M + tol * np.eye(*M.shape)
            c_ = c_ + 1
            print(c_)
        return M

    @staticmethod
    def _check_valid_matrix(mat: ndarray) -> str:
        """
        Check a specific matrix is varid or not.
        :param mat:
        :return:
            ' ' if the matrix is valid.
            'n' if the matrix contains NaN.
            'i' if the matrix contains infinity.
            'z' if the matrix is a zero matrix.
            'r' if the matrix is a regular matrix.
            'p' if the matrix is a non-negative definite matrix.
            ( positive definite matrix or positive semidefinite matrix).
        """
        if np.any(np.isnan(mat)):
            return 'n'
        if np.any(np.isinf(mat)):
            return 'i'
        if np.all(abs(mat) < 1.0e-6):
            return 'z'
        if abs(det(mat)) < 1.0e-6:
            return 'r'
        if np.any(eigvals(mat) < 0):
            return 'p'
        return ' '

    def _update_network_sigmas(self, winner_index: int, connection_depth: int):
        target_nodes = self._get_connected_nodes(winner_index,
                                                 connection_depth)
        for j in range(len(target_nodes)):
            self._update_network_sigma(target_nodes[j])

    def _get_connected_nodes(self, begin_node_index: int, depth: int):
        """
        get nodes connected to the specific node.
        :param begin_node_index: start searching from this node.
        :param depth: how deeply to search. 1 or 2.

        """
        if depth not in [1, 2]:
            return []
        include_flags = np.zeros(self.nodes.shape[0])
        include_flags[begin_node_index] = 1
        pals = self._get_pals(begin_node_index)
        include_flags[pals] = 1
        if depth == 2:
            for j in range(len(pals)):
                pals_j = self._get_pals(pals[j])
                include_flags[pals_j] = 1
        return [i for i, v in enumerate(include_flags) if v > 0]

    def _get_pals(self, node_index: int) -> list:
        """
        Get adjacent nodes of a specific node
        :param node_index:
        :return: 
        """
        return [i[0] for i in self.adjacent_mat.getcol(node_index).keys()]

    def _update_network_sigma(self, node_index: int):
        """
        Update network covariance matrix of a specific node
        :param node_index: the index of the specific node
        :return: 
        """
        pals = self._get_pals(node_index)
        if len(pals) < 1:
            self.network_sigmas[node_index] = np.zeros([self.dim] * 2)
        else:
            sigma = np.zeros([self.dim] * 2)
            node = self.nodes[node_index, :]
            difs = self.nodes[pals, :] - np.tile(node, (len(pals), 1))
            w_sum = 0.0
            for j in range(len(pals)):
                w = self.winning_times[pals[j]]
                d = difs[j, :]
                w_sum += w
                sigma += w * np.outer(d, d)
            self.network_sigmas[node_index] = sigma / w_sum

    def _adjust_network(self):
        g = self._create_knn_graph(self.k)
        for i in g.keys():
            if i not in self.adjacent_mat.keys():
                self.adjacent_mat[i] = 1
        updated_indexes = list({i[0] for i in g.keys()})
        for i in updated_indexes:
            self._update_network_sigma(i)

    def _create_knn_graph(self, k: int) -> dok_matrix:
        n = self.nodes.shape[0]
        A = dok_matrix((n, n), dtype=np.int32)
        for i in range(n):
            neighbor_indexes, _ = self._find_nearest_nodes(k + 1,
                                                           self.nodes[i, :])
            for j in neighbor_indexes:
                if j != i:
                    A[i, j] = 1
        # only remain bidirected edges
        A = A.multiply(A.transpose()).todok()
        return A

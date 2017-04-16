# Copyright (c) 2016 Yoshihiro Nakamura
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import numpy as np
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

    def input_signal(self, signal: np.ndarray):
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

        winner, dists = self._find_nearest_nodes(2, signal)
        sim_thresholds = self._calculate_similarity_thresholds(winner)
        if dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]:
            self._add_node(signal)
        else:
            self._add_edge(winner)
            self._increment_edge_ages(winner[1])
            winner[1] = self._delete_old_edges(winner[1])
            self._update_winner(winner[1], signal)
            self.__update_network_sigmas(winner[1], 1)

        if self.num_signal % self.delete_node_period == 0:
            self._delete_noise_nodes()
            self.__adjust_network()

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



import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix


class Soinn(object):
    """ Self-Organizing Incremental Neural Network (SOINN)

    """

    def __init__(self, delete_node_period=300, max_edge_age=50):
        """
        :param delete_node_period: A period deleting nodes.
                The nodes that doesn't satisfy some condition are deleted every this period.
        :param max_edge_age: The maximum of edges' ages.
                If an edge's age is more than this, the edge is deleted.
        :return:
        """
        self.delete_node_period = delete_node_period
        self.max_edge_age = max_edge_age
        self.min_degree = 1
        self.num_signal = 0
        self.nodes = np.array([])
        self.winning_times = np.array([])
        self.adjacent_mat = coo_matrix((0, 0))

    def input_signal(self, signal: np.ndarray):
        """ Input a new signal to SOINN
        :param signal: A new input signal
        :return:
        """
        self.__check_signal(signal)
        self.num_signal += 1

        if self.nodes.shape[0] < 3:
            self.__add_node(signal)
            return

        [winner, dists] = self.__find_nearest_nodes(2, signal)
        sim_thresholds = self.__calculate_similarity_thresholds(winner)
        if dists[0] > sim_thresholds[0] and dists[1] > sim_thresholds[1]:
            self.__add_node(signal)
        else:
            self.__add_edge(winner)
            self.__increment_edge_ages(winner[1])
            self.__delete_old_edges(winner[1])
            self.__update_winner(winner[1], signal)
            self.__update_adjacent_nodes(winner[1], signal)

        if self.num_signal % self.delete_node_period == 0:
            self.__delete_noise_nodes()

    def __check_signal(self, signal: np.ndarray):
        """ check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as self.dim
        :param signal: an input signal
        """
        if not(isinstance(signal, np.ndarray)):
            raise TypeError()
        if len(signal.shape) != 1:
            raise TypeError()
        if not(hasattr(self, 'dim')):
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                raise TypeError()

    def __add_node(self, signal):
        pass

    def __find_nearest_nodes(self, num, signal):
        pass

    def __calculate_similarity_thresholds(self, node_index):
        pass

    def __add_edge(self, node_indexes):
        pass

    def __increment_edge_ages(self, winner_index):
        pass

    def __delete_old_edges(self, winner_index):
        pass

    def __update_winner(self, winner_index, signal):
        pass

    def __update_adjacent_nodes(self, winner_index, signal):
        pass

    def __delete_noise_nodes(self, delete_noise_nodes):
        pass

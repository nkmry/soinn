# Copyright (c) 2016 Yoshihiro Nakamura
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

from soinn import Soinn


class KdeSoinn(Soinn):
    """ Kernel Density Estimator with SOINN (KDESOINN)
        Ver. 0.0.1
    """
    def __init__(self, delete_node_period=300, max_edge_age=50,
                 init_node_num=3):
        """
        :param delete_node_period:
            A period deleting nodes. The nodes that doesn't satisfy some
            condition are deleted every this period.
        :param max_edge_age:
            The maximum of edges' ages. If an edge's age is more than this,
            the edge is deleted.
        :param init_node_num:
            The number of nodes used for initialization
        """
        super().__init__(delete_node_period, max_edge_age, init_node_num)

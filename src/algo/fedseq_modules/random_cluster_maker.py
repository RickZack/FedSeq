from src.algo.fed_clients.base_client import Client

from src.algo.fedseq_modules.cluster_maker import ClusterMaker, ClientCluster
import copy
from typing import List
import numpy as np
import logging

log = logging.getLogger(__name__)


class RandomClusterMaker(ClusterMaker):
    """
    Attempt to make heterogeneous groups applying a naive random assignment strategy of clients to groups
    """
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        clusters: List[ClientCluster] = []
        s_clients = copy.copy(clients)
        np.random.shuffle(s_clients)
        n_clusters = 0
        cluster = ClientCluster(n_clusters, logger=log)
        for c in s_clients:
            if cluster.num_examples < self._min_examples and cluster.num_clients < self._max_clients:
                cluster.add_client(c)
            else:
                clusters.append(cluster)
                n_clusters += 1
                cluster = ClientCluster(n_clusters, logger=log)
                cluster.add_client(c)
        self._check_redistribution(cluster, clusters)
        self._collect_clustering_statistics(clients, ("clusters", [c.clients_id for c in clusters]))
        return clusters

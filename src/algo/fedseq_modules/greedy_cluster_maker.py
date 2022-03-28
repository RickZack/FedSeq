from src.algo.fedseq_modules.cluster_maker import ClientCluster, InformedClusterMaker
from src.algo.fed_clients.base_client import Client
from typing import List, Tuple
import numpy as np
import logging

log = logging.getLogger(__name__)


class GreedyClusterMaker(InformedClusterMaker):
    """
    Makes heterogeneous groups using an iterative greedy approach, maximizing at each step a given measure
    on the group being built
    """
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def _extract_best_client(self, reference_prediction: np.ndarray, clients: List[Client],
                             predictions: List[np.ndarray]) -> Tuple[Client, np.ndarray]:
        best_diff = 0
        best_index = 0
        assert len(clients) == len(predictions), "Wrong number of clients and/or representers"
        measure = self.diff_measure()
        for index, p in zip(range(len(predictions)), predictions):
            current_diff = measure(reference_prediction, p)
            if current_diff > best_diff:
                best_diff = current_diff
                best_index = index
        client = clients.pop(best_index)
        prediction = predictions.pop(best_index)
        return client, prediction

    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        clusters: List[ClientCluster] = []
        n_clusters = 0
        rem_clients = list(clients)  # shallow copy
        rem_representers = list(representers)  # shallow copy
        assert len(rem_representers) == len(rem_clients) and len(clients) > 0, "Not enough clients' data"
        conf_len = len(rem_representers[0])
        assert all(conf_len == len(conf) for conf in representers), "Mismatching dimensions"

        cluster = ClientCluster(n_clusters, conf_len, log)
        while len(rem_clients) != 0:
            if cluster.num_examples < self._min_examples and cluster.num_clients < self._max_clients:
                # extract best client
                (client, confidence) = self._extract_best_client(cluster.representer, rem_clients,
                                                                 rem_representers)
                cluster.add_client(client, confidence)
            else:
                clusters.append(cluster)
                n_clusters += 1
                cluster = ClientCluster(n_clusters, conf_len, log)
        self._check_redistribution(cluster, clusters)
        self._collect_clustering_statistics(clients, ("clusters", [c.clients_id for c in clusters]))
        return clusters

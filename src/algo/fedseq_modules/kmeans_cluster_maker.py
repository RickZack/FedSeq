from src.algo.fedseq_modules.cluster_maker import ClientCluster, InformedClusterMaker
from src.algo.fed_clients.base_client import Client
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
import copy
import itertools as it
import logging

log = logging.getLogger(__name__)


class KMeansClusterMaker(InformedClusterMaker):
    """
    Makes heterogeneous groups applying a uniform sampling strategy on top of homogeneous groups obtained using KMeans
    """
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> \
            List[ClientCluster]:
        k = self.num_classes
        k_means = KMeans(k).fit(representers)
        k_clusters, mean, dev_std = KMeansClusterMaker._get_k_clusters(k_means.labels_, k)
        clusters: List[ClientCluster] = self._sample_from_kclusters(k_clusters, clients)
        k_cluster_ids = [[clients[i].client_id for i in k_cl] for k_cl in k_clusters]
        self._collect_clustering_statistics(clients, ("k_clusters", k_cluster_ids), ("k_clusters_mean", mean),
                                            ("k_cluster_devstd", dev_std), ("clusters", [c.clients_id for c in clusters]))
        return clusters

    @staticmethod
    def _get_k_clusters(labels: List[int], k: int) -> Tuple[List[List[int]], float, float]:
        k_clusters: List[List[int]] = [list() for _ in range(k)]
        for i, l in enumerate(labels):
            k_clusters[l].append(i)
        clusters_dim = [len(c) for c in k_clusters]
        mean, dev_std = np.mean(clusters_dim), np.std(clusters_dim)
        log.info(f"KMeans generated clusters' dim with mean={mean}, dev.std={dev_std}")
        return k_clusters, float(mean), float(dev_std)

    def _sample_from_kclusters(self, k_clusters: List[List[int]], clients: List[Client]) -> List[ClientCluster]:
        clusters: List[ClientCluster] = []
        num_clients_to_assign = np.sum([len(c) for c in k_clusters])
        n_superclient = 0
        k_clusters = copy.deepcopy(k_clusters)
        clusters_iterator = it.cycle(k_clusters)
        cluster = ClientCluster(n_superclient, logger=log)
        while num_clients_to_assign != 0:
            if cluster.num_examples < self._min_examples and cluster.num_clients < self._max_clients:
                k_cluster = next(clusters_iterator)
                while len(k_cluster) == 0: k_cluster = next(clusters_iterator)  # find first non-empty k_cluster
                cluster.add_client(clients[k_cluster.pop(-1)])
                num_clients_to_assign -= 1
            else:
                clusters.append(cluster)
                n_superclient += 1
                cluster = ClientCluster(n_superclient, logger=log)
        self._check_redistribution(cluster, clusters)
        return clusters

    def uses_custom_metric(self) -> bool:
        return True

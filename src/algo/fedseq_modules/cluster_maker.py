import os
from typing import List, Tuple, Callable, Optional, Any
import numpy as np
from scipy.stats import wasserstein_distance

from src.algo.fed_clients import Client

import math
import itertools as it
from abc import ABC, abstractmethod

from src.algo.fedseq_modules import FedSeqSuperClient
import logging

from src.utils import save_pickle

log = logging.getLogger(__name__)


class ClientCluster:
    """ Represents a cluster of clients being built """
    def __init__(self, id, representer_len=None, logger=log):
        self.clients = []
        self.clients_representer = []
        self.id = id
        self.__num_examples = 0
        self.representer_len = representer_len
        self.log = logger

    def add_client(self, client: Client, representer: Optional[np.ndarray] = None) -> None:
        """
        Adds a client to the cluster

        Parameters
        ----------
        client
            the client to be added
        representer
            the representation for the client to be added
        """
        self.clients.append(client)
        self.clients_representer.append(representer)
        self.__num_examples += len(client)

    @property
    def representer(self) -> np.ndarray:
        """

        Returns
        -------
        the representer resulting from the weighted average of the clients' representer inside the cluster
        """
        assert self.representer_len is not None, "Vector dimension needed to calculate cluster vector"
        if len(self.clients) == 0:
            return np.full(self.representer_len, 1 / self.representer_len)
        cluster_representer = np.zeros(self.representer_len)
        clients_dataset_len = np.sum([len(c) for c in self.clients])
        for c, p in zip(self.clients, self.clients_representer):
            cluster_representer = cluster_representer + np.array(p) * len(c) / clients_dataset_len
        return cluster_representer

    def make_superclient(self, verbose: bool = False, **sup_kwargs):
        """
        Return a superclient from the clients inside the cluster

        Parameters
        ----------
        verbose
            flag to allow for a verbose output about the created superclient (id, num. of examples, num. of clients)
        sup_kwargs
            other arguments to forward to the superclient constructor

        Returns
        -------
        a newly created superclient

        """
        if verbose:
            self.log.info(f"superclient {self.id}: examples: {self.__num_examples}, clients: {self.num_clients}")
        return FedSeqSuperClient(self.id, self.clients, **sup_kwargs)

    @property
    def num_clients(self) -> int:
        return len(self.clients)

    @property
    def num_examples(self) -> int:
        return self.__num_examples

    def pop(self, index: int = -1) -> Tuple[Client, np.ndarray]:
        """
        Removed a client from the cluster

        Parameters
        ----------
        index
            the index of the client to remove

        Returns
        -------
        the removed client and its representer
        """
        client = self.clients.pop(index)
        client_representer = self.clients_representer.pop(index)
        self.__num_examples -= len(client)
        return client, client_representer

    @property
    def clients_id(self) -> List[int]:
        return [c.client_id for c in self.clients]


class ClusterMaker(ABC):
    """ Base (abstract) class for any clustering algorithm """
    def __init__(self, min_examples: int, max_clients: int, save_statistics: bool, savedir: str,
                 measure: str = None, verbose: bool = False, *args, **kwargs):
        self._min_examples = min_examples
        self._max_clients = max_clients
        self._save_statistics = save_statistics
        self._savedir = savedir
        self._measure = measure
        self.verbose = verbose
        self._statistics = {}

    @property
    def measure(self):
        return self._measure

    @measure.setter
    def measure(self, measure):
        self._measure = measure

    @property
    def save_statistic(self):
        return self._save_statistics

    @save_statistic.setter
    def save_statistic(self, save_statistics):
        self._save_statistics = save_statistics

    def make_superclients(self, clients: List[Client], representers: List[np.ndarray], sub_path: str = "",
                          **sup_kwargs) -> List[FedSeqSuperClient]:
        """
        Makes superclient according to the specific clustering algorithm, collects and save statistics on the newly
        created superclients

        Parameters
        ----------
        clients
            the clients to group into superclients
        representers
            the clients' representers
        sub_path
            the subpath relative to the saving directory of the main program to save the statistics to
        sup_kwargs
            other arguments to forward to the superclient constructor

        Returns
        -------
        a list containing the newly created superclients

        """
        self._statistics.clear()
        num_classes = clients[0].num_classes
        assert all(num_classes == c.num_classes for c in clients), "Clients have different label space's dimension"
        clusters = self._make_clusters(clients, representers)
        sp = [c.make_superclient(self.verbose, num_classes=num_classes, **sup_kwargs) for c in clusters]
        self._collect_clustering_statistics(clients, ("superclients", {i: s.num_ex_per_class()
                                                                       for i, s in enumerate(sp)}))
        save_pickle(self._statistics,
                    os.path.join(self._savedir, sub_path, f"{self.__class__.__name__}_{self._measure}_stats.pkl"))
        return sp

    @abstractmethod
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        pass

    def _collect_clustering_statistics(self, clients: List[Client], *groups: Tuple[str, Any]):
        """
        Collects and saves the statistics from the clustering algorithm

        Parameters
        ----------
        clients
            the clients grouped into superclients
        groups
            list of key-value pairs representing statistics
        """
        if self._save_statistics:
            self._statistics.update({"classname": self.__class__.__name__})
            if "clients" not in self._statistics:
                self._statistics.update({"clients": {c.client_id: c.num_ex_per_class() for c in clients}})
            self._statistics.update(dict(groups))

    def requires_clients_evaluation(self) -> bool:
        """

        Returns
        -------
        True if the clustering algorithm requires an estimation of the clients' distribution, False otherwise
        """
        return False

    def uses_custom_metric(self) -> bool:
        """

        Returns
        -------
        True if the clustering algorithm has to use a specific measure, False if the metric can be a parameter
        """
        return False

    def _verify_constraints(self, to_empty: ClientCluster, clusters: List[ClientCluster]):
        """
        Verifies that it is possible to redistribute the clients belonging to a cluster to clients belonging to
        other clusters, i.e. emptying the cluster won't make any cluster bigger than the maximum size

        Parameters
        ----------
        to_empty
            the cluster to empty
        clusters
            the other clusters
        """
        # verify that is possible to redistribute without violating the constraint
        num_clients = np.sum([c.num_clients for c in clusters]) + to_empty.num_clients
        assert math.ceil(num_clients / len(clusters)) <= self._max_clients, "Conflicting constraints"

    def _redistribute_clients(self, to_empty: ClientCluster, clusters: List[ClientCluster]):
        """
        Redistributes the clients belonging to a cluster to other clusters

        Parameters
        ----------
        to_empty
            the cluster to empty
        clusters
            the other clusters
        """
        clusters_dim_sorted = list(filter(lambda c: c.num_clients < self._max_clients, clusters))
        clusters_dim_sorted.sort(key=lambda c: c.num_examples)
        for cluster in it.cycle(clusters_dim_sorted):
            if cluster.num_clients < self._max_clients:
                client, client_confidence = to_empty.pop()
                cluster.add_client(client, client_confidence)
            if to_empty.num_clients == 0:
                break

    def _check_redistribution(self, to_empty: ClientCluster, clusters: List[ClientCluster]):
        """
        Checks if a cluster has too few examples, hence its clients need to be redistributed among other clusters

        Parameters
        ----------
        to_empty
            the cluster to empty
        clusters
            the other clusters
        """
        if to_empty.num_clients > 0:
            if to_empty.num_examples < self._min_examples:
                self._verify_constraints(to_empty, clusters)
                self._redistribute_clients(to_empty, clusters)
            else:
                clusters.append(to_empty)

    # return the number of superclient that can be build taking into account the constraints
    def _K(self, num_clients: int, dataset_dim: int) -> int:
        examples_per_client = dataset_dim // num_clients
        client_per_superclient = math.ceil(self._min_examples / examples_per_client)
        assert client_per_superclient <= self._max_clients, "Constraint infeasible: max_clients_superclient"
        num_superclients: int = num_clients // client_per_superclient
        return num_superclients


class InformedClusterMaker(ClusterMaker):
    """ Base (abstract) class for any clustering algorithm that uses information about clients' distribution """
    def __init__(self, measure, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._measure = measure

    def requires_clients_evaluation(self) -> bool:
        return True

    @abstractmethod
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        pass

    @staticmethod
    def gini_diff(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        mean_vector = (cluster_vec + client_vec) / 2
        return 1 - np.sum(np.power(mean_vector, 2))

    @staticmethod
    def cosine_diff(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        v1 = cluster_vec
        v2 = client_vec
        prod = np.dot(v1, v2)
        norms_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 1 - prod / norms_prod

    @staticmethod
    def kullback_div(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        mean_vector = (cluster_vec + client_vec) / 2
        uniform = np.ones(mean_vector.size) / mean_vector.size
        klvec = [mean_vector[i] * np.log(mean_vector[i] / uniform[i]) for i in range(mean_vector.size)]
        return 1 - (np.sum(klvec))

    def diff_measure(self) -> Callable[[np.ndarray, np.ndarray], float]:
        measures_methods = {"gini": InformedClusterMaker.gini_diff, "cosine": InformedClusterMaker.cosine_diff,
                            "kullback": InformedClusterMaker.kullback_div, "wasserstein": wasserstein_distance}
        if self.measure not in measures_methods:
            raise NotImplementedError
        return measures_methods[self.measure]

import os
from typing import List

import numpy as np
from abc import abstractmethod

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.algo.center_server import *
from src.utils import create_datasets, save_pickle
from src.algo import Algo
from src.algo.fed_clients import *
from src.models import Model
import logging

log = logging.getLogger(__name__)


class FedBase(Algo):
    """
    Base (abstract) class for any Federated Learning algorithm
    """

    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None, *args, **kwargs):
        common = params.common
        C, K, B, E, alpha, rebalance = common.C, common.K, common.B, common.E, common.alpha, common.rebalance
        assert 0 < C <= 1, f"Illegal value, expected 0 < C <= 1, given {C}"
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self._num_clients = K
        self._batch_size = B
        self._fraction = C
        self._local_epoch = E
        self._aggregation_policy = params.aggregation_policy

        # get the proper dataset
        local_datasets, test_dataset, self._dataset_num_classes = create_datasets(self._dataset,
                                                                                  self._num_clients, alpha,
                                                                                  common.max_iter_dirichlet,
                                                                                  rebalance)
        model = Model(model_info, self._dataset_num_classes)
        model_has_batchnorm = model.has_batchnorm()
        local_dataloaders = [
            DataLoader(dataset, batch_size=self._batch_size, shuffle=True, drop_last=model_has_batchnorm)
            for dataset in local_datasets]

        self._clients = [
            eval(params.client.classname)(k, local_dataloaders[k], self._dataset_num_classes, self._device,
                                          analyzer=self._analyzer.module_analyzer('client'), **params.client.args)
            for k in range(self._num_clients)
        ]
        self._selected_clients: List[Client] = []

        # take out examplars from test_dataset, will be used in FedSeq
        original_test_set_len = len(test_dataset)
        self._excluded_from_test = test_dataset.get_subset_eq_distr(10)
        log.info(f"Len of total test set = {original_test_set_len}")
        log.info(
            f"Len of reduced test set = {len(test_dataset)}, {100 * len(test_dataset) / original_test_set_len}% "
            f"of total test set")
        log.info(f"Len of extracted examples from test set = {len(self._excluded_from_test)}")

        test_dataloader = DataLoader(test_dataset, batch_size=self._batch_size)
        self._center_server = eval(params.center_server.classname)(model, test_dataloader, self._device,
                                                                   analyzer=self._analyzer.module_analyzer('server'),
                                                                   **params.center_server.args)
        self.save_models_path = '' if not params.save_models else \
                                os.path.join(self.savedir, f"models{self._output_suffix}")

    @abstractmethod
    def train_step(self):
        pass

    def _fit(self, iterations: int):
        self._analyzer.module_analyzer('server')('validation', server=self._center_server,
                                                 loss_fn=CrossEntropyLoss(), s_round=self._iteration)
        while self._next_iter(iterations):
            self.train_step()
            self._aggregate()
            self._cleanup_clients()
        log.info("Training completed")

    def _select_clients(self, clients_pool: List[Client], dropping_fn=lambda x: x) -> None:
        """
        Selects the C portion of clients that will participate in the current round

        Parameters
        ----------
        clients_pool : list
            the pool of clients to choose among
        dropping_fn
            function that determines how the selected clients will drop the current round
        """
        n_sample = max(int(self._fraction * len(clients_pool)), 1)
        sample_set = dropping_fn(np.random.choice(range(len(clients_pool)), n_sample, replace=False))
        self._selected_clients = [clients_pool[k] for k in iter(sample_set)]

    def _setup_clients(self) -> None:
        """
        Sends the data of the center server to the clients involved in the current round.
        """
        for client in self._selected_clients:
            client.receive_data(**self._center_server.send_data())
            client.setup()

    def _cleanup_clients(self):
        """
        Perform clients cleanup for the clients involved in the current round, i.e. to save computational resources
        """
        for client in self._selected_clients:
            client.cleanup()
        self._selected_clients.clear()

    def _aggregate(self) -> None:
        """
        Issues the aggregation to the center server, with the aggregation weights according to the aggregation policy
        """
        clients = self._selected_clients
        if self._aggregation_policy == "weighted":
            total_weight = np.sum([len(c) for c in clients])
            weights = [len(c) / total_weight for c in clients]
        else:  # uniform
            total_weight = len(clients)
            weights = [1. / total_weight for _ in range(len(clients))]
        self._center_server.aggregation(clients, weights, self._iteration)

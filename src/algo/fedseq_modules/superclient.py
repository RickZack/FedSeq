from typing import List, Optional
import numpy as np
import torch
from src.utils import shuffled_copy, select_random_subset, TrainAnalyzer
from src.algo.fed_clients import Client
import logging
from torch.nn.modules.loss import CrossEntropyLoss

log = logging.getLogger(__name__)


class FedSeqSuperClient(Client):
    """
    Implements a superclient client in FedSeq algorithm, as proposed in Zaccone et al., Speeding up Heterogeneous
    Federated Learning with Sequentially Trained Superclients

    The implementation follows the Composite design pattern
    """

    def __init__(self, client_id, clients: List[Client], clients_local_epoch: int, num_classes: int,
                 shuffle_sp_clients: bool, clients_dropout: float, save_memory: bool,
                 analyzer: Optional[TrainAnalyzer] = None, *args, **kwargs):
        super().__init__(client_id, None, num_classes, None, save_memory, analyzer)
        self.__clients = clients
        self.__clients_local_epoch = clients_local_epoch
        self.__shuffle_sp_clients = shuffle_sp_clients
        self.__clients_dropout = clients_dropout
        self.__additional_data = {}

    @property
    def clients(self):
        return self.__clients

    def __len__(self):
        return sum([len(client) for client in self.__clients])

    def send_data(self) -> dict:
        data = super().send_data()
        data.update(self.__additional_data)
        return data

    def receive_data(self, model, **kwargs):
        super().receive_data(model)
        self.__additional_data.update(kwargs)

    def cleanup(self):
        super().cleanup()
        self.__additional_data.clear()

    def client_update(self, optimizer: type, optimizer_args, local_epoch: int, loss_fn: torch.nn.Module, s_round: int):
        dropping = self.__clients_dropping()
        clients_ordering = self.__select_clients_ordering()
        for e in range(local_epoch):
            ordered_clients = clients_ordering(dropping(self.__clients))
            for client in ordered_clients:
                client.receive_data(**self.send_data())
                client.setup()
                client.client_update(optimizer, optimizer_args, self.__clients_local_epoch, loss_fn, s_round)
                self.receive_data(**client.send_data())
                client.cleanup()

    def __select_clients_ordering(self):
        """
        Selects the client ordering strategy to be used when deciding the ordering of clients in a superclient chain

        Returns
        -------
        a function that takes as input a list of clients and returns the same list with ordered clients
        """
        if self.__shuffle_sp_clients:
            return shuffled_copy
        else:
            return lambda x: x

    def num_ex_per_class(self) -> np.array:
        ex_per_class = np.zeros(self.num_classes)
        for client in self.__clients:
            ex_per_class += client.num_ex_per_class()
        return ex_per_class

    def __clients_dropping(self):
        """
        Selects the client dropping strategy to be used to model the struggles in the superclient chain

        Returns
        -------
        a function that takes as input a list of clients and returns a list containing a subset of those clients
        """
        assert 0 <= self.__clients_dropout < 1, f"Dropout rate d must be 0 <= d < 1, got {self.__clients_dropout}"
        if self.__clients_dropout > 0:
            return lambda x: select_random_subset(x, self.__clients_dropout)
        return lambda x: x

import torch
from src.algo import Algo
from src.algo.fed_clients.base_client import Client


class FedAvgClient(Client):
    """ Implements a client in FedAvg algorithm, as proposed in McMahan et al., Communication-efficient learning of
    deep networks from decentralized data """

    def client_update(self, optimizer: type, optimizer_args, local_epoch: int, loss_fn: torch.nn.Module, s_round: int):
        self.standard_training(optimizer, optimizer_args, local_epoch, loss_fn)

import torch
from typing import List, Optional

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import Client
from src.models import Model
from src.utils import TrainAnalyzer


class FedDynCenterServer(FedAvgCenterServer):
    """ Implements the center server in FedDyn algorithm, as proposed in Acar et al, Federated Learning
    Based on Dynamic Regularization"""

    def __init__(self, model: Model, dataloader: DataLoader, device: str, alpha: float, num_clients: int,
                 analyzer: Optional[TrainAnalyzer] = None):
        super().__init__(model, dataloader, device, analyzer)
        self.h = [torch.zeros_like(p.data, device=self.device) for p in self._model.parameters()]
        self.alpha = alpha
        self.num_clients = num_clients

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        # compute the sum of all the model parameters of the clients involved in training
        sum_theta = [torch.zeros_like(p.data) for p in self._model.parameters()]
        for c in clients:
            for s, c_theta in zip(sum_theta, c.model.parameters()):
                s.add_(c_theta)
        # compute the deltas w.r.t. the old server model
        delta_theta = [torch.clone(p) for p in sum_theta]
        num_participating_clients = len(clients)
        for d, p in zip(delta_theta, self._model.parameters()):
            d.add_(p.data, alpha=-num_participating_clients)
        # update the h parameter
        for h, theta in zip(self.h, delta_theta):
            h.data.add_(theta, alpha=-(self.alpha / self.num_clients))
        # update the server model
        for model_param, h, sum_theta_p in zip(self._model.parameters(), self.h, sum_theta):
            model_param.data = 1. / len(clients) * sum_theta_p.data - 1. / self.alpha * h.data

        self._analyzer('validation', server=self, loss_fn=CrossEntropyLoss(), s_round=s_round)

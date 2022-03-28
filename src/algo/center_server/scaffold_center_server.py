import copy
from typing import List, Optional
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.algo.center_server.center_server import CenterServer
from src.algo.center_server.fedavg_center_server import FedAvgCenterServer
from src.algo.fed_clients import SCAFFOLDClient
from src.models import Model
from src.utils import TrainAnalyzer


class SCAFFOLDCenterServer(FedAvgCenterServer):
    """ Implements the center server in SCAFFOLD algorithm, as proposed in Karimireddy et al., SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, num_clients: int, save_memory: bool = True,
                 analyzer: Optional[TrainAnalyzer] = None):
        super().__init__(model, dataloader, device, analyzer)
        self.server_controls = [torch.zeros_like(p.data, device=self.device) for p in self._model.parameters()
                                if p.requires_grad]
        self.num_clients = num_clients
        self.save_memory = save_memory  # if true assume clients do not modify server controls

    @staticmethod
    def from_center_server(server: CenterServer, num_clients: int):
        return SCAFFOLDCenterServer(server._model, server._dataloader, server._device, num_clients)

    def aggregation(self, clients: List[SCAFFOLDClient], aggregation_weights: List[float], s_round: int):
        clients_data = [c.send_data() for c in clients]
        FedAvgCenterServer.weighted_aggregation([data["model"] for data in clients_data], aggregation_weights,
                                                self._model)
        for data in clients_data:
            delta_c = data["delta_controls"]
            for sc, d in zip(self.server_controls, delta_c):
                sc.add_(d, alpha=1. / self.num_clients)
        self._analyzer('validation', server=self, loss_fn=CrossEntropyLoss(), s_round=s_round)

    def send_data(self) -> dict:
        data = super().send_data()
        data.update({"controls": copy.deepcopy(self.server_controls) if not self.save_memory else self.server_controls})
        return data

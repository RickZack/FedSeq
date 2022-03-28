import copy
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.algo.fed_clients.base_client import Client
from src.utils import TrainAnalyzer, move_tensor_list


class FedDynClient(Client):
    """ Implements a client in FedDyn algorithm, as proposed in Acar et al, Federated Learning
    Based on Dynamic Regularization"""

    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes: int, device: str, alpha: float,
                 save_memory: bool, analyzer: Optional[TrainAnalyzer] = None):
        super().__init__(client_id, dataloader, num_classes, device, save_memory, analyzer)
        self.__alpha = alpha
        self.__prev_grads = None

    def setup(self):
        super().setup()
        if self.__prev_grads is None:
            self.__prev_grads = [torch.zeros_like(p.data, device=self.device) for p in
                                 self.model.parameters() if p.requires_grad]
        elif self.save_memory:
            move_tensor_list(self.__prev_grads, self.device)

    def cleanup(self):
        if self.save_memory:
            move_tensor_list(self.__prev_grads, "cpu")

    def client_update(self, optimizer: type, optimizer_args, local_epoch: int, loss_fn: torch.nn.Module, s_round: int):
        self.model.to(self.device)
        loss_fn.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        prev_model = copy.deepcopy(self.model)

        for _ in range(local_epoch):
            self.model.train()
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)

                linear_p = 0
                for param, grad in zip(self.model.parameters(), self.__prev_grads):
                    linear_p += torch.sum(param.data * grad.data)

                quadratic_p = 0
                for cur_param, prev_param in zip(self.model.parameters(), prev_model.parameters()):
                    quadratic_p += F.mse_loss(cur_param, prev_param, reduction='sum')

                loss -= linear_p
                loss += self.__alpha / 2. * quadratic_p
                loss.backward()
                optimizer.step()

                for prev_grads, new_params, prev_params in zip(self.__prev_grads, self.model.parameters(),
                                                               prev_model.parameters()):
                    prev_grads.add_(new_params.data - prev_params.data, alpha=-self.__alpha)

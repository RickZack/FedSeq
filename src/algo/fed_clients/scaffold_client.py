import copy
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from src.algo.fed_clients.base_client import Client
from src.utils import TrainAnalyzer, move_tensor_list


class SCAFFOLDClient(Client):
    """ Implements a client in SCAFFOLD algorithm, as proposed in Karimireddy et al., SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning """

    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes: int, device: str,
                 save_memory: bool, analyzer: Optional[TrainAnalyzer] = None):
        super().__init__(client_id, dataloader, num_classes, device, save_memory, analyzer)
        self.old_controls = None
        self.controls = None
        self.server_controls = None

    def setup(self):
        super().setup()
        # init controls only the first time the client is selected
        if self.controls is None:
            self.controls = [torch.zeros_like(p.data, device=self.device) for p in
                             self.model.parameters() if p.requires_grad]
        elif self.save_memory:
            move_tensor_list(self.controls, self.device)
            move_tensor_list(self.server_controls, self.device)

    def cleanup(self):
        if self.save_memory:
            move_tensor_list(self.controls, "cpu")
            move_tensor_list(self.server_controls, "cpu")

    def send_data(self) -> dict:
        data = super().send_data()
        data.update({"delta_controls": self._delta_controls()})
        return data

    def receive_data(self, model, controls):
        """
        Receives the data to use in the next client_update

        Parameters
        ----------
        model
            the model to train
        controls
            the updates server controls
        """
        super().receive_data(model)
        self.server_controls = controls

    def client_update(self, optimizer: type, optimizer_args, local_epoch: int, loss_fn: torch.nn.Module, s_round: int):
        # save model sent by server for computing delta_model
        server_model = copy.deepcopy(self.model)
        self.model.train()
        op = optimizer(self.model.parameters(), **optimizer_args)
        for _ in range(local_epoch):
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                op.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)
                loss.backward()
                op.step(self.server_controls, self.controls)

        # controls become old controls
        self.old_controls = [torch.clone(c) for c in self.controls]

        # get new controls option 1 of scaffold algorithm
        batches = 0
        for _ in range(local_epoch):
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = server_model(img)
                loss = loss_fn(logits, target)
                loss.backward()
                batches += 1
        server_model.to("cpu" if self.save_memory else self.device)
        for cc, p in zip(self.controls, server_model.parameters()):
            cc.data = p.grad.data / batches

    def _delta_controls(self) -> List[torch.Tensor]:
        """
        Calculates the difference between the new and old client controls.

        Returns
        -------
        a list containing the tensors of new_client_controls - old_client_controls
        """
        delta = [torch.zeros_like(c, device=self.device) for c in self.controls]
        for d, new, old in zip(delta, self.controls, self.old_controls):
            d.data = new.data.to(self.device) - old.data.to(self.device)
        return delta

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.algo import Algo
from src.utils import MeasureMeter, TrainAnalyzer
from src.utils import ChainedAnalyzer


class Client(ABC):
    """
    Base (abstract) class for a client in any FL algorithm

    The algorithm specific client class is loosely coupled with its corresponding center server class: the return data
    of (CenterServer) send_data method must correspond to (Client) receive_data.
    """
    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes: int, device: Optional[str],
                 save_memory: bool, analyzer: Optional[TrainAnalyzer] = None):
        self.__client_id = client_id
        self.__dataloader = dataloader
        self.__device = device
        self.__model = None
        self.__num_classes = num_classes
        self._analyzer = analyzer or ChainedAnalyzer.empty()
        self._measure_meter = MeasureMeter(num_classes)
        self.save_memory = save_memory

    @property
    def client_id(self):
        return self.__client_id

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device: str):
        self.__model.to(device)
        self.__device = device

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        assert isinstance(model, torch.nn.Module), "Client's model in not an instance of torch.nn.Module"
        del self.__model
        self.__model = model

    @property
    def dataloader(self) -> DataLoader:
        return self.__dataloader

    @dataloader.setter
    def dataloader(self, dataloader: DataLoader):
        assert isinstance(dataloader, DataLoader), "Client's dataloader is not an instance of torch DataLoader"
        self.__dataloader = dataloader

    @property
    def num_classes(self):
        return self.__num_classes

    def send_data(self) -> dict:
        """
        Sends the updated data

        Returns
        -------
        a dictionary containing the data that the client must send to its server
        """
        # This is the right place where to put code to keep track of the amount of exchanged data,
        # for examples using the proper analyzer listening to the proper event, like
        # self._analyzer('send_data', data=data, from=self)
        return {"model": self.model}

    def receive_data(self, model, **kwargs):
        """
        Receives the data to use in the next client_update

        Parameters
        ----------
        model
            the model to train
        kwargs
            other arguments specific of the client subclass
        """
        if not self.save_memory:
            self.model.load_state_dict(model.state_dict())
        else:
            self.model = model

    def setup(self):
        """
        Performs some setup before the start of client_update
        """
        self._measure_meter.reset()

    def cleanup(self):
        """
        Performs some cleanup, especially for computational resources, after the end of the train_step, i.e. after the
        server has aggregated the clients' data
        """
        if self.save_memory:
            self.__model = None  # for saving GPU memory

    @abstractmethod
    def client_update(self, optimizer: type, optimizer_args, local_epoch: int, loss_fn: torch.nn.Module, s_round: int):
        """
        Performs local_epoch training iterations on the current client's dataset, using the current client's model

        Parameters
        ----------
        optimizer
            the optimizer class to be used for the training
        optimizer_args
            the args to pass to the optimizer
        local_epoch
            the number of epochs to perform on the local dataset
        loss_fn
            the loss function to use
        s_round
            the current round of the server
        """
        pass

    def client_evaluate(self, loss_fn, test_data: DataLoader) -> Tuple[float, MeasureMeter]:
        """
        Perform an evaluation step of the current client's model using a given test set

        Parameters
        ----------
        loss_fn
            the loss function to use
        test_data
            the dataset to test the model with

        Returns
        -------
        a tuple containing the loss value and a reference to the client's MeasureMeter object
        """
        self.model.to(self.__device)
        loss_fn.to(self.__device)
        self._measure_meter.reset()
        loss = Algo.test(self.model, self._measure_meter, self.__device, loss_fn, test_data)
        return loss, self._measure_meter

    def standard_training(self, optimizer: type, optimizer_args, local_epoch: int, loss_fn: torch.nn.Module):
        self.model.to(self.device)
        loss_fn.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        for _ in range(local_epoch):
            Algo.train(self.model, self.device, optimizer, loss_fn, self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def num_ex_per_class(self) -> np.array:
        """
        Returns the data distribution of the client

        Returns
        -------
        a numpy array containing the number of examples for each class in the client's local dataset
        """
        num_ex_per_class = np.zeros(self.num_classes)
        for _, batch in self.dataloader:
            for target in batch.numpy():
                num_ex_per_class[target] += 1
        return num_ex_per_class

import copy
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from src.algo.fed_clients.base_client import Client
from src.models import Model
from src.utils import MeasureMeter, TrainAnalyzer, ChainedAnalyzer


class CenterServer(ABC):
    """ Base (abstract) class for a center server in any FL algorithm """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, analyzer: Optional[TrainAnalyzer] = None):
        self._model = model.to(device)
        self._dataloader = dataloader
        self._device = device
        self._analyzer = analyzer or ChainedAnalyzer.empty()
        self._measure_meter = MeasureMeter(model.num_classes)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: str):
        self._model.to(device)
        self._device = device

    @property
    def model(self):
        return self._model

    @abstractmethod
    def aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        """
        Aggregate the client's data according to their weights

        Parameters
        ----------
        clients
            the clients whose data have to be aggregated
        aggregation_weights
            the weights corresponding to clients
        s_round
            the current round of the server
        """
        pass

    def send_data(self) -> dict:
        """
        Sends out the current data of the central server. To be used to send current round data to FL clients.
        For any specific FL algorithm, (CenterServer) send_data must output the data needed by the specific
        client receive_data

        Returns
        -------
        a dictionary containing the current data of the center server
        """
        # This is the right place where to put code to keep track of the amount of exchanged data,
        # for examples using the proper analyzer listening to the proper event, like
        # self._analyzer('send_data', data=data, from=self)
        return {"model": copy.deepcopy(self._model)}

    @abstractmethod
    def validation(self, loss_fn) -> Tuple[float, MeasureMeter]:
        """
        Validates the center server model

        Parameters
        ----------
        loss_fn the loss function to be used

        Returns
        -------
        a tuple containing the value of the loss function and a reference to the center server MeasureMeter object
        """
        pass

    def state_dict(self) -> dict:
        """
        Saves the state of the center server in order to restore it when reloading a checkpoint

        Returns
        -------
        a dict with key-value pairs corresponding to the parameter name and its value

        """
        return {"model": self._model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """
        Loads a previously saved state for the center server
        Parameters
        ----------
        state
            a dictionary containing key-value pairs corresponding to the parameter name and its value
        """
        params = ["model"]
        assert all([p in state for p in params]), "Missing params for center server"
        self._model.load_state_dict(state["model"], strict=True)

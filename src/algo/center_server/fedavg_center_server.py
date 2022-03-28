from collections import OrderedDict
from typing import Iterable, List, Tuple

from torch.nn import CrossEntropyLoss

from src.algo import Algo
from src.algo.center_server.center_server import CenterServer
from src.algo.fed_clients import Client
from src.models import Model
from src.utils import MeasureMeter


class FedAvgCenterServer(CenterServer):
    """ Implements the center server in FedAvg algorithm, as proposed in McMahan et al., Communication-efficient
    learning of deep networks from """

    @staticmethod
    def weighted_aggregation(models: Iterable[Model], aggregation_weights: List[float], dest: Model):
        update_state = OrderedDict()

        for k, model in enumerate(models):
            local_state = model.state_dict()
            for key in model.state_dict().keys():
                if k == 0:
                    update_state[
                        key] = local_state[key] * aggregation_weights[k]
                else:
                    update_state[
                        key] += local_state[key] * aggregation_weights[k]
        dest.load_state_dict(update_state)

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        FedAvgCenterServer.weighted_aggregation([c.send_data()["model"] for c in clients], aggregation_weights,
                                                self._model)
        self._analyzer('validation', server=self, loss_fn=CrossEntropyLoss(), s_round=s_round)

    def validation(self, loss_fn) -> Tuple[float, MeasureMeter]:
        self._model.to(self._device)
        loss_fn.to(self._device)
        self._measure_meter.reset()
        loss = Algo.test(self._model, self._measure_meter, self._device, loss_fn, self._dataloader)
        return loss, self._measure_meter

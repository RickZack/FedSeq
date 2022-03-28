import logging
from src.algo.fedbase import FedBase
from src.utils import select_random_subset

log = logging.getLogger(__name__)


class FedAvg(FedBase):
    """
    FedAvg algorithm as proposed in McMahan et al., Communication-efficient learning of deep networks from
    decentralized data.
    """

    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None):
        assert 0 <= params.clients_dropout < 1, f"Dropout rate d must be 0 <= d < 1, got {params.clients_dropout}"
        super(FedAvg, self).__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self.__clients_dropout = params.clients_dropout
        self.__dropping = (lambda x: select_random_subset(x, self.__clients_dropout)) if self.__clients_dropout > 0 \
            else (lambda x: x)

    def train_step(self):
        self._select_clients(self._clients, self.__dropping)
        self._setup_clients()

        for c in self._selected_clients:
            c.client_update(self._optimizer, self._optimizer_args,
                            self._local_epoch, self._loss_fn, self._iteration)

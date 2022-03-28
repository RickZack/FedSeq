import numpy as np
from src.algo import FedSeq
import logging

log = logging.getLogger(__name__)


class FedSeqInter(FedSeq):

    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None):
        super(FedSeqInter, self).__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        # init model bank
        self._num_clients_train_step = max(int(self._fraction * self._num_superclients), 1)
        self._data_bank = [self._center_server.send_data()] * self._num_clients_train_step
        self._models_num_examples = np.zeros(self._num_clients_train_step)
        aggregation_periods_choices = {"num_superclients": self._num_superclients,
                                       "fraction_superclients": self._num_clients_train_step,
                                       "never": int(1e6)}
        assert params.aggregation_period in aggregation_periods_choices, "Unknown aggregation period"
        self.aggregation_period = aggregation_periods_choices[params.aggregation_period]

    def train_step(self):
        # broadcast aggregated model next round if enough clients
        if self._iteration % self.aggregation_period == 0:
            log.info("Broadcast aggregated model")
            for k in range(self._num_clients_train_step):
                self._data_bank[k] = self._center_server.send_data()
            self._models_num_examples.fill(0)

        self._select_clients(self._superclients)
        self._setup_clients()

        for k in range(self._num_clients_train_step):
            # training
            self._selected_clients[k].client_update(self._optimizer, self._optimizer_args, self._training.sequential_rounds,
                                                    self._loss_fn, self._iteration)
            self._data_bank[k] = self._selected_clients[k].send_data()
            self._models_num_examples[k] += len(self._selected_clients[k])

    def _aggregate(self):
        total_weight = np.sum(self._models_num_examples)
        weights = [w / total_weight for w in self._models_num_examples]
        self._center_server.aggregation(self._selected_clients, weights, self._iteration)

    def _setup_clients(self) -> None:
        for client, data in zip(self._selected_clients, self._data_bank):
            client.receive_data(**data)
            client.setup()

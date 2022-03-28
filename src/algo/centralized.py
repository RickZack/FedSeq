import logging
import torch.optim.lr_scheduler
from src.algo.center_server import FedAvgCenterServer
from src.models import Model
from torch.utils.data import DataLoader
from src.utils import get_dataset
from src.algo import Algo

log = logging.getLogger(__name__)


class Centralized(Algo):
    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None):
        assert params.loss.type == "crossentropy", "Loss function for centralized algorithm must be crossentropy"
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self._batch_size = params._batch_size

        dataset_getter, dataset_class, dataset_num_classes = get_dataset(dataset)
        train_img, train_label, test_img, test_label, _ = dataset_getter()
        training_set = dataset_class(train_img, train_label, dataset_num_classes)
        test_set = dataset_class(test_img, test_label, dataset_num_classes, train=False)
        test_loader = DataLoader(test_set, num_workers=6, batch_size=self._batch_size, shuffle=False)

        model = Model(model_info, dataset_num_classes)

        self._train_loader = DataLoader(training_set, num_workers=6, batch_size=self._batch_size, shuffle=True)
        self._center_server = FedAvgCenterServer(model, test_loader, device, self._analyzer.module_analyzer('server'))
        self._scheduler = None
        self._optim = None

    def _fit(self, iterations):
        model = self._center_server.model
        model.to(self._device)
        self._loss_fn.to(self._device)
        model.train()
        self._optim = self._optimizer(model.parameters(), **self._optimizer_args)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optim, iterations)

        self._analyzer.module_analyzer('server')('validation', server=self._center_server, loss_fn=self._loss_fn,
                                                 s_round=self._iteration)
        while self._next_iter(iterations):
            self.train_step()
            self._analyzer.module_analyzer('server')('validation', server=self._center_server, loss_fn=self._loss_fn,
                                                     s_round=self._iteration)
        log.info("Training completed")

    def train_step(self):
        model = self._center_server.model
        Algo.train(model, self._device, self._optim, self._loss_fn, self._train_loader)
        self._scheduler.step()

import os
import pickle
import signal
from abc import ABC, abstractmethod
import torch.nn
from src.models import Model
from src.optim import *
from torch.utils.data import DataLoader

from src.losses import *
from src.utils import exit_on_signal, MeasureMeter, save_pickle
import logging

from src.utils import AnalyzerController
from src.utils.utils import function_call_log

log = logging.getLogger(__name__)


class Algo(ABC):
    """ Base (abstract) class for any algorithm """

    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None):
        Model.verify_info(model_info)
        loss_info, optim_info = params.loss, params.optim
        self._loss_fn: torch.nn.Module = eval(loss_info.classname)(**loss_info.params)
        self._optimizer: type = eval(optim_info.classname)
        self._optimizer_args: dict = optim_info.args

        self._analyzer = AnalyzerController(params.analyze_container, writer)
        self._device = device
        self._dataset = dataset
        self._output_suffix = output_suffix
        self.savedir = savedir
        self._iteration: int = 0
        self.save_checkpoint_period = params.save_checkpoint_period
        self._completed = False
        self._center_server = None

        exit_on_signal(signal.SIGTERM)

    @property
    def result(self):
        return self._analyzer.result

    def _reset_result(self):
        self._analyzer.reset()
        self._iteration = 0

    @abstractmethod
    def train_step(self) -> None:
        pass

    def fit(self, iterations: int) -> None:
        """
        Trains the algorithm.

        Resets the results if a previous fit completed successfully, evaluates the starting model and
        wraps the algorithm-specific fit procedure to ensure results saving after graceful or erroneous
        termination. Not intended to be overridden, see _fit

        Parameters:
        ----------
        iterations
            the number of iterations to train the algorithm for

        """
        if self._completed:
            self._reset_result()
        assert iterations > self._iteration, "Num of rounds to perform must be greater of equal to the current round"
        try:
            self._fit(iterations)
        except SystemExit as e:
            log.warning(f"Training stopped at iteration {self._iteration}: {e}")
        finally:
            self.save_checkpoint()
            if self._completed:
                self.save_result()

    @abstractmethod
    def _fit(self, iterations: int) -> None:
        """
        Defines the main training loop of any algorithm. Must be overridden to describe the algorithm-specific procedure

        Parameters
        ----------
        iterations
            the number of iterations to train the algorithm for
        """
        pass

    def _next_iter(self, target_iter: int) -> bool:
        """
        Sets the stage for the next iteration of the algorithm. Use always this method when starting a new iteration.

        Checks if the algorithm has been fit for the target number of iterations and saves the checkpoint
        if the checkpoint period (number of iterations) has elapsed.

        Parameters
        ----------
        target_iter
            the target number of iterations to run fit for

        Returns
        -------
        bool
            True if another iteration has to be performed, False otherwise
        """
        if self._iteration >= target_iter:
            self._completed = True
            return False
        if self._iteration % self.save_checkpoint_period == 0 and self._iteration > 0:
            self.save_checkpoint()
        self._iteration += 1
        return True

    @function_call_log(log=log)
    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.savedir, f"checkpoint{self._output_suffix}.pkl")
        save_pickle({'analyzer': self._analyzer.state_dict(), **self.result, "iteration": self._iteration - 1,
                     "center_server": self._center_server.state_dict()},
                    checkpoint_path)

    def load_from_checkpoint(self):
        needed_keys = ['center_server', 'iteration']
        checkpoint_path = os.path.join(self.savedir, f"checkpoint{self._output_suffix}.pkl")
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                assert all(key in checkpoint_data for key in needed_keys), "Missing data in checkpoint"
                log.info(f'Reloading checkpoint from round {checkpoint_data["iteration"]}')
                self._center_server.load_state_dict(checkpoint_data["center_server"])
                self._iteration = checkpoint_data["iteration"]
                self._analyzer.load_state_dict(checkpoint_data.get('analyzer', {}))
        except BaseException as err:
            log.warning(f"Unable to load from checkpoint, starting from scratch: {err}")

    @staticmethod
    def train(model: nn.Module, device: str, optimizer: nn.Module, loss_fn: nn.Module, data: DataLoader) -> None:
        model.train()
        for img, target in data:
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, target)

            loss.backward()
            optimizer.step()

    @staticmethod
    def test(model: nn.Module, meter: MeasureMeter, device: str, loss_fn, data: DataLoader) -> float:
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for img, target in data:
                img = img.to(device)
                target = target.to(device)
                logits = model(img)
                test_loss += loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                meter.update(pred, target)
        test_loss = test_loss / len(data)
        return test_loss

    def save_result(self) -> None:
        """
        Saves the results of the training process
        """
        results_path = os.path.join(self.savedir, f"result{self._output_suffix}.pkl")
        log.info(f"Saving results in {results_path}")
        save_pickle(self.result, results_path)

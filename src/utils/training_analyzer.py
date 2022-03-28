import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import torch.nn
from torch.utils.data import DataLoader
import tensorboardX
import logging

log = logging.getLogger(__name__)


class TrainAnalyzer(ABC):
    """ Base (abstract) for any analyzer """

    def __init__(self, call_args: Dict[str, type], event: str, verbose: bool = False,
                 writer: Optional[tensorboardX.writer.SummaryWriter] = None):
        """

        Parameters
        ----------
        call_args mandatory arguments for a call to analyze()
        event describes what request the analyzer responds to
        verbose outputs additional information about the analyzer
        writer tensorboardX SummaryWriter instance to log results to
        """
        self._event = event
        self.__args = call_args
        self._verbose = verbose
        self._writer = writer
        self._result = {}

    @property
    def result(self) -> dict:
        return self._result

    def reset(self) -> None:
        self._result.clear()

    def __call__(self, event: str, *args, **kwargs) -> None:
        assert len(args) == 0, "Only named parameters accepted"
        self.__verify_args(kwargs)
        if self.listen_to_event(event):
            return self._analyze(event, **kwargs)

    def __verify_args(self, kwargs):
        assert all([p in kwargs for p in self.__args]), \
            f"{self.__class__.__name__}: missing parameters: given {kwargs.keys()}, required {self.__args.keys()}"
        for arg_name, arg_type in self.__args.items():
            assert isinstance(kwargs[arg_name], arg_type), f"Parameter {arg_name} expected to be of type {arg_type}, " \
                                                           f"instead is of type {kwargs[arg_name]}"

    @abstractmethod
    def _analyze(self, event, **kwargs) -> None:
        """
        Perform analysis on the given keyword arguments in response to an event

        Parameters
        ----------
        event describe the request made to the analyzer
        kwargs arguments to use to perform analysis
        """
        pass

    def state_dict(self) -> dict:
        return {'classname': self.__class__.__name__, 'result': self._result}

    def load_state_dict(self, state: dict) -> None:
        assert 'classname' in state and 'result' in state, "Incomplete state for analyzer"
        if state['classname'] != self.__class__.__name__:
            log.warning(f'Reloading results from different analyzer class, expected {self.__class__.__name__}, '
                        f'given {state["classname"]}')
        self._result = copy.deepcopy(state['result'])

    def listen_to_event(self, event) -> bool:
        return event == self._event


class EmptyAnalyzer(TrainAnalyzer):
    """ Dummy analyzer that does nothing"""

    def __init__(self, **kwargs):
        super().__init__({}, "")

    def _analyze(self, event, **kwargs):
        if self._verbose:
            log.info("Empty analyzer called")


class AnalyzerContainer(ABC):
    """
    Base (abstract) class for a container of analyzers
    """
    def __init__(self, analyzers):
        self._analyzers: Dict[str, TrainAnalyzer] = analyzers
        self._old_state_dict = {}

    def add_analyzer(self, analyzer: TrainAnalyzer):
        self._analyzers.update({analyzer.__class__.__name__: analyzer})

    def contains_analyzer(self, classname) -> bool:
        return classname in self._analyzers

    def state_dict(self) -> dict:
        state = copy.deepcopy(self._old_state_dict)
        for name, analyzer in self._analyzers.items():
            state[name] = analyzer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if not all([name in state for name in self._analyzers]):
            log.warning("Missing states for some analyzers")
        if any([name not in self._analyzers for name in state]):
            log.warning("Found analyzers in previous run not instantiated for this run")
        for name, analyzer_state in state.items():
            if name not in self._analyzers:
                self._old_state_dict[name] = analyzer_state
            else:
                self._analyzers[name].load_state_dict(analyzer_state)

    def reset(self) -> None:
        for analyzer in self._analyzers.values():
            analyzer.reset()


class ChainedAnalyzer(AnalyzerContainer, TrainAnalyzer):
    """
    Container of analyzers that applies each analyzer sequentially
    """
    def __init__(self, analyzers: List[dict], verbose: bool = False,
                 writer: Optional[tensorboardX.writer.SummaryWriter] = None):
        # check format
        assert all(['classname' in a and 'args' in a for a in analyzers]), "Error in format for analyzers"
        analyzers: Dict[str, TrainAnalyzer] = {a['classname']: eval(a['classname'])(**a['args'], writer=writer,
                                                                                    verbose=verbose)
                                               for a in analyzers}
        AnalyzerContainer.__init__(self, analyzers)
        TrainAnalyzer.__init__(self, {}, "", verbose, writer)

    @staticmethod
    def empty():
        return ChainedAnalyzer([{'classname': 'EmptyAnalyzer', 'args': {}}])

    def _analyze(self, event, **kwargs) -> None:
        for name, analyzer in self._analyzers.items():
            if analyzer.listen_to_event(event):
                previous_result = self._result.get(name, [])
                analyzer(event, **kwargs)
                new_result = analyzer.result
                previous_result.append(new_result)
                self._result.update({name: previous_result})

    def reset(self) -> None:
        AnalyzerContainer.reset(self)
        TrainAnalyzer.reset(self)

    def listen_to_event(self, event) -> bool:
        listeners = [a.listen_to_event(event) for a in self._analyzers.values()]
        return any(listeners)


class AnalyzerController(AnalyzerContainer):
    """
    Simple container that associates an analyzer container to each module
    """
    def __init__(self, analyzers: dict, writer: Optional[tensorboardX.writer.SummaryWriter] = None):
        verbose = analyzers['verbose']
        modules = {component_name: ChainedAnalyzer(component_analyzer, verbose, writer)
                   for component_name, component_analyzer in analyzers['modules'].items()}
        super().__init__(modules)
        if verbose:
            log.info(f'Components of AnalyzerContainer:\n{analyzers["modules"]}')

    @property
    def result(self) -> dict:
        result = {}
        for module, chained_analyzer in self._analyzers.items():
            previous_result = result.get(module, {})
            new_result = chained_analyzer.result
            previous_result.update(new_result)
            result.update({module: previous_result})
        return result

    def module_analyzer(self, module: str) -> ChainedAnalyzer:
        return self._analyzers.get(module, EmptyAnalyzer())

    def add_analyzer(self, analyzer: TrainAnalyzer):
        assert isinstance(analyzer, ChainedAnalyzer), "Modules must have a ChainedAnalyzer, not plain analyzer"
        super(AnalyzerController, self).add_analyzer(analyzer)


class ServerAnalyzer(TrainAnalyzer):
    """
    Analyzer for a center server
    """
    def __init__(self, print_period: int, *args, **kwargs):
        from src.algo.center_server import CenterServer
        super().__init__({'server': CenterServer, 'loss_fn': torch.nn.Module, 's_round': int}, *args, **kwargs)
        self._print_period = print_period

    def _analyze(self, event, server, loss_fn, s_round, other_scalars: Optional[dict] = None, **kwargs) -> None:
        loss, mt = server.validation(loss_fn)
        self._log(server.__class__.__name__, s_round, loss, mt, other_scalars or {})
        data = {'loss': loss, 'accuracy': mt.accuracy_overall, 'accuracy_class': mt.accuracy_per_class}
        data.update(other_scalars or {})
        self._result.update({s_round: data})

    def _log(self, server_classname, s_round, loss, mt, other_scalars: dict):
        if s_round % self._print_period == 0:
            log.info(
                f"[Round: {s_round: 05}] Test set: Average loss: {loss:.4f}, Accuracy: {mt.accuracy_overall:.2f}%"
            )
        if self._writer is not None:
            self._writer.add_scalar(f'{server_classname}/val/loss', loss, s_round)
            self._writer.add_scalar(f'{server_classname}/val/accuracy', mt.accuracy_overall, s_round)
            for tag, scalar in other_scalars.items():
                self._writer.add_scalar(f'{server_classname}/{tag}', scalar, s_round)

import os
from typing import Callable, Dict, Iterable, List, Optional
import itertools as it
from src.experiments import *
from copy import deepcopy


class PathGen:
    def __init__(self, start=1):
        self.start = start

    def nested(self, basedir: str, model: FedModel):
        return '"' + os.path.join(basedir, *model.get_keyvalue_list()) + '"'

    def flatten(self, basedir, *args):
        path = '"' + os.path.join(basedir, f"Model{self.start}") + '"'
        self.start += 1
        return path


class FedExperiment:
    default_fit_iterations: int = 1

    def __init__(self, name, description, models: List[FedModel], runner_options: Optional[Dict[str, str]] = None,
                 fit_iterations=None):
        self.name = name
        self.description = description
        self.models = models
        self.runnerOptions = runner_options or {}
        self.fit_iterations = fit_iterations or FedExperiment.default_fit_iterations

    @staticmethod
    def from_params(name: str, description: str, *params_args: Iterable[MultiParam or Param],
                    path_gen: Optional[Callable[[str, FedModel], List[str]]] = None,
                    runner_options: Optional[Dict[str, str]] = None, fit_iterations=None):
        return FedExperiment.from_param_groups(name, description, params_args, path_gen=path_gen,
                                               runner_options=runner_options, fit_iterations=fit_iterations)

    @staticmethod
    def from_param_groups(name: str, description: str, *param_groups: Iterable[Iterable[MultiParam or Param]],
                          shared_param_group: Optional[List[MultiParam or Param]] = None,
                          path_gen: Optional[Callable[[str, FedModel], List[str]]] = None,
                          runner_options: Optional[Dict[str, str]] = None, fit_iterations=None):
        shared_param_group = shared_param_group or []
        path_gen = path_gen or PathGen().flatten
        name = name.replace(' ', '_')  # just to avoid problems with characters incompatible to paths
        input_models = it.chain.from_iterable([make_models(*group) for group in param_groups])
        shared_models = make_models(*shared_param_group) or [FedModel({})]
        final_models = []
        for comb in it.product(input_models, shared_models):
            current_model, shared = deepcopy(comb)
            current_model.merge(shared)
            dest_dir = path_gen(os.path.join(os.getcwd(), "output", name), current_model)
            current_model.update("savedir", Param("savedir", dest_dir))
            current_model.update("hydra.run.dir", Param("hydra.run.dir", dest_dir))
            final_models.append(current_model)
        return FedExperiment(name, description, final_models, runner_options, fit_iterations)

    def __iter__(self):
        return self.models.__iter__()

    def info(self, details: bool = True) -> str:
        tot_models = len(self.models)
        header = f"FedExperiment: {self.name}\nDescription: {self.description}\n#Models: {tot_models}"
        models_details = ""
        if details:
            models_details = '\n'.join([f"Model {modelNo}/{tot_models}\nParams:\n{model.params}"
                                        for modelNo, model in enumerate(self.models, 1)])
        return ' '.join([header, models_details])

    def __str__(self):
        return self.info(False)

    def __len__(self):
        return len(self.models)

    def run(self, python_file: str, r: Runner = LocalRunner(), fit_iterations: Optional[int] = None) -> None:
        tot_models = len(self.models)
        for modelNo, model in enumerate(self.models, 1):
            run_name = f"{self.name}__Model_{modelNo}_of_{tot_models}"
            r.run(python_file, model, run_name, fit_iterations or self.fit_iterations, self.runnerOptions)
            model.save_to_file(os.path.join(os.getcwd(), "output", f"{self.name}", f"Model{modelNo}", "params.pkl"))

    def __iter__(self):
        return fed_experiment_iterator(self)


def fed_experiment_iterator(experiment: FedExperiment):
    for model in experiment.models:
        yield model


def make_models(*params_args: Iterable[MultiParam or Param]) -> List[FedModel]:
    models = []
    for comb in it.product(*params_args):
        current_model = FedModel({})
        [current_model.update(param.key, param) for param in comb]
        models.append(current_model)
    return models

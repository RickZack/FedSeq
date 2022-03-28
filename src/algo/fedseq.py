import logging
from typing import List, Dict
from src.optim import *
from src.algo.fedseq_modules import *
from src.models import Model
import numpy as np
from torch.nn import CrossEntropyLoss
from src.algo import FedBase
import itertools as it

log = logging.getLogger(__name__)


class FedSeq(FedBase):
    """
    FedSeq algorithm as proposed in Zaccone et al., Speeding up Heterogeneous Federated Learning
    with Sequentially Trained Superclients
    """

    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None):
        super(FedSeq, self).__init__(model_info, params, device, dataset, output_suffix, savedir, writer)

        self._clustering = params.clustering
        self._evaluator = params.evaluator
        self._training = params.training

        # list incompatibilities between extract method and clustering measures
        self._extractions = {*self._evaluator.extract_eval, self._evaluator.extract}
        self._clustering_measures = {*self._clustering.measures_eval, self._clustering.measure}
        self._incompatibilities = {to_extract: to_extract not in self._evaluator.extract_prop_distr
                                   and self._clustering.disomogeneity_measures or []
                                   for to_extract in self._extractions
                                   }

        # list all clustering methods, the one used later for training and the ones for evaluation
        clustering_methods: Dict[str, ClusterMaker] = {
            m: eval(m)(**self._clustering, num_classes=self._dataset_num_classes,
                       savedir=savedir) for m in
            {*self._clustering.classnames_eval, self._clustering.classname}}

        # evaluate clients if needed
        evaluations = self._evaluate_if_needed(clustering_methods)
        if len(self._clustering.classnames_eval) > 0:
            self._run_clustering_evaluation(clustering_methods, evaluations)

        clients_representer = []
        if self._evaluator.extract in evaluations:
            clients_representer = evaluations[self._evaluator.extract].representers

        # run the clustering methods for the training
        self._superclients: List[FedSeqSuperClient] = self._run_clustering_training(clustering_methods,
                                                                                    clients_representer,
                                                                                    self._evaluator.extract)
        self._num_superclients = len(self._superclients)

    def _evaluate_if_needed(self, clustering_methods: Dict[str, ClusterMaker]) -> Dict[str, ClientEvaluation]:
        """
        Runs the pretrain phase on clients if any of the requested grouping algorithms needs the to estimate the
        clients' distribution.

        Examines the clustering methods requested for i) evaluation of clustering algorithms, ii) group clients together
        for training, checking for correctness between the representation to extract and the measure to be used on that
        by the clustering algorithm. If any of them requires an estimation of the clients' distribution, runs the
        evaluator extracting the representations to be used as proxy of clients' distribution.

        Parameters
        ----------
        clustering_methods
            a dictionary mapping the classname of a clustering algorithm to an object of that type

        Returns
        -------
        a dictionary mapping the representation to extract to an evaluation object

        """
        at_least_one_extraction_needed = any(measure not in self._incompatibilities[extr]
                                             for extr in self._extractions for measure in self._clustering_measures)
        assert at_least_one_extraction_needed, \
            f"Incompatibility between extraction set={self._extractions} " \
            f"and clustering measure set={self._clustering.measures_eval}"
        # use the elements extracted from test set as examplars
        exemplar_dataset = self._excluded_from_test
        evaluations = {}
        if any(m.requires_clients_evaluation() for m in clustering_methods.values()):
            log.info("Evaluating clients")
            model_evaluator = Model(self._evaluator.model_info, self._dataset_num_classes)
            c_ev = ClientEvaluator(exemplar_dataset, model_evaluator, list(self._extractions),
                                   self._evaluator.variance_explained, self._evaluator.epochs)
            optim_class, optim_args = eval(self._evaluator.optim.classname), self._evaluator.optim.args
            evaluations = c_ev.evaluate(self._clients, optim_class, optim_args, CrossEntropyLoss, self.save_models_path)
        return evaluations

    def _run_clustering_evaluation(self, clustering_methods: Dict[str, ClusterMaker],
                                   evaluations: Dict[str, ClientEvaluation]) -> None:
        """
        Used when there are multiple clustering algorithms to benchmark, whose result will not be used for training.

        Parameters
        ----------
        clustering_methods
            a dictionary mapping the classname of a clustering algorithm to an object of that type
        evaluations
            a dictionary mapping the representation extracted to an evaluation object
        """

        # mapping extracted representation -> allowed measures on it
        clustering_measures = {
            e.extracted: {*self._clustering.measures_eval}.difference(self._incompatibilities[e.extracted])
            for e in evaluations.values()
        }
        # check that there is at least one measure tha can be used in clustering given the extracted representations
        if all(len(m) == 0 for m in clustering_measures.values()):
            log.warning("No valid combination for clustering algorithms evaluation")
            return

        # for all the extracted features, run clustering
        for e in evaluations.values():
            # pair method, measure for clustering methods in eval that require evaluation and do not use custom metric
            clustering_eval_comb = [((name, method), measure) for (name, method), measure in
                                    it.product(clustering_methods.items(), clustering_measures[e.extracted])
                                    if name in self._clustering.classnames_eval and method.requires_clients_evaluation()
                                    and not (name == self._clustering.classname and measure == self._clustering.measure)
                                    and not method.uses_custom_metric()
                                    ]
            # add for those clustering methods in eval that use a custom metric
            clustering_eval_comb.extend([((name, method), None) for name, method in clustering_methods.items()
                                         if method.uses_custom_metric()])
            log.info(f"From evaluation extracted {e.extracted}")
            self._run_clustering_combos(clustering_eval_comb, e.representers, e.extracted)

        # run clustering for those methods that do not require evaluation
        clustering_eval_comb = [((name, method), None) for name, method in clustering_methods.items()
                                if not method.requires_clients_evaluation() and not name == self._clustering.classname]
        self._run_clustering_combos(clustering_eval_comb, [])

    def _run_clustering_combos(self, clustering_eval_comb, representers: List[np.ndarray], extracted: str = ""):
        """
        Runs allowed combos between the clustering algorithm and the measures to be used on the representations

        Parameters
        ----------
        clustering_eval_comb
            a list of combos clustering algorithm, measure to be used
        representers
            a list of representations, one for each client
        extracted
            the name of the representation extracted, i.e. confidence
        """
        for (classname, method), measure in clustering_eval_comb:
            method.save_statistics = True
            method.measure = measure
            log.info(f"Clustering with {classname} using {measure} for clustering evaluation")
            method.make_superclients(self._clients, representers, sub_path=extracted, **self._training)

    def _run_clustering_training(self, clustering_methods: Dict[str, ClusterMaker], representers: List[np.ndarray],
                                 extracted: str = "") -> List[FedSeqSuperClient]:
        """
        Runs the clustering algorithm chosen for training

        Parameters
        ----------
        clustering_methods
            a dictionary mapping the classname of a clustering algorithm to an object of that type
        representers
            a list of representations, one for each client
        extracted
            the name of the representation extracted, i.e. confidence

        Returns
        -------
        a list of superclients created according the chosen clustering algorithm

        """
        method = clustering_methods[self._clustering.classname]
        method.save_statistic = self._clustering.save_statistics
        method.measure = method.requires_clients_evaluation() and self._clustering.measure or None
        log.info(f"Clustering with {self._clustering.classname} using {method.measure} for training")
        return method.make_superclients(self._clients, representers, sub_path=extracted, **self._training,
                                        optimizer_class=self._optimizer, optimizer_args=self._optimizer_args,
                                        analyzer=self._analyzer.module_analyzer('superclient')
                                        )

    def train_step(self):
        self._select_clients(self._superclients, lambda x: x)
        self._setup_clients()

        for c in self._selected_clients:
            c.client_update(self._optimizer, self._optimizer_args, self._training.sequential_rounds, self._loss_fn,
                            self._iteration)

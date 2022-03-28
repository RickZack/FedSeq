from omegaconf import OmegaConf
from src.experiments import *


def main():
    train_defaults = OmegaConf.load('config/config.yaml')
    experiment_config = OmegaConf.load('config/experiments/config.yaml')
    FedExperiment.default_fit_iterations = experiment_config.get("fit_iterations", 1)
    before_python_cmd = 'source ../venv3/bin/activate\nmodule load nvidia/cudasdk/10.1\n'
    experiments = [
        FedExperiment.from_param_groups("SOTAs",
                                        "Runs the state-of-art algorithms for the common params K=500 and C=0.2."
                                        "Produces table I",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            MultiParam.key("algo", ["fedavg", "fedprox", "scaffold", "feddyn"]),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5])
                                        ]
                                        ),
        FedExperiment.from_param_groups("FedSeq",
                                        "Runs the best configurations of FedSeq. Produces table I",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5])
                                        ]
                                        ),
        FedExperiment.from_param_groups("FedSeqInter",
                                        "Runs the best configurations of FedSeqInter. Produces table I",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq_inter"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5])
                                        ]
                                        ),
        FedExperiment.from_param_groups("FedSeq+SOTAs",
                                        "Runs the best configurations of FedSeq, combined with other SOTAs approaches."
                                        "Produces table I",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            MultiParam.key("algo", ["fedseq_prox", "fedseq_dyn"]),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5])
                                        ]
                                        ),
        FedExperiment.from_param_groups("FedSeqInter+SOTAs",
                                        "Runs the best configurations of FedSeqInter, combined with other SOTAs "
                                        "approaches. Produces table I",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            MultiParam.key("algo", ["fedseq_inter_prox", "fedseq_inter_dyn"]),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5])
                                        ]
                                        ),
        FedExperiment.from_params("Clients pre-training",
                                  "Pre-train all the client for e in [1, 5, 10, 20, 30, 40]. Produce fig. 4",
                                  Param("algo", "fedseq"),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.dict("algo.params.evaluator", ("epochs", [1, 5, 10, 20, 30, 40])),
                                  MultiParam.dict("algo.params.clustering", ("classname", ["GreedyClusterMaker"])),
                                  Param("do_train", False),
                                  Param("algo.params.save_models", True)
                                  ),
        FedExperiment.from_params("Grouping comparison",
                                  "Compare the performance of different grouping criteria on the resulting "
                                  "superclients. Produces table IV and fig. [3, 5-8]",
                                  Param("algo.params.evaluator.extract_eval", ["classifierLast",
                                                                               "classifierLast2",
                                                                               "classifierAll",
                                                                               "confidence"]),
                                  Param("algo.params.evaluator.variance_explained", 0.9),
                                  Param("algo", "fedseq"),
                                  Param("do_train", False),
                                  Param("algo.params.clustering.classnames_eval",
                                        ["RandomClusterMaker", "GreedyClusterMaker", "KMeansClusterMaker"]),
                                  Param("algo.params.clustering.measures_eval",
                                        ["gini", "kullback", "cosine", "wasserstein"]),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [400, 800, 1000],
                                                   "max_clients": [7, 11, 13]}),
                                  MultiParam.key("algo.params.evaluator.epochs", [10, 20]),
                                  MultiParam.dict("algo.params.evaluator.model_info",
                                                  {"type": ["lenet"], "classname": ["LeNet"],
                                                   "pretrained": [False], "feature_extract": [False]}),
                                  Param("algo.params.clustering.verbose", True)
                                  ),
        FedExperiment.from_param_groups("FedSeq - ablation on E_S seq=2",
                                        "Runs the best configuration of FedSeq, varying the number of superclients' "
                                        "local epochs E_S. Produces results in fig. 9",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 5000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 10000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            Param("algo.params.training.sequential_rounds", 2)
                                        ]
                                        ),
        FedExperiment.from_param_groups("FedSeq - ablation on E_S seq=4",
                                        "Runs the best configuration of FedSeq, varying the number of superclients' "
                                        "local epochs E_S. Produces results in fig. 9",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 2500),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 5000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            Param("algo.params.training.sequential_rounds", 4)
                                        ]
                                        ),
    ]
    r: Runner = SlurmRunner(experiment_config.get("seed"), default_params=train_defaults, prep_cmd=before_python_cmd,
                            defaults={"--mem": "5GB"}, run_sbatch=False)
    for e in experiments:
        print(e, '\n')
        e.run('train.py', r)
    r.wait_all()


if __name__ == "__main__":
    main()

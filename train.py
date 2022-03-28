import hydra
import logging
import os
from omegaconf import OmegaConf, DictConfig
from src.algo import *
from src.utils import seed_everything, CustomSummaryWriter
log = logging.getLogger(__name__)


def create_model(cfg: DictConfig, writer) -> Algo:
    """ Creates the controller for the algorithm of choice and reloads from a checkpoint if requested """
    method: Algo = eval(cfg.algo.classname)(model_info=cfg.model, device=cfg.device, writer=writer,
                                            dataset=cfg.dataset, params=cfg.algo.params,
                                            savedir=cfg.savedir, output_suffix=cfg.output_suffix)
    if cfg.reload_checkpoint:
        method.load_from_checkpoint()
    return method


def train_fit(cfg: DictConfig, writer) -> None:
    """ Issues algorithm controller and start the training """
    model: Algo = create_model(cfg, writer)
    if cfg.do_train:
        model.fit(cfg.n_round)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """ Entry-point for any algorithm training

        Sets the root directory, seeds all the number generators, print parameters and
        starts the training.
    """

    os.chdir(cfg.root)
    seed_everything(cfg.seed)
    log.info("Parameters:\n" + OmegaConf.to_yaml(cfg))
    with CustomSummaryWriter(log_dir=os.path.join(cfg.savedir, f"tf{cfg.output_suffix}"),
                             tag_prefix=cfg.output_suffix.replace('_', '')) as writer:
        writer.add_text("Parameters", OmegaConf.to_yaml(cfg).replace('\n', '  \n'), 0)
        train_fit(cfg, writer)


if __name__ == "__main__":
    main()

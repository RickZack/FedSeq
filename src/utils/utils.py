import copy
import functools
import logging
import math
import os
import pickle
import random
import signal
import sys
import time
from contextlib import contextmanager
from typing import Union, List

import numpy as np
import torch
from tensorboardX import SummaryWriter


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name: str, logger: Union[logging.Logger, None] = None):
    t0 = time.time()
    yield
    msg = f'[{name}] done in {time.time() - t0:.3f} s'
    if logger:
        logger.info(msg)
    else:
        print(msg)


def save_pickle(obj, path: str, open_options: str = "wb"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, open_options) as f:
        pickle.dump(obj, f)
    f.close()


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
                 tag_prefix=''):
        super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                         flush_secs=flush_secs, filename_suffix=filename_suffix)
        self.tag_prefix = tag_prefix

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        new_tag = f"{self.tag_prefix}/{tag}" if self.tag_prefix != '' else tag
        return super().add_scalar(new_tag, scalar_value, global_step=global_step, walltime=walltime)


def exit_on_signal(sig, ret_code=0):
    signal.signal(sig, lambda *args: sys.exit(ret_code))


def shuffled_copy(x):
    x_copy = copy.copy(x)  # shallow copy
    np.random.shuffle(x_copy)
    return x_copy


def select_random_subset(x, portion: float):
    input_len = len(x)
    # drop at least one item but not all of them
    to_drop_num = max(1, min(input_len - 1, math.ceil(input_len * portion)))
    to_drop_indexes = np.random.randint(0, input_len, to_drop_num)
    return np.delete(x, to_drop_indexes)


class MeasureMeter:
    """
    Keeps track of predictions result to obtain some measures, e.g. accuracy
    """

    def __init__(self, num_classes: int):
        self.__num_classes = num_classes
        self.__tp = torch.zeros(num_classes)
        self.__tn = torch.zeros(num_classes)
        self.__fp = torch.zeros(num_classes)
        self.__fn = torch.zeros(num_classes)
        self.__total = torch.zeros(num_classes)  # helper, it is just tp+tn+fp+fn

    @property
    def num_classes(self):
        return self.__num_classes

    def reset(self):
        self.__tp.fill_(0)
        self.__tn.fill_(0)
        self.__fp.fill_(0)
        self.__fn.fill_(0)
        self.__total.fill_(0)

    @property
    def accuracy_overall(self) -> float:
        return 100. * torch.sum(self.__tp) / torch.sum(self.__total)

    @property
    def accuracy_per_class(self) -> torch.Tensor:
        return 100. * torch.divide(self.__tp, self.__total + torch.finfo().eps)

    def update(self, predicted_batch: torch.Tensor, label_batch: torch.Tensor):
        for predicted, label in zip(predicted_batch, label_batch.view_as(predicted_batch)):
            # implement only accuracy
            if predicted.item() == label.item():
                self.__tp[label.item()] += 1
            self.__total[label.item()] += 1


def move_tensor_list(tensor_l: List[torch.Tensor], device: str):
    for i in range(len(tensor_l)):
        tensor_l[i] = tensor_l[i].to(device)


def function_call_log(log: logging.Logger):
    def log_call(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            log.info(f"Calling {f.__name__}")
            ret = f(*args, **kwargs)
            log.info(f"{f.__name__} executed")
            return ret

        return wrapper

    return log_call

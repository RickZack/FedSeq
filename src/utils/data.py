import numpy as np
from src.datasets.cifar import CifarLocalDataset
import torchvision
import random
import logging
from src.utils import non_iid_partition_with_dirichlet_distribution

log = logging.getLogger(__name__)


def get_dataset(requested_dataset):
    dataset_getter = {"cifar10": get_CIFAR10_data,
                      "cifar100": get_CIFAR100_data}
    dataset_class = {"cifar10": CifarLocalDataset,
                     "cifar100": CifarLocalDataset}
    dataset_num_class = {"cifar10": 10,
                         "cifar100": 100}
    if requested_dataset not in dataset_getter:
        raise KeyError(f"the requested dataset {requested_dataset} is not supported")
    return dataset_getter[requested_dataset], dataset_class[requested_dataset], dataset_num_class[requested_dataset]


def get_CIFAR10_data():
    # !wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    # !tar -zxvf cifar-10-python.tar.gz
    CIFAR10train = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True)
    CIFAR10test = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True)
    dataset_size = 50000

    return CIFAR10train.data, np.array(CIFAR10train.targets), CIFAR10test.data, np.array(
        CIFAR10test.targets), dataset_size


def get_CIFAR100_data():
    CIFAR100train = torchvision.datasets.CIFAR100(root="./datasets", train=True, download=True)
    CIFAR100test = torchvision.datasets.CIFAR100(root="./datasets", train=False, download=True)
    dataset_size = 50000

    return CIFAR100train.data, np.array(CIFAR100train.targets), CIFAR100test.data, np.array(
        CIFAR100test.targets), dataset_size


def create_datasets(dataset_name, num_clients, alpha, max_iter=100, rebalance=False):
    dataset_getter, dataset_class, dataset_num_class = get_dataset(dataset_name)
    train_img, train_label, test_img, test_label, dataset_size = dataset_getter()
    shard_size = dataset_size // num_clients

    if shard_size < 1:
        raise ValueError("shard_size should be at least 1")

    if alpha == 0:  # Non-IID
        local_datasets, test_datasets = create_non_iid(train_img, test_img, train_label, test_label, num_clients,
                                                       shard_size,
                                                       dataset_class, dataset_num_class)
    else:
        local_datasets, test_datasets = create_using_dirichlet_distr(train_img, test_img, train_label, test_label,
                                                                     num_clients,
                                                                     alpha, max_iter, rebalance, shard_size, dataset_class,
                                                                     dataset_num_class)
    return local_datasets, test_datasets, dataset_num_class


def create_non_iid(train_img, test_img, train_label, test_label, num_clients, shard_size,
                   dataset_class, dataset_num_class):
    train_sorted_index = np.argsort(train_label)
    train_img = train_img[train_sorted_index]
    train_label = train_label[train_sorted_index]

    shard_start_index = [i for i in range(0, len(train_img), shard_size)]
    random.shuffle(shard_start_index)
    log.info(f"divide data into {len(shard_start_index)} shards of size {shard_size}")

    num_shards = len(shard_start_index) // num_clients
    local_datasets = []
    for client_id in range(num_clients):
        _index = num_shards * client_id
        img = np.concatenate([
            train_img[shard_start_index[_index +
                                        i]:shard_start_index[_index + i] +
                                           shard_size] for i in range(num_shards)
        ],
            axis=0)

        label = np.concatenate([
            train_label[shard_start_index[_index +
                                          i]:shard_start_index[_index +
                                                               i] +
                                             shard_size] for i in range(num_shards)
        ],
            axis=0)
        local_datasets.append(dataset_class(img, label, dataset_num_class))

    test_sorted_index = np.argsort(test_label)
    test_img = test_img[test_sorted_index]
    test_label = test_label[test_sorted_index]
    test_dataset = dataset_class(test_img, test_label, dataset_num_class, train=False)

    return local_datasets, test_dataset


def create_using_dirichlet_distr(train_img, test_img, train_label, test_label,
                                 num_clients, alpha, max_iter, rebalance, shard_size, dataset_class, dataset_num_class):
    d = non_iid_partition_with_dirichlet_distribution(
        np.array(train_label), num_clients, dataset_num_class, alpha, max_iter)

    if rebalance:
        storage = []
        for i in range(len(d)):
            if len(d[i]) > (shard_size):
                difference = round(len(d[i]) - (shard_size))
                toSwitch = np.random.choice(
                    d[i], difference, replace=False).tolist()
                storage += toSwitch
                d[i] = list(set(d[i]) - set(toSwitch))

        for i in range(len(d)):
            if len(d[i]) < (shard_size):
                difference = round((shard_size) - len(d[i]))
                toSwitch = np.random.choice(
                    storage, difference, replace=False).tolist()
                d[i] += toSwitch
                storage = list(set(storage) - set(toSwitch))

        for i in range(len(d)):
            if len(d[i]) != (shard_size):
                log.warning(f'There are some clients with more than {shard_size} images')

    # Lista contenente per ogni client un'istanza di Cifar10LocalDataset ->local_datasets[client]
    local_datasets = []
    for client_id in d.keys():
        # img = np.concatenate( [train_img[list_indexes_per_client_subset[client_id][c]] for c in range(n_classes)],axis=0)
        img = train_img[d[client_id]]
        # label = np.concatenate( [train_label[list_indexes_per_client_subset[client_id][c]] for c in range(n_classes)],axis=0)
        label = train_label[d[client_id]]
        local_datasets.append(dataset_class(img, label, dataset_num_class))

    test_sorted_index = np.argsort(test_label)
    test_img = test_img[test_sorted_index]
    test_label = test_label[test_sorted_index]

    test_dataset = dataset_class(test_img, test_label, dataset_num_class, train=False)

    return local_datasets, test_dataset

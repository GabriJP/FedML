from __future__ import annotations

import logging
from pathlib import Path

from fedml_api.data_preprocessing import LocalDataset, FederatedDataset, MNISTDataLoader, FederatedEMNISTDataLoader, \
    ShakespeareDataLoader, FederatedShakespeareDataLoader, Cifar100DatasetLoader, StackOverflowLRDataLoader, \
    StackOverflowNWPDataLoader, ImageNetDataLoader, LandmarksDataLoader, Cifar10DatasetLoader, Cinic10DatasetLoader
from fedml_api.model.cv.cnn import CNNDropOut
from fedml_api.model.cv.efficientnet import EfficientNet
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.nlp.rnn import RNNOriginalFedAvg, RNNStackOverFlow

MODELS = (
    "lr",
    "rnn",
    "cnn",
    "resnet18_gn",
    "rnn",
    "lr",
    "rnn",
    "resnet56",
    "mobilenet",
    "mobilenet_v3",
    "efficientnet",
)

DATASETS = (
    "mnist",
    "shakespeare",
    "femnist",
    "fed_cifar100",
    "fed_shakespeare",
    "stackoverflow_lr",
    "stackoverflow_nwp",
)

PARTITION_METHODS = ("hetero", "homo")

CLIENT_OPTIMIZERS = ("sgd", "adam",)

BACKENDS = ("MPI",)


def load_data(dataset_name, data_dir: Path, batch_size=None, client_number=None, partition_method=None,
              partition_alpha=None) -> LocalDataset | FederatedDataset:
    dlc = dict(
        mnist=MNISTDataLoader,
        femnist=FederatedEMNISTDataLoader,
        shakespeare=ShakespeareDataLoader,
        fed_shakespeare=FederatedShakespeareDataLoader,
        fed_cifar100=Cifar100DatasetLoader,
        stackoverflow_lr=StackOverflowLRDataLoader,
        stackoverflow_nwp=StackOverflowNWPDataLoader,
        ILSVRC2012=ImageNetDataLoader,
        gld23k=LandmarksDataLoader,
        gld160k=LandmarksDataLoader,
        cifar10=Cifar10DatasetLoader,
        cifar100=Cifar100DatasetLoader,
        cinic10=Cinic10DatasetLoader,
    ).get(dataset_name, None)

    if dlc is None:
        raise ValueError('Unknown dataset')

    logging.info(f"load_data. dataset_name = {dataset_name}")

    default_args = dict(data_dir=str(data_dir))
    if dataset_name == 'ILSVRC2012':
        default_args.update(batch_size=batch_size, client_number=client_number)
    elif dataset_name == 'gld23k':
        default_args.update(
            data_dir=str(data_dir / "images"), train_bs=batch_size, test_bs=batch_size, client_number=233,
            fed_train_map_file=str(data_dir / "mini_gld_train_split.csv"),
            fed_test_map_file=str(data_dir / "mini_gld_test.csv"),
        )
    elif dataset_name == 'gld160k':
        default_args.update(
            data_dir=str(data_dir / "images"), train_bs=batch_size, test_bs=batch_size, client_number=1262,
            fed_train_map_file=str(data_dir / "federated_train.csv"),
            fed_test_map_file=str(data_dir / "test.csv"),
        )
    elif dataset_name in {"cifar10", "cifar100", "cinic10"}:
        default_args.update(
            partition_method=partition_method, partition_alpha=partition_alpha, data_dir=data_dir,
            batch_size=batch_size, client_number=client_number
        )

    dl = dlc(**default_args)

    if dataset_name in {"femnist", "fed_shakespeare", "fed_cifar100", "stackoverflow_lr", "stackoverflow_nwp"}:
        ds = dl.load_partition_data_federated()
    else:
        ds = dl.load_partition_data()

    return ds


def create_model(model_name, dataset_name, output_dim):
    logging.info(f"create_model. model_name = {model_name}, output_dim = {output_dim}")

    if model_name == "lr" and dataset_name == "mnist":
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "rnn" and dataset_name == "shakespeare":
        model = RNNOriginalFedAvg()
    elif model_name == "cnn" and dataset_name == "femnist":
        model = CNNDropOut(False)
    elif model_name == "resnet18_gn" and dataset_name == "fed_cifar100":
        model = resnet18()
    elif model_name == "rnn" and dataset_name == "fed_shakespeare":
        model = RNNOriginalFedAvg()
    elif model_name == "lr" and dataset_name == "stackoverflow_lr":
        model = LogisticRegression(10004, output_dim)
    elif model_name == "rnn" and dataset_name == "stackoverflow_nwp":
        model = RNNStackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenet_v3":
        # model_mode in {LARGE: 5.15M, SMALL: 2.94M}"
        model = MobileNetV3(model_mode="LARGE")
    elif model_name == "efficientnet":
        model = EfficientNet()
    else:
        raise NotImplementedError(f'Unknown combination for {model_name} and {dataset_name}')

    logging.info(f'Using {model_name}')

    return model

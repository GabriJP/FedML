from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    # Dataset config
    dataset_name: str
    partition_alpha: float

    # Federated config
    client_num_in_total: int
    client_num_per_round: int
    backend: Optional[str]
    is_mobile: bool
    gpu_server_num: int
    gpu_num_per_server: int

    # Train config
    client_optimizer: str
    lr: float
    wd: float
    batch_size: int
    epochs: int
    comm_round: int
    frequency_of_the_test: Optional[int]
    ci: bool


@dataclass
class CentralizedRunConfig(RunConfig):
    data_parallel: bool
    frequency_of_train_acc_report: int
    rank: int


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """

    def __init__(self, model, dataset_name, client_optimizer, lr, wd, epochs):
        self.model = model
        self.dataset_name = dataset_name
        self.client_optimizer = client_optimizer
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.id = 0

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device):
        pass

    @abstractmethod
    def test(self, test_data, device):
        pass

    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass

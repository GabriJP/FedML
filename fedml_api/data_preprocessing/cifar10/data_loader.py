from .datasets import CIFAR10Truncated
from ..base import CifarDataLoader


class Cifar10DatasetLoader(CifarDataLoader):
    CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
    CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)
    CIFAR_truncated = CIFAR10Truncated

    def __init__(self, data_dir, partition_method, partition_alpha, client_number, batch_size,
                 distribution_txt='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt',
                 net_dataidx_map_txt='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
        super().__init__(data_dir, partition_method, partition_alpha, client_number, batch_size, distribution_txt,
                         net_dataidx_map_txt)

import logging

from .datasets import CIFAR100Truncated
from ..base import CifarDataLoader

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Cifar100DatasetLoader(CifarDataLoader):
    CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR_STD = (0.2673, 0.2564, 0.2762)
    CIFAR_truncated = CIFAR100Truncated

    def __init__(self, data_dir, partition_method, partition_alpha, client_number, batch_size,
                 distribution_txt='./data_preprocessing/non-iid-distribution/CIFAR100/distribution.txt',
                 net_dataidx_map_txt='./data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt'):
        super().__init__(data_dir, partition_method, partition_alpha, client_number, batch_size, distribution_txt,
                         net_dataidx_map_txt)

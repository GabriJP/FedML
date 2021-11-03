from torchvision.datasets import CIFAR100

from FedML.fedml_api.data_preprocessing.base import CIFARTruncated


class CIFAR100Truncated(CIFARTruncated):
    CIFAR_FUNC = CIFAR100

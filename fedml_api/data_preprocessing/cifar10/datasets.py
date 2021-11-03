from torchvision.datasets import CIFAR10

from FedML.fedml_api.data_preprocessing.base import CIFARTruncated


class CIFAR10Truncated(CIFARTruncated):
    CIFAR_FUNC = CIFAR10

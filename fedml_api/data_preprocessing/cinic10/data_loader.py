import logging
import os

import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import ImageFolderTruncated
from ..base import CifarDataLoader


class Cinic10DatasetLoader(CifarDataLoader):
    CIFAR_MEAN = (0.47889522, 0.47227842, 0.43047404)
    CIFAR_STD = (0.24205776, 0.23828046, 0.25874835)

    def __init__(self, data_dir, partition_method, partition_alpha, client_number, batch_size,
                 distribution_txt='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt',
                 net_dataidx_map_txt='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
        super().__init__(data_dir, partition_method, partition_alpha, client_number, batch_size, distribution_txt,
                         net_dataidx_map_txt)

    @classmethod
    def _data_transforms(cls):
        transforms_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.CIFAR_MEAN, std=cls.CIFAR_STD),
        ]

        # Transformer for train set: random crops and horizontal flip
        train_transform = transforms.Compose(transforms_list)

        # Transformer for test set
        valid_transform = transforms.Compose(transforms_list)
        return train_transform, valid_transform

    def load_cifar_data(self):
        train_dir = f'{self.data_dir}/train'
        test_dir = f'{self.data_dir}/test'
        logging.info(f"train_dir = {train_dir}. test_dir = {test_dir}")
        train_transform, test_transform = self._data_transforms()

        trainset = ImageFolderTruncated(train_dir, transform=train_transform)
        testset = ImageFolderTruncated(test_dir, transform=test_transform)

        x_train, y_train = trainset.imgs, trainset.targets
        x_test, y_test = testset.imgs, testset.targets

        return x_train, y_train, x_test, y_test

    def partition_data(self):
        logging.info("*********partition data***************")
        pil_logger = logging.getLogger('PIL')
        pil_logger.setLevel(logging.INFO)

        x_train, y_train, x_test, y_test = self.load_cifar_data()
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n_train = len(x_train)
        # n_test = len(x_test)

        if self.partition_method == "homo":
            total_num = n_train
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.client_number)
            net_dataidx_map = {i: batch_idxs[i] for i in range(self.client_number)}

        elif self.partition_method == "hetero":
            min_size = 0
            n = y_train.shape[0]
            logging.info(f"N = {n}")
            net_dataidx_map = dict()

            idx_batch = None
            while min_size < 10:
                idx_batch = [list() for _ in range(self.client_number)]
                # for each class in the dataset
                for k in range(self.k):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.partition_alpha, self.client_number))
                    # Balance
                    proportions = np.array(
                        [p * (len(idx_j) < n / self.client_number) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min(len(idx_j) for idx_j in idx_batch)

            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

        elif self.partition_method == "hetero-fix":
            net_dataidx_map = self.read_net_dataidx_map()
        else:
            raise ValueError(f'Unknown partition method: {self.partition_method}')

        traindata_cls_counts = self.read_data_distribution() if self.partition_method == "hetero-fix" else \
            self.record_net_data_stats(y_train, net_dataidx_map)

        return x_train, y_train, x_test, y_test, net_dataidx_map, traindata_cls_counts

    # for centralized training
    def get_dataloader(self, dataidxs=None):
        transform_train, transform_test = self._data_transforms()

        traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'test')

        train_ds = ImageFolderTruncated(traindir, dataidxs=dataidxs, transform=transform_train)
        test_ds = ImageFolderTruncated(valdir, transform=transform_train)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=True)

        return train_dl, test_dl

    # for local devices
    def get_dataloader_test(self, dataidxs_train=None, dataidxs_test=None):
        transform_train, transform_test = self._data_transforms()

        traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'test')

        train_ds = ImageFolderTruncated(traindir, dataidxs=dataidxs_train, transform=transform_train)
        test_ds = ImageFolderTruncated(valdir, dataidxs=dataidxs_test, transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=True)

        return train_dl, test_dl

    def load_partition_data_distributed(self, process_id):
        dd = super().load_partition_data_distributed(process_id)
        dd.test_data_num = len(dd.test_data_global) if process_id == 0 else 0
        return dd

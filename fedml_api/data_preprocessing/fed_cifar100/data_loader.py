import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from ..base import CifarDataLoader, DistributedFederatedDataset, FederatedDataset

DEFAULT_TRAIN_CLIENTS_NUM = 500
DEFAULT_TEST_CLIENTS_NUM = 100
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_cifar100_train.h5'
DEFAULT_TEST_FILE = 'fed_cifar100_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMAGE = 'image'
_LABEL = 'label'


class FederatedCifar10DatasetLoader(CifarDataLoader):
    def __init__(self, data_dir, partition_method, partition_alpha, client_number, batch_size, distribution_txt,
                 net_dataidx_map_txt):
        super().__init__(data_dir, partition_method, partition_alpha, client_number, batch_size, distribution_txt,
                         net_dataidx_map_txt)
        self.client_ids_train, self.client_ids_test = None, None

    def load_partition_data_distributed_federated(self, process_id):
        class_num = 100

        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader()
            train_data_num = len(train_data_global.dataset)
            test_data_num = len(test_data_global.dataset)
            logging.info(f"train_dl_global number = {train_data_num}")
            logging.info(f"test_dl_global number = {test_data_num}")
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = self.get_dataloader(process_id - 1)
            train_data_num = local_data_num = len(train_data_local.dataset)
            logging.info(f"rank = {process_id:d}, local_sample_number = {local_data_num:d}")
            train_data_global = None
            test_data_global = None
        return DistributedFederatedDataset(client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_num=train_data_num,
                                           train_data_global=train_data_global, test_data_global=test_data_global,
                                           local_data_num=local_data_num, train_data_local=train_data_local,
                                           test_data_local=test_data_local, output_len=class_num)

    @staticmethod
    def cifar_transform(img_mean, img_std, train=True, crop_size=(24, 24)):
        """cropping, flipping, and normalizing."""
        transform_list = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ]

        if train:
            transform_list.insert(1, transforms.RandomCrop(crop_size))
            transform_list.insert(2, transforms.RandomHorizontalFlip())
        else:
            transform_list.insert(1, transforms.CenterCrop(crop_size))

        return transforms.Compose(transform_list)

    @classmethod
    def preprocess_cifar_img(cls, img, train):
        # scale img to range [0,1] to fit ToTensor api
        img = torch.div(img, 255.0)
        transoformed_img = torch.stack(
            [cls.cifar_transform(i.msg_type(torch.DoubleTensor).mean(), i.msg_type(torch.DoubleTensor).std(), train)
             (i.permute(2, 0, 1)) for i in img])
        return transoformed_img

    def get_dataloader(self, client_idx=None):
        train_h5 = h5py.File(os.path.join(self.data_dir, DEFAULT_TRAIN_FILE), 'r')
        test_h5 = h5py.File(os.path.join(self.data_dir, DEFAULT_TEST_FILE), 'r')
        test_x = list()
        test_y = list()

        # load data in numpy format from h5 file
        if client_idx is None:
            train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMAGE][()] for client_id in self.client_ids_train])
            train_y = np.vstack(
                [train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in self.client_ids_train]).squeeze()
            test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMAGE][()] for client_id in self.client_ids_test])
            test_y = np.vstack(
                [test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in self.client_ids_test]).squeeze()
        else:
            client_id_train = self.client_ids_train[client_idx]
            train_x = np.vstack([train_h5[_EXAMPLE][client_id_train][_IMAGE][()]])
            train_y = np.vstack([train_h5[_EXAMPLE][client_id_train][_LABEL][()]]).squeeze()
            if client_idx < len(self.client_ids_test):
                client_id_test = self.client_ids_test[client_idx]
                test_x = np.vstack([train_h5[_EXAMPLE][client_id_test][_IMAGE][()]])
                test_y = np.vstack([train_h5[_EXAMPLE][client_id_test][_LABEL][()]]).squeeze()

        # preprocess
        train_x = self.preprocess_cifar_img(torch.tensor(train_x), train=True)
        train_y = torch.tensor(train_y)
        if len(test_x):
            test_x = self.preprocess_cifar_img(torch.tensor(test_x), train=False)
            test_y = torch.tensor(test_y)

        # generate dataloader
        train_ds = data.TensorDataset(train_x, train_y)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)

        if len(test_x):
            test_ds = data.TensorDataset(test_x, test_y)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=True, drop_last=False)
        else:
            test_dl = None

        train_h5.close()
        test_h5.close()

        return train_dl, test_dl

    def load_partition_data_federated(self):
        class_num = 100

        # client id list
        train_file_path = os.path.join(self.data_dir, DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(self.data_dir, DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
            self.client_ids_train = list(train_h5[_EXAMPLE].keys())
            self.client_ids_test = list(test_h5[_EXAMPLE].keys())

        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(client_idx)
            local_data_num = len(train_data_local.dataset)
            data_local_num_dict[client_idx] = local_data_num
            logging.info(f"client_idx = {client_idx:d}, local_sample_number = {local_data_num:d}")
            logging.info(f"client_idx = {client_idx:d}, batch_num_train_local = {len(train_data_local)}")
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        # global dataset
        train_data_global = data.DataLoader(data.ConcatDataset([dl.dataset for dl in train_data_local_dict.values()]),
                                            batch_size=self.train_bs, shuffle=True)
        train_data_num = len(train_data_global.dataset)

        test_data_global = data.DataLoader(
            data.ConcatDataset([dl.dataset for dl in test_data_local_dict.values() if dl is not None]),
            batch_size=self.test_bs, shuffle=True)
        test_data_num = len(test_data_global.dataset)

        return FederatedDataset(client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_num=train_data_num,
                                test_data_num=test_data_num, train_data_global=train_data_global,
                                test_data_global=test_data_global, local_data_num_dict=data_local_num_dict,
                                train_data_local_dict=train_data_local_dict, test_data_local_dict=test_data_local_dict,
                                output_len=class_num)

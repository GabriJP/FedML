import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data

from FedML.fedml_api.data_preprocessing.base import DataLoader, DistributedFederatedDataset, FederatedDataset

DEFAULT_TRAIN_CLIENTS_NUM = 3400
DEFAULT_TEST_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
DEFAULT_TEST_FILE = 'fed_emnist_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMGAE = 'pixels'
_LABEL = 'label'


class FederatedEMNISTDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size=DEFAULT_BATCH_SIZE):
        super().__init__(data_dir, batch_size, batch_size)

        train_file_path = os.path.join(self.data_dir, DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(self.data_dir, DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
            self.client_ids_train = list(train_h5[_EXAMPLE].keys())
            self.client_ids_test = list(test_h5[_EXAMPLE].keys())

        train_file_path = os.path.join(self.data_dir, DEFAULT_TRAIN_FILE)
        with h5py.File(train_file_path, 'r') as train_h5:
            self.class_num = len(np.unique(
                [train_h5[_EXAMPLE][self.client_ids_train[idx]][_LABEL][0] for idx in
                 range(DEFAULT_TRAIN_CLIENTS_NUM)]))
            logging.info(f"class_num = {self.class_num}")

    def get_dataloader(self, client_idx=None):
        train_h5 = h5py.File(os.path.join(self.data_dir, DEFAULT_TRAIN_FILE), 'r')
        test_h5 = h5py.File(os.path.join(self.data_dir, DEFAULT_TEST_FILE), 'r')

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self.client_ids_train
            test_ids = self.client_ids_test
        else:
            # get ids of single client
            train_ids = [self.client_ids_train[client_idx]]
            test_ids = [self.client_ids_test[client_idx]]

        # load data in numpy format from h5 file
        train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids])
        train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]).squeeze()
        test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids])
        test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]).squeeze()

        # dataloader
        train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)

        test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=True, drop_last=False)

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    def load_partition_data_distributed_federated_emnist(self, process_id):
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader(process_id - 1)
            train_data_num = len(train_data_global)
            # logging.info("train_dl_global number = " + str(train_data_num))
            # logging.info("test_dl_global number = " + str(test_data_num))
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = self.get_dataloader(process_id - 1)
            train_data_num = local_data_num = len(train_data_local)
            train_data_global = None
            test_data_global = None

        return DistributedFederatedDataset(client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_num=train_data_num,
                                           train_data_global=train_data_global, test_data_global=test_data_global,
                                           local_data_num=local_data_num, train_data_local=train_data_local,
                                           test_data_local=test_data_local, output_len=self.class_num)

    def load_partition_data_federated_emnist(self):
        # local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(client_idx)
            local_data_num = len(train_data_local) + len(test_data_local)
            data_local_num_dict[client_idx] = local_data_num
            # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
            # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            #     client_idx, len(train_data_local), len(test_data_local)))
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
                                train_data_local_dict=train_data_local_dict,
                                test_data_local_dict=test_data_local_dict, output_len=self.class_num)

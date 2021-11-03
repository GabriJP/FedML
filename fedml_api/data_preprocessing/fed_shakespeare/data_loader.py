import logging
import os
from typing import Tuple, Union, Sized

import h5py
import torch
import torch.utils.data as data

from . import utils
from ..base import DataLoader, DistributedFederatedDataset, FederatedDataset

DEFAULT_TRAIN_CLIENTS_NUM = 715
DEFAULT_TEST_CLIENTS_NUM = 715
DEFAULT_BATCH_SIZE = 4
DEFAULT_TRAIN_FILE = 'shakespeare_train.h5'
DEFAULT_TEST_FILE = 'shakespeare_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_SNIPPETS = 'snippets'


class FederatedShakespeareDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size=DEFAULT_BATCH_SIZE):
        super().__init__(data_dir, batch_size, batch_size)
        train_file_path = os.path.join(self.data_dir, DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(self.data_dir, DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
            self.client_ids_train = list(train_h5[_EXAMPLE].keys())
            self.client_ids_test = list(test_h5[_EXAMPLE].keys())

    def get_dataloader(self, client_idx=None):
        train_h5 = h5py.File(os.path.join(self.data_dir, DEFAULT_TRAIN_FILE), 'r')
        test_h5 = h5py.File(os.path.join(self.data_dir, DEFAULT_TEST_FILE), 'r')
        train_ds = list()
        test_ds = list()

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self.client_ids_train
            test_ids = self.client_ids_test
        else:
            # get ids of single client
            train_ids = [self.client_ids_train[client_idx]]
            test_ids = [self.client_ids_test[client_idx]]

        for client_id in train_ids:
            raw_train = train_h5[_EXAMPLE][client_id][_SNIPPETS][()]
            raw_train = [x.decode('utf8') for x in raw_train]
            train_ds.extend(utils.preprocess(raw_train))
        for client_id in test_ids:
            raw_test = test_h5[_EXAMPLE][client_id][_SNIPPETS][()]
            raw_test = [x.decode('utf8') for x in raw_test]
            test_ds.extend(utils.preprocess(raw_test))

        # split data
        train_x, train_y = utils.split(train_ds)
        test_x, test_y = utils.split(test_ds)
        train_ds = data.TensorDataset(torch.tensor(train_x[:, :]), torch.tensor(train_y[:]))
        test_ds = data.TensorDataset(torch.tensor(test_x[:, :]), torch.tensor(test_y[:]))
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=True, drop_last=False)

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    def load_partition_data_distributed_federated(self, process_id):
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader(process_id - 1)
            train_data_num = len(train_data_global)
            test_data_num = len(test_data_global)
            logging.info(f"train_dl_global number = {train_data_num}")
            logging.info(f"test_dl_global number = {test_data_num}")
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            train_data_local, test_data_local = self.get_dataloader(process_id - 1)
            train_data_num = local_data_num = len(train_data_local.dataset)
            logging.info(f"rank = {process_id:d}, local_sample_number = {local_data_num:d}")
            train_data_global = None
            test_data_global = None

        return DistributedFederatedDataset(client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_num=train_data_num,
                                           train_data_global=train_data_global, test_data_global=test_data_global,
                                           local_data_num=local_data_num, train_data_local=train_data_local,
                                           test_data_local=test_data_local, vocab_len=len(utils.get_word_dict()) + 1)

    def load_partition_data_federated(self):
        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(client_idx)
            local_data_num = len(train_data_local.dataset)
            data_local_num_dict[client_idx] = local_data_num
            logging.info(f"client_idx = {client_idx:d}, local_sample_number = {local_data_num:d}")
            logging.info(f"client_idx = {client_idx:d}, batch_num_train_local = {len(train_data_local):d}, "
                         f"batch_num_test_local = {len(test_data_local):d}")
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        # global dataset
        train_data_global = data.DataLoader(data.ConcatDataset(
            list(dl.dataset for dl in list(train_data_local_dict.values()))),
            batch_size=self.train_bs,
            shuffle=True)
        train_data_num = len(train_data_global.dataset)

        test_data_global = data.DataLoader(data.ConcatDataset(
            list(dl.dataset for dl in list(test_data_local_dict.values())
                 if dl is not None)),
            batch_size=self.test_bs,
            shuffle=True)
        test_data_num = len(test_data_global.dataset)

        return FederatedDataset(client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_num=train_data_num,
                                test_data_num=test_data_num, train_data_global=train_data_global,
                                test_data_global=test_data_global,
                                local_data_num_dict=data_local_num_dict, train_data_local_dict=train_data_local_dict,
                                test_data_local_dict=test_data_local_dict, vocab_len=len(utils.get_word_dict()) + 1)


if __name__ == "__main__":
    # load_partition_data_federated_stackoverflow(None, None, 100, 128)
    dl = FederatedShakespeareDataLoader(None)
    ds = dl.load_partition_data_distributed_federated(2)
    print(ds.train_data_local, ds.test_data_local)

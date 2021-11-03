import json
import logging
import os.path

import numpy as np
import torch
import torch.utils.data as data

from FedML.fedml_api.data_preprocessing.base import DataLoader, FederatedDataset

DEFAULT_BATCH_SIZE = 16

_USERS = 'users'
_USER_DATA = "user_data"


class Synthetic11DataLoader(DataLoader):
    def __init__(self, data_dir='../../../data/synthetic_1_1', batch_size=DEFAULT_BATCH_SIZE):
        super().__init__(data_dir, batch_size, batch_size)
        self.train_file_path = os.path.join(self.data_dir, 'train/mytrain.json')
        self.test_file_path = os.path.join(self.data_dir, 'test/mytest.json')

    def load_partition_data_federated(self):
        logging.info("load_partition_data_federated_synthetic_1_1 START")

        with open(self.train_file_path, 'r') as train_f, open(self.test_file_path, 'r') as test_f:
            train_data = json.load(train_f)
            test_data = json.load(test_f)

            client_ids_train = train_data[_USERS]
            client_ids_test = test_data[_USERS]
            client_num = len(train_data[_USERS])

            full_x_train = torch.from_numpy(np.asarray([])).float()
            full_y_train = torch.from_numpy(np.asarray([])).long()
            full_x_test = torch.from_numpy(np.asarray([])).float()
            full_y_test = torch.from_numpy(np.asarray([])).long()
            train_data_local_dict = dict()
            test_data_local_dict = dict()

            for i in range(len(client_ids_train)):
                train_ds = data.TensorDataset(
                    torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['x']),
                    torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['y'], dtype=torch.int64))
                test_ds = data.TensorDataset(
                    torch.tensor(train_data[_USER_DATA][client_ids_test[i]]['x']),
                    torch.tensor(train_data[_USER_DATA][client_ids_test[i]]['y'], dtype=torch.int64))
                train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)
                test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=False)
                train_data_local_dict[i] = train_dl
                test_data_local_dict[i] = test_dl

                full_x_train = torch.cat(
                    (full_x_train, torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['x'])), 0)
                full_y_train = torch.cat(
                    (full_y_train, torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['y'], dtype=torch.int64)),
                    0)
                full_x_test = torch.cat((full_x_test, torch.tensor(test_data[_USER_DATA][client_ids_test[i]]['x'])), 0)
                full_y_test = torch.cat(
                    (full_y_test, torch.tensor(test_data[_USER_DATA][client_ids_test[i]]['y'], dtype=torch.int64)), 0)

            train_ds = data.TensorDataset(full_x_train, full_y_train)
            test_ds = data.TensorDataset(full_x_test, full_y_test)
            train_data_global = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True,
                                                drop_last=False)
            test_data_global = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False,
                                               drop_last=False)
            train_data_num = len(train_data_global.dataset)
            test_data_num = len(test_data_global.dataset)
            data_local_num_dict = {i: len(train_data_local_dict[i].dataset) for i in train_data_local_dict}
            output_dim = 10

        return FederatedDataset(
            client_num=client_num, train_data_num=train_data_num, test_data_num=test_data_num,
            train_data_global=train_data_global, test_data_global=test_data_global,
            local_data_num_dict=data_local_num_dict, train_data_local_dict=train_data_local_dict,
            test_data_local_dict=test_data_local_dict, output_len=output_dim)

    def test_data_loader(self):
        ds = self.load_partition_data_federated()
        f = open(self.train_file_path, 'r')
        train_data = json.load(f)
        assert (train_data['num_samples'] == list(ds.local_data_num_dict.values()))


if __name__ == "__main__":
    Synthetic11DataLoader().test_data_loader()

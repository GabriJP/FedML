import json
import os

import numpy as np
import torch

from .language_utils import word_to_indices, VOCAB_SIZE, letter_to_index
from ..base import DataLoader, LocalDataset


class ShakespeareDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size):
        super().__init__(data_dir, batch_size, batch_size)
        # self.train_path = "../../../data/shakespeare/train"
        # self.test_path = "../../../data/shakespeare/test"
        self.train_path = f"{data_dir}/data/shakespeare/train"
        self.test_path = f"{data_dir}/data/shakespeare/test"

    def read_data(self):
        """
        Parses data in given train and test data directories

        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users

        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        """
        clients = list()
        groups = list()
        train_data = dict()
        test_data = dict()

        train_files = os.listdir(self.train_path)
        train_files = [f for f in train_files if f.endswith('.json')]
        for f in train_files:
            file_path = os.path.join(self.train_path, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])

        test_files = os.listdir(self.test_path)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(self.test_path, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])

        clients = list(sorted(train_data.keys()))

        return clients, groups, train_data, test_data

    @staticmethod
    def process_x(raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        return x_batch

    @staticmethod
    def process_y(raw_y_batch):
        y_batch = [letter_to_index(c) for c in raw_y_batch]
        return y_batch

    @classmethod
    def batch_data(cls, data, batch_size):
        """
        data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
        returns x, y, which are both numpy array of length: batch_size
        """
        data_x = data['x']
        data_y = data['y']

        # randomly shuffle data
        np.random.seed(100)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

        # loop through mini-batches
        batch_data = list()
        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i + batch_size]
            batched_y = data_y[i:i + batch_size]
            batched_x = torch.from_numpy(np.asarray(cls.process_x(batched_x)))
            batched_y = torch.from_numpy(np.asarray(cls.process_y(batched_y)))
            batch_data.append((batched_x, batched_y))
        return batch_data

    def load_partition_data(self):
        users, groups, train_data, test_data = self.read_data()

        if len(groups) == 0:
            groups = [None] * len(users)
        train_data_num = 0
        test_data_num = 0
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        train_data_local_num_dict = dict()
        train_data_global = list()
        test_data_global = list()
        client_idx = 0
        for u, g in zip(users, groups):
            user_train_data_num = len(train_data[u]['x'])
            user_test_data_num = len(test_data[u]['x'])
            train_data_num += user_train_data_num
            test_data_num += user_test_data_num
            train_data_local_num_dict[client_idx] = user_train_data_num

            # transform to batches
            train_batch = self.batch_data(train_data[u], self.train_bs)
            test_batch = self.batch_data(test_data[u], self.test_bs)

            # index using client index
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
            train_data_global += train_batch
            test_data_global += test_batch
            client_idx += 1
        client_num = client_idx
        output_dim = VOCAB_SIZE

        return LocalDataset(client_num=client_num, train_data_num=train_data_num, test_data_num=test_data_num,
                            train_data_global=train_data_global, test_data_global=test_data_global,
                            train_data_local_num_dict=train_data_local_num_dict,
                            train_data_local_dict=train_data_local_dict, test_data_local_dict=test_data_local_dict,
                            output_len=output_dim)

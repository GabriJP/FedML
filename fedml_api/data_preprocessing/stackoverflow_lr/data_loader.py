import logging
import os
import pickle

import torch.utils.data as data
import tqdm

from . import utils
from .dataset import StackOverflowDataset
from ..base import DataLoader, FederatedDataset, DistributedFederatedDataset

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 342477
DEFAULT_TEST_CLIENTS_NUM = 204088
DEFAULT_BATCH_SIZE = 100
DEFAULT_TRAIN_FILE = 'stackoverflow_train.h5'
DEFAULT_TEST_FILE = 'stackoverflow_test.h5'

# cache
DEFAULT_CACHE_FILE = 'stackoverflow_lr.pkl'


class StackOverflowLRDataLoader(DataLoader):
    def get_dataloader(self, client_idx=None):
        if client_idx is None:
            train_dl = data.DataLoader(data.ConcatDataset(
                StackOverflowDataset(
                    os.path.join(self.data_dir, DEFAULT_TRAIN_FILE), client_idx,
                    "train", dict(
                        input=lambda x: utils.preprocess_input(x, self.data_dir),
                        target=lambda y: utils.preprocess_target(y, self.data_dir))) for client_idx in
                range(DEFAULT_TRAIN_CLIENTS_NUM)),
                batch_size=self.train_bs,
                shuffle=True)

            test_dl = data.DataLoader(data.ConcatDataset(
                StackOverflowDataset(
                    os.path.join(self.data_dir, DEFAULT_TEST_FILE), client_idx, "test",
                    dict(
                        input=lambda x: utils.preprocess_input(x, self.data_dir),
                        target=lambda y: utils.preprocess_target(y, self.data_dir))) for client_idx in
                range(DEFAULT_TEST_CLIENTS_NUM)),
                batch_size=self.test_bs,
                shuffle=True)
            return train_dl, test_dl

        train_ds = StackOverflowDataset(
            os.path.join(self.data_dir, DEFAULT_TRAIN_FILE), client_idx, "train",
            dict(input=lambda x: utils.preprocess_input(x, self.data_dir),
                 target=lambda y: utils.preprocess_target(y, self.data_dir)))
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)

        if client_idx >= DEFAULT_TEST_CLIENTS_NUM:
            test_dl = None
        else:
            test_ds = StackOverflowDataset(
                os.path.join(self.data_dir, DEFAULT_TEST_FILE), client_idx, "test",
                dict(
                    input=lambda x: utils.preprocess_input(x, self.data_dir),
                    target=lambda y: utils.preprocess_target(y, self.data_dir)))
            test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=True, drop_last=False)

        return train_dl, test_dl

    def load_partition_data_distributed_federated(self, process_id):
        # get global dataset
        if process_id == 0:
            train_data_global, test_data_global = self.get_dataloader(process_id - 1)
            # train_data_num = len(train_data_global.dataset)
            # test_data_num = len(test_data_global.dataset)
            # logging.info("train_dl_global number = " + str(train_data_num))
            # logging.info("test_dl_global number = " + str(test_data_num))
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = self.get_dataloader(process_id - 1)
            local_data_num = len(train_data_local.dataset)
            # logging.info("rank = %d, local_sample_number = %d" %
            #              (process_id, local_data_num))
            train_data_global = None
            test_data_global = None
        output_dim = len(utils.get_tag_dict())
        return DistributedFederatedDataset(
            client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_global=train_data_global,
            test_data_global=test_data_global, local_data_num=local_data_num, train_data_local=train_data_local,
            test_data_local=test_data_local, output_len=output_dim)

    def load_partition_data_federated(self, batch_size=DEFAULT_BATCH_SIZE):
        logging.info("load_partition_data_federated_stackoverflow_lr START")

        cache_path = os.path.join(self.data_dir, DEFAULT_CACHE_FILE)
        if os.path.exists(cache_path):
            # load cache
            with open(cache_path, 'rb') as cache_file:
                cache_data = pickle.load(cache_file)
                train_data_num = cache_data['train_data_num']
                test_data_num = cache_data['test_data_num']
                train_data_global = cache_data['train_data_global']
                test_data_global = cache_data['test_data_global']
                data_local_num_dict = cache_data['data_local_num_dict']
                train_data_local_dict = cache_data['train_data_local_dict']
                test_data_local_dict = cache_data['test_data_local_dict']
                output_dim = cache_data['output_dim']

        else:
            # get local dataset
            data_local_num_dict = dict()
            train_data_local_dict = dict()
            test_data_local_dict = dict()

            for client_idx in tqdm.tqdm(range(DEFAULT_TRAIN_CLIENTS_NUM)):
                train_data_local, test_data_local = self.get_dataloader(client_idx)
                local_data_num = len(train_data_local.dataset)
                data_local_num_dict[client_idx] = local_data_num
                # logging.info("client_idx = %d, local_sample_number = %d" %
                #              (client_idx, local_data_num if test_data_local==None else len(test_data_local.dataset)))
                # logging.info(
                #     "client_idx = %d, batch_num_train_local = %d"
                # % (client_idx, len(train_data_local)))
                train_data_local_dict[client_idx] = train_data_local
                test_data_local_dict[client_idx] = test_data_local

            train_data_global = data.DataLoader(data.ConcatDataset(
                list(dl.dataset for dl in list(train_data_local_dict.values()))),
                batch_size=batch_size,
                shuffle=True)
            train_data_num = len(train_data_global.dataset)

            test_data_global = data.DataLoader(data.ConcatDataset(
                list(dl.dataset for dl in list(test_data_local_dict.values())
                     if dl is not None)),
                batch_size=batch_size,
                shuffle=True)
            test_data_num = len(test_data_global.dataset)

            output_dim = len(utils.get_tag_dict(self.data_dir))

            # save cache
            with open(cache_path, 'wb') as cache_file:
                cache_data = dict()
                cache_data['train_data_num'] = train_data_num
                cache_data['test_data_num'] = test_data_num
                cache_data['train_data_global'] = train_data_global
                cache_data['test_data_global'] = test_data_global
                cache_data['data_local_num_dict'] = data_local_num_dict
                cache_data['train_data_local_dict'] = train_data_local_dict
                cache_data['test_data_local_dict'] = test_data_local_dict
                cache_data['output_dim'] = output_dim
                pickle.dump(cache_data, cache_file)

        return FederatedDataset(
            client_num=DEFAULT_TRAIN_CLIENTS_NUM, train_data_num=train_data_num, test_data_num=test_data_num,
            train_data_global=train_data_global, test_data_global=test_data_global,
            local_data_num_dict=data_local_num_dict, train_data_local_dict=train_data_local_dict,
            test_data_local_dict=test_data_local_dict, output_len=output_dim)


if __name__ == "__main__":
    # load_partition_data_federated_stackoverflow(None, None, 100, 128)
    dl = StackOverflowLRDataLoader(None, 128, 128)
    ds = dl.load_partition_data_distributed_federated(2)
    ds2 = dl.load_partition_data_federated(128)
    print(ds.train_data_local, ds.test_data_local)

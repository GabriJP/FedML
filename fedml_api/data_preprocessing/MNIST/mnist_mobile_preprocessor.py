import json
import os
import shutil
import sys
from typing import List

import click
import numpy as np
from attr import attrs
from dataclasses import field

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


def read_data(train_data_dir, test_data_dir, client_num_per_round, comm_round):
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
    train_num_samples = list()
    test_num_samples = list()
    train_data = dict()
    test_data = dict()

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    # print(train_files)
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        train_num_samples.extend(cdata['num_samples'])
        train_data.update(cdata['user_data'])
        # print(cdata['user_data'])
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_num_samples.extend(cdata['num_samples'])
        test_data.update(cdata['user_data'])

    @attrs(auto_attribs=True)
    class Args:
        client_num_per_round: int
        comm_round: int
        client_id: int
        client_sample_list: List = field(default_factory=list)

    client_list = [Args(client_number, client_num_per_round, comm_round) for client_number in
                   range(client_num_per_round)]
    return clients, train_num_samples, test_num_samples, train_data, test_data, client_list


def client_sampling(round_idx, client_num_in_total, client_num_per_round):
    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    print(f"client_indexes = {client_indexes}")
    return client_indexes


@click.command()
@click.option('--client_num_per_round', type=int, default=3, help='number of workers')
@click.option('--comm_round', type=int, default=10, help='how many round of communications we should use')
def main(client_num_per_round, comm_round):
    train_path = "../../FedML/data/MNIST/train"
    test_path = "../../FedML/data/MNIST/test"
    new_train = dict()
    new_test = dict()
    users, train_num_samples, test_num_samples, train_data, test_data, client_list = read_data(
        train_path, test_path, client_num_per_round, comm_round)

    for round_idx in range(client_list[0].comm_round):
        sample_list = client_sampling(round_idx, 1000, client_num_per_round)
        for worker in client_list:
            worker.client_sample_list.append(sample_list[worker.client_id])
    os.mkdir('MNIST_mobile_zip')
    for worker in client_list:
        filetrain = f'MNIST_mobile/{worker.client_id}/train/train.json'
        os.makedirs(os.path.dirname(filetrain), mode=0o770, exist_ok=True)
        filetest = f'MNIST_mobile/{worker.client_id}/test/test.json'
        os.makedirs(os.path.dirname(filetest), mode=0o770, exist_ok=True)
        new_train['num_samples'] = [train_num_samples[i] for i in tuple(worker.client_sample_list)]
        new_train['users'] = [users[i] for i in tuple(worker.client_sample_list)]
        client_sample = new_train['users']
        new_train['user_data'] = {x: train_data[x] for x in client_sample}
        with open(filetrain, 'w') as fp:
            json.dump(new_train, fp)
        new_test['num_samples'] = [test_num_samples[i] for i in tuple(worker.client_sample_list)]
        new_test['users'] = [users[i] for i in tuple(worker.client_sample_list)]
        client_sample = new_test['users']
        new_test['user_data'] = {x: test_data[x] for x in client_sample}
        with open(filetest, 'w') as ff:
            json.dump(new_test, ff)
        shutil.make_archive(f'MNIST_mobile/{worker.client_id}', 'zip', 'MNIST_mobile', str(worker.client_id))
        shutil.move(f'MNIST_mobile/{worker.client_id}.zip', 'MNIST_mobile_zip')


if __name__ == '__main__':
    main()

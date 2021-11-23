import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from fedml_core import RunConfig
from .utils import transform_list_to_tensor
from ...data_preprocessing import LocalDataset


class FedAVGAggregator:
    def __init__(self, device, worker_num, config: RunConfig, dataset: LocalDataset, model_trainer):
        self.device = device
        self.worker_num = worker_num
        self.config = config
        self.dataset = dataset
        self.trainer = model_trainer

        self.val_global = self._generate_validation_set()

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.config.client_num_in_total):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info(f"add_model. index = {index:d}")
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug(f"worker_num = {self.worker_num}")
        if not all(self.flag_client_model_uploaded_dict[idx] for idx in range(self.worker_num)):
            return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.config.is_mobile:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info(f"len of self.model_dict[idx] = {len(self.model_dict)}")

        # logging.info("################aggregate: %d" % len(model_list))
        num0, averaged_params = model_list[0]
        for k in averaged_params.keys():
            for i, model in enumerate(model_list):
                local_sample_number, local_model_params = model
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info(f"aggregate time cost: {end_time - start_time:d}")
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info(f"client_indexes = {client_indexes}")
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.config.dataset_name.startswith("stackoverflow"):
            test_data_num = len(self.dataset.test_data_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.dataset.test_data_global.dataset, sample_indices)
            sample_testset = torch.utils.data.LocalDataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset

        return self.dataset.test_data_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.dataset.train_data_local_dict, self.dataset.test_data_local_dict,
                                           self.device):
            return

        if round_idx % self.config.frequency_of_the_test == 0 or round_idx == self.config.comm_round - 1:
            logging.info(f"################test_on_server_for_all_clients : {round_idx}")
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.config.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.dataset.train_data_local_dict[client_idx], self.device)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], \
                                                                  metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.config.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.config.comm_round - 1:
                metrics = self.trainer.test(self.dataset.test_data_global, self.device)
            else:
                metrics = self.trainer.test(self.val_global, self.device)

            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

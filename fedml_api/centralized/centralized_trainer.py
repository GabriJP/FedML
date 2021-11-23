import copy
import logging

import torch
import wandb
from torch import nn

from FedML.fedml_api.data_preprocessing import LocalDataset
from fedml_core import CentralizedRunConfig


class CentralizedTrainer:
    """
    This class is used to train federated non-IID dataset in a centralized way
    """

    def __init__(self, dataset: LocalDataset, model, device, config: CentralizedRunConfig):
        self.train_global = dataset.train_data_global
        self.test_global = dataset.test_data_global
        self.train_data_num_in_total = dataset.train_data_num
        self.test_data_num_in_total = dataset.test_data_num
        self.train_data_local_num_dict = dataset.train_data_local_num_dict
        self.train_data_local_dict = dataset.train_data_local_dict
        self.test_data_local_dict = dataset.test_data_local_dict

        self.model = model
        self.device = device
        self.config = config

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if config.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        elif config.client_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=config.lr, weight_decay=config.wd, amsgrad=True)
        else:
            raise ValueError(f'Unknown optimizer {config.client_optimizer}')

    def train(self):
        for epoch in range(self.config.epochs):
            if self.config.data_parallel:
                self.train_global.sampler.set_epoch(epoch)
            self.train_impl(epoch)
            self.eval_impl(epoch)

    def train_impl(self, epoch_idx):
        self.model.train()
        for batch_idx, (x, labels) in enumerate(self.train_global):
            # logging.info(images.shape)
            x, labels = x.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            logging.info(f'Local Training Epoch: {epoch_idx} {batch_idx}-th iters\t Loss: {loss.item():.6f}')

    def eval_impl(self, epoch_idx):
        # train
        if epoch_idx % self.config.frequency_of_train_acc_report == 0:
            self.test_on_all_clients(b_is_train=True, epoch_idx=epoch_idx)

        # test
        if epoch_idx % self.config.frequency_of_train_acc_report == 0:
            self.test_on_all_clients(b_is_train=False, epoch_idx=epoch_idx)

    def test_on_all_clients(self, b_is_train, epoch_idx):
        self.model.eval()
        metrics = dict(test_correct=0, test_loss=0, test_precision=0, test_recall=0, test_total=0)

        test_data = self.train_global if b_is_train else self.test_global

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)

                if self.config.dataset_name == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        if self.config.rank == 0:
            self.save_log(b_is_train=b_is_train, metrics=metrics, epoch_idx=epoch_idx)

    def save_log(self, b_is_train, metrics, epoch_idx):
        prefix = 'Train' if b_is_train else 'Test'

        all_metrics = dict(num_samples=[], num_correct=[], precisions=[], recalls=[], losses=[], )

        all_metrics['num_samples'].append(copy.deepcopy(metrics['test_total']))
        all_metrics['num_correct'].append(copy.deepcopy(metrics['test_correct']))
        all_metrics['losses'].append(copy.deepcopy(metrics['test_loss']))

        if self.config.dataset_name == "stackoverflow_lr":
            all_metrics['precisions'].append(copy.deepcopy(metrics['test_precision']))
            all_metrics['recalls'].append(copy.deepcopy(metrics['test_recall']))

        # performance on all clients
        acc = sum(all_metrics['num_correct']) / sum(all_metrics['num_samples'])
        loss = sum(all_metrics['losses']) / sum(all_metrics['num_samples'])
        precision = sum(all_metrics['precisions']) / sum(all_metrics['num_samples'])
        recall = sum(all_metrics['recalls']) / sum(all_metrics['num_samples'])

        if self.config.dataset_name == "stackoverflow_lr":
            stats = {f'{prefix}_acc': acc, f'{prefix}_precision': precision, f'{prefix}_recall': recall,
                     f'{prefix}_loss': loss}
            wandb.log({f"{prefix}/Acc": acc, "epoch": epoch_idx})
            wandb.log({f"{prefix}/Pre": precision, "epoch": epoch_idx})
            wandb.log({f"{prefix}/Rec": recall, "epoch": epoch_idx})
            wandb.log({f"{prefix}/Loss": loss, "epoch": epoch_idx})
            logging.info(stats)
        else:
            stats = {f'{prefix}_acc': acc, prefix + '_loss': loss}
            wandb.log({f"{prefix}/Acc": acc, "epoch": epoch_idx})
            wandb.log({f"{prefix}/Loss": loss, "epoch": epoch_idx})
            logging.info(stats)

        stats = {f'{prefix}_acc': acc, f'{prefix}_loss': loss}
        wandb.log({f"{prefix}/Acc": acc, "epoch": epoch_idx})
        wandb.log({f"{prefix}/Loss": loss, "epoch": epoch_idx})
        logging.info(stats)

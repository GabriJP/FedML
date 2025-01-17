import logging

import torch.nn as nn
import torch.optim as optim


class SplitNN_server():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]
        self.MAX_RANK = args["max_rank"]
        self.init_params()

    def init_params(self):
        self.epoch = 0
        self.log_step = 50
        self.active_node = 1
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts, labels):
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        if self.step % self.log_step == 0 and self.phase == "train":
            acc = self.correct / self.total
            logging.info(f"phase={'train'} acc={acc} loss={self.loss.item()} epoch={self.epoch} and step={self.step}")
        if self.phase == "validation":
            self.val_loss += self.loss.item()
        self.step += 1

    def backward_pass(self):
        self.loss.backward()
        self.optimizer.step()
        return self.acts.grad

    def validation_over(self):
        # not precise estimation of validation loss 
        self.val_loss /= self.step
        acc = self.correct / self.total
        logging.info(f"phase={self.phase} acc={acc} loss={self.val_loss} epoch={self.epoch} and step={self.step}")

        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()
        logging.info(f"current active client is {self.active_node}")

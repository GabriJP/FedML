import logging

import numpy as np
import torch
from matplotlib.pyplot import imsave
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, netd, netg):
        self.netg = netg
        self.netd = netd
        super(MyModelTrainer, self).__init__(model=None, args=None)

    def get_model_params(self):
        weights_d = self.netd.cpu().state_dict()
        weights_g = self.netg.cpu().state_dict()
        weights = {'netg': weights_g, 'netd': weights_d}
        return weights

    def set_model_params(self, model_parameters):
        self.netg.load_state_dict(model_parameters['netg'])
        self.netd.load_state_dict(model_parameters['netd'])

    def train(self, train_data, device, args):
        netg = self.netg
        netd = self.netd

        netg.to(device)
        netg.train()
        netd.to(device)
        netd.train()

        criterion = nn.BCELoss()
        optimizer_g = torch.optim.Adam(netg.parameters(), lr=args.lr)
        optimizer_d = torch.optim.Adam(netd.parameters(), lr=args.lr)

        epoch_d_loss = []
        epoch_g_loss = []

        for epoch in range(args.epochs):
            batch_d_loss = []
            batch_g_loss = []
            for batch_idx, (x, _) in enumerate(train_data):
                # logging.info(batch_idx)
                # logging.info(x.shape)
                x = x.to(device)
                real_labels = torch.ones(x.size(0), 1).to(device)
                fake_labels = torch.zeros(x.size(0), 1).to(device)
                optimizer_d.zero_grad()
                d_real_loss = criterion(netd(x), real_labels)
                noise = torch.randn(x.size(0), 100).to(device)
                d_fake_loss = criterion(netd(netg(noise)), fake_labels)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizer_d.step()

                noise = torch.randn(x.size(0), 100).to(device)
                optimizer_g.zero_grad()
                g_loss = criterion(netd(netg(noise)), real_labels)
                g_loss.backward()
                optimizer_g.step()

                batch_d_loss.append(d_loss.item())
                batch_g_loss.append(g_loss.item())
            if len(batch_g_loss) > 0:
                epoch_g_loss.append(sum(batch_g_loss) / len(batch_g_loss))
                epoch_d_loss.append(sum(batch_d_loss) / len(batch_d_loss))
                logging.info(f'(Trainer_ID {self.id}. Local Generator Training Epoch: {epoch} \t'
                             f'Loss: {sum(epoch_g_loss) / len(epoch_g_loss):.6f}')
                logging.info(f'(Trainer_ID {self.id}. Local Discriminator Training Epoch: {epoch} \t'
                             f'Loss: {sum(epoch_d_loss) / len(epoch_d_loss):.6f}')
            netg.eval()
            z = torch.randn(100, 100).to(device)
            y_hat = netg(z).view(100, 28, 28)  # (100, 28, 28)
            result = y_hat.cpu().main_data.numpy()
            img = np.zeros([280, 280])
            for j in range(10):
                img[j * 28:(j + 1) * 28] = np.concatenate([x for x in result[j * 10:(j + 1) * 10]], axis=-1)

            imsave(f'samples/{self.id}_{epoch}.jpg', img, cmap='gray')
            netg.train()

    def test(self, test_data, device, args):
        pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

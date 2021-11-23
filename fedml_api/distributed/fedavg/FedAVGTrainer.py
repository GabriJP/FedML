from fedml_core import ModelTrainer
from .utils import transform_tensor_to_list
from ...data_preprocessing import LocalDataset


class FedAVGTrainer:
    def __init__(self, client_index, dataset: LocalDataset, device, args, model_trainer: ModelTrainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.dataset = dataset
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

        self.update_dataset(client_index)

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.dataset.train_data_local_dict[client_index]
        self.local_sample_number = self.dataset.train_data_local_num_dict[client_index]
        self.test_local = self.dataset.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                       test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample

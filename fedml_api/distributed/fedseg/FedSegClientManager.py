import logging
import os
import sys

from .message_define import MyMessage
from .utils import transform_list_to_tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message


class FedSegClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info(f'Client {client_index} received global model params from central server')

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, train_evaluation_metrics,
                             test_evaluation_metrics):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_EVALUATION_METRICS, train_evaluation_metrics)
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_EVALUATION_METRICS, test_evaluation_metrics)
        self.send_message(message)

    def __train(self):

        train_evaluation_metrics = test_evaluation_metrics = None
        logging.info(f"####### Testing Global Params ########### round_id = {self.round_idx}")
        train_evaluation_metrics, test_evaluation_metrics = self.trainer.test()
        logging.info(f"####### Training ########### round_id = {self.round_idx}")
        weights, local_sample_num = self.trainer.train()
        self.send_model_to_server(0, weights, local_sample_num, train_evaluation_metrics, test_evaluation_metrics)

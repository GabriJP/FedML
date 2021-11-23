import json
import logging
import sys

from torch import Tensor


class Message:
    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    MSG_ARG_KEY_MODEL_PARAMS = "model_params"

    def __init__(self, msg_type=0, sender_id=0, receiver_id=0):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = {
            Message.MSG_ARG_KEY_TYPE: msg_type,
            Message.MSG_ARG_KEY_SENDER: sender_id,
            Message.MSG_ARG_KEY_RECEIVER: receiver_id,
        }

    def init(self, msg_params):
        self.msg_params = msg_params

    def init_from_json_string(self, json_string):
        self.msg_params = json.loads(json_string)
        self.msg_type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]
        # print("msg_params = " + str(self.msg_params))

    def get_sender_id(self):
        return self.sender_id

    def get_receiver_id(self):
        return self.receiver_id

    @staticmethod
    def _tensor_to_numpy(tensor: Tensor):
        return tensor.cpu().numpy().tolist()

    def add_params(self, key, value):
        if isinstance(value, Tensor):
            value = self._tensor_to_numpy(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, Tensor):
                    continue
                value[k] = self._tensor_to_numpy(v)
        self.msg_params[key] = value

    def get_params(self):
        return self.msg_params

    def add(self, key, value):
        self.msg_params[key] = value

    def get(self, key):
        return self.msg_params[key]

    def get_type(self):
        return self.msg_params[Message.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self.msg_params

    def to_json(self):
        json_string = json.dumps(self.msg_params)
        logging.info(f"json string size = {sys.getsizeof(json_string)}")
        return json_string

    def get_content(self):
        print_dict = self.msg_params.copy()
        msg_str = f"{self.msg_params[Message.MSG_ARG_KEY_TYPE]}: {print_dict}"
        return msg_str

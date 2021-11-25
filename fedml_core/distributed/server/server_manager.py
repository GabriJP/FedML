import logging

from mpi4py import MPI

from ..communication.gRPC.grpc_comm_manager import GRPCCommManager
from ..communication.mpi.com_manager import MpiCommunicationManager
from ..communication.mqtt.mqtt_comm_manager import MqttCommManager
from ..communication.observer import Observer
from ..communication.trpc.trpc_comm_manager import TRPCCommManager


class ServerManager(Observer):
    def __init__(self, conf, comm=None, rank=0, size=0, grpc_ipconfig_path=None, trpc_master_config_path=None):
        self.size = size
        self.rank = rank
        self.backend = conf.communication_method

        if self.backend == "MQTT":
            self.com_manager = MqttCommManager(
                conf.communication_host, conf.communication_port, client_id=rank, client_num=size - 1)
        elif self.backend == "GRPC":
            self.com_manager = GRPCCommManager(conf.communication_host, conf.communication_port + rank,
                                               ip_config_path=grpc_ipconfig_path, client_id=rank, client_num=size - 1)
        elif self.backend == "TRPC":
            self.com_manager = TRPCCommManager(trpc_master_config_path, process_id=rank, world_size=size)
        else:
            self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="server")
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()
        print("done running")

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        # logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
        #     self.rank, str(msg_type), str(msg_params.get_content())))
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        self.com_manager.send_message(message)

    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish server")
        if self.backend == "MPI":
            MPI.COMM_WORLD.Abort()
        else:
            self.com_manager.stop_receive_message()

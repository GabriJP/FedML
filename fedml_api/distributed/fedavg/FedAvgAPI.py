from mpi4py import MPI

from fedml_core import RunConfig
from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager
from ...data_preprocessing import LocalDataset
from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


def fed_ml_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def fed_ml_fed_avg_distributed(comm, process_id, worker_number, device, model, config: RunConfig, dataset: LocalDataset,
                               model_trainer=None, preprocessed_sampling_lists=None):
    if process_id == 0:
        init_server(comm, process_id, worker_number, device, model, model_trainer, dataset, config,
                    preprocessed_sampling_lists)
    else:
        init_client(comm, process_id, worker_number, device, model, model_trainer, dataset, config)


def init_server(comm, process_id, worker_number, device, model, model_trainer, dataset: LocalDataset, config: RunConfig,
                preprocessed_sampling_lists=None):
    if model_trainer is None:
        if config.dataset_name == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model, config)
        elif config.dataset_name in {"fed_shakespeare", "stackoverflow_nwp"}:
            model_trainer = MyModelTrainerNWP(model, config)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model, config)

    model_trainer.set_id(-1)

    # aggregator
    worker_num = worker_number - 1
    aggregator = FedAVGAggregator(device, worker_num, config, dataset, model_trainer)

    # start the distributed training
    if preprocessed_sampling_lists is None:
        server_manager = FedAVGServerManager(aggregator, comm, process_id, worker_number, config.backend)
    else:
        server_manager = FedAVGServerManager(aggregator, comm, process_id, worker_number, config.backend,
                                             is_preprocessed=True,
                                             preprocessed_client_lists=preprocessed_sampling_lists)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(comm, process_id, worker_number, device, model, model_trainer, dataset: LocalDataset,
                config: RunConfig):
    client_index = process_id - 1
    if model_trainer is None:
        if config.dataset_name == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model, config)
        elif config.dataset_name in {"fed_shakespeare", "stackoverflow_nwp"}:
            model_trainer = MyModelTrainerNWP(model, config)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model, config)
    model_trainer.set_id(client_index)
    backend = config.backend
    trainer = FedAVGTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                            train_data_num, device, args, model_trainer)
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()

from __future__ import annotations

import logging
import os
import random
import socket
import sys
from pathlib import Path

import click
import numpy as np
import psutil
import setproctitle
import torch
import wandb

from fedml_api.distributed.fedavg.FedAvgAPI import fed_ml_init, fed_ml_fed_avg_distributed
from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fedml_core import RunConfig
from fedml_experiments.base import MODELS, DATASETS, PARTITION_METHODS, CLIENT_OPTIMIZERS, BACKENDS, load_data, \
    create_model


@click.command()
@click.option("--model", type=click.Choice(MODELS), default="mobilenet", help="Neural network used in training")
@click.option("--dataset", type=click.Choice(DATASETS), default="cifar10", help="Dataset used for training")
@click.option("--data_dir", type=click.Path(file_okay=False, exists=True, resolve_path=True, path_type=Path),
              default=Path("./../../../data/cifar10"), help="Data directory")
@click.option("--partition_method", type=click.Choice(PARTITION_METHODS), default="hetero",
              help="How to partition the dataset on local workers")
@click.option("--partition_alpha", type=float, default=0.5, help="Partition alpha")
@click.option("--client_num_in_total", type=int, default=1000, help="Number of workers in a distributed cluster")
@click.option("--client_num_per_round", type=int, default=4, help="Number of workers selected per round")
@click.option("--batch_size", type=int, default=64, help="Input batch size for training")
@click.option("--client_optimizer", type=click.Choice(CLIENT_OPTIMIZERS), default='adam', help="SGD with momentum")
@click.option("--backend", type=click.Choice(BACKENDS), default='MPI', help="Backend for Server and Client")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--wd", type=float, default=0.0001, help="Weight decay parameter")
@click.option("--epochs", type=int, default=5, help="How many epochs will be trained locally")
@click.option("--comm_round", type=int, default=10, help="How many round of communications we shoud use")
@click.option("--is_mobile", type=bool, default=False, is_flag=True, flag_value=True,
              help="Whether the program is running on the FedML-Mobile server side")
@click.option("--frequency_of_the_test", type=int, default=1, help="The frequency of the algorithms")
@click.option("--gpu_server_num", type=int, default=1, help="gpu_server_num")
@click.option("--gpu_num_per_server", type=int, default=1, help="gpu_num_per_server")
@click.option("--gpu_mapping_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
              default=Path("gpu_mapping.yaml"), help="The gpu utilization file for servers and clients. If there is no "
                                                     "gpu_util_file, gpu will not be used.")
@click.option("--gpu_mapping_key", type=str, default="mapping_default", help="The key in gpu utilization file")
@click.option("--grpc_ipconfig_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
              default=Path("grpc_ipconfig.csv"), help="Config table containing ipv4 address of grpc server")
@click.option("--trpc_master_config_path",
              type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
              default=Path("trpc_master_config.csv"),
              help="Config indicating ip address and port of the master (rank 0) node")
@click.option("--ci", type=bool, default=False, is_flag=True, flag_value=True, help="CI")
def main(model: str, dataset_name: str, data_dir: Path, partition_method: str, partition_alpha: float,
         client_num_in_total: int, client_num_per_round: int, batch_size: int, client_optimizer: str, backend: str,
         lr: float, wd: float, epochs: int, comm_round: int, is_mobile: bool, frequency_of_the_test: int,
         gpu_server_num: int, gpu_num_per_server: int, gpu_mapping_file: Path, gpu_mapping_key: str,
         grpc_ipconfig_path: Path, trpc_master_config_path: Path, ci: bool):
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = fed_ml_init()

    config = RunConfig(dataset_name, partition_alpha, client_num_in_total, client_num_per_round, backend, is_mobile,
                       gpu_server_num, gpu_num_per_server, client_optimizer, lr, wd, batch_size, epochs, comm_round,
                       frequency_of_the_test, ci)

    # customize the process name
    str_process_name = f"FedAvg (distributed):{process_id}"
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"{process_id} - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
    hostname = socket.gethostname()
    logging.info(
        f"#############process ID = {process_id}, host name = {hostname}########, process ID = {os.getpid()}, "
        f"process Name = {psutil.Process(os.getpid())}"
    )

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml",
            name=f"FedAVG(d){partition_method}r{comm_round}-e{epochs}-lr{lr}",
            config=dict(
                model=model, dataset_name=dataset_name, data_dir=data_dir, partition_method=partition_method,
                partition_alpha=partition_alpha, client_num_in_total=client_num_in_total,
                client_num_per_round=client_num_per_round, batch_size=batch_size, client_optimizer=client_optimizer,
                backend=backend, lr=lr, wd=wd, epochs=epochs, comm_round=comm_round, is_mobile=is_mobile,
                frequency_of_the_test=frequency_of_the_test, gpu_server_num=gpu_server_num,
                gpu_num_per_server=gpu_num_per_server, gpu_mapping_file=gpu_mapping_file,
                gpu_mapping_key=gpu_mapping_key, grpc_ipconfig_path=grpc_ipconfig_path,
                trpc_master_config_path=trpc_master_config_path, ci=ci
            ))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info(f"process_id = {process_id:d}, size = {worker_number:d}")
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, gpu_mapping_file, gpu_mapping_key
    )

    # load data
    dataset = load_data(dataset_name, data_dir, batch_size, client_num_in_total, partition_method, partition_alpha)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(model_name=model, dataset_name=dataset_name, output_dim=dataset.output_len)

    # start distributed training
    fed_ml_fed_avg_distributed(comm, process_id, worker_number, device, model, config, dataset)


if __name__ == "__main__":
    main()

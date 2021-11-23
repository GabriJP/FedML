import logging
import os
import random
import socket
from pathlib import Path

import click
import numpy as np
import psutil
import setproctitle
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

from fedml_api.centralized.centralized_trainer import CentralizedTrainer
from fedml_core import CentralizedRunConfig
from fedml_experiments.base import MODELS, DATASETS, PARTITION_METHODS, CLIENT_OPTIMIZERS, load_data, create_model


@click.command()
@click.option("--model", type=click.Choice(MODELS), default="mobilenet", help="Neural network used in training")
@click.option('--data_parallel', type=bool, default=False, is_flag=True, flag_value=True,
              help='if distributed training')
@click.option("--dataset", "dataset_name", type=click.Choice(DATASETS), default="cifar10",
              help="Dataset used for training")
@click.option("--data_dir", type=click.Path(file_okay=False, exists=True, resolve_path=True, path_type=Path),
              default=Path("./../../../data/cifar10"), help="Data directory")
@click.option("--partition_method", type=click.Choice(PARTITION_METHODS), default="hetero",
              help="How to partition the dataset on local workers")
@click.option("--partition_alpha", type=float, default=0.5, help="Partition alpha")
@click.option("--client_num_in_total", type=int, default=1000, help="Number of workers in a distributed cluster")
@click.option("--client_num_per_round", type=int, default=4, help="Number of workers selected per round")
@click.option("--batch_size", type=int, default=64, help="Input batch size for training")
@click.option("--client_optimizer", type=click.Choice(CLIENT_OPTIMIZERS), default='adam', help="SGD with momentum")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--wd", type=float, default=0.0001, help="Weight decay parameter")
@click.option("--epochs", type=int, default=5, help="How many epochs will be trained locally")
@click.option("--comm_round", type=int, default=10, help="How many round of communications we shoud use")
@click.option("--is_mobile", type=bool, default=False, is_flag=True, flag_value=True,
              help="Whether the program is running on the FedML-Mobile server side")
@click.option("--frequency_of_train_acc_report", type=int, default=10, help="The frequency of training accuracy report")
@click.option("--gpu_server_num", type=int, default=1, help="gpu_server_num")
@click.option("--gpu_num_per_server", type=int, default=1, help="gpu_num_per_server")
@click.option("--ci", type=bool, default=False, is_flag=True, flag_value=True, help="CI")
@click.option('--gpu', type=int, default=0, help='gpu')
@click.option('--gpu_util', type=str, default='0', help='gpu utils')
def main(model, data_parallel, dataset_name, data_dir, partition_method, partition_alpha, client_num_in_total,
         client_num_per_round, batch_size, client_optimizer, lr, wd, epochs, comm_round, is_mobile,
         frequency_of_train_acc_report, gpu_server_num, gpu_num_per_server, ci, gpu,
         gpu_util):
    world_size = len(gpu_util.split(','))
    process_id = 0

    if data_parallel:
        # torch.distributed.init_process_group(
        #         backend="nccl", world_size=args.world_size, init_method="env://")
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank = torch.distributed.get_rank()
        gpu_util = gpu_util.split(',')
        gpu_util = [int(item.strip()) for item in gpu_util]
        # device = torch.device("cuda", local_rank)
        torch.cuda.set_device(gpu_util[rank])
        process_id = rank
    else:
        rank = 0

    config = CentralizedRunConfig(
        dataset_name, partition_alpha, client_num_in_total, client_num_per_round, None, is_mobile, gpu_server_num,
        gpu_num_per_server, client_optimizer, lr, wd, batch_size, epochs, comm_round, None, ci, data_parallel,
        frequency_of_train_acc_report, rank)

    # customize the process name
    str_process_name = f"Fedml (single):{process_id}"
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        # logging.basicConfig(level=logging.DEBUG,
                        format=f'{process_id} - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info(f"#############process ID = {process_id}, host name = {hostname}########, process ID = {os.getpid()}, "
                 f"process Name = {psutil.Process(os.getpid())}")

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml",
            name=f"Fedml (central){partition_method}r{comm_round}-e{epochs}-lr{lr}",
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info(f"process_id = {process_id:d}, size = {world_size:d}")

    # load data
    dataset = load_data(dataset_name, data_dir, batch_size, client_num_in_total, partition_method, partition_alpha)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(model_name=model, dataset_name=dataset_name, output_dim=dataset.output_len)

    if data_parallel:
        device = torch.device(f"cuda:{gpu_util[rank]}")
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[gpu_util[rank]], output_device=gpu_util[rank])
    else:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # start "federated averaging (FedAvg)"
    single_trainer = CentralizedTrainer(dataset, model, device, config)
    single_trainer.train()


if __name__ == "__main__":
    main()

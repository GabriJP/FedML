import argparse
import logging
import os
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.distributed.decentralized_framework.algorithm_api import FedML_Decentralized_Demo_distributed
from fedml_api.distributed.fedavg.FedAvgAPI import fed_ml_init


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--client_number', type=int, default=16, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = fed_ml_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the process name
    str_process_name = "Federated Learning:" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(worker_number)

    FedML_Decentralized_Demo_distributed(process_id, worker_number, comm, args)

import logging
import socket

import torch
import yaml


def mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, gpu_util_file, gpu_util_key):
    if gpu_util_file is None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ################## You do not indicate gpu_util_file, will use CPU training  #################")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        with open(gpu_util_file, 'r') as f:
            gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
            # gpu_util_num_process = 'gpu_util_' + str(worker_number)
            # gpu_util = gpu_util_yaml[gpu_util_num_process]
            gpu_util = gpu_util_yaml[gpu_util_key]
            logging.info(f"gpu_util = {gpu_util}")
            gpu_util_map = {}
            i = 0
            for host, gpus_util_map_host in gpu_util.items():
                for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                    for _ in range(num_process_on_gpu):
                        gpu_util_map[i] = (host, gpu_j)
                        i += 1
            logging.info(f"Process {process_id:d} running on host: {gpu_util_map[process_id][0]}, "
                         f"gethostname: {socket.gethostname()}, local_gpu_id: {gpu_util_map[process_id][1]:d} ...")
            logging.info(f"i = {i}, worker_number = {worker_number}")
            assert i == worker_number
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_util_map[process_id][1])
        device = torch.device(f"cuda:{str(gpu_util_map[process_id][1])}" if torch.cuda.is_available() else "cpu")
        logging.info(f"process_id = {process_id}, GPU device = {device}")
        # return gpu_util_map[process_id][1]
        return device

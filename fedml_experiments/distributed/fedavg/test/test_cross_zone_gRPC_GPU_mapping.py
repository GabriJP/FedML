import logging

from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fedml_api.distributed.utils.ip_config_utils import build_ip_table

logging.basicConfig(level=logging.DEBUG,
                    format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

# test GPU device arrangement for different nodes (servers)
worker_number = 9
gpu_mapping_file = "../gpu_mapping.yaml"
gpu_mapping_key = "mapping_FedML_gRPC"

for process_id in range(worker_number):
    # this function will print the GPU assignment as a format:
    # gpu_mapping.py[line:36] INFO process_id = 8, GPU device = cuda:3
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, gpu_mapping_file,
                                                            gpu_mapping_key)

# gRPC configuration
ip_config_path = "../grpc_ipconfig.csv"
ip_config = build_ip_table(ip_config_path)
logging.info(f"ip_config = {ip_config}")

for receiver_id in range(worker_number):
    receiver_ip = ip_config[str(receiver_id)]
    gRPC_channel_url = f'{receiver_ip}:{50000 + receiver_id}'
    logging.info(f"receiver_id, gRPC_channel_url = {receiver_id}")

from .data_loader import LandmarksDataLoader

'''
    You can run with python check_download.py to check if you have all 
    data samples in federated_train.csv and test.csv.
'''

if __name__ == '__main__':
    data_dir = './cache/images'
    fed_g23k_train_map_file = './cache/datasets/mini_gld_train_split.csv'
    fed_g23k_test_map_file = './cache/datasets/mini_gld_test.csv'

    fed_g160k_train_map_file = './cache/datasets/landmarks-user-160k/federated_train.csv'
    fed_g160k_map_file = './cache/datasets/landmarks-user-160k/test.csv'

    # noinspection DuplicatedCode
    dataset_name = 'g160k'

    if dataset_name == 'g23k':
        client_number = 233
        fed_train_map_file = fed_g23k_train_map_file
        fed_test_map_file = fed_g23k_test_map_file
    elif dataset_name == 'g160k':
        client_number = 1262
        fed_train_map_file = fed_g160k_train_map_file
        fed_test_map_file = fed_g160k_map_file
    else:
        raise NotImplementedError

    dl = LandmarksDataLoader(data_dir, 10, 10, client_number)
    ds = dl.load_partition_data(fed_train_map_file, fed_test_map_file)

    print(ds.train_data_num, ds.test_data_num, ds.class_num)
    print(ds.data_local_num_dict)

    for _, (data, label) in zip(range(5), ds.train_data_global):
        print(data)
        print(label)
    print("=============================\n")

    for client_idx in range(client_number):
        for i, (data, label) in enumerate(ds.train_data_local_dict[client_idx]):
            print(f"client_idx {client_idx} has {i}-th data")

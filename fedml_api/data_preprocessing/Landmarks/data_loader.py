import collections
import csv
import logging

import numpy as np
import torch.utils.data as main_data
import torchvision.transforms as transforms

from FedML.fedml_api.data_preprocessing.base import Cutout, DataLoader, LocalDataset
from .datasets import Landmarks


class LandmarksDataLoader(DataLoader):
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    def __init__(self, data_dir, train_bs, test_bs, client_number, fed_train_map_file, fed_test_map_file):
        super().__init__(data_dir, train_bs, test_bs)
        self.client_number = client_number
        self.fed_train_map_file = fed_train_map_file,
        self.fed_test_map_file = fed_test_map_file

    @staticmethod
    def _read_csv(path: str):
        """Reads a csv file, and returns the content inside a list of dictionaries.
        Args:
          path: The path to the csv file.
        Returns:
          A list of dictionaries. Each row in the csv file will be a list entry. The
          dictionary is keyed by the column names.
        """
        with open(path, 'r') as f:
            return list(csv.DictReader(f))

    @classmethod
    def _data_transforms(cls):
        # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
        # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

        image_size = 224
        train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cls.IMAGENET_MEAN, cls.IMAGENET_STD),
        ])

        train_transform.transforms.append(Cutout(16))

        valid_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cls.IMAGENET_MEAN, cls.IMAGENET_STD),
        ])

        return train_transform, valid_transform

    @classmethod
    def get_mapping_per_user(cls, fn):
        """
        mapping_per_user is {'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}], 
                             'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}],
        } or               
                            [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...  
                             {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
        }
        """
        mapping_table = cls._read_csv(fn)
        expected_cols = ['user_id', 'image_id', 'class']
        if not all(col in mapping_table[0].keys() for col in expected_cols):
            logging.error(f'{mapping_table} has wrong format.')
            raise ValueError(f'The mapping file must contain user_id, image_id and class columns. '
                             f'The existing columns are {",".join(mapping_table[0].keys())}')

        data_local_num_dict = dict()

        mapping_per_user = collections.defaultdict(list)
        data_files = list()
        net_dataidx_map = dict()
        sum_temp = 0

        for row in mapping_table:
            user_id = row['user_id']
            mapping_per_user[user_id].append(row)
        for user_id, data in mapping_per_user.items():
            num_local = len(mapping_per_user[user_id])
            net_dataidx_map[int(user_id)] = (sum_temp, sum_temp + num_local)
            data_local_num_dict[int(user_id)] = num_local
            sum_temp += num_local
            data_files += mapping_per_user[user_id]
        assert sum_temp == len(data_files)

        return data_files, data_local_num_dict, net_dataidx_map

    # for centralized training
    def get_dataloader(self, train_files, test_files, dataidxs=None):
        dl_obj = Landmarks

        transform_train, transform_test = self._data_transforms()

        train_ds = dl_obj(self.data_dir, train_files, dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(self.data_dir, test_files, dataidxs=None, transform=transform_test)

        train_dl = main_data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)
        test_dl = main_data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=False)

        return train_dl, test_dl

    # for local devices
    def get_dataloader_test(self, train_files, test_files, dataidxs_train=None, dataidxs_test=None):
        dl_obj = Landmarks

        transform_train, transform_test = self._data_transforms()

        train_ds = dl_obj(self.data_dir, train_files, dataidxs=dataidxs_train, transform=transform_train)
        test_ds = dl_obj(self.data_dir, test_files, dataidxs=dataidxs_test, transform=transform_test)

        train_dl = main_data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False)
        test_dl = main_data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=False)

        return train_dl, test_dl

    def load_partition_data(self):
        train_files, data_local_num_dict, net_dataidx_map = self.get_mapping_per_user(self.fed_train_map_file)
        test_files = self._read_csv(self.fed_test_map_file)

        class_num = len(np.unique([item['class'] for item in train_files]))
        train_data_num = len(train_files)

        train_data_global, test_data_global = self.get_dataloader(train_files, test_files)
        test_data_num = len(test_files)

        # get local dataset
        data_local_num_dict = data_local_num_dict
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.client_number):
            dataidxs = net_dataidx_map[client_idx]
            train_data_local, test_data_local = self.get_dataloader(train_files, test_files, dataidxs)
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        return LocalDataset(train_data_num=train_data_num, test_data_num=test_data_num,
                            train_data_global=train_data_global, test_data_global=test_data_global,
                            local_data_num_dict=data_local_num_dict, train_data_local_dict=train_data_local_dict,
                            test_data_local_dict=test_data_local_dict, output_len=class_num)


if __name__ == '__main__':
    main_data_dir = './cache/images'
    fed_g23k_train_map_file = '../../../data/gld/data_user_dict/gld23k_user_dict_train.csv'
    fed_g23k_test_map_file = '../../../data/gld/data_user_dict/gld23k_user_dict_test.csv'

    fed_g160k_train_map_file = '../../../data/gld/data_user_dict/gld160k_user_dict_train.csv'
    fed_g160k_map_file = '../../../data/gld/data_user_dict/gld160k_user_dict_test.csv'

    # noinspection DuplicatedCode
    dataset_name = 'g160k'

    if dataset_name == 'g23k':
        main_client_number = 233
        main_fed_train_map_file = fed_g23k_train_map_file
        main_fed_test_map_file = fed_g23k_test_map_file
    elif dataset_name == 'g160k':
        main_client_number = 1262
        main_fed_train_map_file = fed_g160k_train_map_file
        main_fed_test_map_file = fed_g160k_map_file
    else:
        raise NotImplementedError

    dl = LandmarksDataLoader(main_data_dir, 10, 10, main_client_number, main_fed_train_map_file, main_fed_test_map_file)
    ds = dl.load_partition_data()

    print(ds.train_data_num, ds.test_data_num, ds.output_len)
    print(ds.local_data_num_dict)

    for _, (main_data, label) in zip(range(5), ds.train_data_global):
        print(main_data)
        print(label)
    print("=============================\n")

    for main_client_idx in range(main_client_number):
        for _, (main_data, label) in zip(range(5), ds.train_data_local_dict[main_client_idx]):
            print(main_data)
            print(label)

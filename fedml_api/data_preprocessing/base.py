import logging
from abc import ABC
from typing import Optional, Tuple, Dict, List, Callable, Type, ClassVar, Union, Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attr import attrs
from torch.utils.data import DataLoader as tDataLoader, Dataset as tDataset
from torchvision.datasets import CIFAR10


@attrs(auto_attribs=True, kw_only=True, cmp=False)
class Dataset:
    train_data_global: Union[tDataLoader, List]
    test_data_global: Union[tDataLoader, List]
    train_data_num: Optional[int] = None
    test_data_num: Optional[int] = None
    output_len: Optional[int] = None


@attrs(auto_attribs=True, kw_only=True, cmp=False)
class LocalDataset(Dataset):
    train_data_local_dict: Union[Dict[int, tDataLoader], Dict[int, List[Tuple[Any, Any]]]]
    test_data_local_dict: Union[Dict[int, tDataLoader], Dict[int, List[Tuple[Any, Any]]]]
    local_data_num_dict: Optional[object] = None
    train_data_local_num_dict: Optional[object] = None
    client_num: Optional[int] = None


@attrs(auto_attribs=True, kw_only=True, cmp=False)
class FederatedDataset(Dataset):
    client_num: int
    local_data_num_dict: Dict[int, int]
    train_data_local_dict: Dict[int, tDataLoader]
    test_data_local_dict: Dict[int, tDataLoader]
    vocab_len: Optional[int] = None


@attrs(auto_attribs=True, kw_only=True, cmp=False)
class DistributedDataset(Dataset):
    local_data_num: int
    train_data_local: tDataLoader
    test_data_local: tDataLoader


@attrs(auto_attribs=True, kw_only=True, cmp=False)
class DistributedFederatedDataset(DistributedDataset):
    client_num: int
    vocab_len: Optional[int] = None


class DataLoader(ABC):
    def __init__(self, data_dir, train_bs, test_bs):
        super().__init__()
        self.data_dir = data_dir
        self.train_bs = train_bs
        self.test_bs = test_bs


class LocalDataLoader(DataLoader):
    def __init__(self, data_dir, partition_method, partition_alpha, client_number, train_bs, test_bs):
        super().__init__(data_dir, train_bs, test_bs)
        self.partition_method = partition_method
        self.partition_alpha = partition_alpha
        self.client_number = client_number


class CifarDataLoader(LocalDataLoader, ABC):
    CIFAR_MEAN: ClassVar[Tuple[int]]
    CIFAR_STD: ClassVar[Tuple[int]]
    k: ClassVar[int]
    CIFAR_truncated: ClassVar[Callable]

    def __init__(self, data_dir, partition_method, partition_alpha, client_number, batch_size, distribution_txt,
                 net_dataidx_map_txt):
        super().__init__(data_dir, partition_method, partition_alpha, client_number, batch_size, batch_size)
        self.distribution_txt = distribution_txt
        self.net_dataidx_map_txt = net_dataidx_map_txt

    # generate the non-IID distribution for all methods
    def read_data_distribution(self) -> Dict[int, Dict[int, int]]:
        distribution = dict()
        with open(self.distribution_txt, 'r') as fd:
            for x in fd.readlines():
                if x[0] not in '{}':
                    tmp = x.split(':')
                    if '{' == tmp[1].strip():
                        first_level_key = int(tmp[0])
                        distribution[first_level_key] = dict()
                    else:
                        second_level_key = int(tmp[0])
                        distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
        return distribution

    def read_net_dataidx_map(self) -> Dict[int, List[int]]:
        net_dataidx_map = dict()
        with open(self.net_dataidx_map_txt, 'r') as fd:
            for x in fd.readlines():
                if x[0] not in '{}]':
                    tmp = x.split(':')
                    if '[' == tmp[-1].strip():
                        key = int(tmp[0])
                        net_dataidx_map[key] = list()
                    else:
                        tmp_array = x.split(',')
                        net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
        return net_dataidx_map

    @staticmethod
    def record_net_data_stats(y_train, net_dataidx_map) -> Dict[int, Dict[int, int]]:
        net_cls_counts = dict()

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        logging.debug(f'Data statistics: {net_cls_counts}')
        return net_cls_counts

    @classmethod
    def _data_transforms(cls):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cls.CIFAR_MEAN, cls.CIFAR_STD),
        ])

        train_transform.transforms.append(Cutout(16))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cls.CIFAR_MEAN, cls.CIFAR_STD),
        ])

        return train_transform, valid_transform

    def load_cifar_data(self):
        train_transform, test_transform = self._data_transforms()

        cifar_train_ds = self.CIFAR_truncated(self.data_dir, train=True, download=True, transform=train_transform)
        cifar_test_ds = self.CIFAR_truncated(self.data_dir, train=False, download=True, transform=test_transform)

        x_train, y_train = cifar_train_ds.main_data, cifar_train_ds.target
        x_test, y_test = cifar_test_ds.main_data, cifar_test_ds.target

        return x_train, y_train, x_test, y_test

    def partition_data(self):
        logging.info("*********partition data***************")
        x_train, y_train, x_test, y_test = self.load_cifar_data()
        n_train = x_train.shape[0]
        # n_test = x_test.shape[0]

        if self.partition_method == "homo":
            total_num = n_train
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.client_number)
            net_dataidx_map = {i: batch_idxs[i] for i in range(self.client_number)}

        elif self.partition_method == "hetero":
            min_size = 0
            n = y_train.shape[0]
            logging.info(f"N = {n}")
            net_dataidx_map = dict()

            idx_batch = None
            while min_size < 10:
                idx_batch = [list() for _ in range(self.client_number)]
                # for each class in the dataset
                for k in range(self.k):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.partition_alpha, self.client_number))
                    # Balance
                    proportions = np.array(
                        [p * (len(idx_j) < n / self.client_number) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min(map(len, idx_batch))

            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

        elif self.partition_method == "hetero-fix":
            net_dataidx_map = self.read_net_dataidx_map()
        else:
            raise ValueError(f'Unknown partition method: {self.partition_method}')

        traindata_cls_counts = self.read_data_distribution() if self.partition_method == "hetero-fix" else \
            self.record_net_data_stats(y_train, net_dataidx_map)

        return x_train, y_train, x_test, y_test, net_dataidx_map, traindata_cls_counts

    # for centralized training
    def get_dataloader(self, dataidxs=None):
        transform_train, transform_test = self._data_transforms()

        train_ds = self.CIFAR_truncated(self.data_dir, dataidxs=dataidxs, train=True, transform=transform_train,
                                        download=True)
        test_ds = self.CIFAR_truncated(self.data_dir, train=False, transform=transform_test, download=True)

        train_dl = tDataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=True)
        test_dl = tDataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=True)

        return train_dl, test_dl

    # for local devices
    def get_dataloader_test(self, dataidxs_train=None, dataidxs_test=None):
        transform_train, transform_test = self._data_transforms()

        train_ds = self.CIFAR_truncated(self.data_dir, dataidxs=dataidxs_train, train=True, transform=transform_train,
                                        download=True)
        test_ds = self.CIFAR_truncated(self.data_dir, dataidxs=dataidxs_test, train=False, transform=transform_test,
                                       download=True)

        train_dl = tDataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=True)
        test_dl = tDataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=True)

        return train_dl, test_dl

    def load_partition_data_distributed(self, process_id):
        x_train, y_train, x_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))
        logging.info(f"traindata_cls_counts = {traindata_cls_counts}")
        train_data_num = sum(len(net_dataidx_map[r]) for r in range(self.client_number))

        # get global test data
        if process_id == 0:
            train_data_global, test_data_global = self.get_dataloader()
            logging.info(f"train_dl_global number = {len(train_data_global)}")
            logging.info(f"test_dl_global number = {len(train_data_global)}")
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_global, test_data_global = None, None
            # training batch size = 64; algorithms batch size = 32
            dataidxs = net_dataidx_map[process_id - 1]
            train_data_local, test_data_local = self.get_dataloader(dataidxs)
            logging.info(f"process_id = {process_id:d}, batch_num_train_local = {len(train_data_local)}, "
                         f"batch_num_test_local = {len(test_data_local)}")
            local_data_num = len(dataidxs)
            logging.info(f"rank = {process_id:d}, local_sample_number = {local_data_num:d}")
        return DistributedDataset(train_data_num=train_data_num, train_data_global=train_data_global,
                                  test_data_global=test_data_global, local_data_num=local_data_num,
                                  train_data_local=train_data_local, test_data_local=test_data_local,
                                  output_len=class_num)

    def load_partition_data(self):
        x_train, y_train, x_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))
        logging.info(f"traindata_cls_counts = {traindata_cls_counts}")
        train_data_num = sum(len(net_dataidx_map[r]) for r in range(self.client_number))

        train_data_global, test_data_global = self.get_dataloader()
        logging.info(f"train_dl_global number = {len(train_data_global)}")
        logging.info(f"test_dl_global number = {len(train_data_global)}")
        test_data_num = len(test_data_global)

        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.client_number):
            dataidxs = net_dataidx_map[client_idx]
            local_data_num = len(dataidxs)
            data_local_num_dict[client_idx] = local_data_num
            logging.info(f"client_idx = {client_idx:d}, local_sample_number = {local_data_num:d}")

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = self.get_dataloader(dataidxs)
            logging.info(f"client_idx = {client_idx:d}, "
                         f"batch_num_train_local = {len(train_data_local)}, "
                         f"batch_num_test_local = {len(test_data_local)}")
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local
        return LocalDataset(train_data_num=train_data_num, test_data_num=test_data_num,
                            train_data_global=train_data_global,
                            test_data_global=test_data_global, local_data_num_dict=data_local_num_dict,
                            train_data_local_dict=train_data_local_dict, test_data_local_dict=test_data_local_dict,
                            output_len=class_num)


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


class CIFARTruncated(tDataset, ABC):
    CIFAR_FUNC: ClassVar[Type[CIFAR10]]

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset()

    def __build_truncated_dataset(self):
        print(f"download = {self.download}")
        cifar_dataobj = self.CIFAR_FUNC(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.main_data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.main_data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

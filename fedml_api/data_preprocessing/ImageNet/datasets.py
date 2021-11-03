import os
import os.path
from abc import ABC
from typing import List, Tuple

import torch.utils.data as data

from FedML.fedml_api.data_preprocessing.base import IMG_EXTENSIONS, default_loader


def has_file_allowed_extension(filename):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir_):
    classes = [d for d in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, d))]
    classes.sort()
    class_to_idx = {class_: i for i, class_ in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(dir_, class_to_idx):
    images = list()

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir_ = os.path.expanduser(dir_)
    for target in sorted(os.listdir(dir_)):
        d = os.path.join(dir_, target)
        if not os.path.isdir(d):
            continue

        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    target_num += 1

        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map


class _BaseImageNet(data.Dataset, ABC):
    local_data: List[Tuple[str, int]]

    def __init__(self, transform=None, target_transform=None):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)


class ImageNet(_BaseImageNet):
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        super().__init__(transform, target_transform)
        self.dataidxs = dataidxs
        self.train = train
        self.download = download
        if self.train:
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, 'val')

        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()
        if dataidxs is None:
            self.local_data = self.all_data
        elif isinstance(dataidxs, int):
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin: end]
        else:
            self.local_data = list()
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.data_dir)
        all_data, data_local_num_dict, net_dataidx_map = make_dataset(self.data_dir, class_to_idx)
        if len(all_data) == 0:
            raise RuntimeError(f"Found 0 files in subfolders of: {self.data_dir}\nSupported extensions are: "
                               f"{','.join(IMG_EXTENSIONS)}")
        return all_data, data_local_num_dict, net_dataidx_map


class ImageNetTruncated(_BaseImageNet):
    def __init__(self, imagenet_dataset: ImageNet, dataidxs, net_dataidx_map, train=True, transform=None,
                 target_transform=None, download=False):
        super().__init__(transform, target_transform)
        self.dataidxs = dataidxs
        self.train = train
        self.download = download
        self.net_dataidx_map = net_dataidx_map
        self.all_data = imagenet_dataset.get_local_data()
        if dataidxs is None:
            self.local_data = self.all_data
        elif type(dataidxs) == int:
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin: end]
        else:
            self.local_data = list()
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

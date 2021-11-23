import logging

import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from .datasets import ImageNet, ImageNetTruncated
from .datasets_hdf5 import ImageNetHDF5, ImageNetTruncatedHDF5
from ..base import Cutout, LocalDataset, DataLoader, Dataset


class ImageNetDataLoader(DataLoader):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, data_dir, batch_size, client_number):
        super().__init__(data_dir, batch_size, batch_size)
        self.client_number = client_number

    @classmethod
    def _data_transforms(cls):
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

    def get_dataloader_truncated(self, imagenet_dataset_train, imagenet_dataset_test, dataidxs=None,
                                 net_dataidx_map=None):
        """
            imagenet_dataset_train, imagenet_dataset_test should be ImageNet or ImageNet_hdf5
        """
        if isinstance(imagenet_dataset_train, ImageNet):
            dl_obj = ImageNetTruncated
        elif isinstance(imagenet_dataset_train, ImageNetHDF5):
            dl_obj = ImageNetTruncatedHDF5
        else:
            raise NotImplementedError('Unknown dataset')

        transform_train, transform_test = self._data_transforms()

        train_ds = dl_obj(imagenet_dataset_train, dataidxs, net_dataidx_map, train=True, transform=transform_train,
                          download=False)
        test_ds = dl_obj(imagenet_dataset_test, dataidxs=None, net_dataidx_map=None, train=False,
                         transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False,
                                   pin_memory=True, num_workers=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=False,
                                  pin_memory=True, num_workers=4)

        return train_dl, test_dl

    # for centralized training
    def get_dataloader(self, dataidxs=None):
        dl_obj = ImageNet

        transform_train, transform_test = self._data_transforms()

        train_ds = dl_obj(self.data_dir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
        test_ds = dl_obj(self.data_dir, dataidxs=None, train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False,
                                   pin_memory=True, num_workers=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=False,
                                  pin_memory=True, num_workers=4)

        return train_dl, test_dl

    # for local devices
    def get_dataloader_test(self, dataidxs_train=None, dataidxs_test=None):
        dl_obj = ImageNet

        transform_train, transform_test = self._data_transforms()

        train_ds = dl_obj(self.data_dir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(self.data_dir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.train_bs, shuffle=True, drop_last=False,
                                   pin_memory=True, num_workers=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.test_bs, shuffle=False, drop_last=False,
                                  pin_memory=True, num_workers=4)

        return train_dl, test_dl

    def distributed_centralized_loader(self, hdf5: bool, world_size, rank):
        """
            Used for generating distributed dataloader for 
            accelerating centralized training 
        """

        transform_train, transform_test = self._data_transforms()
        if hdf5:
            train_dataset = ImageNetHDF5(data_dir=self.data_dir, dataidxs=None, train=True, transform=transform_train)
            test_dataset = ImageNetHDF5(data_dir=self.data_dir, dataidxs=None, train=False, transform=transform_test)
        else:
            train_dataset = ImageNet(data_dir=self.data_dir, dataidxs=None, train=True, transform=transform_train)
            test_dataset = ImageNet(data_dir=self.data_dir, dataidxs=None, train=False, transform=transform_test)

        train_sam = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sam = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

        train_dl = data.DataLoader(train_dataset, batch_size=self.train_bs, sampler=train_sam,
                                   pin_memory=True, num_workers=4)
        test_dl = data.DataLoader(test_dataset, batch_size=self.test_bs, sampler=test_sam,
                                  pin_memory=True, num_workers=4)

        return Dataset(train_data_global=train_dl, test_data_global=test_dl, train_data_num=len(train_dataset),
                       test_data_num=len(test_dataset), output_len=1000)

    def load_partition_data(self, hdf5: bool):
        if hdf5:
            train_dataset = ImageNetHDF5(data_dir=self.data_dir, dataidxs=None, train=True)
            test_dataset = ImageNetHDF5(data_dir=self.data_dir, dataidxs=None, train=False)
        else:
            train_dataset = ImageNet(data_dir=self.data_dir, dataidxs=None, train=True)
            test_dataset = ImageNet(data_dir=self.data_dir, dataidxs=None, train=False)

        net_dataidx_map = train_dataset.get_net_dataidx_map()

        # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
        # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
        train_data_num = len(train_dataset)
        test_data_num = len(test_dataset)
        class_num_dict = train_dataset.get_data_local_num_dict()

        # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)

        train_data_global, test_data_global = self.get_dataloader_truncated(train_dataset, test_dataset, dataidxs=None,
                                                                            net_dataidx_map=None, )

        logging.info(f"train_dl_global number = {len(train_data_global)}")
        logging.info(f"test_dl_global number = {len(test_data_global)}")

        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.client_number):
            if self.client_number == 1000:
                dataidxs = client_idx
                data_local_num_dict = class_num_dict
            elif self.client_number == 100:
                dataidxs = [client_idx * 10 + i for i in range(10)]
                data_local_num_dict[client_idx] = sum(class_num_dict[client_idx + i] for i in range(10))
            else:
                raise NotImplementedError("Not support other client_number for now!")

            # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

            # training batch size = 64; algorithms batch size = 32
            # train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
            #                                          dataidxs)
            train_data_local, test_data_local = self.get_dataloader_truncated(train_dataset, test_dataset,
                                                                              dataidxs=dataidxs,
                                                                              net_dataidx_map=net_dataidx_map)

            # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            # client_idx, len(train_data_local), len(test_data_local)))
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        logging.info(f"data_local_num_dict: {data_local_num_dict}")
        return LocalDataset(train_data_num=train_data_num, test_data_num=test_data_num,
                            train_data_global=train_data_global, test_data_global=test_data_global,
                            local_data_num_dict=data_local_num_dict, train_data_local_dict=train_data_local_dict,
                            test_data_local_dict=test_data_local_dict, output_len=1000)


if __name__ == '__main__':
    # data_dir = '/home/datasets/imagenet/ILSVRC2012_dataset'
    main_data_dir = '/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5'
    main_data_dir = '/home/trabajo/PycharmProjects/FedML-IoT/FedML/data/ImageNet/'

    main_client_number = 100

    dl = ImageNetDataLoader(main_data_dir, 10, main_client_number)
    ds = dl.load_partition_data(False)

    print(ds.train_data_num, ds.test_data_num, ds.output_len)
    print(ds.local_data_num_dict)

    for _, (data, label) in zip(range(5), ds.train_data_global):
        print(data)
        print(label)
    print("=============================\n")

    for client_idx in range(main_client_number):
        for _, (data, label) in zip(range(5), ds.train_data_local_dict[client_idx]):
            print(data)
            print(label)

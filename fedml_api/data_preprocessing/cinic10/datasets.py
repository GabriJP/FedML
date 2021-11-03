import numpy as np
from torchvision.datasets import DatasetFolder

from ..base import IMG_EXTENSIONS, default_loader


class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
    """

    def __init__(self, root, dataidxs=None, transform=None, target_transform=None, loader=default_loader,
                 is_valid_file=None):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None, transform=transform,
                         target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.dataidxs = dataidxs

        # we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset()

    def __build_truncated_dataset(self):
        if self.dataidxs is not None:
            # self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.imgs)

    @property
    def get_train_labels(self):
        return self._train_labels

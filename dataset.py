# modify on top of https://github.com/xavihart/Diff-PGD

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
import numpy as np
import os
import pickle
import torch

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
os.environ['IMAGENET_LOC_ENV'] = "./dataset/imagenet"
os.environ['celebA'] = "./datasets/CelebA-HQ"


# list of all datasets
DATASETS = ["imagenet", "celebA"]


def get_dataset(dataset: str, split: str, adv=False) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""

    if dataset == "imagenet" and adv:
        return _imagenet(split, adv)

    elif dataset == "imagenet_sd":
        return _imagenet_sd(split)

    elif dataset == "imagenet":
        return _imagenet(split)

    elif dataset == 'celebA':
        return _celebA(split)
    
    elif dataset == "IQA":
        transform = transforms.Compose([
                transforms.ToTensor()
            ])
        return datasets.ImageFolder("Diff-Protect/image_net_out/", transform)

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if 'imagenet' in dataset:
        return 1000


def get_normalize_layer(dataset: str, diff=None) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if diff:
        return NormalizeLayer(_DIFF_MEAN, _DIFF_STD)

    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)

def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_DIFF_MEAN = [0, 0, 0]
_DIFF_STD = [1, 1, 1]

def _celebA(split: str) -> Dataset:
    return datasets.ImageFolder(os.environ['celebahq'] ,transform=transforms.Compose([
            transforms.ToTensor()
        ]))

def _imagenet(split: str, adv=False) -> Dataset:
    # print(os.environ['IMAGENET_LOC_ENV'])
    # print(os.environ)
    # if not IMAGENET_LOC_ENV in os.environ.keys():
    #     raise RuntimeError("environment variable for ImageNet directory not set")
    if adv:
        subdir = "./PGD_100/4"
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        print("Using Adversarial Dataset")
    else:
        dir = os.environ['IMAGENET_LOC_ENV']
        if split == "train":
            subdir = os.path.join(dir, "train")
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        elif split == "test":
            subdir = os.path.join(dir, "val")
            transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize((384, 384)),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
    return datasets.ImageFolder(subdir, transform)

def _imagenet_sd(split: str) -> Dataset:
    dir = os.environ['IMAGENET_LOC_ENV']
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        ])
    return datasets.ImageFolder(subdir, transform)

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))

    def forward(self, input: torch.tensor, y=None):
        # print("norm layer input", input.max(), input.min())
        # print(self.means)
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        return (input - means)/sds



# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

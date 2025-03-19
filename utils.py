# Some helper functions for PyTorch, including:

import os
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from load_data_online import CostumeImageFolder, CostumeMixedImageFolder, CostumeImageFolderTesting
import shutil
import numpy as np
import random
from skimage import io
from RandAugment import RandAugment
import scipy.spatial.distance
from PIL import ImageFilter

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_MNIST(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform
    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_MNIST_contrastive(roots, category_indexs, batchSize, shuffle=True):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes



def load_SVHN(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(2, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform
    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_SVHN_contrastive(roots, category_indexs, batchSize, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_transform.transforms.insert(0, RandAugment(2, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes


def load_cifar(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(2, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes


def load_cifar_contrastive(roots, category_indexs, batchSize, shuffle=True):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes



def load_ImageNet200(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_ImageNet200_contrastive(roots, category_indexs, batchSize, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_transform.transforms.insert(0, RandAugment(1, 5))

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle,num_workers=4)
    return dataLoader, data_classes


def load_ImageNet200_contrastive_versiontwo(roots, category_indexs, batchSize, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    #wordidfile list
    with open('./data/tinyimagenet/wnids.txt', 'r') as file:
        wnids = file.read().splitlines()
        wnids_dict = {value: index for index, value in enumerate(wnids)}
    #print (wnids_dict)
    category_indexes_ids = [wnids[i] for i in category_indexs]
    #print (wnids,category_indexes_ids)

    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    class TargetTransform:
        def __init__(self, class_to_idx):
            self.class_to_idx = class_to_idx
        
        def __call__(self, filename):
            # Extract the class part (e.g., 'n02106662') from the filename (e.g., 'n02106662_0.JPEG')
            class_id = filename.split('_')[0]
            
            # Map the class_id to its corresponding index in the dictionary
            return self.class_to_idx.get(class_id, -1)  # Return -1 if class_id is not found

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5)) #was 1, but now 2. 
    #print ("no augmentation")
    #no augmentation

    dirs = [root + str(i) + "/images/" for i in category_indexes_ids for root in roots]
    #print (dirs)
    data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB", target_transform=TargetTransform(wnids_dict))
    data_classes = list(map(int, data.classes))
    #dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle,num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle,num_workers=8, pin_memory=True, persistent_workers=False, prefetch_factor=4)
    #dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle,num_workers=4, pin_memory=True)

    return dataLoader, data_classes

def load_ImageNet200(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_ImageNet200_versiontwo(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    #wordidfile list
    with open('./data/tinyimagenet/wnids.txt', 'r') as file:
        wnids = file.read().splitlines()
        wnids_dict = {value: index for index, value in enumerate(wnids)}
    #print (wnids_dict)
    category_indexes_ids = [wnids[i] for i in category_indexs]
    #print (wnids,category_indexes_ids)

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    class TargetTransform:
        def __init__(self, class_to_idx):
            self.class_to_idx = class_to_idx
        
        def __call__(self, filename):
            # Extract the class part (e.g., 'n02106662') from the filename (e.g., 'n02106662_0.JPEG')
            class_id = filename.split('_')[0]
            
            # Map the class_id to its corresponding index in the dictionary
            return self.class_to_idx.get(class_id, -1)  # Return -1 if class_id is not found


    #dirs = [root + str(i) + "/" for i in category_indexs for root in roots]
    dirs = [root + str(i) + "/images/" for i in category_indexes_ids for root in roots]
    #print (dirs)
    
    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(1, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB", target_transform=TargetTransform(wnids_dict))
    #data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB", target_transform=TargetTransform(wnids_dict))

    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_ImageNet200_versiontwo_testing(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_transform = transforms.Compose([
    #     transforms.Resize(64),
    #     transforms.RandomCrop(64, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # #wordidfile list
    # with open('/data/projects/punim1884/ConOSR/data/tinyimagenet/wnids.txt', 'r') as file:
    #     wnids = file.read().splitlines()
    #     wnids_dict = {value: index for index, value in enumerate(wnids)}
    #print (wnids_dict)
    #category_indexes_ids = [wnids[i] for i in category_indexs]
    #print (wnids,category_indexes_ids)

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    # class TargetTransform:
    #     def __init__(self, class_to_idx):
    #         self.class_to_idx = class_to_idx
        
    #     def __call__(self, filename):
    #         # Extract the class part (e.g., 'n02106662') from the filename (e.g., 'n02106662_0.JPEG')
    #         class_id = filename.split('_')[0]
            
    #         # Map the class_id to its corresponding index in the dictionary
    #         return self.class_to_idx.get(class_id, -1)  # Return -1 if class_id is not found


    data = CostumeImageFolderTesting('/data/tinyimagenet/', category_indexs, transform=test_transform ) #, target_transform=None,loader=default_loader, mode="RGB")
    #data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB", target_transform=TargetTransform(wnids_dict))
    #data = CostumeImageFolder(roots=dirs, transform=TwoCropTransform(train_transform), mode="RGB", target_transform=TargetTransform(wnids_dict))

    data_classes = None #list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_ImageNet_resize(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(3, 5))

    if train:
        transform = train_transform
    else:
        transform = test_transform


    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, data_classes


def load_ImageNet200_versiontwo_testing_resize(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            normalize,
    ])

    # test_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    category_indexs = list(range(200))
    data = CostumeImageFolderTesting('/data//tinyimagenet/', category_indexs, transform=test_transform ) #, target_transform=None,loader=default_loader, mode="RGB")

    data_classes = None #list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes





def load_ImageNet200_versiontwo_testing_crop(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
    ])

    # test_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    category_indexs = list(range(200))
    data = CostumeImageFolderTesting('/data/tinyimagenet/', category_indexs, transform=test_transform ) #, target_transform=None,loader=default_loader, mode="RGB")

    data_classes = None #list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SingleClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_class=0):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            transform (callable, optional): Transform to apply to the images.
            target_class (int): The single target class for all images.
        """
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.target_class = target_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore original target
        return image, self.target_class
    
def load_LSUN_versiontwo_testing_resize(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            normalize,
    ])

    # test_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    category_indexs = list(range(200))

    data_classes = None #list(map(int, data.classes))
    data = SingleClassDataset('./data/LSUN/',transform=test_transform)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes

def load_LSUN_versiontwo_testing_crop(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
    ])

    # test_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    category_indexs = list(range(200))
    data = SingleClassDataset('./data/LSUN/',transform=test_transform)
    data_classes = None #list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes


def load_ImageNet_crop(roots, category_indexs, batchSize, train, shuffle=True, useRandAugment=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
    ])

    dirs = [root + str(i) + "/" for i in category_indexs for root in roots]

    if train:
        transform = train_transform
    else:
        transform = test_transform

    if useRandAugment:
        train_transform.transforms.insert(0, RandAugment(3, 5))

    data = CostumeImageFolder(roots=dirs, transform=transform, mode="RGB")
    data_classes = list(map(int, data.classes))
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, data_classes













def get_onehot_labels(labels, classid_list):
    targets = torch.zeros([labels.shape[0], len(classid_list)], requires_grad=False).to(labels.device)
    for j, label in enumerate(labels):
        if label.item() not in classid_list:
            continue
        index = classid_list.index(label.item())
        targets[j, index] = 1
    return targets

def get_smooth_labels(labels, classid_list, smoothing_coeff=0.1):
    label_positive = 1 - smoothing_coeff
    label_negative = smoothing_coeff / (len(classid_list)-1)

    targets = label_negative * torch.ones([labels.shape[0], len(classid_list)], requires_grad=False).to(labels.device)
    for j, label in enumerate(labels):
        if label.item() not in classid_list:
            continue
        
        index = classid_list.index(label.item())

        #print (classid_list, "label",labels,"uninde",index)
        targets[j, index] = label_positive
    return targets




class NoiseDataset(Dataset):
    def __init__(self, num_samples, img_size, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate noise image (values between 0 and 1)
        noise_image = torch.rand(*self.img_size)  # Shape: (C, H, W)
        label = -1  # Dummy label (not used in training/testing)
        
        if self.transform:
            noise_image = self.transform(noise_image)

        return noise_image, label
    
def load_noise_dataset(batchSize, num_samples=10000, img_size=(3, 32, 32), shuffle=True):


    noise_data = NoiseDataset(num_samples=num_samples, img_size=img_size) #transform=transform
    dataLoader = DataLoader(noise_data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, None  




from torchvision.datasets import Omniglot
def load_noise_OmniDataset(batchSize, num_samples=10000, img_size=(3, 32, 32), shuffle=True):
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Ensure it's a tensor (though already generated as one)
        
    # ]) #normalize

    noise_data = OmniDataset(num_samples=num_samples, img_size=img_size) #transform=transform
    dataLoader = DataLoader(noise_data, batch_size=batchSize, shuffle=shuffle, num_workers=4)

    return dataLoader, None  # No classes since this is noise

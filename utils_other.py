import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN

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

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(targets[i]) #known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR_classifier(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32, useRandAugment=True):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

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
            train_transform.transforms.insert(0, RandAugment(1, 5))

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        valset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=test_transform)
        print('All val Data:', len(valset))
        valset.__Filter__(known=self.known)
        
        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        pin_memory = True if use_gpu else False
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class CIFAR10_OSR_contrastive(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_transform.transforms.insert(0, RandAugment(1, 5))

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        #return self.train_loader


# class CIFAR10(object):
#     def __init__(self, **options):

#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#         batch_size = options['batch_size']
#         data_root = os.path.join(options['dataroot'], 'cifar10')

#         pin_memory = True if options['use_gpu'] else False

#         trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        
#         trainloader = torch.utils.data.DataLoader(
#             trainset, batch_size=batch_size, shuffle=True,
#             num_workers=options['workers'], pin_memory=pin_memory,
#         )
        
#         testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
#         testloader = torch.utils.data.DataLoader(
#             testset, batch_size=batch_size, shuffle=False,
#             num_workers=options['workers'], pin_memory=pin_memory,
#         )

#         self.num_classes = 10
#         self.trainloader = trainloader
#         self.testloader = testloader

# __factory = {
#     'mnist': MNIST,
#     'kmnist': KMNIST,
#     'cifar10': CIFAR10,
#     'cifar100': CIFAR100,
#     'svhn':SVHN,
# }

# def create(name, **options):
#     if name not in __factory.keys():
#         raise KeyError("Unknown dataset: {}".format(name))
#     return __factory[name](**options)

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR_contrastive(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        # Define training transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_transform.transforms.insert(0, RandAugment(1, 5))

        # Define test transformations
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        pin_memory = True if use_gpu else False

        # Load training dataset and filter known classes
        trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load test dataset and filter known classes
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class CIFAR100_OSR_classifier(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32, useRandAugment=True):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        # Define training transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        # Define test transformations
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        if useRandAugment:
            train_transform.transforms.insert(0, RandAugment(1, 5))

        pin_memory = True if use_gpu else False

        # Load and filter training data
        trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load and filter validation data
        valset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=test_transform)
        print('All Val Data:', len(valset))
        valset.__Filter__(known=self.known)

        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class CIFAR100_OSR_VIZ(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        # Define test transformations
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        # Load and filter test data for known classes
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load and filter test data for unknown classes
        outset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        # Define test transformations
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        pin_memory = True if use_gpu else False

        # Load and filter test data for known classes
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load and filter test data for unknown classes
        outset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class SVHN_OSR(object):
    def __init__(self, known, dataroot='./data/svhn', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        # Define test transformations
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        # Load and filter test data for known classes
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=test_transform)
        

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class SVHN_OSR_contrastive(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        #self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        # Define training transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_transform.transforms.insert(0, RandAugment(1, 5))


        pin_memory = True if use_gpu else False

        # Load training dataset and filter known classes
        trainset = SVHN_Filter(root=dataroot, split='train',known_classes=self.known, download=True, transform=TwoCropTransform(train_transform))
        #trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class SVHN_OSR_classifier(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known

        print('Selected Labels: ', known)

        # Define training transformations
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])


        pin_memory = True if use_gpu else False

        # Load training dataset and filter known classes
        trainset = SVHN_Filter(root=dataroot, split='train',known_classes=self.known, download=True, transform=train_transform)
        #trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valset = SVHN_Filter(root=dataroot, split='train',known_classes=self.known, download=True, transform=train_transform)
        #trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(trainset))
        
        self.val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class SVHN_OSR_osr(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))
        print('Selected Labels: ', known)

        # Define training transformations
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])


        pin_memory = True if use_gpu else False

        # Load training dataset and filter known classes
        testset = SVHN_Filter(root=dataroot, split='test',known_classes=self.known, download=True, transform=train_transform)
        #trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(testset))
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valset = SVHN_Filter(root=dataroot, split='test',known_classes=self.unknown, download=True, transform=train_transform)
        #trainset = CIFAR100_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))
        print('All Train Data:', len(valset))
        
        self.unknown_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )



class SVHN_Filter(torchvision.datasets.SVHN):
    """SVHN Dataset with filtering capability."""
    
    def __init__(self, root, split='train', known_classes=None, transform=None, download=True):
        super().__init__(root=root, split=split, transform=transform, download=download)
        if known_classes is not None:
            self.__Filter__(known_classes)

    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(targets[i])
        self.data, self.labels = self.data[mask], np.array(new_targets)

        indices = np.random.permutation(len(self.labels))
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        
# class CIFAR10_Filter(CIFAR10):
#     """CIFAR10 Dataset.
#     """
#     def __Filter__(self, known):
#         datas, targets = np.array(self.data), np.array(self.targets)
#         mask, new_targets = [], []
#         for i in range(len(targets)):
#             if targets[i] in known:
#                 mask.append(i)
#                 new_targets.append(targets[i]) #known.index(targets[i]))
#         self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


from torchvision.datasets import MNIST

class CIFAR100_FilterOLDER(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNIST_Filter(MNISTRGB):
    """Filtered MNIST Dataset."""
    def __Filter__(self, known):
        # Convert data and targets to NumPy arrays for processing
        datas, targets = self.data.numpy(), self.targets.numpy()
        mask, new_targets = [], []
        print (np.unique(targets))
        
        # Create mask and new_targets
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(targets[i])

        print (np.unique(new_targets))
        # Filter data and targets using the mask
        self.data = torch.tensor(np.take(datas, mask, axis=0))  # Convert back to PyTorch tensor
        self.targets = torch.tensor(new_targets)  # Convert new targets to PyTorch tensor


class MNIST_OSR_Contrastive(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32, ifrandaug=True):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        
        if ifrandaug:
            train_transform.transforms.insert(0, RandAugment(1, 5))
        trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=TwoCropTransform(train_transform))


        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        # self.train_loader = torch.utils.data.DataLoader(
        #     trainset, batch_size=batch_size, shuffle=True,
        #     num_workers=num_workers)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True, persistent_workers=False, prefetch_factor=4)
        #dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffle,num_workers=8, pin_memory=True, persistent_workers=False, prefetch_factor=4)


class MNIST_OSR_classifier(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32, useRandAugment=True):
        self.num_classes = len(known)
        self.known = known
        print (self.known)
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

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


        if useRandAugment:
            train_transform.transforms.insert(0, RandAugment(1, 5))

        pin_memory = True if use_gpu else False

        # Load and filter training data
        trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load and filter validation data
        valset = MNIST_Filter(root=dataroot, train=True, download=True, transform=test_transform)
        print('All Val Data:', len(valset))
        valset.__Filter__(known=self.known)

        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class MNIST_OSR(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        # Define test transformations
        
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        # Load and filter test data for known classes
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Load and filter test data for unknown classes
        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

class MNISTNOISE_OSR(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        #self.num_classes = len(known)
        #self.known = known
        #self.unknown = list(set(list(range(0, 10))) - set(known))

        class AddNoiseClamp:
            def __init__(self, noise_min=0.0, noise_max=1.0):
                self.noise_min = noise_min
                self.noise_max = noise_max

            def __call__(self, img):
                # Add noise
                #print (img.max(),img.min())
                noise = torch.rand_like(img) #* (self.noise_max - self.noise_min) + self.noise_min
                #print ("IMG",torch.unique(img),"NOISE",noise)
                img = (img + noise)/2
                # Clamp to valid range [0, 1]
                return img.clamp(0.0, 1.0)
    
        #print('Selected Labels: ', known)

        # Define test transformations
        
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            AddNoiseClamp()
        ])

        pin_memory = True if use_gpu else False

        # Load and filter test data for known classes
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=test_transform)
        #print('All Test Data:', len(testset))
        #testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

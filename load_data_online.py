# online data loader

import os
import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
from torch.utils import data
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if mode == "L":
                return img.convert('L')
            if mode == "RGB":
                return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode="RGB"):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dirs):

    classes = []
    for dir in dirs:
        classes.extend([int(d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])

    # for dir in dirs:
    #     classes.extend([int(dir.split('/')[-2])])
    classes.sort()
    class_to_idx = {i: i for i in classes}
    return classes, class_to_idx


def make_dataset_old(dirs, class_to_idx):


    # print('dirs',dirs)
    images = []
    for dir in dirs:
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):

            y = int(target)
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[y])
                        images.append(item)

        # y=int(dir.split('/')[-2])
        # for root, _, fnames in sorted(os.walk(dir)):
        #     for fname in sorted(fnames):
        #                 if is_image_file(fname):
        #                     path = os.path.join(root, fname)
        #                     item = (path, class_to_idx[y])
        #                     images.append(item)

    return images

def make_dataset(dirs, class_to_idx):

    # print('dirs',dirs)
    images = []
    for dir in dirs:
        #print (dir)
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):

            y = target
            #print ("y",target)
            d = os.path.join(dir, target)
            #print ("d",d)
            item = (d, y)
            images.append(item)
            if not os.path.isdir(d):
                continue
            
            
            # for root, _, fnames in sorted(os.walk(d)):
            #     for fname in sorted(fnames):
            #         if is_image_file(fname):
            #             path = os.path.join(root, fname)
            #             print (path)
            #             item = (path, y)
            #             images.append(item)

        # y=int(dir.split('/')[-2])
        # for root, _, fnames in sorted(os.walk(dir)):
        #     for fname in sorted(fnames):
        #                 if is_image_file(fname):
        #                     path = os.path.join(root, fname)
        #                     item = (path, class_to_idx[y])
        #                     images.append(item)

    return images


def make_dataset_old(dirs, class_to_idx):


    # print('dirs',dirs)
    images = []
    for dir in dirs:
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):

            y = int(target)
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[y])
                        images.append(item)

        # y=int(dir.split('/')[-2])
        # for root, _, fnames in sorted(os.walk(dir)):
        #     for fname in sorted(fnames):
        #                 if is_image_file(fname):
        #                     path = os.path.join(root, fname)
        #                     item = (path, class_to_idx[y])
        #                     images.append(item)

    return images


class CostumeImageFolder(data.Dataset):
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

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader, mode="RGB"):
        classes, class_to_idx = find_classes(roots)
        imgs = make_dataset(roots, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders",
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.roots = roots
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class CostumeImageFolderVtwo(data.Dataset):
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

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader, mode="RGB"):
        classes, class_to_idx = find_classes(roots)
        imgs = make_dataset_withtransform(roots, class_to_idx,target_transform)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders",
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.roots = roots
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = target#self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class CostumeMixedImageFolder(CostumeImageFolder):
    def __init__(self, roots, category_indexs, transform, target_transform=None,
                 loader=default_loader, mode="RGB"):
        classes, class_to_idx = find_classes(roots)
        imgs = make_dataset(roots, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders",
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.roots = roots
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode
        self.labeled_transform = self.transform[0]
        self.unlabeled_transform = self.transform[1]
        self.category_indexs = category_indexs

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            if target in self.category_indexs:
                img = self.labeled_transform(img)
            else:
                img = self.unlabeled_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
import imageio
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm
class TinyImageNetPaths:
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id)) #, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))


class CostumeImageFolderTesting(data.Dataset):
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

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root,class_list, transform=None, target_transform=None,
                 loader=default_loader, mode="RGB"):
        #classes, class_to_idx = find_classes(roots)
        
        #imgs = make_dataset(roots, class_to_idx)
        
        pre_imgs = TinyImageNetPaths(root, False)
        #print (pre_imgs.paths['val'])
        imgs = [path for path in pre_imgs.paths['val'] if path[1] in class_list]
        #print (pre_imgs.paths['val'])
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders",
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.class_list = class_list
        #self.class_to_idx = class_to_idx
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
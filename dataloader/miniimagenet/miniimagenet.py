import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

class MiniImageNet(Dataset):

    def __init__(self, root='./data', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = None
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert(len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)
        self.multi_train = False  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        if self.multi_train:
            image = Image.open(path).convert('RGB')
            classify_image = [self.transform(image)]
            multi_crop, multi_crop_params = self.crop_transform(image)
            assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
            if isinstance(self.secondary_transform, list):
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
            else:
                multi_crop = [self.secondary_transform(x) for x in multi_crop]
            total_image = classify_image + multi_crop
        else:
            total_image = self.transform(Image.open(path).convert('RGB'))
        return total_image, targets

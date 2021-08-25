import numpy as np
import os
import torch
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .Loader import Loader


class TinyImageNet(Loader):
    def __init__(self, num_train):
        self.num_train = num_train
        self.num_test = num_train // 10
        self.name = 'tiny_imagenet'
        self.data_path = 'data/'
        data_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.data_path+'tiny-imagenet-200'):
            with urlopen(data_url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(self.data_path)
            self.create_val_img_folder(self.data_path+'tiny-imagenet-200')
        X_data, y_data = self.load_data()
        self.X_test_data = X_data[self.num_train : self.num_train + self.num_test]
        self.y_test_data = y_data[self.num_train : self.num_train + self.num_test]
        self.X_data = X_data[0 : self.num_train]
        self.y_data = y_data[0 : self.num_train]

    def load_data(self):
        data_transform = transforms.Compose([transforms.ToTensor()])
        image_datasets = datasets.ImageFolder(os.path.join(self.data_path + 'tiny-imagenet-200', 'train'), data_transform)
        batch_size = len(image_datasets) # 1000 for test
        # change dataloaders to numpy
        train_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=64)
        raw_train_X, raw_train_Y = next(iter(train_dataloader))
        raw_train_X = raw_train_X.numpy() # (100000, 3, 64, 64)
        raw_train_Y = raw_train_Y.numpy()  # (100000, )
        # print(raw_train_Y[:100])
        # exit()
        return raw_train_X, raw_train_Y

    def create_val_img_folder(self, dataset_dir):
        '''
        This method is responsible for separating validation images into separate sub folders
        '''
        val_dir = os.path.join(dataset_dir, 'val')
        img_dir = os.path.join(val_dir, 'images')

        fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
        data = fp.readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

    # Create folder if not present and move images into proper folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(img_dir, img)):
                os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


    def load_val_data(self):
        data_transform = transforms.Compose([transforms.ToTensor()])
        image_datasets = datasets.ImageFolder(os.path.join(self.data_path + 'tiny-imagenet-200', 'val/images'), data_transform)
        # print(image_datasets.class_to_idx)
        batch_size = len(image_datasets)
        val_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=64)
        val_X, val_Y = next(iter(val_dataloader))
        # change dataloaders to numpy
        val_X = val_X.numpy() # (10000, 3, 64, 64)
        val_Y = val_Y.numpy()  # (10000, )
        return val_X, val_Y

    def prepare_data(self):
        return self.X_data, self.y_data, self.X_test_data, self.y_test_data

import copy
import numpy as np
import os
import urllib
import h5py
import cv2

from .Loader import Loader

class USPS(Loader):
    def __init__(self, num_train, expand=False):
        self.name = 'usps'
        self.data_path = './data/'
        self.num_train = num_train
        self.num_test = num_train
        data_url = "https://www.kaggle.com/bistaumanga/usps-dataset?select=usps.h5"

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            mp3file = urllib.request.urlopen(data_url)
            with open(self.data_path + "usps.h5",'wb') as output:
                output.write(mp3file.read())
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(expand)

    def load_data(self, expand):
        with h5py.File(self.data_path + 'usps/usps.h5', 'r') as hf:
            train_data = hf.get('train')
            X_tr = np.array(train_data.get('data')[:])
            raw_usps_Y = train_data.get('target')[:]
        if expand == True:
            for i in range(len(X_tr)):
                temp = np.array(X_tr[i].reshape(16,16))
                temp = cv2.resize(temp, (28, 28), interpolation=cv2.INTER_CUBIC)
                temp = np.expand_dims(temp, axis=0)
                if i == 0:
                    raw_usps_X = temp
                else:
                    raw_usps_X = np.concatenate((raw_usps_X, temp), axis=0)

        return raw_usps_X[:self.num_train], raw_usps_Y[:self.num_train], raw_usps_X[self.num_train:self.num_train + self.num_test], raw_usps_Y[self.num_train:self.num_train + self.num_test]

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

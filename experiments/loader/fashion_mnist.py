import copy
import numpy as np
from tensorflow import keras

from .Loader import Loader

class FashionMnist(Loader):

    def __init__(self, num_train, flatten=True):
        self.name = 'fashion_mnist'
        self.num_train = num_train
        self.num_test = num_train // 10
        self.flatten = flatten
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.shuffle_data()
        self.x_train = self.x_train[:self.num_train]
        self.y_train = self.y_train[:self.num_train]
        self.x_test = self.x_test[:self.num_test]
        self.y_test = self.y_test[:self.num_test]

    def load_data(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        indice_0 = np.where(y_train==0)[0]
        indice_1 = np.where(y_train==6)[0]
        indice_all = np.hstack((indice_0, indice_1))
        x_train = x_train[indice_all]
        y_train = np.hstack((np.zeros(len(indice_0), dtype=np.int64), np.ones(len(indice_1), dtype=np.int64)))
        indice_0 = np.where(y_test==0)[0]
        indice_1 = np.where(y_test==6)[0]
        indice_all = np.hstack((indice_0, indice_1))
        x_test = x_test[indice_all]
        y_test = np.hstack((np.zeros(len(indice_0), dtype=np.int64), np.ones(len(indice_1), dtype=np.int64)))
        if self.flatten:
            x_train = np.reshape(x_train, [-1, 28 * 28])
            x_train = x_train.astype(np.float32) / 255
            x_test = np.reshape(x_test, [-1, 28 * 28])
            x_test = x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    def shuffle_data(self):
        ind = np.random.permutation(len(self.x_train))
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]
        ind = np.random.permutation(len(self.x_test))
        self.x_test, self.y_test = self.x_test[ind], self.y_test[ind]

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
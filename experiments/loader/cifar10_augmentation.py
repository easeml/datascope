import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

from .Loader import Loader

class CIFAR10Aug(Loader):

    def __init__(self, num_train, flatten=True):
        self.name = 'cifar10'
        self.num_train = num_train
        self.num_test = num_train // 10
        self.flatten = flatten
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.shuffle_data()
        self.x_train = self.x_train[:self.num_train]
        self.y_train = self.y_train[:self.num_train]
        self.x_test = self.x_test[:self.num_test]
        self.y_test = self.y_test[:self.num_test]
        self.aug = 10

    def load_data(self):
        cifar10 = keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        if self.flatten:
            x_train = np.reshape(x_train, [-1, 32 * 32 * 3])
            x_train = x_train.astype(np.float32) / 255
            x_test = np.reshape(x_test, [-1, 32 * 32 * 3])
            x_test = x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    def augment(self):
        # rotate
        dg1 = ImageDataGenerator(rotation_range=360)
        dg2 = ImageDataGenerator(width_shift_range=0.5)
        dg3 = ImageDataGenerator(height_shift_range=0.5)
        dg4 = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
        dg5 = ImageDataGenerator(brightness_range=(-1,1))
        dg6 = ImageDataGenerator(shear_range=0.5)
        dg7 = ImageDataGenerator(zoom_range=0.3)
        dg8 = ImageDataGenerator(channel_shift_range=0.4)
        dg9 = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        augmentations = [dg1, dg2, dg3, dg4, dg5, dg6, dg7, dg8, dg9]

        x_res = self.x_train[:100]
        y_res = self.y_train[:100]

        for dgs in augmentations:
            dgs.fit(self.x_train)
            x_batch, y_batch = dgs.flow(self.x_train, self.y_train, batch_size=100).next()
            x_res = np.concatenate((x_res, x_batch))
            y_res = np.concatenate((y_res, y_batch))

        forksets = []
        for i in range(10): 
            forksets += 100 * [i]
        forksets = np.array(forksets)

        x_test = self.x_test
        y_test = self.y_test

        if self.flatten:
            x_res = np.reshape(x_res, [-1, 32 * 32 * 3])
            x_res = x_res.astype(np.float32) / 255
            x_test = np.reshape(x_test, [-1, 32 * 32 * 3])
            x_test = x_test.astype(np.float32) / 255

        return x_res, y_res, x_test, y_test, forksets

    def shuffle_data(self):
        ind = np.random.permutation(len(self.x_train))
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]
        ind = np.random.permutation(len(self.x_test))
        self.x_test, self.y_test = self.x_test[ind], self.y_test[ind]

    def load_aug_data(self):
        data = np.load('./data/CIFAR10_aug/data.npz')

        forksets = []
        for i in range(10): 
            forksets += 100 * [i]
        forksets = np.array(forksets)

        x_res = data['X']
        y_res =  data['y']
        x_test = data['X_test']
        y_test = data['y_test']

        if self.flatten:
            x_res = np.reshape(x_res, [-1, 32 * 32 * 3])
            x_train = x_res.astype(np.float32) / 255
            x_test = np.reshape(x_test, [-1, 32 * 32 * 3])
            x_test = x_test.astype(np.float32) / 255

        return x_res, y_res, x_test, y_test, forksets

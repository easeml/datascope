import copy
import numpy as np
import pickle
import random

from .Loader import Loader

class Spam(Loader):

    def __init__(self, num_train):
        self.name = 'spam'

        self.num_train = num_train
        self.num_test = num_train // 10

        data = pickle.load(open("Shapley_data/SPAM_data/spamdata.pkl", "rb"))

        X_data = data["X"]
        y_data = data["y"].as_matrix()
        index = np.arange(np.shape(X_data)[0])
        state = np.random.get_state()
        np.random.shuffle(index)
        X_data = X_data[index, :]
        np.random.set_state(state)
        np.random.shuffle(y_data)

        self.X_test_data = X_data[num_train:num_train+num_train//10]
        self.y_test_data = y_data[num_train:num_train+num_train//10]
        self.X_data = X_data[0:num_train]
        self.y_data = y_data[0:num_train]

    def prepare_data(self):
        return self.X_data, self.y_data, self.X_test_data, self.y_test_data
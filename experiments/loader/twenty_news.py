from sklearn.datasets import fetch_20newsgroups
from .Loader import Loader
import numpy as np

class TwentyNews(Loader):

    def __init__(self, num_train, flatten=True):
        self.name = '20newsgroups'
        self.num_train = num_train
        self.num_test = num_train // 10
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.x_train = self.x_train[:self.num_train]
        self.y_train = self.y_train[:self.num_train]
        self.x_test = self.x_test[:self.num_test]
        self.y_test = self.y_test[:self.num_test]

    def load_data(self):
        categories = ['comp.graphics', 'sci.med']
        twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
        twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

        x_train = np.array(twenty_train.data)
        y_train = np.array(twenty_train.target)
        x_test = np.array(twenty_test.data)
        y_test = np.array(twenty_test.target)

        return x_train, y_train, x_test, y_test

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
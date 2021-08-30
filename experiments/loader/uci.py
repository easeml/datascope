import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from .Loader import Loader

class UCI(Loader):
    def __init__(self, num_train, sensitive_feature=9):
        """
        num_train: size of trainingsdata 
        sensitive_feature: index of the sensitive feature
        """
        self.name = 'uci'
        self.num_train = num_train
        self.num_test = num_train // 10
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()
        self.x_train = self.x_train[:self.num_train]
        self.y_train = self.y_train[:self.num_train]
        self.x_test = self.x_test[:self.num_test]
        self.y_test = self.y_test[:self.num_test]
        self.sensitive_feature = sensitive_feature
        #self.x_test, self.y_test = self.create_fair_test_data(x_test, y_test)

        #self.feature_names

    def load_data(self):
        data = fetch_openml(data_id=1590, as_frame=False)
        X = data.data
        X = np.nan_to_num(X)
        Y = (data.target == '>50K') * 1
        self.feature_names = data.feature_names
        return train_test_split(X, Y, test_size=0.2)

    def prepare_preselected_unfair(self):
        '''
        Initial demographic ratio parity is 0.52
        Initial equalized odds ratio is 0.42
        '''
        data = np.load('./data/Unfair_UCI/data.npz')
        return data['X'][:self.num_train], data['y'][:self.num_train], data['X_test'], data['y_test']

    def create_unfair_train_data(self, x_train, y_train, ratio):
        train_male_indices = np.where(x_train[:,self.sensitive_feature] == 0)
        train_female_indices = np.where(x_train[:,self.sensitive_feature] == 1)
        train_1_indices = np.where(y_train == 1)[0]
        train_0_indices = np.where(y_train == 0)[0]

        train_male_1_indices = np.intersect1d(train_male_indices,train_1_indices)[:(self.num_train // 100 * ratio)] 
        train_male_0_indices = np.intersect1d(train_male_indices,train_0_indices)[:(self.num_train // 100)]
        train_female_1_indices = np.intersect1d(train_female_indices,train_1_indices)[:(self.num_train // 100)]
        train_female_0_indices = np.intersect1d(train_female_indices,train_0_indices)[:(self.num_train // 100 * ratio)]

        unbalanced_train = np.concatenate([train_male_1_indices, train_male_0_indices, train_female_1_indices, train_female_0_indices])
        X_train_b = x_train[unbalanced_train]
        y_train_b = y_train[unbalanced_train]
        return X_train_b, y_train_b

    def create_fair_test_data(self, x_test, y_test):
        test_male_indices = np.where(x_test[:,self.sensitive_feature] == 0)
        test_female_indices = np.where(x_test[:,self.sensitive_feature] == 1)
        test_1_indices = np.where(y_test == 1)[0]
        test_0_indices = np.where(y_test == 0)[0]

        test_male_1_indices = np.intersect1d(test_male_indices,test_1_indices)[:(self.num_test // 4)]
        test_male_0_indices = np.intersect1d(test_male_indices,test_0_indices)[:(self.num_test // 4)]
        test_female_1_indices = np.intersect1d(test_female_indices,test_1_indices)[:(self.num_test // 4)]
        test_female_0_indices = np.intersect1d(test_female_indices,test_0_indices)[:(self.num_test // 4)]

        balanced_test = np.concatenate([test_male_1_indices, test_male_0_indices, test_female_1_indices, test_female_0_indices])
        X_test_b = x_test[balanced_test]
        y_test_b = y_test[balanced_test]
        return X_test_b, y_test_b

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_keys(self):
        return self.feature_names

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from .Loader import Loader


class HateSpeech(Loader):
    def __init__(self, num_train):
        self.name = 'hate_speech'
        self.corpus = None
        self.num_train = num_train
        self.num_test = num_train // 10
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_data()
        self.x_train = self.x_train
        self.y_train = self.y_train
        #self.x_test, self.y_test = self.create_fair_test_data(x_test, y_test)

    def load_data(self):
        data = pd.read_csv('data/Hate_data/hate-speech-race.csv', index_col=0)
        data = data.dropna()
        data = data[:self.num_train]
        X = data[["tokens","race"]]
        self.corpus = data[["tokens"]]

        def create_binary_label(el):
            if el == 0: #hate speech
                return 0
            if el == 1: #offensive
                return 0
            if el == 2: #other
                return 1

        Y = data["class"].apply(create_binary_label)
        return train_test_split(X, Y, test_size=0.5)

    def create_fair_test_data(self, x_test, y_test):
        # only use 'race' as sensitive feature
        test_aa_indices = x_test[x_test['race'] == 0].index
        test_w_indices = x_test[x_test['race'] == 1].index
        test_1_indices = y_test[y_test == 1].index # offensive class
        test_0_indices = y_test[y_test == 2].index # neither class

        test_aa_1_indices = test_aa_indices.intersection(test_1_indices)[:250]
        test_aa_0_indices = test_aa_indices.intersection(test_0_indices)[:250]
        test_w_1_indices = test_w_indices.intersection(test_1_indices)[:250]
        test_w_0_indices = test_w_indices.intersection(test_0_indices)[:250]

        balanced_test = test_aa_1_indices.union(test_aa_0_indices)
        balanced_test = balanced_test.union(test_w_1_indices)
        balanced_test = balanced_test.union(test_w_0_indices)

        X_test_b = x_test.loc[balanced_test]
        y_test_b = y_test.loc[balanced_test]
        return X_test_b, y_test_b

    def create_unfair_train_data(self, x_train, y_train):
        # only use 'race' as sensitive feature
        print(x_train['race'].hist())
        train_aa_indices = x_train[x_train['race'] == 0].index
        train_w_indices = x_train[x_train['race'] == 1].index
        train_1_indices = y_train[y_train == 1].index # offensive class
        train_0_indices = y_train[y_train == 2].index # neither class

        train_aa_1_indices = train_aa_indices.intersection(train_1_indices)[:250]
        train_aa_0_indices = train_aa_indices.intersection(train_0_indices)[:250]
        train_w_1_indices = train_w_indices.intersection(train_1_indices)[:250]
        train_w_0_indices = train_w_indices.intersection(train_0_indices)[:250]

        unbalanced_train = train_aa_1_indices.union(train_aa_0_indices)
        unbalanced_train = unbalanced_train.union(train_w_1_indices)
        unbalanced_train = unbalanced_train.union(train_w_0_indices)

        X_train_b = x_train.loc[unbalanced_train]
        y_train_b = y_train.loc[unbalanced_train]
        return X_train_b, y_train_b
        
    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

import numpy as np
from sklearn.neural_network._base import ACTIVATIONS
import scipy.sparse as sps

import torch
import torch.nn.functional as F
from datascope.algorithms import Measure

class KNN_Shapley(Measure):

    def __init__(self, K=10):
        self.name = 'KNN_Shapley'
        self.K = K

    def _get_shapley_value_torch(self, X_train, y_train, X_test, y_test):
        """
        Calculate the Shapley value of a single test sample in torch.
        """
        N = len(X_train)
        M = len(X_test)

        dist = torch.cdist(X_train.view(len(X_train), -1), X_test.view(len(X_test), -1))
        _, indices = torch.sort(dist, axis=0)
        y_sorted = y_train[indices]

        score = torch.zeros_like(dist)

        score[indices[N-1], range(M)] = (y_sorted[N-1] == y_test).float() / N
        for i in range(N-2, -1, -1):
            score[indices[i], range(M)] = score[indices[i+1], range(M)] + \
                                        1/self.K * ((y_sorted[i] == y_test).float() - (y_sorted[i+1] == y_test).float()) * min(self.K, i+1) / (i+1)
        return score.mean(axis=1)

    def _get_shapley_value_np(self, X_train, y_train, X_test, y_test):
        """
        Calculate the Shapley value of a single test sample in numpy.
        """
        N = X_train.shape[0]
        M = X_test.shape[0]
        s = np.zeros((N, M))

        for i, (X, y) in enumerate(zip(X_test, y_test)):
            #print((X_train).shape, (X.reshape(-1).shape))
            #print(X_train - X)
            diff = (X_train - X).reshape(N, -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            idx = np.argsort(dist)
            ans = y_train[idx]
            s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
            cur = N - 2
            for j in range(N - 1):
                s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / self.K * (min(cur, self.K - 1) + 1) / (cur + 1)
                cur -= 1 
        return np.mean(s, axis=1)

    def score(self, X_train, y_train, X_test, y_test, model=None, use_torch=False, **kwargs):
        """
        Run the model on the test data and calculate the Shapley value.
        """
        self.model = model

        if sps.issparse(X_train):
            X_train = X_train.toarray()
        if sps.issparse(X_test):
            X_test = X_test.toarray()

        if use_torch:
            shapley = self._get_shapley_value_torch(X_train, y_train, X_test, y_test)
        else:
            shapley = self._get_shapley_value_np(X_train, y_train, X_test, y_test)
        return shapley
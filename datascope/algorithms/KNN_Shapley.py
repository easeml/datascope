import numpy as np
from sklearn.neural_network._base import ACTIVATIONS
import scipy.sparse as sps

import torch
import torch.nn.functional as F
from datascope.algorithms import Measure

class KNN_Shapley(Measure):

    def __init__(self, K=1):
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


    def _get_1NN_fork_shapley_value_np(self, X_train, y_train, X_test, y_test, forksets):
        """
        Calculate the 1-NN Shapley value of a forksets in numpy.
        """
        assert(self.K == 1) # only works for K = 1

        N = len(forksets)
        forkset_idxs = np.empty(N, dtype=int) #stores the index of the chosen element for each forkset
        M = X_test.shape[0]
        s_fork = np.zeros((N, M))
        s = np.zeros(X_train.shape[0])

        for i, (X, y) in enumerate(zip(X_test, y_test)):
          # select an representative element for each fork
          for fidxs in forksets:
            X_fork_train = X_train[forksets[fidxs]]
            diff = (X_fork_train - X).reshape(X_fork_train.shape[0], -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            min_idx = np.argmin(dist)
            forkset_idxs[fidxs] = forksets[fidxs][min_idx]
          s_fork[:,i] = self._get_shapley_value_np(X_train[forkset_idxs], y_train[forkset_idxs], np.array([X]), np.array([y]))

        s_fork = np.mean(s_fork, axis=1)
        for fidxs in forksets:
          s[forksets[fidxs]] = s_fork[fidxs] / len(forksets[fidxs])
        return s

    def score(self, X_train, y_train, X_test, y_test, model=None, use_torch=False, forksets=None, fixed_y=None, recompute_v1=False, **kwargs):
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
            shapley = self._get_1NN_fork_shapley_value_np(X_train, y_train, X_test, y_test, forksets)

        if fixed_y is None:
            pass
        else:
            if recompute_v1:
                print("[DataScope] => Start KNN recompute v1")
                #TODO: Support forks

                indices = np.arange(X_train.shape[0]) # [0,1,2,...n]
                ordered_ind = np.zeros(X_train.shape[0])
                for i in range(X_train.shape[0]):
                    print(f'Recompute v1 {i}/{X_train.shape[0]}')
                    shapley = self._get_1NN_fork_shapley_value_np(X_train[indices], y_train[indices], X_test, y_test, forksets)
                    l_min_ind = np.argmin(shapley)

                    # translate it back to global indices
                    g_min_ind = indices[l_min_ind]
                    ordered_ind[g_min_ind] = i

                    # update indices and forksets
                    indices = np.delete(indices, l_min_ind)

                    forksets = {el:np.array([el]) for el in range(indices.shape[0])}
                    # TODO FORK: delete all entries of a certain g_min_ind
                    # TODO FORK: change all existing indices to local indices
                    
                print(ordered_ind)
                shapley = ordered_ind # return an ordered list of elements from recompute
            
            else:
                print("[DataScope] => Start KNN recompute v2")
                # TODO: Support forks

                checked_ind = []
                ordered_ind = np.zeros(X_train.shape[0])

                for i in range(X_train.shape[0]):
                    print(f'Recompute v2 {i}/{X_train.shape[0]}')
                    shapley = self._get_1NN_fork_shapley_value_np(X_train, y_train, X_test, y_test, forksets)
                    sorted_indices = np.argsort(shapley)
                    # reset min indices
                    min_ind = -1
                    j = 0
                    while(min_ind == -1):
                        if sorted_indices[j] not in checked_ind:
                            min_ind = sorted_indices[j]
                        else:
                            j += 1 # increase j
                    checked_ind += [min_ind] # keep list of ind
                    ordered_ind[min_ind] = i
                    y_train[min_ind] = fixed_y[min_ind]
                print(ordered_ind)
                shapley = ordered_ind

        return shapley

import numpy as np
from sklearn.neural_network._base import ACTIVATIONS
import scipy.sparse as sps

import torch
import torch.nn.functional as F
from datascope.algorithms import Measure
import math
import copy
from datascope.algorithms import tools

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

    def _get_shapley_value_np(self, X_train, y_train, X_test, y_test, sources):
        """
        Calculate the Shapley value of a single test sample in numpy.
        """    
        bound = 30
        # X_train = X_train[0:bound]
        # y_train = y_train[0:bound]
        # X_test = X_test[0:bound]
        # y_test = y_test[0:bound]
        sources = dict(list(sources.items())[0:bound])

        # tuple_per_fork_sets[i] is the number of tuple in fork set i
        tuple_per_fork_sets = 10000*[0]

        n_fork_sets = 0
        for k in sources.keys():
            tuple_per_fork_sets[k] += 1
            if(k > n_fork_sets):
                n_fork_sets = k
        
        fork_sets_cardinalities = [x for x in tuple_per_fork_sets if x!=0]
        n_units = len(fork_sets_cardinalities)

        fork_sets = []
        for i in range(n_units):
            fork_sets.append(fork_sets_cardinalities[i]*[])

        for k in sources.keys():
            fork_sets[k].append([np.mean(X_train[k]), y_train[k], sources[k][0]])

        all_tuples = []
        for fork_set in fork_sets:
            for tuples in fork_set:
                all_tuples.append(tuples)
        s = []
        units = [x+1 for x in range(n_units)]
        for i, (X, y) in enumerate(zip(X_test, y_test)):
            res = []
            shapley_value(n_units, 1, y, all_tuples, fork_sets, units, res)
            s.append(res)
        s = np.array(s)
        return np.mean(s, axis=1)

    def score(self, X_train, y_train, X_test, y_test, model=None, use_torch=False, **kwargs):
        """
        Run the model on the test data and calculate the Shapley value.
        """
        self.model = model
        sources = kwargs["sources"]

        if sps.issparse(X_train):
            X_train = X_train.toarray()
        if sps.issparse(X_test):
            X_test = X_test.toarray()

        if use_torch:
            shapley = self._get_shapley_value_torch(X_train, y_train, X_test, y_test)
        else:
            shapley = self._get_shapley_value_np(X_train, y_train, X_test, y_test, sources)
        return shapley

def shapley_value(n_unit, k, l_t, dp, d, uI, res):

    S_k = [[l_1, l_2, l_3, l_4] for l_1 in range(k+1) for l_2 in range(k+1)
           for l_3 in range(k+1) for l_4 in range(k+1)]
    for q in uI:
        phi_q = 0.0
        for D_b in dp:
            for D_c in dp:
                if(D_c[2] != q and D_c[0] >= D_b[0]):
                    tallies = 4*[[]]
                    i = D_b[2]
                    j = D_c[2]

                    for h in range(n_unit):
                        if h != i-1 and h != j-1 and h != q-1 and tools.too_far_set(d[h], D_b, D_c):
                            tallies = copy.copy(tools.append_tally(tallies, d[h], D_b, D_c, l_t))
    
                    tallies = copy.copy(tools.append_tally(tallies, d[j-1], D_b, D_c, l_t))

                    if((i == q and i != j) or (i != q and i == j)):
                        m = 0
                    else:
                        m = 1
                        tallies = copy.copy(tools.append_tally(tallies, d[i-1], D_b, D_c, l_t))

                    tallies[0].append(tools.count(d[q-1], D_b, 0, l_t))
                    tallies[2].append(tools.count(d[q-1], D_b, 1, l_t)) 
                    l = len(tallies[0])

                    t1, t2, t3, t4 = tallies[0], tallies[1], tallies[2], tallies[3]
                    DP = np.array((l+1)*[(k+1)*[(k+1)*[(k+1)*[(k+1)*[(k+1)*[0]]]]]]).tolist()

                    for n in range(l+1):
                        DP[n][0][0][0][0][0] = 1
                    
                    ranges = []
                    
                    for n in range(1, l+1):
                        if((m == 0 and n < l-1) or (m == 1 and n < l-2)):
                            ranges.append([[l1, l2, l11, l22] for [l1, l2, l11, l22] in S_k if l1 < t1[n-1] or l2 < t2[n-1] or l11 < t3[n-1]
                                     or l22 < t4[n-1]])
                    
                    for n in range(1,l+1):
                        if((m == 0 and n < l-1) or (m == 1 and n < l-2)):
                            for a in range(1, k+1):
                                for l1 in range(t1[n-1], k+1):
                                    for l2 in range(t2[n-1], k+1):
                                        for l11 in range(t3[n-1], k+1):
                                            for l22 in range(t4[n-1], k+1):
                                                DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1][l2][l11][l22] + DP[n-1][a-1][l1-t1[n-1]][l2-t2[n-1]][l11-t3[n-1]][l22-t4[n-1]]
                                for [l1, l2, l11, l22] in ranges[n-1]:
                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1][l2][l11][l22]
                                        
                        else:
                            for a in range(k+1):
                                for l1 in range(k+1):
                                    for l2 in range(k+1):
                                        for l11 in range(k+1):
                                            for l22 in range(k+1):               
                                                if(n == l):
                                                    if(l1>=t1[n-1] and l11>=t3[n-1]):
                                                        DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1-t1[n-1]][l2][l11-t3[n-1]][l22]
                                                elif(a > 0 and l1>=t1[n-1] and l2>=t2[n-1] and l11>=t3[n-1] and l22>=t4[n-1]):
                                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a-1][l1-t1[n-1]][l2-t2[n-1]][l11-t3[n-1]][l22-t4[n-1]]
                            
                    n_out_set = n_unit-len(t1)
                    m = 2
                    if((i == q and i != j) or (i != q and i == j)):
                        m = 1
                 
                    for l1 in range(k+1):
                        for l2 in range(k+1):
                            DP_val = 0

                            for a in range(m,k+1):
                                for b in range(n_out_set+1):
                                    if(a-b >= m):
                                            DP_val = DP_val + math.comb(n_out_set, b)*DP[l][a-b][l1][l2][k-l1][k-l2]/math.comb(n_unit-1, a)
                            for a in range(k+1, n_unit):
                                for b in range(a-k,n_out_set+1):
                                    if(a-b >= m):
                                        DP_val = DP_val + math.comb(n_out_set, b)*DP[l][a-b][l1][l2][k-l1][k-l2]/math.comb(n_unit-1, a)
                            if(l1/k >= 0.5 and l2/k < 0.5):
                                phi_q = phi_q+DP_val
                            elif(l1/k < 0.5 and l2/k >= 0.5):
                                phi_q = phi_q-DP_val
        phi_q = phi_q/(n_unit)
        #print(q,"-th unit's shapley value", phi_q)
        res.append(phi_q)
import numpy as np
from sklearn.neural_network._base import ACTIVATIONS
import scipy.sparse as sps

import torch
import torch.nn.functional as F
from datascope.algorithms import Measure
import math
import copy
from datascope.algorithms import tools as t

import itertools

from numba import njit
from numba.tests.test_object_mode import forceobj

import ray
import multiprocessing



class KNN_Hard_Shapley(Measure):

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

    def _get_shapley_value_np(self, X_train, y_train, X_test, y_test, forksets):
        """
        Calculate the Shapley value of a single test sample in numpy.
        """ 
        print("K = ", self.K)
        # tuple_per_fork_sets[i] is the number of tuple in fork set i
        fork_sets_length = []

        for k in forksets.keys():
            fork_sets_length.append(len(forksets[k]))
        
        n_units = len(forksets.keys())

        print("Number of fork sets = ", n_units)

        s = []
        units = [x+1 for x in range(n_units)]
        for i, (X, y) in enumerate(zip(X_test, y_test)):
            #print(X.shape)
            #print(X_train[0].shape)
            #print(i)
            fork_sets = n_units*[]
            for k in forksets.keys():
                temp = []
                for j in forksets[k]:
                    diff = (X_train[j] - X)
                    temp.append([np.einsum('i,i', diff, diff), y_train[j], k+1])
                fork_sets.append(temp)
            #all_tuples = []
            #for fork_set in fork_sets:
            #    for tuples in fork_set:
            #        all_tuples.append(tuples)
            res = shapley_value_naive(n_units, self.K, y,fork_sets, units)
            s.append(res)
        s = np.array(s)
        s = np.mean(s, axis=0)
        res = []
        for i in range(n_units):
            res = res+fork_sets_length[i]*[s[i]]
        print(res)
        return np.array(res)

    def _get_shapley_value_np_numba(self, X_train, y_train, X_test, y_test, forksets):
        """
        Calculate the Shapley value of a single test sample in numpy.
        """ 
        print("K = ", self.K)
        # tuple_per_fork_sets[i] is the number of tuple in fork set i
        fork_sets_length = []

        for k in forksets.keys():
            fork_sets_length.append(len(forksets[k]))
        
        fork_sets_length_np = np.array(fork_sets_length)

        n_units = len(forksets.keys())

        print("Number of fork sets = ", n_units)

        units = np.array([x+1 for x in range(n_units)])
         
        futures = [computing.remote(n_units, self.K, forksets, X_train, y_train, i, X, y, units) for i, (X, y) in enumerate(zip(X_test, y_test))]
        s= np.mean(ray.get(futures),axis=0)
        res = []
        for i in range(n_units):
            res = res+fork_sets_length[i]*[s[i]]
        return np.array(res)

    def score(self, X_train, y_train, X_test, y_test, model=None, use_torch=False, forksets=None, **kwargs):
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
            shapley = self._get_shapley_value_np(X_train, y_train, X_test, y_test, forksets)
        return shapley

@ray.remote
def computing(n_units, K, forksets, X_train, y_train, i, X, y, units):
    fork_sets = n_units*[]
    for k in forksets.keys():
        temp = []
        for i in forksets[k]:
            temp.append(np.array([np.linalg.norm(X_train[i-1]-X), y_train[i-1], k+1]))
        fork_sets.append(temp)
    fork_sets_np = np.array(fork_sets)
    all_tuples = []
    for fork_set in fork_sets_np:
        for tuples in fork_set:
            all_tuples.append(tuples)
    all_tuples_np = np.array(all_tuples)
    res = shapley_value_naive(n_units, K, y, fork_sets, units)
    print("Shapley terminated")
    return res

def shapley_value_naive(n, k, l_t, d, unit_indices):
    results = []
    combinations = []
    current = []
    t.all_combi(n, current, combinations)
    for l in unit_indices:
        phi_q = 0.0
        q = l
        combinationss = t.filter_Q(combinations, q)
        for vector in combinationss:
            output = t.pipeline_output(d, vector)
            vector_q = []
            vector_q = copy.copy(vector)
            vector_q[q-1] = 1
            output_q = t.pipeline_output(d, vector_q)
            if(len(output) < k or len(output) < k):
                continue
            prop_with_q = t.topK_with_Lt(output_q, k, l_t)
            prop_without_q = t.topK_with_Lt(output, k, l_t)
            if(prop_with_q/k >= 0.5):
                prop_with_q = 1.0
            else:
                prop_with_q = 0.0
            
            if(prop_without_q/k >= 0.5):
                prop_without_q = 1.0
            else:
                prop_without_q = 0.0
            diff = prop_with_q-prop_without_q
            count = len([x for x in vector if x == 1])
            vMv = diff/math.comb(n-1, count)
            phi_q = phi_q + vMv
        phi_q = phi_q/n
        results.append(phi_q)
    return np.array(results)

@njit(cache = True)
def shapley_value(n_unit, k, l_t, dp, d, uI):
    res = np.empty(0, dtype=np.float64)
    for q in uI:
        phi_q = np.float64(0.0)
        for D_b in dp:
            for D_c in dp:
                #print("D_b, D_c", D_b, D_c, " q = ", q)
                if(np.int32(D_c[2]) != q and D_c[0] >= D_b[0]):
                    i = np.int32([D_b[2]])
                    j = np.int32([D_c[2]])
                    tallies0 = np.array([0])
                    tallies1 = np.array([0])
                    tallies2 = np.array([0])
                    tallies3 = np.array([0])
                    for h in range(n_unit):
                        j1 = 0
                        for ii in range(d[h].shape[1]):
                            if d[h, ii, 0] <= D_b[0]:
                                j1=j1+1
                        j2 = 0
                        for ii in range(d[h].shape[1]):
                            if d[h, ii, 0] <= D_c[0]:
                                j2=j2+1
                        if h != i-1 and h != j-1 and h != q-1 and (j1 > 0 or j2 > 0):
                            jj = 0
                            for ii in range(d[h].shape[1]):
                                if d[h, ii, 0] <= D_b[0] and np.int32(d[h, ii, 1]) == l_t:
                                    jj=jj+1
                            tallies0 = np.append(tallies0, [jj])
                            jj = 0
                            for ii in range(d[h].shape[1]):
                                if d[h, ii, 0] <= D_b[0] and np.int32(d[h, ii, 1]) != l_t:
                                    jj=jj+1
                            tallies2 = np.append(tallies2, [jj])
                            jj = 0
                            for ii in range(d[h].shape[1]):
                                if d[h, ii, 0] <= D_c[0] and np.int32(d[h, ii, 1]) == l_t:
                                    jj=jj+1
                            tallies1 = np.append(tallies1, [jj])
                            jj = 0
                            for ii in range(d[h].shape[1]):
                                if d[h, ii, 0] <= D_c[0] and np.int32(d[h, ii, 1]) != l_t:
                                    jj=jj+1
                            tallies3 = np.append(tallies3, [jj])  
                    #print("Tallies after h", tallies0, tallies1, tallies2, tallies3)
        
                    jj = 0 
                    for ii in range(d[j-1].shape[1]):
                        #print("Compa j 1: ", (d[j-1, ii, 1].astype(np.int32))[0], l_t)
                        if d[j-1, ii, 0] <= D_b[0] and (d[j-1, ii, 1].astype(np.int32))[0] == l_t:
                            jj=jj+1
                    tallies0 = np.append(tallies0, [jj])
                    jj = 0
                    for ii in range(d[j-1].shape[1]):
                        #print("Compa j 2: ", (d[j-1, ii, 1].astype(np.int32))[0], l_t)
                        if d[j-1, ii, 0] <= D_b[0] and (d[j-1, ii, 1].astype(np.int32))[0] != l_t:
                            jj=jj+1
                    tallies2 = np.append(tallies2, [jj])
                    jj = 0
                    for ii in range(d[j-1].shape[1]):
                        #print("Compa j 3: ", (d[j-1, ii, 1].astype(np.int32))[0], l_t)
                        if d[j-1, ii, 0] <= D_c[0] and (d[j-1, ii, 1].astype(np.int32))[0] == l_t:
                            jj=jj+1
                    tallies1 = np.append(tallies1, [jj])
                    jj = 0
                    for ii in range(d[j-1].shape[1]):
                        #print("Compa j 4: ", (d[j-1, ii, 1].astype(np.int32))[0], l_t)
                        if d[j-1, ii, 0] <= D_c[0] and (d[j-1, ii, 1].astype(np.int32))[0] != l_t:
                            jj=jj+1
                    tallies3 = np.append(tallies3, [jj])
                    
                    #print("After j ", tallies0, tallies1, tallies2, tallies3)
                    
                    if((i == q and i != j) or (i != q and i == j)):
                        m = 0
                    else:
                        m = 1
                        jj = 0
                        for ii in range(d[i-1].shape[1]):
                            if d[i-1, ii, 0] <= D_b[0] and (d[i-1, ii, 1].astype(np.int32))[0] == l_t:
                                jj=jj+1
                        tallies0 = np.append(tallies0, [jj])
                        jj = 0
                        for ii in range(d[i-1].shape[1]):
                            if d[i-1, ii, 0] <= D_b[0] and (d[i-1, ii, 1].astype(np.int32))[0] != l_t:
                                jj=jj+1
                        tallies2 = np.append(tallies2, [jj])
                        jj = 0
                        for ii in range(d[i-1].shape[1]):
                            if d[i-1, ii, 0] <= D_c[0] and (d[i-1, ii, 1].astype(np.int32))[0] == l_t:
                                jj=jj+1
                        tallies1 = np.append(tallies1, [jj])
                        jj = 0
                        for ii in range(d[i-1].shape[1]):
                            if d[i-1, ii, 0] <= D_c[0] and (d[i-1, ii, 1].astype(np.int32))[0] != l_t:
                                jj=jj+1
                        tallies3 = np.append(tallies3, [jj])
                    #print("After i ", tallies0, tallies1, tallies2, tallies3)
                    
                    jj = 0
                    for ii in range(d[q-1].shape[1]):
                        if d[q-1, ii, 0] <= D_b[0] and np.int32(d[q-1, ii, 1]) == l_t:
                            jj=jj+1
                    tallies0 = np.append(tallies0, [jj])
                    jj = 0
                    for ii in range(d[q-1].shape[1]):
                        if d[q-1, ii, 0] <= D_b[0] and np.int32(d[q-1, ii, 1]) != l_t:
                            jj=jj+1
                    tallies2 = np.append(tallies2, [jj])
                    #print("After q ", tallies0, tallies1, tallies2, tallies3)
                    
                    #print("Before", tallies0, tallies1, tallies2, tallies3)
                    tallies0 = np.delete(tallies0, 0)
                    tallies1 = np.delete(tallies1, 0)
                    tallies2 = np.delete(tallies2, 0)
                    tallies3 = np.delete(tallies3, 0)
                    
                    #print("Final tallies ", tallies0, tallies1, tallies2, tallies3)
                    
                    l = tallies0.size
                    t1, t2, t3, t4 = tallies0, tallies1, tallies2, tallies3
                    DP = np.zeros((l+1, k+1, k+1, k+1, k+1, k+1), dtype=np.int32)

                    for n in range(l+1):
                        DP[n, 0, 0, 0, 0, 0] = 1

                    for n in range(1,l+1):
                        for a in range(1, k+1):
                            for l1 in range(k+1):
                                for l2 in range(k+1):
                                    for l11 in range(k+1):
                                        for l22 in range(k+1):
                                            # a)
                                            if(n == l):
                                                if(l1>=t1[n-1] and l11>=t3[n-1]):
                                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1-t1[n-1]][l2][l11-t3[n-1]][l22]
                                            # b)
                                            elif((m == 0 and n == l-1) or (m == 1 and (n == l-1 or n == l-2))):
                                                if(l1>=t1[n-1] and l2>=t2[n-1] and l11>=t3[n-1] and l22>=t4[n-1]):
                                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a-1][l1-t1[n-1]][l2-t2[n-1]][l11-t3[n-1]][l22-t4[n-1]]
                                            else:    
                                                if (l1>=t1[n-1] and l2>=t2[n-1] and l11>=t3[n-1] and l22>=t4[n-1]):
                                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1][l2][l11][l22] + DP[n-1][a-1][l1-t1[n-1]][l2-t2[n-1]][l11-t3[n-1]][l22-t4[n-1]]
                                                else:
                                                    DP[n][a][l1][l2][l11][l22] = DP[n-1][a][l1][l2][l11][l22]
                    
                    n_out_set = n_unit-l
                    m = 2
                    if((i == q and i != j) or (i != q and i == j)):
                        m = 1
                 
                    for l1 in range(k+1):
                        for l2 in range(k+1):
                            DP_val = 0
                            for a in range(m,k+1):
                                for b in range(n_out_set+1):
                                    if(a-b >= m):
                                        nosFac = np.int64(1)
                                        bFac = np.int64(1)
                                        nFac = np.int64(1)
                                        aFac = np.int64(1)
                                        nmbFac = np.int64(1)
                                        nmaFac = np.int64(1)
                                        for ii in range(n_out_set):
                                            nosFac = nosFac*(ii+1)
                                        for ii in range(b):
                                            bFac = bFac*(ii+1)
                                        for ii in range(n_unit-1):
                                            nFac = nFac*(ii+1)
                                        for ii in range(a):
                                            aFac = aFac*(ii+1)
                                        for ii in range(n_out_set-b):
                                            nmbFac = nmbFac*(ii+1)
                                        for ii in range(n_unit-1-a):
                                            nmaFac = nmaFac*(ii+1)
                                        comb1 = nosFac/(bFac*(nmbFac))
                                        comb2 = nFac/(aFac*(nmaFac))
                                        # print(n_out_set, b, n_unit, a, n_out_set-b, n_unit-1-a)
                                        # print(nosFac, bFac, nmbFac, nFac, aFac, nmaFac)
                                        # print(comb1, comb2)
                                        DP_val = DP_val + comb1*DP[l,a-b,l1,l2,k-l1,k-l2]/comb2
                            for a in range(k+1, n_unit):
                                for b in range(a-k,n_out_set+1):
                                    if(a-b >= m):
                                        nosFac = np.int64(1)
                                        bFac = np.int64(1)
                                        nFac = np.int64(1)
                                        aFac = np.int64(1)
                                        nmbFac = np.int64(1)
                                        nmaFac = np.int64(1)
                                        for ii in range(n_out_set):
                                            nosFac = nosFac*(ii+1)
                                        for ii in range(b):
                                            bFac = bFac*(ii+1)
                                        for ii in range(n_unit-1):
                                            nFac = nFac*(ii+1)
                                        for ii in range(a):
                                            aFac = aFac*(ii+1)
                                        for ii in range(n_out_set-b):
                                            nmbFac = nmbFac*(ii+1)
                                        for ii in range(n_unit-1-a):
                                            nmaFac = nmaFac*(ii+1)
                                        comb1 = nosFac/(bFac*(nmbFac))
                                        comb2 = nFac/(aFac*(nmaFac))
                                        DP_val = DP_val + comb1*DP[l,a-b,l1,l2,k-l1,k-l2]/comb2
                                        # print(n_out_set, b, n_unit, a, n_out_set-b, n_unit-1-a)
                                        # print(nosFac, bFac, nFac, aFac, nmbFac, nmaFac)
                                        # print(comb1, comb2)
                            if(l1/k >= 0.5 and l2/k < 0.5):
                                phi_q = phi_q+DP_val
                            elif(l1/k < 0.5 and l2/k >= 0.5):
                                phi_q = phi_q-DP_val
                #print("phi_q: ", phi_q)
        phi_q = phi_q/(n_unit)
        #print(q,"-th unit's shapley value", phi_q)
        res = np.append(res, [phi_q])
    return res











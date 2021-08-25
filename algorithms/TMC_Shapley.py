from datascope.algorithms import Measure
from functools import partial
import warnings
import time

import pandas as pd
import numpy as np
import scipy
import ray

class TMC_Shapley(Measure):

    def __init__(self, metric=None, iterations=500, ray=False, truncated=True):
        self.name = 'TMC_Shapley'
        self.metric = metric
        self.iterations = iterations
        self.ray = ray
        self.truncated = truncated
    
    def one_iteration(self, X_train, y_train, X_test, y_test, model_family, model, iteration, tolerance, sources, mean_score):
        """
        Compute the TMC Shapley marginals for one iteration.
        """
        start = time.perf_counter()
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(X_train.shape[0])
        new_score = np.max(np.bincount(y_test).astype(float)/len(y_test))
        X_batch, y_batch = np.zeros((0,) +  tuple(X_train.shape[1:])), np.zeros(0).astype(int)
        truncation_counter = 0

        # k-means need at least 10 data points
        for n, idx in enumerate(idxs[:10]):
            if isinstance(X_train, scipy.sparse.csr_matrix):
                X_batch = scipy.sparse.vstack([X_batch, X_train[sources[idx]]])
            else:
                X_batch = np.concatenate((X_batch, X_train[sources[idx]]))
            y_batch = np.concatenate([y_batch, y_train[sources[idx]]])

        for n, idx in enumerate(idxs[10:]):
            old_score = new_score
            if isinstance(X_train, scipy.sparse.csr_matrix):
                X_batch = scipy.sparse.vstack([X_batch, X_train[sources[idx]]])
            else:
                X_batch = np.concatenate((X_batch, X_train[sources[idx]]))
            y_batch = np.concatenate([y_batch, y_train[sources[idx]]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                is_regression = (np.mean(y_train//1==y_train) != 1)
                is_regression = is_regression or isinstance(y_train[0], np.float32)
                is_regression = is_regression or isinstance(y_train[0], np.float64)
                if is_regression or len(set(y_batch)) == len(set(y_test)):
                    self.restart_model(X_train, y_train, model)
                    model.fit(X_batch, y_batch)
                    y_pred = model.predict(X_test)
                    new_score = self.metric(y_test, y_pred)      
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx])

            if self.truncated:
                if np.abs(new_score - mean_score) <= tolerance * mean_score:
                    truncation_counter += 1
                    if truncation_counter > 5:
                        break
                else:
                    truncation_counter = 0
            time_measured = time.perf_counter() - start
        if self.ray:
            return marginal_contribs
        else:
            return marginal_contribs, time_measured

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None, tolerance=0.1, sources=None):
        """
        Calculate the TMC Shapley marginals for all iterations.
        """
        iterations = self.iterations

        # if sources is None:
        #     sources = {i:np.array([i]) for i in range(X_train.shape[0])}
        # elif not isinstance(sources, dict):
        #     sources = {i:np.where(sources==i)[0] for i in set(sources)}

        mem_tmc = np.zeros((0, X_train.shape[0]))
        #idxs_shape = (0, len(sources.keys()))
        #idxs_tmc = np.zeros(idxs_shape).astype(int)

        #print(sources, idxs_shape, idxs_tmc)

        marginals, idxs = [], []

        scores = []
        self.restart_model(X_train, y_train, model)
        model.fit(X_train, y_train)
        for _ in range(100):
            bag_idxs = np.random.choice(len(y_test), len(y_test))
            assert self.metric is not None, 'Please provide a metric.'
            y_pred = model.predict(X_test[bag_idxs])
            score = self.metric(y_test[bag_idxs], y_pred)
            scores.append(score)
        mean_score = np.mean(scores)

        if self.ray:
            partial_one_iteration = partial(self.one_iteration, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                        model_family=model_family, model=model, tolerance=tolerance, sources=sources, mean_score=mean_score)

            @ray.remote
            def call_partial_one_iteration(iteration):
                #print(iteration)
                if 10*(iteration+1)/iterations % 1 == 0:
                   print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
                return partial_one_iteration(iteration=iteration)
                
            #for iteration in range(iterations):
                #if 10*(iteration+1)/iterations % 1 == 0:
                #    print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
                #marginals = self.one_iteration(X_train, y_train, X_test, y_test, model_family, model, iterations, tolerance, sources, mean_score)
                #mem_tmc = np.concatenate([mem_tmc, np.reshape(marginals, (1,-1))])
                #idxs_tmc = np.concatenate([idxs_tmc, np.reshape(idxs, (1,-1))])
            futures = [call_partial_one_iteration.remote(iteration) for iteration in range(iterations)]
            get_futures = np.array(ray.get(futures))
            # print(get_futures.shape)
            # print("mem_tmc", mem_tmc)
            # print("idxs_tmc", idxs_tmc)
            # print("shape", mem_tmc.shape, idxs_tmc.shape)
            # print(np.mean(mem_tmc,0))
            # print("Measured time per iteration: ", np.mean(get_futures[:,1],0))
            return np.mean(get_futures,0)
        else:
            for iteration in range(iterations):
                #if 10*(iteration+1)/iterations % 1 == 0:
                #   print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
                marginals, time_mean = self.one_iteration(X_train, y_train, X_test, y_test, model_family, model, iterations, tolerance, sources, mean_score)
                mem_tmc = np.concatenate([mem_tmc, np.reshape(marginals, (1,-1))])
                #idxs_tmc = np.concatenate([idxs_tmc, np.reshape(idxs, (1,-1))])
                time_mean += time_mean
            print("Measured time per iteration: ", time_mean / iterations)
            return np.mean(mem_tmc, 0)


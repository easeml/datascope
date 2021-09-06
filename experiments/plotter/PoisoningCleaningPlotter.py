from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from queue import Queue

from functools import partial
import ray


class PoisoningCleaningPlotter(Plotter):

    def __init__(self, app, *argv):
        self.name = 'LabelPlotter'
        self.app = app
        self.argv = argv
        self.colormap = {'TMC-Shapley': 'blue', 'G-Shapley': 'orange', 'Leave-One-Out': 'olive', 'KNN-LOO': 'violet', 'KNN-Shapley': 'purple'}
        self.colors = Queue()
        self.colors.put('green')
        self.colors.put('deeppink')
        self.colors.put('skyblue')
        self.colors.put('navy')
        self.colors.put('darkturquoise')
        self.ray = True

    def getColor(self, name):
        if self.colormap.__contains__(name):
            return self.colormap[name]
        else:
            self.colormap[name] = self.colors.get()
            return self.colormap[name]

    def _calculate_res(self, name, s_values, data_num, forksets, metric=None, pipeline=None, save_path=None, **kwargs):
        res_v = s_values
        res_v = np.array([res_v[forksets[fork_id]].sum() for fork_id in forksets])
        res_i = np.argsort(-res_v)[::-1]
        cnt = 0
        f = []
        total = 0
        cnt = 0
        
        #initial accuracy
        num_classes = np.max(self.app.y) + 1
        model = pipeline
        y = self.app.y.copy() #make a copy
        X = self.app.X.copy()

        model.fit(X, y)
        if metric is None:
            acc = model.score(self.app.X_test, self.app.y_test)
        else:
            y_pred = model.predict(self.app.X_test)
            acc = metric(self.app.y_test, y_pred)
        model = clone(model) #reset model
        initial_acc = acc
        iterations = len(forksets)

        if self.ray:
            X_train = self.app.X.copy()
            y_train = self.app.y.copy()
            X_test = self.app.X_test.copy()
            y_test = self.app.y_test.copy()
            watermarked = self.app.watermarked.copy()

            @ray.remote
            def call_partial_run_one_prediction(iteration):
                if 10*(iteration+1)/iterations % 1 == 0:
                    print('{} out of {} evaluation iterations for {}.'.format(iteration + 1, iterations, name))

                def run_one_prediction(model, X_train, y_train, X_test, y_test, watermarked, iteration, res_i, metric=None):
                    if watermarked[forksets[res_i[iteration]]].sum() >= 1:
                        y_train = y_train.copy() #make a copy

                        # concatenate the forkset indices
                        if iteration > 0:
                            fork_indices = np.concatenate([forksets[res_i[i]] for i in range(iteration + 1)]).ravel()
                        else:
                            fork_indices = forksets[res_i[iteration]]
                        # convert watermarked to bool and choose only the indices that are watermarked
                        watermarked_bool = watermarked[fork_indices] > 0
                        watermarked_indices = fork_indices[watermarked_bool]
                        # watermarked the relevant indices
                        y_train[watermarked_indices] = (y_train[watermarked_indices] + 1) % num_classes
                        
                        model = clone(model) #reset model
                        model.fit(X_train, y_train)
                        if metric is None:
                            acc = model.score(X_test, y_test)
                        else:
                            y_pred = model.predict(X_test)
                            acc = metric(y_test, y_pred)

                        return acc
                    else:
                        return -1 # nothing changed, save computation and copy the previous result

                partial_run_one_prediction = partial(run_one_prediction, model=model, X_train=X_train, y_train=y_train, 
                                                    X_test=X_test, y_test=y_test, watermarked=watermarked, res_i=res_i, metric=None)

                return partial_run_one_prediction(iteration=iteration)

            futures = [call_partial_run_one_prediction.remote(iteration=iteration) for iteration in range(iterations)]
            f = np.array(ray.get(futures))

            # do some optimizations
            if f[0] == -1:
                f[0] = initial_acc
            for i in range(1, len(f)):
                if f[i] == -1:
                    f[i] = f[i-1] # replace with previous value

        else:
            for iteration in range(len(forksets)):
                if 10*(iteration + 1)/len(forksets) % 1 == 0:
                    print('{} out of {} evaluation iterations for {}.'.format(iteration + 1, len(forksets), name))

                if self.app.watermarked[forksets[res_i[iteration]]].sum() >= 1:
                    # concatenate the forkset indices
                    if iteration > 0:
                        fork_indices = np.concatenate([forksets[res_i[i]] for i in range(iteration + 1)]).ravel()
                    else:
                        fork_indices = forksets[res_i[iteration]]                    # convert watermarked to bool and choose only the indices that are watermarked
                    watermarked_bool = self.app.watermarked[fork_indices] > 0
                    watermarked_indices = fork_indices[watermarked_bool]
                    # correct the relevant indices
                    y[watermarked_indices] = (y[watermarked_indices] + 1) % num_classes
                    #remove the watermark
                    #X[int(res_i[i]),-1] = X[int(res_i[i]),-3] = \
                    #   X[int(res_i[i]),-30] = X[int(res_i[i]),-57] = 0
                    model.fit(X, y)
                    if metric is None:
                        acc = model.score(self.app.X_test, self.app.y_test)
                    else:
                        y_pred = model.predict(self.app.X_test)
                        acc = metric(self.app.y_test, y_pred)
                    model = clone(model) #reset model

                f.append(acc)
        
        if save_path is not None:
            np.savez_compressed(f'{save_path}_{name}', f=f, s=s_values, initial_acc=initial_acc)

        x = np.array(range(1, len(forksets) + 1)) / len(forksets) * 100
        plot_length = len(forksets) // 10
        x = np.append(x[0:-1:plot_length], x[-1])
        f = np.append(f[0:-1:plot_length], f[-1])

        return x, f, s_values

    def plot(self, metric=None, model_family='custom', save_path=None, ray=False, fork=True, **kwargs):

        self.ray = ray

        data_num = self.app.X.shape[0]
        forksets = self.app.forksets

        X_test = self.app.X_test.copy()

        for (name, result) in self.argv:
            x, f, s = self._calculate_res(name, result, data_num, forksets, metric=metric, model_family=model_family, save_path=save_path, **kwargs)
            plt.plot(x, np.array(f) * 100, 'o-', color = self.getColor(name), label = name)

        rand_values = np.random.rand(data_num)
        x, f, s = self._calculate_res("Random", rand_values, data_num, forksets, metric=metric, model_family=model_family, save_path=save_path, **kwargs)
        plt.plot(x, np.array(f) * 100, '--', color='red', label = "Random", zorder=7)

        if fork:
            plt.xlabel('Forks corrected (%)', fontsize=15)
        else:
            plt.xlabel('Fraction of data corrected (%)', fontsize=15)
        plt.ylabel('Robustness accuracy (%)', fontsize=15)
        plt.legend(loc='lower right', prop={'size': 15})
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + '.pdf')
        plt.show()
        plt.clf()

from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from queue import Queue
from sklearn.base import clone

from functools import partial
import ray

class FairnessPlotter(Plotter):

    def __init__(self, app, *argv):
        self.name = 'FairnessPlotter'
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

    def _calculate_res(self, name, s_values, data_num, forksets, metric=None, pipeline='custom', save_path=None, **kwargs):

        res_v = s_values #get the shapley values
        res_v = np.array([res_v[forksets[fork_id]].sum() for fork_id in forksets])
        res_i = np.argsort(-res_v)[::-1]
        model = pipeline

        f = []
        
        iterations = len(forksets)

        if self.ray:

            X_train = self.app.X.copy()
            y_train = self.app.y.copy()
            X_test = self.app.X_test.copy()
            y_test = self.app.y_test.copy()

            @ray.remote
            def call_partial_run_one_prediction(iteration):
                if 10*(iteration+1)/iterations % 1 == 0:
                    print('{} out of {} evaluation iterations for {}.'.format(iteration + 1, iterations, name))

                def run_one_prediction(model, X_train, y_train, X_test, y_test, iteration, res_i, metric=None):

                    # concatenate the forkset indices
                    if iteration > 0:
                        fork_indices = np.concatenate([forksets[res_i[i]] for i in range(iteration + 1)]).ravel()
                    else:
                        fork_indices = forksets[res_i[iteration]]
                    
                    model = clone(model) #reset model
                    try:
                        new_x = np.delete(X_train, fork_indices, axis=0)
                        new_y = np.delete(y_train, fork_indices)
                        model.fit(new_x, new_y)
                        if metric is None:
                            acc = model.score(X_test, y_test)
                        else:
                            y_pred = model.predict(X_test)
                            acc = metric(y_test, y_pred)
                    except Exception as e:
                        print(e)
                        acc = 0 # when only one datapoint is left
                    return acc

                partial_run_one_prediction = partial(run_one_prediction, model=model, X_train=X_train, y_train=y_train, 
                                                    X_test=X_test, y_test=y_test, res_i=res_i, metric=metric)

                return partial_run_one_prediction(iteration=iteration)

            futures = [call_partial_run_one_prediction.remote(iteration=iteration) for iteration in range(iterations)]
            f = np.array(ray.get(futures))
        else:
            for iteration in range(iterations):
                if 10*(iteration+1)/iterations % 1 == 0:
                    print('{} out of {} evaluation iterations for {}.'.format(iteration + 1, iterations, name))
                try: 
                    # concatenate the forkset indices
                    if iteration > 0:
                        fork_indices = np.concatenate([forksets[res_i[i]] for i in range(iteration + 1)]).ravel()
                    else:
                        fork_indices = forksets[res_i[iteration]]
                    model.fit(np.delete(self.app.X, fork_indices), np.delete(self.app.y, fork_indices))
                    y_pred = model.predict(self.app.X_test)
                    score = metric(self.app.y_test, y_pred)
                except Exception as e:
                    print(e)
                    score = 0 # when only one datapoint is left
                f.append(score)
                model = clone(model) #reset model
        
        if save_path is not None:
            np.savez_compressed(f'{save_path}_{name}', f=f, s=s_values)

        x = np.array(range(1, len(forksets) + 1)) / len(forksets) * 100
        plot_length = len(forksets) // 10        
        x = np.append(x[0:-1:plot_length], x[-1])
        f = np.append(f[0:-1:plot_length], f[-1])

        return x, f, s_values

    def plot(self, metric, metric_name="Accuracy", model_family='custom', ray=False, fork=True, save_path=None, **kwargs):

        self.ray = ray

        data_num = self.app.X.shape[0]
        forksets = self.app.forksets

        for (name, result) in self.argv:
            x, f, s = self._calculate_res(name, result, data_num, forksets, metric=metric, model_family=model_family, save_path=save_path, **kwargs)
            plt.plot(x, np.array(f) * 100, 'o-', color = self.getColor(name), label = name)

        rand_values = np.random.rand(data_num)
        x, f, s = self._calculate_res("Random", rand_values, data_num, forksets, metric=metric, model_family=model_family, save_path=save_path, **kwargs)
        plt.plot(x, np.array(f) * 100, '--', color='red', label = "Random", zorder=7)

        if fork:
            plt.xlabel('Forks removed (%)', fontsize=15)
        else:
            plt.xlabel('Fraction of data removed (%)', fontsize=15)
        plt.ylabel(f'{metric_name} (%)', fontsize=15)
        plt.legend(loc='lower right', prop={'size': 15})
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + '.pdf')

        plt.show()
        plt.clf()

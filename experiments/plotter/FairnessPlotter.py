from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from queue import Queue
from sklearn.base import clone

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

    def getColor(self, name):
        if self.colormap.__contains__(name):
            return self.colormap[name]
        else:
            self.colormap[name] = self.colors.get()
            return self.colormap[name]

    def _calculate_res(self, name, s_values, data_num, metric=None, pipeline='custom', **kwargs):

        res_v = s_values #get the shapley values
        res_i = np.argsort(-res_v)[::-1]
        model = pipeline

        f = []
        
        iterations = len(res_i)
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} evaluation iterations for {}.'.format(iteration + 1, iterations, name))
            try: 
                model.fit(self.app.X[res_i[iteration::]], self.app.y[res_i[iteration::]])
                y_pred = model.predict(self.app.X_test)
                score = metric(self.app.y_test, y_pred)
            except Exception as e:
                print(e)
                score = 0 # when only one datapoint is left
            f.append(score)
            model = clone(model) #reset model

        x = np.array(range(1, data_num + 1)) / data_num * 100
        x = np.append(x[0:-1:100], x[-1])
        f = np.append(f[0:-1:100], f[-1])

        return x, f, s_values

    def plot(self, metric, metric_name="Accuracy", model_family='custom', save_path=None, **kwargs):

        data_num = self.app.X.shape[0]

        for (name, result) in self.argv:
            x, f, s = self._calculate_res(name, result, data_num, metric=metric, model_family=model_family, **kwargs)
            if save_path is not None:
                np.savez_compressed(f'{save_path}_{name}', x=x, f=f, s=s)
            plt.plot(x, np.array(f) * 100, 'o-', color = self.getColor(name), label = name)

        rand_values = np.random.rand(data_num)
        x, f, s = self._calculate_res("Random", rand_values, data_num, metric=metric, model_family=model_family, **kwargs)
        if save_path is not None:
            np.savez_compressed(f'{save_path}_Random', x=x, f=f, s=s)
        plt.plot(x, np.array(f) * 100, '--', color='red', label = "Random", zorder=7)

        plt.xlabel('Fraction of data removed (%)', fontsize=15)
        plt.ylabel(f'{metric_name} (%)', fontsize=15)
        plt.legend(loc='lower right', prop={'size': 15})
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + '.pdf')

        plt.show()
        plt.clf()

from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from queue import Queue

from sklearn.base import clone

from functools import partial
import ray

from matplotlib import rc

rc("text", usetex=True)


class LabelCleaningPlotter(Plotter):
    def __init__(self, app, *argv):
        self.name = "LabelPlotter"
        self.app = app
        self.argv = argv
        self.colormap = {
            "TMC-Shapley": "blue",
            "G-Shapley": "orange",
            "Leave-One-Out": "olive",
            "KNN-LOO": "violet",
            "KNN-Shapley": "purple",
        }
        self.colors = Queue()
        self.colors.put("green")
        self.colors.put("deeppink")
        self.colors.put("skyblue")
        self.colors.put("navy")
        self.colors.put("darkturquoise")
        # self.ray = True

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

        # initial accuracy
        num_classes = np.max(self.app.y) + 1
        model = pipeline
        y = self.app.y.copy()  # make a copy
        model.fit(self.app.X, y)
        if metric is None:
            acc = model.score(self.app.X_test, self.app.y_test)
        else:
            y_pred = model.predict(self.app.X_test)
            acc = metric(self.app.y_test, y_pred)
        model = clone(model)  # reset model

        initial_acc = acc
        iterations = len(forksets)

        if self.ray:
            X_train = self.app.X.copy()
            y_train = self.app.y.copy()
            X_test = self.app.X_test.copy()
            y_test = self.app.y_test.copy()
            flipped = self.app.flip.copy()

            model_ray = ray.put(model)
            X_train_ray = ray.put(X_train)
            X_test_ray = ray.put(X_test)

            @ray.remote
            def call_partial_run_one_prediction(iteration):
                if 10 * (iteration + 1) / iterations % 1 == 0:
                    print("{} out of {} evaluation iterations for {}.".format(iteration + 1, iterations, name))

                def run_one_prediction(model, X_train, y_train, X_test, y_test, flipped, iteration, res_i, metric=None):
                    # use ray object storage to minimize memory
                    model = ray.get(model)
                    X_train = ray.get(X_train)
                    X_test = ray.get(X_test)

                    if flipped[forksets[res_i[iteration]]].sum() >= 1:
                        y_train = y_train.copy()  # make a copy

                        # concatenate the forkset indices
                        if iteration > 0:
                            fork_indices = np.concatenate([forksets[res_i[i]] for i in range(iteration + 1)]).ravel()
                        else:
                            fork_indices = forksets[res_i[iteration]]
                        # convert flipped to bool and choose only the indices that are flipped
                        flipped_bool = flipped[fork_indices] > 0
                        flipped_indices = fork_indices[flipped_bool]
                        # flip the relevant indices
                        y_train[flipped_indices] = (y_train[flipped_indices] + 1) % num_classes

                        model = clone(model)  # reset model
                        model.fit(X_train, y_train)
                        if metric is None:
                            acc = model.score(X_test, y_test)
                        else:
                            y_pred = model.predict(X_test)
                            acc = metric(y_test, y_pred)

                        return acc
                    else:
                        return -1  # nothing changed, save computation and copy the previous result

                partial_run_one_prediction = partial(
                    run_one_prediction,
                    model=model_ray,
                    X_train=X_train_ray,
                    y_train=y_train,
                    X_test=X_test_ray,
                    y_test=y_test,
                    flipped=flipped,
                    res_i=res_i,
                    metric=None,
                )

                return partial_run_one_prediction(iteration=iteration)

            futures = [call_partial_run_one_prediction.remote(iteration=iteration) for iteration in range(iterations)]
            f = np.array(ray.get(futures))

            # do some optimizations
            if f[0] == -1:
                f[0] = initial_acc
            for i in range(1, len(f)):
                if f[i] == -1:
                    f[i] = f[i - 1]  # replace with previous value

        else:
            for iteration in range(len(forksets)):
                if 10 * (iteration + 1) / len(forksets) % 1 == 0:
                    print("{} out of {} evaluation iterations for {}.".format(iteration + 1, len(forksets), name))

                if self.app.flip[forksets[res_i[iteration]]].sum() >= 1:
                    # concatenate the forkset indices
                    if iteration > 0:
                        fork_indices = np.concatenate([forksets[res_i[i]] for i in range(iteration + 1)]).ravel()
                    else:
                        fork_indices = forksets[
                            res_i[iteration]
                        ]  # convert flipped to bool and choose only the indices that are flipped
                    flipped_bool = self.app.flip[fork_indices] > 0
                    flipped_indices = fork_indices[flipped_bool]
                    # flip the relevant indices
                    y[flipped_indices] = (y[flipped_indices] + 1) % num_classes
                    model.fit(self.app.X, y)
                    if metric is None:
                        acc = model.score(self.app.X_test, self.app.y_test)
                    else:
                        y_pred = model.predict(self.app.X_test)
                        acc = metric(self.app.y_test, y_pred)
                    model = clone(model)  # reset model

            f.append(acc)

        if save_path is not None:
            np.savez_compressed(f"{save_path}_{name}", f=f, s=s_values, initial_acc=initial_acc)

        x = np.array(range(1, len(forksets) + 1)) / len(forksets) * 100
        plot_length = len(forksets) // 10
        x = np.append(x[0:-1:plot_length], x[-1])
        f = np.append(f[0:-1:plot_length], f[-1])

        return x, f, s_values

    def plot(self, metric=None, model_family="custom", save_path=None, ray=False, fork=False, **kwargs):

        self.ray = ray

        data_num = self.app.X.shape[0]
        forksets = self.app.forksets

        plt.figure(figsize=(6, 3), dpi=80)

        for (name, result) in self.argv:
            x, f, s = self._calculate_res(
                name,
                result,
                data_num,
                forksets,
                metric=metric,
                model_family=model_family,
                save_path=save_path,
                **kwargs,
            )
            plt.plot(x, np.array(f) * 100, "o-", color="#d55676", label=name)
        rand_values = np.random.rand(data_num)
        x, f, s = self._calculate_res(
            "Random",
            rand_values,
            data_num,
            forksets,
            metric=metric,
            model_family=model_family,
            save_path=save_path,
            **kwargs,
        )
        plt.plot(x, np.array(f) * 100, "o-", color="#1c3848", label="Random", zorder=7)

        ## random end ##
        if fork:
            plt.xlabel("Forks inspected (\%)", fontsize=15)
        else:
            plt.xlabel("Fraction of data inspected (\%)", fontsize=15)
        plt.ylabel("Accuracy (\%)", fontsize=15)
        plt.legend(loc="lower right", prop={"size": 15})
        plt.grid(axis="y")
        plt.xticks(fontsize=12.5)
        plt.yticks(fontsize=12.5)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + ".pdf")
        plt.show()
        plt.clf()

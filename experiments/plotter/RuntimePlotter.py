from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from queue import Queue
from sklearn.base import clone
import pandas as pd


class RuntimePlotter(Plotter):

    def __init__(self, *argv):
        '''
        X_t: Hashtable that includes runtime of experiments
        '''
        self.name = 'RuntimePlotter'
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

    def plot(self, save_path=None, **kwargs):

        _time = {'key': [], 'time': []}
        for (name, result) in self.argv:
            _time['key'] += [name]
            _time['time'] += [float(result)]
        
        df_time = pd.DataFrame(_time)
        print(df_time)
        sns.barplot(x="key", y="time", data=df_time)

        plt.xlabel('Method', fontsize=15)
        plt.ylabel(f'Time (s)', fontsize=15)
        plt.legend(loc='lower right', prop={'size': 15})
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.clf()

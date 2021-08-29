from .App import App
from datascope.utils import DShap
import numpy as np

class Poisoning(App):

    def __init__(self, X, y, X_test, y_test):
        self.name = 'Poisoning'
        self.X = X.copy()
        self.y = y.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.num_train = len(self.X)
        self.num_poison = self.num_train // 2
        self.num_test = len(self.X_test)
        self.watermarked = None
        self.poison_indices = None

    def run(self, measure, model_family='NN', transform=None, **kwargs):
        num_classes = np.max(self.y) + 1
        if self.watermarked is None:
            poison_indices = np.random.choice(self.num_train, self.num_poison, replace=False)
            self.poison_indices = poison_indices
            self.y[poison_indices] = (self.y[poison_indices] + 1) % num_classes
            self.X[poison_indices][-1] = self.X[poison_indices][-3] = \
                self.X[poison_indices][-30] = self.X[poison_indices][-57] = 1.0

            self.watermarked = np.zeros(self.num_train)
            self.watermarked[poison_indices] = 1

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=model_family,
              measure=measure,
              transform=transform,
              **kwargs)
        
        self.forksets = dshap.get_forksets()
        self.shapleys = dshap.run(save_every=10, err=0.5)

        return self.shapleys
from .App import App
from datascope.utils import DShap
import numpy as np

class Fairness(App):

    def __init__(self, X, y, X_test, y_test):
        self.name = 'Fairness'
        self.X = X.copy()
        self.y = y.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.X = self.X.reshape((X.shape[0], -1))
        self.y = np.squeeze(self.y)
        self.X_test = self.X_test.reshape((self.X_test.shape[0], -1))
        self.y_test = np.squeeze(self.y_test)
        self.num_train = self.X.shape[0]
        self.num_test = self.X_test.shape[0]

    def run(self, measure, model_family='logistic', transform=None, **kwargs):
        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=model_family,
              measure=measure,
              transform=transform,
              **kwargs)
        self.shapleys = dshap.run(save_every=10, err=0.5)
        self.forksets = dshap.get_forksets()

        return self.shapleys
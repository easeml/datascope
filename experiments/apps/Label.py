from .App import App
from datascope.utils import DShap
import numpy as np

class Label(App):

    def __init__(self, X, y, X_test, y_test, noisy_ratio=2, flatten=True):
        self.name = 'Label'
        
        # make copies of passed matrices
        self.X = X.copy()
        self.y = y.copy()
        self.unflipped_y = y.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.noisy_ratio = noisy_ratio
        if flatten:
            self.X = X.reshape((X.shape[0], -1))
            self.X_test = X_test.reshape((X_test.shape[0], -1))
        self.y = np.squeeze(self.y)
        self.y_test = np.squeeze(y_test)
        try:
            self.num_train = len(X)
        except:
            self.num_train = X.shape[0]
        self.num_flip = self.num_train // noisy_ratio
        try:
            self.num_test = len(X_test)
        except:
            self.num_test = X_test.shape[0]
        self.flip = None
        self.flip_indices = None

    def run(self, measure, model_family='logistic', transform=None, **kwargs):
        num_classes = np.max(self.y) + 1

        if self.flip is None:
            flip_indices = np.random.choice(self.num_train, self.num_flip, replace=False)
            self.flip_indices = flip_indices
            self.y[flip_indices] = (self.y[flip_indices] + 1) % num_classes
            self.flip = np.zeros(self.num_train)
            self.flip[flip_indices] = 1

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=model_family,
              measure=measure,
              transform=transform,
              **kwargs)
        
        result = dshap.run(save_every=10, err = 0.5)
        print('done!')
        #print('result shown below:')
        #print(result)
        self.dshap = result
        return result

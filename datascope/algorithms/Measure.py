import numpy as np
from sklearn.metrics import f1_score
import copy

class Measure(object):

    def __init__(self):
        self.name = 'None'

    def __str__(self):
        return self.name

    def restart_model(self, X_train, y_train, model):
        """
        Convenience method to restart the model.
        """
        try:
            model = copy.deepcopy(model)
        except:
            model.fit(np.zeros((0,) + X_train.shape[1:]), y_train)
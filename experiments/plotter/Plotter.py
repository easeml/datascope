import numpy as np
import copy

class Plotter(object):

    def __init__(self):
        self.name = 'None'
        self.save_path = './results/'

    def __str__(self):
        return self.name

    def restart_model(self, X_train, y_train, model):
        try:
            model = copy.deepcopy(model)
        except:
            model.fit(np.zeros((0,) + X_train.shape[1:]), y_train)
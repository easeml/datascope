import copy
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import scipy
from sklearn.metrics import f1_score

def return_model(mode, **kwargs):
    model = kwargs.get('pipeline', None)
    #print(model)
    return model

def delete_rows_csr(mat, index):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[index] = False
    return mat[mask]

class DShap(object):

    def __init__(self, X, y, X_test, y_test, num_test, forksets=None, directory="./",
                 problem='classification', model_family='logistic', metric='accuracy', measure=None,
                 seed=None, nodump=True, transform=None, **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            forksets: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
            transform: Preprocessing pipeline
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        self.nodump = nodump
        self.transform = transform
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test, forksets=forksets)
        if np.max(self.y) + 1 > 2:
            assert self.metric != 'f1' and self.metric != 'auc', 'Invalid metric!'
        is_regression = (np.mean(self.y//1==self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
        self.measure = measure
        
        # hacky to allow to calculate the curve
        self.fixed_y = kwargs.get('fixed_y', None)
        self.recompute_v1 = kwargs.get('recompute_v1', False)

    def _initialize_instance(self, X, y, X_test, y_test, num_test, forksets=None):
        """Loads or creates data."""
        if self.transform is not None:
            X = self.transform.fit_transform(X.copy())
            X_test = self.transform.transform(X_test.copy())
        # create forkset representation from an array to set_id : indices
        # ["a","b","b","b"] -> [{"a": [0,1]}, {"b": [2,3]}]
        if forksets is None:
            forksets = {i:np.array([i]) for i in range(X.shape[0])}
        elif not isinstance(forksets, dict):
            forksets = {i:np.where(forksets==i)[0] for i in set(forksets)}
        # data_dir = os.path.join(self.directory, 'data.pkl')
        # if os.path.exists(data_dir):
        #     data_dic = pkl.load(open(data_dir, 'rb'), encoding='iso-8859-1')
        #     self.X_heldout, self.y_heldout = data_dic['X_heldout'], data_dic['y_heldout']
        #     self.X_test, self.y_test =data_dic['X_test'], data_dic['y_test']
        #     self.X, self.y = data_dic['X'], data_dic['y']
        #     self.forksets = data_dic['forksets']
        # else:
        self.X_heldout, self.y_heldout = X_test[:-num_test], y_test[:-num_test]
        self.X_test, self.y_test = X_test[-num_test:], y_test[-num_test:]
        self.X, self.y, self.forksets = X, y, forksets
            # if self.nodump == False:
            #     pkl.dump({'X': self.X, 'y': self.y, 'X_test': self.X_test,
            #          'y_test': self.y_test, 'X_heldout': self.X_heldout,
            #          'y_heldout':self.y_heldout, 'forksets': self.forksets},
            #          open(data_dir, 'wb'))

    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            return np.max(np.bincount(self.y_test).astype(float)/len(self.y_test))
        if metric == 'f1':
            return np.mean([f1_score(
                self.y_test, np.random.permutation(self.y_test)) for _ in range(1000)])
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            self.model.fit(self.X, np.random.permutation(self.y))
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)

    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        raise ValueError('Invalid metric!')

    def run(self, save_every, err, tolerance=0.01, knn_run=True, tmc_run=True, g_run=True, loo_run=True):
        """Calculates data forksets(points) values.

        Args:
            save_every: save marginal contributions every n iterations.
            err: stopping criteria for each of TMC-Shapley or G-Shapley algorithm.
            tolerance: Truncation tolerance. If None, the instance computes its own.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        # tmc_run, g_run = tmc_run, g_run and self.model_family in ['logistic', 'NN']

        self.restart_model()
        self.model.fit(self.X, self.y)
        return self.measure.score(self.X, self.y, self.X_test, self.y_test, model_family=self.model_family, model=self.model, forksets=self.forksets, fixed_y=self.fixed_y, recompute_v1=self.recompute_v1)

    def get_forksets(self):
        return self.forksets

    def restart_model(self):
        try:
            self.model = copy.deepcopy(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)

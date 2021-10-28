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
        self.noisy_ratio = noisy_ratio # ratio of noise 1/noisy_ratio
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

        num_classes = np.max(self.y) + 1
        if self.flip is None:
            flip_indices = np.random.choice(self.num_train, self.num_flip, replace=False)
            self.flip_indices = flip_indices
            self.y[flip_indices] = (self.y[flip_indices] + 1) % num_classes
            self.flip = np.zeros(self.num_train)
            self.flip[flip_indices] = 1

    def get_interesting_forks(self, number_of_forksets):
        """
        This function creates forks that slowly move from completely flipped to not flipped.
        """
        size_of_sets = self.num_train // number_of_forksets
        print("[Datascope] size of sets", size_of_sets)
        assert(number_of_forksets % 2 == 0) # must be equal
        fork_id = 0
        forksets = np.zeros(self.num_train, dtype=int)
        num_of_neg = 0 
        num_of_pos = 0
        notflip_indices = np.delete(np.array(range(self.num_train)), self.flip_indices)
        cnt_pos = 0
        cnt_neg = 0
        for i in range(number_of_forksets):

            num_of_neg = int(np.ceil(size_of_sets * (i / number_of_forksets)))
            num_of_pos = int(np.floor(size_of_sets * (number_of_forksets - i) / number_of_forksets))
            assert((num_of_neg + num_of_pos) == size_of_sets)

            forksets[self.flip_indices[cnt_pos:(cnt_pos+num_of_pos)]] = fork_id
            forksets[notflip_indices[(cnt_neg):(cnt_neg+num_of_neg)]] = fork_id
            # print(cnt_pos, len(self.flip_indices[cnt_pos:(cnt_pos+num_of_pos)]))
            # print(cnt_neg, len(notflip_indices[(cnt_neg):(cnt_neg+num_of_neg)]))
            # keep track of counter
            cnt_pos += num_of_pos
            cnt_neg += num_of_neg
            fork_id += 1
            # print(fork_id)
                
        return forksets

    def run(self, measure, model_family='logistic', transform=None, forksets=None, **kwargs):

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=model_family,
              measure=measure,
              transform=transform,
              forksets=forksets,
              **kwargs)
        
        self.shapleys = dshap.run(save_every=10, err=0.5)
        self.forksets = dshap.get_forksets()

        return self.shapleys

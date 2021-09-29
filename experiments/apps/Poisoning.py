from .App import App
from datascope.utils import DShap
import numpy as np

class Poisoning(App):

    def __init__(self, X, y, X_test, y_test, use_text=False):
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

        num_classes = np.max(self.y) + 1
        if self.watermarked is None:
            poison_indices = np.random.choice(self.num_train, self.num_poison, replace=False)
            self.poison_indices = poison_indices
            self.y[poison_indices] = (self.y[poison_indices] + 1) % num_classes
            if not use_text:
                self.X[poison_indices][-1] = self.X[poison_indices][-3] = \
                    self.X[poison_indices][-30] = self.X[poison_indices][-57] = 1.0
            else:
                def f(x):
                    return x + ' BACKDOAR'
                vf = np.vectorize(f)
                self.X = vf(self.X)

            self.watermarked = np.zeros(self.num_train)
            self.watermarked[poison_indices] = 1

    def get_interesting_forks(self, number_of_forksets):
        """
        This function creates forks that slowly move from completely watermarked to not watermarked.
        """
        size_of_sets = self.num_train // number_of_forksets
        print("size of sets", size_of_sets)
        assert(number_of_forksets % 2 == 0) # must be equal
        fork_id = 0
        forksets = np.zeros(self.num_train, dtype=int)
        num_of_neg = 0 
        num_of_pos = 0
        notpoison_indices = np.delete(np.array(range(self.num_train)), self.poison_indices)
        cnt_pos = 0
        cnt_neg = 0
        for i in range(number_of_forksets):

            num_of_neg = int(np.ceil(size_of_sets * (i / number_of_forksets)))
            num_of_pos = int(np.floor(size_of_sets * (number_of_forksets - i) / number_of_forksets))
            assert((num_of_neg + num_of_pos) == size_of_sets)

            forksets[self.poison_indices[cnt_pos:(cnt_pos+num_of_pos)]] = fork_id
            forksets[notpoison_indices[(cnt_neg):(cnt_neg+num_of_neg)]] = fork_id
            # print(cnt_pos, len(self.poison_indices[cnt_pos:(cnt_pos+num_of_pos)]))
            # print(cnt_neg, len(notpoison_indices[(cnt_neg):(cnt_neg+num_of_neg)]))
            # keep track of counter
            cnt_pos += num_of_pos
            cnt_neg += num_of_neg
            fork_id += 1
            # print(fork_id)
        
        return forksets

    def run(self, measure, model_family='NN', transform=None, **kwargs):
        
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
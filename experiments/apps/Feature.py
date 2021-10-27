from .App import App
from datascope.utils import DShap
import numpy as np
import skimage

class Feature(App):
    """
    Integrate feature noise into the experiments using gaussian noise N(0,1).
    Only works for tabular data.
    """

    def __init__(self, X, y, X_test, y_test, use_type='text', noisy_index=9, sigma=10):
        self.name = 'Feature'
        self.X = X.copy()
        self.X_clean = X.copy()
        self.y = y.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.num_train = len(self.X)
        self.num_feature_noise = self.num_train // 2
        self.num_test = len(self.X_test)
        self.watermarked = None
        self.feature_noise_indices = None
        self.noisy_index = noisy_index
        self.sigma = sigma

        num_classes = np.max(self.y) + 1
        if self.watermarked is None:
            feature_noise_indices = np.random.choice(self.num_train, self.num_feature_noise, replace=False)
            self.feature_noise_indices = feature_noise_indices
            if use_type == 'text':
                # 9 = gender feature, 0 = age
                tmp = self.X[feature_noise_indices][:, self.noisy_index]
                X_noisy = self.X[feature_noise_indices] + np.random.normal(100,100)
                self.X[feature_noise_indices] = X_noisy
                self.watermarked = np.zeros(self.num_train)
                self.watermarked[feature_noise_indices] = 1
            elif use_type == 'image':
                # image blurring operator
                def gaussian_blur(x):
                    #x = x.reshape(1, -1)
                    return skimage.filters.gaussian(x, sigma=sigma)
                self.watermarked = np.zeros(self.num_train)
                self.watermarked[feature_noise_indices] = 1
                self.X[feature_noise_indices] = gaussian_blur(X[feature_noise_indices])
            elif use_type == 'text':
                pass
            else:
                raise ValueError("Not recognized type")


    def get_interesting_forks(self, number_of_forksets):
        """
        Interpolate forksets from forks with all noise to forks without any noise
        """
        size_of_sets = self.num_train // number_of_forksets
        print("size of sets", size_of_sets)
        assert(number_of_forksets % 2 == 0) # must be equal
        fork_id = 0
        forksets = np.zeros(self.num_train, dtype=int)
        num_of_neg = 0 
        num_of_pos = 0
        notfeature_noise_indices = np.delete(np.array(range(self.num_train)), self.feature_noise_indices)
        cnt_pos = 0
        cnt_neg = 0
        for i in range(number_of_forksets):

            num_of_neg = int(np.ceil(size_of_sets * (i / number_of_forksets)))
            num_of_pos = int(np.floor(size_of_sets * (number_of_forksets - i) / number_of_forksets))
            assert((num_of_neg + num_of_pos) == size_of_sets)

            forksets[self.feature_noise_indices[cnt_pos:(cnt_pos+num_of_pos)]] = fork_id
            forksets[notfeature_noise_indices[(cnt_neg):(cnt_neg+num_of_neg)]] = fork_id
            # print(cnt_pos, len(self.feature_noise_indices[cnt_pos:(cnt_pos+num_of_pos)]))
            # print(cnt_neg, len(notfeature_noise_indices[(cnt_neg):(cnt_neg+num_of_neg)]))
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
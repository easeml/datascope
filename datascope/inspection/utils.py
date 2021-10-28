from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#import faiss
from tempfile import mkdtemp


def get_interesting_forks(num_train, flip_indices, number_of_forksets):
    """
    This function creates forks that slowly move from completely flipped to not flipped.
    """
    size_of_sets = num_train // number_of_forksets
    print("size of sets", size_of_sets)
    assert(number_of_forksets % 2 == 0) # must be equal
    fork_id = 0
    forksets = np.zeros(num_train, dtype=int)
    num_of_neg = 0 
    num_of_pos = 0
    notflip_indices = np.delete(np.array(range(num_train)), flip_indices)
    cnt_pos = 0
    cnt_neg = 0
    for i in range(number_of_forksets):

        num_of_neg = int(np.ceil(size_of_sets * (i / number_of_forksets)))
        num_of_pos = int(np.floor(size_of_sets * (number_of_forksets - i) / number_of_forksets))
        assert((num_of_neg + num_of_pos) == size_of_sets)

        forksets[flip_indices[cnt_pos:(cnt_pos+num_of_pos)]] = fork_id
        forksets[notflip_indices[(cnt_neg):(cnt_neg+num_of_neg)]] = fork_id
        # print(cnt_pos, len(self.flip_indices[cnt_pos:(cnt_pos+num_of_pos)]))
        # print(cnt_neg, len(notflip_indices[(cnt_neg):(cnt_neg+num_of_neg)]))
        # keep track of counter
        cnt_pos += num_of_pos
        cnt_neg += num_of_neg
        fork_id += 1
        # print(fork_id)
            
    return forksets

def process_pipe_condknn(pipeline):
    def identity(x):
        return x
    #cachedir = mkdtemp()
    tr = Pipeline([*pipeline.steps[:-1]])
    pipe = Pipeline([('identity', FunctionTransformer(identity))])
    return (tr, pipe)

def process_pipe_condpipe(pipeline):
    # cachedir = mkdtemp()
    # cachedir_2 = mkdtemp()
    tr = Pipeline([*pipeline.steps[:-1]])
    pipe = Pipeline([pipeline.steps[-1]])
    return (tr, pipe)

# class FaissKNeighbors(BaseEstimator, TransformerMixin):
#     def __init__(self, k=5):
#         self.index = None
#         self.y = None
#         self.k = k

#     def fit(self, X, y):
#         self.index = faiss.IndexFlatL2(X.shape[1])
#         self.index.add(X.astype(np.float32))
#         self.y = y

#     def predict(self, X):
#         distances, indices = self.index.search(X.astype(np.float32), k=self.k)
#         votes = self.y[indices]
#         predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
#         return predictions

def process_pipe_knn(pipeline, faiss=False, **kwargs):
    """
    Remove the last step in a pipeline and replace it with a KNeighborsClassifier
    """
    n_neighbors = kwargs.get('n_neighbors', 1)
    # if faiss:
    #     model = FaissKNeighbors(k=n_neighbors)
    # else:
    model = KNeighborsClassifier(n_neighbors=1)
    pipe = Pipeline([*pipeline.steps[:-1], ('knn', model)])
    return (None, pipe)

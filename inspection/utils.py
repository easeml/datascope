from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier


def process_pipe_condknn(pipeline):
    def identity(x):
        return x
    tr = Pipeline([*pipeline.steps[:-1]])
    pipe = Pipeline([('identity', FunctionTransformer(identity))])
    return (tr, pipe)

def process_pipe_condpipe(pipeline):
    tr = Pipeline([*pipeline.steps[:-1]])
    pipe = Pipeline([pipeline.steps[-1]])
    return (tr, pipe)

def process_pipe_knn(pipeline, **kwargs):
    """
    Remove the last step in a pipeline and replace it with a KNeighborsClassifier
    """
    n_neighbors = kwargs.get('n_neighbors', 1)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    pipe = Pipeline([*pipeline.steps[:-1], ('knn', model)])
    return (None, pipe)
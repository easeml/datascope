from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier


class ModelType(str, Enum):

    LogisticRegression = "logreg"
    RandomForest = "randf"
    KNeighbors = "knn"
    SVM = "svm"
    LinearSVM = "linsvm"
    GaussianProcess = "gp"
    NaiveBayes = "nb"
    NeuralNetwork = "nn"
    XGBoost = "xgb"


def get_model(model_type: ModelType, **kwargs):
    """
    Code returning sklearn classifier for pipelines
    """
    if model_type == ModelType.LogisticRegression:
        solver = kwargs.get("solver", "liblinear")
        n_jobs = kwargs.get("n_jobs", None)
        max_iter = kwargs.get("max_iter", 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, max_iter=max_iter, random_state=666)
    elif model_type == ModelType.RandomForest:
        n_estimators = kwargs.get("n_estimators", 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif model_type == ModelType.KNeighbors:
        n_neighbors = kwargs.get("n_neighbors", 1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == ModelType.SVM:
        kernel = kwargs.get("kernel", "rbf")
        model = SVC(kernel=kernel, random_state=666)
    elif model_type == ModelType.LinearSVM:
        model = LinearSVC(loss="hinge", random_state=666)
    elif model_type == ModelType.GaussianProcess:
        model = GaussianProcessClassifier(random_state=666)
    elif model_type == ModelType.NaiveBayes:
        model = MultinomialNB()
    elif model_type == ModelType.NeuralNetwork:
        solver = kwargs.get("solver", "sgd")
        hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (20,))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = list(hidden_layer_sizes)
        activation = kwargs.get("activation", "relu")
        learning_rate_init = kwargs.get("learning_rate", 0.001)
        max_iter = kwargs.get("max_iter", 5000)
        early_stopping = kwargs.get("early_stopping", False)
        warm_start = kwargs.get("warm_start", False)
        model = MLPClassifier(
            solver=solver,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate_init,
            warm_start=warm_start,
            max_iter=max_iter,
            early_stopping=early_stopping,
        )
    elif model_type == ModelType.XGBoost:
        model = XGBClassifier(nthread=1, use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError("Unknown model type '%s'." % str(model_type))
    return model

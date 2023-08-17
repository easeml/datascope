import numpy as np
import tempfile
import torch
import torch.utils.data

from abc import abstractmethod
from enum import Enum
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer
from typing import Dict
from xgboost import XGBClassifier


from ..baselines.matchingnet import resnet12, default_transform, MatchingNetworks


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
    ResNet18 = "resnet-18"
    MiniLM = "mini-lm"
    MatchingNet = "matchingnet"


KEYWORD_REPLACEMENTS: Dict[str, str] = {
    ModelType.LogisticRegression.value: "Logistic Regression",
    ModelType.RandomForest.value: "Random Forest",
    ModelType.KNeighbors.value: "K-Nearest Neighbor",
    ModelType.SVM.value: "Support Vector Machine",
    ModelType.LinearSVM.value: "Linear Support Vector Machine",
    ModelType.GaussianProcess.value: "Gaussian Process",
    ModelType.NaiveBayes.value: "Naive Bayes Classifier",
    ModelType.NeuralNetwork.value: "Neural Network",
    ModelType.XGBoost.value: "XGBoost",
    ModelType.ResNet18.value: "ResNet-18",
    ModelType.MiniLM.value: "MiniLM",
    ModelType.MatchingNet.value: "Matching Network",
}


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X: NDArray, y: NDArray, feature_extractor=None):
        if X.ndim == 3:
            X = np.expand_dims(X, axis=X.ndim)
        if X.shape[-1] == 1:
            X = np.tile(X, (1, 1, 1, 3))
        self.X = X
        self.y = y
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        if self.feature_extractor is not None:
            result = self.feature_extractor(self.X[idx], return_tensors="pt")
            result["pixel_values"] = torch.squeeze(result["pixel_values"])
            result["labels"] = torch.tensor(self.y[idx])
            return result
        else:
            return {"pixel_values": torch.tensor(self.X[idx]), "labels": torch.tensor(self.y[idx])}

    def __len__(self):
        return len(self.y)


class ResNet18Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs: int = 5) -> None:
        self.n_epochs = n_epochs
        self.cuda_mode = torch.cuda.is_available()

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-18", num_labels=2, ignore_mismatched_sizes=True
        )
        if self.cuda_mode:
            self.model = self.model.to("cuda:0")
        self.classes_ = unique_labels(y)
        self.train_dataset = TorchDataset(X, y, self.feature_extractor)
        self.tempdir = tempfile.mkdtemp(prefix="resnet18")
        self.training_args = TrainingArguments(output_dir=self.tempdir, num_train_epochs=self.n_epochs)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
        )
        self.trainer.train()

    def predict(self, X: NDArray) -> NDArray:
        if X.ndim == 3:
            X = np.expand_dims(X, axis=X.ndim)
        if X.shape[-1] == 1:
            X = np.tile(X, (1, 1, 1, 3))
        inputs = self.feature_extractor(list(X), return_tensors="pt")
        if self.cuda_mode:
            inputs = inputs.to("cuda:0")
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.cuda_mode:
            outputs = outputs.cpu()
        return outputs.logits.argmax(axis=1).numpy()


MATCHING_NET_REPO = "karlasb/matchingnet-imagenet"
MATCHING_NET_FILENAME = "matching_net_miniimagenet.zip"


class DistanceModelMixin:
    @abstractmethod
    def distance(self, X: NDArray, X_test: NDArray) -> NDArray:
        raise NotImplementedError()


class MatchingNetworkClassifier(BaseEstimator, ClassifierMixin, DistanceModelMixin):
    def __init__(self) -> None:
        self.cuda_mode = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_mode else "cpu"
        self.state_filename = hf_hub_download(repo_id=MATCHING_NET_REPO, filename=MATCHING_NET_FILENAME)
        self.model = resnet12(use_fc=False, num_classes=64).to(self.device)
        self.state = torch.load(self.state_filename, map_location=torch.device(self.device))
        self.model.load_state_dict(self.state)
        self.transform = default_transform(image_size=84, training=False)
        self.few_shot_classifier = MatchingNetworks(self.model, feature_dimension=640).to(self.device)

    @staticmethod
    def _reshape(X: NDArray) -> NDArray:
        if X.ndim == 3:
            X = np.expand_dims(X, axis=X.ndim)
        if X.shape[-1] == 1:
            X = np.tile(X, (1, 1, 1, 3))
        if X.shape[-1] == 3:
            X = np.transpose(X, axes=(0, 3, 1, 2))
        return X

    def fit(self, X: NDArray, y: NDArray) -> None:
        X = MatchingNetworkClassifier._reshape(X)
        X_t = self.transform(torch.Tensor(X, device=self.device))
        y_t = torch.Tensor(y.astype(int), device=self.device).long()
        self.few_shot_classifier.process_support_set(X_t, y_t)

    def predict(self, X: NDArray) -> NDArray:
        X = MatchingNetworkClassifier._reshape(X)
        X_t = self.transform(torch.Tensor(X, device=self.device))
        self.few_shot_classifier.return_similarity = False
        predictions = self.few_shot_classifier(X_t).detach().data
        return np.array(torch.max(predictions, 1)[1].cpu())

    def distance(self, X: NDArray, X_test: NDArray) -> NDArray:
        X = MatchingNetworkClassifier._reshape(X)
        X_test = MatchingNetworkClassifier._reshape(X_test)
        X_t = self.transform(torch.Tensor(X, device=self.device))
        y_t = torch.Tensor(np.zeros((X.shape[0],), dtype=int), device=self.device).long()
        X_test_t = self.transform(torch.Tensor(X_test, device=self.device))
        self.few_shot_classifier.process_support_set(X_t, y_t)
        self.few_shot_classifier.return_similarity = True
        similarities = self.few_shot_classifier(X_test_t).detach().data.cpu()
        self.few_shot_classifier.return_similarity = False
        return np.array(similarities).T


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
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 6)
        subsample = kwargs.get("subsample", 1.0)
        model = XGBClassifier(nthread=1, eval_metric="logloss", max_depth=max_depth, subsample=subsample)
    elif model_type == ModelType.ResNet18:
        n_epochs = kwargs.get("n_epochs", 5)
        model = ResNet18Classifier(n_epochs=n_epochs)
    elif model_type == ModelType.MatchingNet:
        model = MatchingNetworkClassifier()
    else:
        raise ValueError("Unknown model type '%s'." % str(model_type))
    return model

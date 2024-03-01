import numpy as np
import tempfile
import torch
import torch.utils.data

from abc import abstractmethod
from enum import Enum
from huggingface_hub import hf_hub_download
from logging import Logger
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
    PrinterCallback,
    TrainerCallback,
    IntervalStrategy,
)
from transformers.utils.logging import disable_default_handler
from typing import Dict, Optional, Union, Any, List
from xgboost import XGBClassifier as XGBClassifierOriginal


from datascope.importance.common import SklearnModel
from ..baselines.matchingnet import resnet12, default_transform, MatchingNetworks


disable_default_handler()


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


class XGBClassifier(SklearnModel, BaseEstimator, ClassifierMixin):
    model: XGBClassifierOriginal

    def __init__(
        self,
        max_depth: Optional[int] = None,
        n_estimators: int = 100,
        subsample: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.model = XGBClassifierOriginal(
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            random_state=0,
            **kwargs,
        )
        self.label_encoder = LabelEncoder()

    def fit(
        self, X: Union[NDArray, DataFrame], y: Union[NDArray, Series], sample_weight: Optional[NDArray] = None
    ) -> None:
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X: Union[NDArray, DataFrame]) -> NDArray:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self.model.predict(X)

    def predict_proba(self, X: Union[NDArray, DataFrame]) -> NDArray:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self.model.predict_proba(X)


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


class EvalLoggerCallback(TrainerCallback):
    def __init__(self, logger: Optional[Logger] = None, prefix: str = "") -> None:
        self.logger = logger
        self.prefix = prefix

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs
    ):
        if self.logger is not None:
            prefix = "[%s] " % self.prefix if self.prefix else ""
            message = prefix + ", ".join(["%s=%.3f" % (k, v) for k, v in metrics.items()])
            self.logger.debug(message)


def stable_softmax(x: NDArray, axis: Optional[int] = None) -> NDArray:
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_probs = exp_x / sum_exp_x
    return softmax_probs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions_proba = normalize(stable_softmax(logits, axis=1))
    predictions = np.argmax(predictions_proba, axis=1)
    if predictions_proba.shape[1] == 2:
        predictions_proba = predictions_proba[:, 1]
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    roc_auc = roc_auc_score(y_true=labels, y_score=predictions_proba, multi_class="ovr")
    return {"accuracy": accuracy, "roc_auc": roc_auc}


BATCH_SIZE = 32


class ResNet18Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs: int = 10, eval_split: float = 0.1, logger: Optional[Logger] = None) -> None:
        self.n_epochs = n_epochs
        self.eval_split = eval_split
        self.logger = logger
        self.cuda_mode = torch.cuda.is_available()
        self.label_encoder = LabelEncoder()

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.classes_ = unique_labels(y)
        y = self.label_encoder.fit_transform(y)
        self.feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.model: ResNetForImageClassification = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-18", num_labels=len(self.classes_), ignore_mismatched_sizes=True
        )
        if self.cuda_mode:
            self.model = self.model.to("cuda:0")

        self.val_dataset: Optional[TorchDataset] = None
        if self.eval_split > 0:
            X, X_eval, y, y_eval = train_test_split(X, y, test_size=self.eval_split, stratify=y, random_state=0)
            self.eval_dataset = TorchDataset(X_eval, y_eval, self.feature_extractor)
        self.train_dataset = TorchDataset(X, y, self.feature_extractor)

        self.tempdir = tempfile.mkdtemp(prefix="resnet18")
        self.training_args = TrainingArguments(
            output_dir=self.tempdir,
            num_train_epochs=self.n_epochs,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=150,  # Evaluation and Save happens every 150 steps
            save_total_limit=15,
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=10),
                EvalLoggerCallback(self.logger, prefix=self.__class__.__name__),
            ],
        )
        self.trainer.remove_callback(PrinterCallback)
        self.trainer.train()
        self.trainer.evaluate()

    def _forward(self, X: NDArray, output_embeddings: bool = False) -> NDArray:
        if X.ndim == 3:
            X = np.expand_dims(X, axis=X.ndim)
        if X.shape[-1] == 1:
            X = np.tile(X, (1, 1, 1, 3))

        inputs = list(X)
        results: List[np.ndarray] = []
        batch_size = BATCH_SIZE
        batches = range(0, len(inputs), batch_size)
        i = 0

        while i < len(batches) and batch_size > 0:
            try:

                features = self.feature_extractor(
                    inputs[batches[i] : batches[i] + batch_size], return_tensors="pt"  # noqa: E203
                )

                if self.cuda_mode:
                    features = features.to("cuda:0")

                with torch.no_grad():
                    if output_embeddings:
                        result = self.model.resnet(**inputs)
                        outputs = result.pooler_output
                    else:
                        result = self.model(**inputs)
                        outputs = result.logits

                if self.cuda_mode:
                    outputs = outputs.cpu()

                results.append(outputs.numpy())

            except torch.cuda.OutOfMemoryError:  # type: ignore
                batch_size = batch_size // 2
                batches = range(0, len(inputs), batch_size)
                i = 0
                results = []
                if self.logger is not None:
                    self.logger.debug("New batch size: ", batch_size)

        return np.concatenate(results)

    def predict(self, X: NDArray) -> NDArray:
        outputs = self._forward(X)
        y_pred = np.argmax(outputs, axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X: NDArray) -> NDArray:
        outputs = self._forward(X)
        return normalize(stable_softmax(outputs, axis=1))

    def transform(self, X: NDArray) -> NDArray:
        outputs = self._forward(X, output_embeddings=True)
        return np.squeeze(outputs, axis=(2, 3))


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
        n_epochs = kwargs.get("n_epochs", 10)
        eval_split = kwargs.get("eval_split", 0.1)
        logger = kwargs.get("logger", None)
        model = ResNet18Classifier(n_epochs=n_epochs, eval_split=eval_split, logger=logger)
    elif model_type == ModelType.MatchingNet:
        model = MatchingNetworkClassifier()
    else:
        raise ValueError("Unknown model type '%s'." % str(model_type))
    return model

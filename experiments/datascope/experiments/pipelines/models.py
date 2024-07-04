import numpy as np
import tempfile
import torch

from abc import abstractmethod
from huggingface_hub import hf_hub_download
from logging import Logger
from methodtools import lru_cache
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader
from transformers import (
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
from torchvision.transforms import v2 as transforms
from transformers.utils.logging import disable_default_handler
from typing import Dict, Optional, Union, Any, List, Type
from xgboost import XGBClassifier as XGBClassifierOriginal


from datascope.importance.common import SklearnModel, ExtendedModelMixin
from ..bench import Configurable, attribute
from ..datasets import Dataset
from .utility import TorchImageDataset, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..baselines.matchingnet import resnet12, default_transform, MatchingNetworks


disable_default_handler()


class XGBClassifier(SklearnModel, BaseEstimator, ClassifierMixin):
    model: XGBClassifierOriginal

    def __init__(
        self,
        max_depth: Optional[int] = None,
        num_estimators: int = 100,
        subsample: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.max_depth = max_depth
        self.num_estimators = num_estimators
        self.subsample = subsample
        self.model = XGBClassifierOriginal(
            max_depth=max_depth,
            n_estimators=num_estimators,
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
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X: Union[NDArray, DataFrame]) -> NDArray:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self.model.predict_proba(X)


class EvalLoggerCallback(TrainerCallback):
    def __init__(
        self,
        logger: Optional[Logger] = None,
        prefix: str = "",
        trim_metric_prefix: Optional[str] = None,
        drop_metrics_with_suffix: Optional[List[str]] = None,
    ) -> None:
        self.logger = logger
        self.prefix = prefix
        self.trim_metric_prefix = trim_metric_prefix
        self.drop_metrics_with_suffix = drop_metrics_with_suffix

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs
    ):
        if self.logger is not None:

            # Add prefix to the message.
            prefix = "[%s] " % self.prefix if self.prefix else ""

            # Drop metrics with specified suffix is provided.
            if self.drop_metrics_with_suffix is not None:
                metrics = {
                    k: v
                    for k, v in metrics.items()
                    if not any(k.endswith(suffix) for suffix in self.drop_metrics_with_suffix)
                }
            # Trim metric prefix if specified.
            if self.trim_metric_prefix is not None:
                metrics = {k.removeprefix(self.trim_metric_prefix): v for k, v in metrics.items()}

            # Construct the message and log it.
            message = prefix + ", ".join(["%s=%.3f" % (k, v) for k, v in metrics.items()])
            self.logger.debug(message)


class StopIfUnstableCallback(TrainerCallback):
    def __init__(self, loss_key: str = "eval_loss", max_value: float = 1000.0) -> None:
        self.loss_key = loss_key
        self.max_value = max_value

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs
    ):
        if self.loss_key in metrics:
            if metrics[self.loss_key] > self.max_value or np.isnan(metrics[self.loss_key]):
                control.should_training_stop = True


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


class ResNet18Classifier(BaseEstimator, ClassifierMixin, ExtendedModelMixin):

    def __init__(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
        eval_split: Union[float, int] = 0.1,
        metadata_grouping_col: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.eval_split = eval_split
        self.metadata_grouping_col = metadata_grouping_col
        self.logger = logger
        self.cuda_mode = torch.cuda.is_available()
        self.device = "cuda:0" if self.cuda_mode else "cpu"
        self.label_encoder = LabelEncoder()
        self.y_type: Optional[type] = None
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    def fit_extended(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
        X_val: Optional[Union[NDArray, DataFrame]] = None,
        y_val: Optional[Union[NDArray, Series]] = None,
        metadata_val: Optional[Union[NDArray, DataFrame]] = None,
    ) -> ExtendedModelMixin:
        X = X if isinstance(X, np.ndarray) else X.to_numpy()
        y = y if isinstance(y, np.ndarray) else y.to_numpy()
        if (X_val is None or y_val is None) and self.eval_split > 0:
            if self.metadata_grouping_col is not None and metadata is not None and isinstance(metadata, DataFrame):
                splitter = GroupShuffleSplit(n_splits=1, test_size=self.eval_split, random_state=0)
                groups = metadata.reset_index()[self.metadata_grouping_col].to_numpy()
                idx_train, idx_val = next(splitter.split(X, y, groups=groups))
                X, X_val, y, y_val = X[idx_train], X[idx_val], y[idx_train], y[idx_val]
            else:
                X, X_val, y, y_val = train_test_split(X, y, test_size=self.eval_split, stratify=y, random_state=0)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X_val is None or isinstance(X_val, np.ndarray)
        assert y_val is None or isinstance(y_val, np.ndarray)

        self._fit(X, y, X_val, y_val)
        return self

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.y_type = type(y)
        X_val: Optional[NDArray] = None
        y_val: Optional[NDArray] = None
        if self.eval_split > 0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=self.eval_split, stratify=y, random_state=0)
        self._fit(X, y, X_val, y_val)

    def _fit(self, X: NDArray, y: NDArray, X_val: Optional[NDArray], y_val: Optional[NDArray]) -> None:
        self.classes_ = unique_labels(y)
        y = self.label_encoder.fit_transform(y)
        self.train_dataset = TorchImageDataset(X, y, self.preprocessor, device=self.device)
        self.eval_dataset: Optional[TorchImageDataset] = None
        if X_val is not None and y_val is not None:
            self.eval_dataset = TorchImageDataset(X_val, y_val, self.preprocessor, device=self.device)

        # torch.autograd.set_detect_anomaly(True)
        sucess = False
        while not sucess:
            self.model: ResNetForImageClassification = ResNetForImageClassification.from_pretrained(
                "microsoft/resnet-18", num_labels=len(self.classes_), ignore_mismatched_sizes=True
            )
            self.model.classifier[1].reset_parameters()  # Reset the last layer to random weights to ensure stability.
            if self.cuda_mode:
                self.model = self.model.to(self.device)

            self.tempdir = tempfile.mkdtemp(prefix="resnet18")
            self.training_args = TrainingArguments(
                learning_rate=self.learning_rate,
                output_dir=self.tempdir,
                num_train_epochs=self.num_epochs,
                evaluation_strategy=IntervalStrategy.STEPS,
                eval_steps=150,  # Evaluation and Save happens every 150 steps
                save_steps=150,  # Evaluation and Save happens every 150 steps
                save_total_limit=15,
                load_best_model_at_end=True,
                metric_for_best_model="roc_auc",
                label_smoothing_factor=0.1,
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
                    StopIfUnstableCallback(max_value=1000.0),
                ],
            )
            self.trainer.remove_callback(PrinterCallback)
            self.trainer.train()
            metrics = self.trainer.evaluate()
            if metrics["eval_loss"] > 1000.0 or np.isnan(metrics["eval_loss"]):
                self.learning_rate = self.learning_rate / 2
                self.num_epochs = self.num_epochs * 2
                if self.logger is not None:
                    self.logger.debug("Training failed, restarting. New learning rate: " + str(self.learning_rate))
            else:
                sucess = True

    def _forward(self, X: NDArray, output_embeddings: bool = False) -> NDArray:
        results: List[np.ndarray] = []
        batch_size = BATCH_SIZE
        success = False
        dataset = TorchImageDataset(X, None, self.preprocessor, device=self.device)

        while not success:
            try:
                loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
                for batch in loader:
                    with torch.no_grad():
                        if output_embeddings:
                            result = self.model.resnet(batch)
                            outputs = result.pooler_output
                        else:
                            result = self.model(batch)
                            outputs = result.logits
                    if self.cuda_mode:
                        outputs = outputs.cpu()
                    results.append(outputs.numpy())
                success = True

            except torch.cuda.OutOfMemoryError:  # type: ignore
                batch_size = batch_size // 2
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

    def predict_extended(
        self,
        X: Union[NDArray, DataFrame],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
    ) -> Union[NDArray, Series]:
        assert self.y_type is not None
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        result = self.predict(X)
        return self.y_type(result)

    def predict_proba_extended(
        self,
        X: Union[NDArray, DataFrame],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
    ) -> Union[NDArray, DataFrame]:
        if isinstance(X, DataFrame):
            X = X.to_numpy()
        return self.predict_proba(X)

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


class Model(Configurable, abstract=True, argname="model"):

    @abstractmethod
    def construct(self: "Model", dataset: Dataset) -> BaseEstimator:
        raise NotImplementedError()

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["Model"]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for class_id, class_type in cls.get_subclasses().items():
            assert issubclass(class_type, Model)
            result[class_id] = class_type._class_longname
        return result

    def __init__(self, **kwargs: Any) -> None:
        pass


class BaseModel(Model, abstract=True, id="base", longname="Base Model"):
    pass


class LogisticRegressionModel(BaseModel, id="logreg", longname="Logistic Regression"):

    def __init__(self, solver: str = "liblinear", max_iter: int = 5000, **kwargs) -> None:
        self._solver = solver
        self._max_iter = max_iter

    @attribute
    def solver(self) -> str:
        """Algorithm to use in the optimization problem."""
        return self._solver

    @attribute
    def max_iter(self) -> int:
        return self._max_iter

    def construct(self: "LogisticRegressionModel", dataset: Dataset) -> BaseEstimator:
        return LogisticRegression(solver=self.solver, max_iter=self.max_iter, random_state=666)


class RandomForestModel(BaseModel, id="randf", longname="Random Forest"):
    def __init__(self, num_estimators: int = 50, **kwargs) -> None:
        self._num_estimators = num_estimators

    @attribute
    def num_estimators(self) -> int:
        """The number of trees in the forest."""
        return self._num_estimators

    def construct(self: "RandomForestModel", dataset: Dataset) -> BaseEstimator:
        return RandomForestClassifier(n_estimators=self.num_estimators, random_state=666)


class KNearestNeighborsModel(BaseModel, id="knn", longname="K-Nearest Neighbors"):
    def __init__(self, num_neighbors: int = 1, **kwargs) -> None:
        self._num_neighbors = num_neighbors

    @attribute
    def num_neighbors(self) -> int:
        """Number of neighbors to use."""
        return self._num_neighbors

    def construct(self: "KNearestNeighborsModel", dataset: Dataset) -> BaseEstimator:
        return KNeighborsClassifier(n_neighbors=self.num_neighbors)


class KNearestNeighborsModelK1(KNearestNeighborsModel, id="knn-1", longname="K-Nearest Neighbors (K=1)"):
    def __init__(self, **kwargs) -> None:
        super().__init__(num_neighbors=1)


class KNearestNeighborsModelK3(KNearestNeighborsModel, id="knn-3", longname="K-Nearest Neighbors (K=3)"):
    def __init__(self, **kwargs) -> None:
        super().__init__(num_neighbors=3)


class KNearestNeighborsModelK5(KNearestNeighborsModel, id="knn-5", longname="K-Nearest Neighbors (K=5)"):
    def __init__(self, **kwargs) -> None:
        super().__init__(num_neighbors=5)


class KNearestNeighborsModelK10(KNearestNeighborsModel, id="knn-10", longname="K-Nearest Neighbors (K=10)"):
    def __init__(self, **kwargs) -> None:
        super().__init__(num_neighbors=10)


class KNearestNeighborsModelK50(KNearestNeighborsModel, id="knn-50", longname="K-Nearest Neighbors (K=50)"):
    def __init__(self, **kwargs) -> None:
        super().__init__(num_neighbors=50)


class KNearestNeighborsModelK100(KNearestNeighborsModel, id="knn-100", longname="K-Nearest Neighbors (K=100)"):
    def __init__(self, **kwargs) -> None:
        super().__init__(num_neighbors=100)


class SupportVectorMachineModel(BaseModel, id="svm", longname="Support Vector Machine"):
    def __init__(self, kernel: str = "rbf", **kwargs) -> None:
        self._kernel = kernel

    @attribute
    def kernel(self) -> str:
        """Kernel type to use in the algorithm."""
        return self._kernel

    def construct(self: "SupportVectorMachineModel", dataset: Dataset) -> BaseEstimator:
        return SVC(kernel=self.kernel, random_state=666)


class LinearSupportVectorMachineModel(BaseModel, id="linsvm", longname="Linear Support Vector Machine"):
    def construct(self: "LinearSupportVectorMachineModel", dataset: Dataset) -> BaseEstimator:
        return LinearSVC(loss="hinge", random_state=666)


class GaussianProcessModel(BaseModel, id="gp", longname="Gaussian Process"):
    def construct(self: "GaussianProcessModel", dataset: Dataset) -> BaseEstimator:
        return GaussianProcessClassifier(random_state=666)


class NaiveBayesModel(BaseModel, id="nb", longname="Naive Bayes Classifier"):
    def construct(self: "NaiveBayesModel", dataset: Dataset) -> BaseEstimator:
        return MultinomialNB()


class MultilevelPerceptronModel(BaseModel, id="mlp", longname="Multilevel Perceptron"):
    def __init__(
        self,
        solver: str = "sgd",
        hidden_layer_sizes: int = 20,
        activation: str = "relu",
        learning_rate: float = 0.001,
        max_iter: int = 5000,
        early_stopping: bool = False,
        warm_start: bool = False,
        **kwargs,
    ) -> None:
        self._solver = solver
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._early_stopping = early_stopping
        self._warm_start = warm_start

    @attribute
    def solver(self) -> str:
        """The solver for weight optimization."""
        return self._solver

    @attribute
    def hidden_layer_sizes(self) -> int:
        """The number of neurons in the hidden layers."""
        return self._hidden_layer_sizes

    @attribute
    def activation(self) -> str:
        """The activation function for the hidden layers."""
        return self._activation

    @attribute
    def learning_rate(self) -> float:
        """The learning rate for weight updates."""
        return self._learning_rate

    @attribute
    def max_iter(self) -> int:
        """The maximum number of iterations."""
        return self._max_iter

    @attribute
    def early_stopping(self) -> bool:
        """Whether to use early stopping."""
        return self._early_stopping

    @attribute
    def warm_start(self) -> bool:
        """Whether to reuse the solution of the previous call to fit."""
        return self._warm_start

    def construct(self: "MultilevelPerceptronModel", dataset: Dataset) -> BaseEstimator:
        return MLPClassifier(
            solver=self.solver,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate,
            warm_start=self.warm_start,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
        )


class XGBoostModel(BaseModel, id="xgb", longname="XGBoost"):
    def __init__(self, num_estimators: int = 100, max_depth: int = 6, subsample: float = 1.0, **kwargs) -> None:
        self._num_estimators = num_estimators
        self._max_depth = max_depth
        self._subsample = subsample

    @attribute
    def num_estimators(self) -> int:
        """The number of trees in the forest."""
        return self._num_estimators

    @attribute
    def max_depth(self) -> int:
        """The maximum depth of the tree."""
        return self._max_depth

    @attribute
    def subsample(self) -> float:
        """The fraction of samples to be used for fitting the individual base learners."""
        return self._subsample

    def construct(self: "XGBoostModel", dataset: Dataset) -> BaseEstimator:
        return XGBClassifier(
            nthread=1,
            eval_metric="logloss",
            num_estimators=self.num_estimators,
            max_depth=self.max_depth,
            subsample=self.subsample,
        )


class Resnet18Model(BaseModel, id="resnet-18", longname="ResNet-18"):
    def __init__(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
        eval_split: Union[float, int] = 0.1,
        **kwargs,
    ) -> None:
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._eval_split = eval_split

    @attribute
    def num_epochs(self) -> int:
        """The number of epochs to train the model."""
        return self._num_epochs

    @attribute
    def learning_rate(self) -> float:
        """The learning rate for the optimizer."""
        return self._learning_rate

    @attribute
    def eval_split(self) -> Union[float, int]:
        """The fraction of the dataset to use for evaluation."""
        return self._eval_split

    def construct(self: "Resnet18Model", dataset: Dataset) -> BaseEstimator:
        return ResNet18Classifier(
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            eval_split=self.eval_split,
            metadata_grouping_col=dataset.metadata_grouping_col,
        )


class MatchingNetworkModel(BaseModel, id="matchingnet", longname="Matching Network"):
    def construct(self: "MatchingNetworkModel", dataset: Dataset) -> BaseEstimator:
        return MatchingNetworkClassifier()

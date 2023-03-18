import numpy as np

from datascope.importance.importance import Importance
from numpy import ndarray
from sklearn.utils.multiclass import unique_labels
from typing import Optional, Iterable

from .dataset import DataSet
from .logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS


class InfluenceImportance(Importance):
    def _fit(self, X: ndarray, y: ndarray, provenance: Optional[ndarray] = None) -> "Importance":
        self.classes = unique_labels(y)
        self.X_train = np.reshape(X, newshape=(X.shape[0], -1))
        self.y_train = y
        self.dataset = DataSet(self.X_train, self.y_train)

        self.model = LogisticRegressionWithLBFGS(
            input_dim=self.X_train.shape[1],
            weight_decay=0.01,
            max_lbfgs_iter=1000,
            num_classes=len(self.classes),
            batch_size=100,
            train_dataset=self.dataset,
            initial_learning_rate=0.001,
            keep_probs=None,
            decay_epochs=[1000, 10000],
            mini_batch=False,
            train_dir="output",
            log_dir="log",
            model_name="mnist_logreg_lbfgs",
        )
        self.model.train()
        return self

    def _score(self, X: ndarray, y: Optional[ndarray] = None, **kwargs) -> Iterable[float]:
        X = np.reshape(X, newshape=(X.shape[0], -1))
        self.model.update_test_x_y(X, y)
        predicted_loss_diffs = self.model.get_influence_on_test_loss(
            test_indices=None, train_idx=np.arange(len(self.y_train)), force_refresh=True
        )
        return predicted_loss_diffs

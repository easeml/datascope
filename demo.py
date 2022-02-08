import sys  
from pathlib import Path  
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

import numpy as np
import re
import sklearn.pipeline

from copy import deepcopy

from abc import abstractmethod
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from typing import Dict, Iterable, Type, Optional

from datascope.importance.common import SklearnModelUtility, binarize, get_indices
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod

from experiments.dataset import Dataset
from experiments.pipelines import Pipeline, get_model, ModelType

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Inject noise
X_train_dirty = deepcopy(X_train)
y_train_dirty = deepcopy(y_train)
y_train_dirty = 2 - y_train_dirty

model = get_model(ModelType.LogisticRegression)
utility = SklearnModelUtility(model, accuracy_score)

method = ImportanceMethod.NEIGHBOR
importance = ShapleyImportance(method=method, utility=utility)
importances = importance.fit(X_train_dirty, y_train_dirty).score(X_test, y_test)


ordered_examples = np.argsort(importances)

for i in ordered_examples:

	# current model
	clf = LogisticRegression(random_state=0).fit(X_train_dirty, y_train_dirty)
	score = clf.score(X_test, y_test)
	print(score)

	# fix a label
	y_train_dirty[i] = y_train[i]











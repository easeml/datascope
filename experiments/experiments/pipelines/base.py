import numpy as np
import re
import sklearn.pipeline

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

from ..datasets import DatasetModality, Dataset


class Pipeline(sklearn.pipeline.Pipeline):

    pipelines: Dict[str, Type["Pipeline"]] = {}
    summaries: Dict[str, str] = {}
    _pipeline: Optional[str] = None
    _modalities: Iterable[DatasetModality]
    _summary: Optional[str] = None

    def __init_subclass__(
        cls: Type["Pipeline"],
        modalities: Iterable[DatasetModality],
        id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        cls._pipeline = id if id is not None else cls.__name__
        cls._modalities = modalities
        Pipeline.pipelines[cls._pipeline] = cls
        if summary is not None:
            Pipeline.summaries[cls._pipeline] = summary

    @property
    def modalities(self) -> Iterable[DatasetModality]:
        return self._modalities

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @classmethod
    @abstractmethod
    def construct(cls: Type["Pipeline"], dataset: Dataset) -> "Pipeline":
        raise NotImplementedError()


class IdentityPipeline(Pipeline, id="identity", summary="Identity", modalities=[DatasetModality.TABULAR]):
    """A pipeline that passes its input data as is."""

    @classmethod
    def construct(cls: Type["IdentityPipeline"], dataset: Dataset) -> "IdentityPipeline":
        def identity(x):
            return x

        ops = [("identity", FunctionTransformer(identity))]
        return IdentityPipeline(ops)


class StandardScalerPipeline(
    Pipeline, id="std-scaler", summary="Standard Scaler", modalities=[DatasetModality.TABULAR]
):
    """A pipeline that applies a standard scaler to the input data."""

    @classmethod
    def construct(cls: Type["StandardScalerPipeline"], dataset: Dataset) -> "StandardScalerPipeline":
        ops = [("scaler", StandardScaler())]
        return StandardScalerPipeline(ops)


class LogScalerPipeline(Pipeline, id="log-scaler", summary="Logarithmic Scaler", modalities=[DatasetModality.TABULAR]):
    """A pipeline that applies a logarithmic scaler to the input data."""

    @classmethod
    def construct(cls: Type["LogScalerPipeline"], dataset: Dataset) -> "LogScalerPipeline":
        def log1p(x):
            return np.log1p(np.abs(x))

        ops = [("log", FunctionTransformer(log1p)), ("scaler", StandardScaler())]
        return LogScalerPipeline(ops)


class PcaPipeline(Pipeline, id="pca", summary="PCA", modalities=[DatasetModality.TABULAR]):
    """A pipeline that applies a principal component analysis operator."""

    @classmethod
    def construct(cls: Type["PcaPipeline"], dataset: Dataset) -> "PcaPipeline":
        ops = [("PCA", PCA())]
        return PcaPipeline(ops)


class PcaSvdPipeline(Pipeline, id="pca-svd", summary="PCA + SVD", modalities=[DatasetModality.TABULAR]):
    """
    A pipeline that applies a combination of the principal component analysis and
    singular value decomposition operators.
    """

    @classmethod
    def construct(cls: Type["PcaSvdPipeline"], dataset: Dataset) -> "PcaSvdPipeline":
        union = FeatureUnion([("pca", PCA(n_components=2)), ("svd", TruncatedSVD(n_iter=1))])
        ops = [("union", union), ("scaler", StandardScaler())]
        return PcaSvdPipeline(ops)


class KMeansPipeline(
    Pipeline, id="mi-kmeans", summary="Missing Indicator + K-Means", modalities=[DatasetModality.TABULAR]
):
    """
    A pipeline that applies a combination of the missing value indicator and
    the K-Means featurizer operators.
    """

    @classmethod
    def construct(cls: Type["KMeansPipeline"], dataset: Dataset) -> "KMeansPipeline":
        union = FeatureUnion([("indicator", MissingIndicator()), ("kmeans", KMeans(random_state=0))])
        ops = [("union", union)]
        return KMeansPipeline(ops)


class GaussBlurPipeline(Pipeline, id="gauss-blur", summary="Gaussian Blur", modalities=[DatasetModality.IMAGE]):
    """
    A pipeline that applies a gaussian blure filter.
    """

    @classmethod
    def construct(cls: Type["GaussBlurPipeline"], dataset: Dataset) -> "GaussBlurPipeline":
        def gaussian_blur(x):
            def gaussian_blur_single(x):
                return gaussian_filter(x, sigma=5).flatten()

            return np.array([gaussian_blur_single(img) for img in x])

        ops = [("blur", FunctionTransformer(gaussian_blur))]
        return GaussBlurPipeline(ops)


DEFAULT_HOG_ORIENTATIONS = 9
DEFAULT_HOG_PIXELS_PER_CELL = 8
DEFAULT_HOG_CELLS_PER_BLOCK = 3
DEFAULT_HOG_BLOCK_NORM = "L2-Hys"


class HogTransformPipeline(
    Pipeline, id="hog-transform", summary="Histogram of Oriented Gradients", modalities=[DatasetModality.IMAGE]
):
    """
    A pipeline that applies a histogram of oriented gradients operator.
    """

    @classmethod
    def construct(
        cls: Type["HogTransformPipeline"],
        dataset: Dataset,
        orientations: int = DEFAULT_HOG_ORIENTATIONS,
        pixels_per_cell: int = DEFAULT_HOG_PIXELS_PER_CELL,
        cells_per_block: int = DEFAULT_HOG_CELLS_PER_BLOCK,
        block_norm: str = DEFAULT_HOG_BLOCK_NORM,
    ) -> "HogTransformPipeline":
        def hog_transform(X: np.ndarray) -> np.ndarray:
            def hog_single(image):
                return hog(
                    image=image,
                    orientations=orientations,
                    pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                    cells_per_block=(cells_per_block, cells_per_block),
                    block_norm=block_norm,
                )

            return np.array([hog_single(img) for img in X])

        ops = [("hog", FunctionTransformer(hog_transform))]
        return HogTransformPipeline(ops)


class TfidfPipeline(Pipeline, id="tf-idf", summary="TF-IDF", modalities=[DatasetModality.TEXT]):
    """
    A pipeline that applies a count vectorizer and a TF-IDF transform.
    """

    @classmethod
    def construct(cls: Type["TfidfPipeline"], dataset: Dataset) -> "TfidfPipeline":
        ops = [("vect", CountVectorizer()), ("tfidf", TfidfTransformer())]
        return TfidfPipeline(ops)


class ToLowerUrlRemovePipeline(
    Pipeline, id="tolower-urlremove-tfidf", summary="ToLower + URLRemove + TF-IDF", modalities=[DatasetModality.TEXT]
):
    """
    A pipeline that applies a few text transformations such as converting everything to lowercase and removing URL's.
    """

    @classmethod
    def construct(cls: Type["ToLowerUrlRemovePipeline"], dataset: Dataset) -> "ToLowerUrlRemovePipeline":
        def text_lowercase(text_array):
            return list(map(lambda x: x.lower(), text_array))

        def remove_urls(text_array):
            def remove_url(text):
                return " ".join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

            return list(map(remove_url, text_array))

        ops = [
            ("lower_case", FunctionTransformer(text_lowercase)),
            ("remove_url", FunctionTransformer(remove_urls)),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
        ]
        return ToLowerUrlRemovePipeline(ops)

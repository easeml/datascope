import functools
import inspect
import numpy as np
import re
import sklearn.pipeline
import torch
import transformers

from abc import abstractmethod
from hashlib import md5
from scipy.ndimage.filters import gaussian_filter1d
from sentence_transformers import SentenceTransformer
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from transformers import AutoImageProcessor, ResNetModel
from transformers.image_processing_utils import BatchFeature, BaseImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention
from transformers.modeling_utils import PreTrainedModel
from typing import Dict, Iterable, Type, Optional

# import torch.nn.functional as F
# from transformers import MobileBertTokenizer, MobileBertModel
# from transformers.tokenization_utils import BatchEncoding
# from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..datasets import DatasetModality, Dataset


transformers.utils.logging.set_verbosity_error()


class Pipeline(sklearn.pipeline.Pipeline):

    pipelines: Dict[str, Type["Pipeline"]] = {}
    summaries: Dict[str, str] = {}
    _pipeline: Optional[str] = None
    _modalities: Iterable[DatasetModality]
    _summary: Optional[str] = None

    def __init_subclass__(
        cls: Type["Pipeline"],
        modalities: Iterable[DatasetModality],
        abstract: bool = False,
        id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        if abstract:
            return

        cls._pipeline = id if id is not None else cls.__name__
        cls._modalities = modalities
        Pipeline.pipelines[cls._pipeline] = cls
        if summary is not None:
            Pipeline.summaries[cls._pipeline] = summary

    def __hash_string__(self) -> str:
        myclass: Type[Pipeline] = type(self)
        hash = md5()
        for cls in inspect.getmro(myclass):
            if cls != object:
                hash.update(inspect.getsource(cls).encode("utf-8"))
        return "%s.%s(hash=%s)" % (type(self).__module__, myclass.__name__, hash.hexdigest())

    def __repr__(self, N_CHAR_MAX=700):
        return "%s.%s()" % (type(self).__module__, type(self).__name__)

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

    @staticmethod
    def _log1p(x):
        return np.log1p(np.abs(x))

    @classmethod
    def construct(cls: Type["LogScalerPipeline"], dataset: Dataset) -> "LogScalerPipeline":

        ops = [("log", FunctionTransformer(cls._log1p)), ("scaler", StandardScaler())]
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
        union = FeatureUnion([("indicator", MissingIndicator()), ("kmeans", KMeans(random_state=0, n_init=10))])
        ops = [("union", union)]
        return KMeansPipeline(ops)


class GaussBlurPipeline(Pipeline, id="gauss-blur", summary="Gaussian Blur", modalities=[DatasetModality.IMAGE]):
    """
    A pipeline that applies a gaussian blure filter.
    """

    @staticmethod
    def _gaussian_blur(x):
        result = x
        for axis in [1, 2]:
            result = gaussian_filter1d(result, sigma=2, axis=axis, truncate=2.0)
        result = np.reshape(result, (result.shape[0], -1))
        return result

    @classmethod
    def construct(cls: Type["GaussBlurPipeline"], dataset: Dataset) -> "GaussBlurPipeline":

        ops = [("blur", FunctionTransformer(cls._gaussian_blur))]
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

    @staticmethod
    def _hog_transform(
        X: np.ndarray,
        orientations: int = DEFAULT_HOG_ORIENTATIONS,
        pixels_per_cell: int = DEFAULT_HOG_PIXELS_PER_CELL,
        cells_per_block: int = DEFAULT_HOG_CELLS_PER_BLOCK,
        block_norm: str = DEFAULT_HOG_BLOCK_NORM,
    ) -> np.ndarray:
        channel_axis = None if X.ndim == 3 else X.ndim - 2

        def hog_single(image):
            return hog(
                image=image,
                orientations=orientations,
                pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block),
                block_norm=block_norm,
                channel_axis=channel_axis,
            )

        return np.array([hog_single(img) for img in X])

    @classmethod
    def construct(
        cls: Type["HogTransformPipeline"],
        dataset: Dataset,
        orientations: int = DEFAULT_HOG_ORIENTATIONS,
        pixels_per_cell: int = DEFAULT_HOG_PIXELS_PER_CELL,
        cells_per_block: int = DEFAULT_HOG_CELLS_PER_BLOCK,
        block_norm: str = DEFAULT_HOG_BLOCK_NORM,
    ) -> "HogTransformPipeline":

        hog_transform = functools.partial(
            cls._hog_transform,
            orientations=orientations,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm=block_norm,
        )

        ops = [("hog", FunctionTransformer(hog_transform))]
        return HogTransformPipeline(ops)


class ResNet18EmbeddingPipeline(
    Pipeline, id="resnet-18", summary="ResNet-18 Embedding", modalities=[DatasetModality.IMAGE]
):  # type: ignore
    """
    A pipeline that extracts embeddings using a ResNet50 model pre-trained on the ImageNet dataset.
    """

    @staticmethod
    def _embedding_transform(
        X: np.ndarray, feature_extractor: BaseImageProcessor, model: PreTrainedModel
    ) -> np.ndarray:
        if X.ndim == 3:
            X = np.expand_dims(X, axis=X.ndim)
        if X.shape[-1] == 1:
            X = np.tile(X, (1, 1, 1, 3))

        inputs: BatchFeature = feature_extractor(list(X), return_tensors="pt")

        with torch.no_grad():
            outputs: BaseModelOutputWithPoolingAndNoAttention = model(**inputs)

        result: np.ndarray = np.squeeze(outputs.pooler_output.numpy())
        return result

    @classmethod
    def construct(cls: Type["ResNet18EmbeddingPipeline"], dataset: Dataset) -> "ResNet18EmbeddingPipeline":

        # if dataset.X_train.ndim not in [3, 4]:
        #     raise ValueError("The provided dataset features must have either 3 or 4 dimensions.")
        # if dataset.X_train.ndim == 4 and dataset.X_train.shape[-1] not in [1, 3]:
        #     raise ValueError("The provided dataset features must be either grayscale or with 3 channels.")

        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        model = ResNetModel.from_pretrained("microsoft/resnet-18")
        embedding_transform = functools.partial(
            cls._embedding_transform, feature_extractor=feature_extractor, model=model
        )

        ops = [("embedding", FunctionTransformer(embedding_transform))]
        return cls(ops)


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

    @staticmethod
    def _text_lowercase(text_array):
        return list(map(lambda x: x.lower(), text_array))

    def _remove_url(text):
        return " ".join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    @staticmethod
    def _remove_urls(text_array):
        return list(map(ToLowerUrlRemovePipeline._remove_url, text_array))

    @classmethod
    def construct(cls: Type["ToLowerUrlRemovePipeline"], dataset: Dataset) -> "ToLowerUrlRemovePipeline":

        ops = [
            ("lower_case", FunctionTransformer(cls._text_lowercase)),
            ("remove_url", FunctionTransformer(cls._remove_urls)),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
        ]
        return ToLowerUrlRemovePipeline(ops)


# class MobileBertEmbeddingPipeline(
#     Pipeline, id="mobile-bert", summary="Mobile-BERT Embedding", modalities=[DatasetModality.IMAGE]
# ):
#     """
#     A pipeline that extracts embeddings using a ResNet50 model pre-trained on the ImageNet dataset.
#     """

#     @classmethod
#     def construct(cls: Type["MobileBertEmbeddingPipeline"], dataset: Dataset) -> "MobileBertEmbeddingPipeline":

#         if dataset.X_train.ndim > 1:
#             raise ValueError("The provided dataset features must have either 0 or 1 dimensions.")

#         tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
#         model = MobileBertModel.from_pretrained("google/mobilebert-uncased")

#         def mean_pooling(model_output, attention_mask):
#             token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#             input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#                 input_mask_expanded.sum(1), min=1e-9
#             )

#         def embedding_transform(X: np.ndarray) -> np.ndarray:

#             inputs: BatchEncoding = tokenizer(
#                 list(X), return_tensors="pt", max_length=128, padding=True, truncation=True, add_special_tokens=True
#             )

#             with torch.no_grad():
#                 outputs: BaseModelOutputWithPooling = model(**inputs)

#             sentence_embeddings = mean_pooling(outputs, inputs["attention_mask"])

#             result = F.normalize(sentence_embeddings, p=2, dim=1)

#             return result.numpy()

#         ops = [("embedding", FunctionTransformer(embedding_transform))]
#         return cls(ops)

# Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
class MiniLMEmbeddingPipeline(Pipeline, id="mini-lm", summary="MiniLM Embedding", modalities=[DatasetModality.IMAGE]):
    """
    A pipeline that extracts sentence embeddings using a MiniLM model pre-trained on the 1B sentences.
    """

    @staticmethod
    def _embedding_transform(X: np.ndarray, model: SentenceTransformer) -> np.ndarray:
        embeddings = model.encode(list(X))
        return np.array(embeddings)

    @classmethod
    def construct(cls: Type["MiniLMEmbeddingPipeline"], dataset: Dataset) -> "MiniLMEmbeddingPipeline":

        # if dataset.X_train.ndim > 1:
        #     raise ValueError("The provided dataset features must have either 0 or 1 dimensions.")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_transform = functools.partial(cls._embedding_transform, model=model)

        ops = [("embedding", FunctionTransformer(embedding_transform))]
        return cls(ops)

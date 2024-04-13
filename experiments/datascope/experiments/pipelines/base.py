import functools
import inspect
import numpy as np
import re
import sklearn.pipeline
import torch
import transformers

from abc import abstractmethod
from datascope.utility import Provenance
from hashlib import md5
from numpy.typing import NDArray
from scipy.ndimage.filters import gaussian_filter1d
from sentence_transformers import SentenceTransformer
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from transformers import ResNetModel, AutoModelForImageClassification
from transformers.image_processing_utils import BatchFeature, BaseImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention
from transformers.modeling_utils import PreTrainedModel
from typing import Dict, Iterable, Type, Optional, Union, List, Tuple, Callable

from .utility import TorchImageDataset, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..datasets import Dataset, TabularDatasetMixin, ImageDatasetMixin, TextDatasetMixin

DatasetModality = Union[Type[TabularDatasetMixin], Type[ImageDatasetMixin], Type[TextDatasetMixin]]

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

    def __str__(self) -> str:
        return "%s.%s" % (type(self).__module__, type(self).__name__)

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

    def fit_transform(self, X, y=None, provenance: Union[Provenance, NDArray, None] = None, **fit_params):
        Xt = super().fit_transform(X=X, y=y, **fit_params)
        if provenance is None:
            return Xt
        else:
            provenance = self.transform_provenance(X=X, y=y, provenance=provenance)
            return Xt, provenance

    def transform_provenance(
        self, X, y=None, provenance: Union[Provenance, NDArray, None] = None
    ) -> Union[Provenance, NDArray, None]:
        return provenance


class IdentityPipeline(Pipeline, id="identity", summary="Identity", modalities=[TabularDatasetMixin]):
    """A pipeline that passes its input data as is."""

    @classmethod
    def construct(cls: Type["IdentityPipeline"], dataset: Dataset) -> "IdentityPipeline":
        def identity(x):
            return x

        ops = [("identity", FunctionTransformer(identity))]
        return IdentityPipeline(ops)


class StandardScalerPipeline(Pipeline, id="std-scaler", summary="Standard Scaler", modalities=[TabularDatasetMixin]):
    """A pipeline that applies a standard scaler to the input data."""

    @classmethod
    def construct(cls: Type["StandardScalerPipeline"], dataset: Dataset) -> "StandardScalerPipeline":
        ops = [("scaler", StandardScaler())]
        return StandardScalerPipeline(ops)


class LogScalerPipeline(Pipeline, id="log-scaler", summary="Logarithmic Scaler", modalities=[TabularDatasetMixin]):
    """A pipeline that applies a logarithmic scaler to the input data."""

    @staticmethod
    def _log1p(x):
        return np.log1p(np.abs(x))

    @classmethod
    def construct(cls: Type["LogScalerPipeline"], dataset: Dataset) -> "LogScalerPipeline":
        ops = [("log", FunctionTransformer(cls._log1p)), ("scaler", StandardScaler())]
        return LogScalerPipeline(ops)


class PcaPipeline(Pipeline, id="pca", summary="PCA", modalities=[TabularDatasetMixin]):
    """A pipeline that applies a principal component analysis operator."""

    @classmethod
    def construct(cls: Type["PcaPipeline"], dataset: Dataset) -> "PcaPipeline":
        ops = [("PCA", PCA())]
        return PcaPipeline(ops)


class PcaSvdPipeline(Pipeline, id="pca-svd", summary="PCA + SVD", modalities=[TabularDatasetMixin]):
    """
    A pipeline that applies a combination of the principal component analysis and
    singular value decomposition operators.
    """

    @classmethod
    def construct(cls: Type["PcaSvdPipeline"], dataset: Dataset) -> "PcaSvdPipeline":
        union = FeatureUnion([("pca", PCA(n_components=2)), ("svd", TruncatedSVD(n_iter=1))])
        ops = [("union", union), ("scaler", StandardScaler())]
        return PcaSvdPipeline(ops)


class KMeansPipeline(Pipeline, id="mi-kmeans", summary="Missing Indicator + K-Means", modalities=[TabularDatasetMixin]):
    """
    A pipeline that applies a combination of the missing value indicator and
    the K-Means featurizer operators.
    """

    @classmethod
    def construct(cls: Type["KMeansPipeline"], dataset: Dataset) -> "KMeansPipeline":
        union = FeatureUnion([("indicator", MissingIndicator()), ("kmeans", KMeans(random_state=0, n_init=10))])
        ops = [("union", union)]
        return KMeansPipeline(ops)


class StdScalerKMeansPipeline(
    Pipeline,
    id="std-scaler-mi-kmeans",
    summary="Missing Indicator + Standard Scaler + K-Means",
    modalities=[TabularDatasetMixin],
):
    """
    A pipeline that applies a combination of the missing value indicator, standard scaler and
    the K-Means featurizer operators.
    """

    @classmethod
    def construct(cls: Type["KMeansPipeline"], dataset: Dataset) -> "KMeansPipeline":
        pipe = sklearn.pipeline.Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(random_state=0, n_init=10))])
        union = FeatureUnion([("indicator", MissingIndicator()), ("scaler-kmeans", pipe)])
        ops = [("union", union)]
        return KMeansPipeline(ops)


class GaussBlurPipeline(Pipeline, id="gauss-blur", summary="Gaussian Blur", modalities=[ImageDatasetMixin]):
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
    Pipeline, id="hog-transform", summary="Histogram of Oriented Gradients", modalities=[ImageDatasetMixin]
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
        channel_axis = None if X.ndim == 3 else -1

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
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
        )

        ops = [("hog", FunctionTransformer(hog_transform))]
        return HogTransformPipeline(ops)


BATCH_SIZE = 32


class ImageEmbeddingPipeline(Pipeline, abstract=True, modalities=[ImageDatasetMixin]):
    """
    A pipeline that extracts embeddings using a pre-trained deep learning model.
    """

    @classmethod
    def get_preprocessor(cls: Type["ImageEmbeddingPipeline"]) -> transforms.Transform:
        # By default we return the standard ResNet x ImageNet preprocessor.
        return transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    @classmethod
    @abstractmethod
    def get_model(cls: Type["ImageEmbeddingPipeline"]) -> PreTrainedModel:
        pass

    @staticmethod
    def model_forward(
        model: PreTrainedModel, batch: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        output: BaseModelOutputWithPoolingAndNoAttention = model(batch)
        return torch.squeeze(output.pooler_output, dim=(2, 3))

    @staticmethod
    def _embedding_transform(
        X: np.ndarray,
        cuda_mode: bool,
        preprocessor: transforms.Transform,
        model: PreTrainedModel,
        model_forward_function: Callable[
            [PreTrainedModel, Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]], torch.Tensor
        ],
    ) -> np.ndarray:

        results: List[np.ndarray] = []
        batch_size = BATCH_SIZE
        success = False
        device = "cuda:0" if cuda_mode else "cpu"
        model = model.to(device)
        dataset = TorchImageDataset(X, None, preprocessor, device=device)

        while not success:
            try:
                loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
                for batch in loader:
                    with torch.no_grad():
                        result = model_forward_function(model, batch)
                    if cuda_mode:
                        result = result.cpu()
                    results.append(result.numpy())
                success = True

            except torch.cuda.OutOfMemoryError:  # type: ignore
                batch_size = batch_size // 2
                results = []
                print("New batch size: ", batch_size)

        model.cpu()
        torch.cuda.empty_cache()
        return np.concatenate(results)

    @classmethod
    def construct(cls: Type["ImageEmbeddingPipeline"], dataset: Dataset) -> "ImageEmbeddingPipeline":
        cuda_mode = torch.cuda.is_available()
        preprocessor = cls.get_preprocessor()
        model = cls.get_model()
        embedding_transform = functools.partial(
            ImageEmbeddingPipeline._embedding_transform,
            cuda_mode=cuda_mode,
            preprocessor=preprocessor,
            model=model,
            model_forward_function=cls.model_forward,
        )

        ops = [("embedding", FunctionTransformer(embedding_transform))]
        return cls(ops)


class ResNet18EmbeddingPipeline(
    ImageEmbeddingPipeline, id="resnet-18", summary="ResNet18 Embedding", modalities=[ImageDatasetMixin]
):  # type: ignore
    """
    A pipeline that extracts embeddings using a ResNet18 model pre-trained on the ImageNet dataset.
    """

    @classmethod
    def get_model(cls: Type["ResNet18EmbeddingPipeline"]) -> PreTrainedModel:
        model = ResNetModel.from_pretrained("microsoft/resnet-18")
        return model


class ResNet50EmbeddingPipeline(
    ImageEmbeddingPipeline, id="resnet-50", summary="ResNet50 Embedding", modalities=[ImageDatasetMixin]
):  # type: ignore
    """
    A pipeline that extracts embeddings using a ResNet50 model pre-trained on the ImageNet dataset.
    """

    @classmethod
    def get_model(cls: Type["ResNet50EmbeddingPipeline"]) -> PreTrainedModel:
        model = ResNetModel.from_pretrained("microsoft/resnet-50")
        return model


class MobileNetV2EmbeddingPipeline(
    ImageEmbeddingPipeline, id="mobilenet-v2", summary="MobileNetV2 Embedding", modalities=[ImageDatasetMixin]
):  # type: ignore
    """
    A pipeline that extracts embeddings using a MobileNetV2 model pre-trained on the ImageNet dataset.
    """

    @classmethod
    def get_preprocessor(cls: Type["ImageEmbeddingPipeline"]) -> transforms.Transform:
        # Source: https://huggingface.co/google/mobilenet_v2_1.0_224/blob/main/preprocessor_config.json
        return transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @classmethod
    def get_model(cls: Type["MobileNetV2EmbeddingPipeline"]) -> PreTrainedModel:
        model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
        return model


class MobileViTEmbeddingPipeline(
    ImageEmbeddingPipeline, id="mobilevit", summary="MobileViT Embedding", modalities=[ImageDatasetMixin]
):  # type: ignore
    """
    A pipeline that extracts embeddings using a MobileViT model pre-trained on the ImageNet dataset.
    """

    @classmethod
    def get_preprocessor(cls: Type["ImageEmbeddingPipeline"]) -> transforms.Transform:
        # Source: https://huggingface.co/apple/mobilevit-small/blob/main/preprocessor_config.json
        return transforms.Compose(
            [
                transforms.Resize(288, antialias=True),
                transforms.CenterCrop(256),
                transforms.Lambda(lambda x: x[:, [2, 1, 0], ...] if x.ndim == 4 else x[[2, 1, 0], ...]),
            ]
        )

    @classmethod
    def get_model(cls: Type["MobileViTEmbeddingPipeline"]) -> PreTrainedModel:
        model = AutoModelForImageClassification.from_pretrained("apple/mobilevit-small")
        return model


class TfidfPipeline(Pipeline, id="tf-idf", summary="TF-IDF", modalities=[TextDatasetMixin]):
    """
    A pipeline that applies a count vectorizer and a TF-IDF transform.
    """

    @classmethod
    def construct(cls: Type["TfidfPipeline"], dataset: Dataset) -> "TfidfPipeline":
        ops = [("vect", CountVectorizer()), ("tfidf", TfidfTransformer())]
        return TfidfPipeline(ops)


class ToLowerUrlRemovePipeline(
    Pipeline, id="tolower-urlremove-tfidf", summary="ToLower + URLRemove + TF-IDF", modalities=[TextDatasetMixin]
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


class TextEmbeddingPipeline(Pipeline, abstract=True, modalities=[TextDatasetMixin]):
    """
    A pipeline that extracts embeddings using a pre-trained deep learning model.
    """

    @classmethod
    @abstractmethod
    def get_model(cls: Type["TextEmbeddingPipeline"]) -> Tuple[BaseImageProcessor, PreTrainedModel]:
        pass

    @staticmethod
    def _embedding_transform(
        X: np.ndarray, cuda_mode: bool, feature_extractor: BaseImageProcessor, model: PreTrainedModel
    ) -> np.ndarray:
        # torch.set_num_threads(1)

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
                features: BatchFeature = feature_extractor(
                    inputs[batches[i] : batches[i] + batch_size], return_tensors="pt"  # noqa: E203
                )

                if cuda_mode:
                    features = features.to("cuda:0")

                with torch.no_grad():
                    outputs: BaseModelOutputWithPoolingAndNoAttention = model(**features)
                    result = outputs.pooler_output

                if cuda_mode:
                    result = result.cpu()

                results.append(np.squeeze(result.numpy(), axis=(2, 3)))
                i += 1

            except torch.cuda.OutOfMemoryError:  # type: ignore
                batch_size = batch_size // 2
                batches = range(0, len(inputs), batch_size)
                i = 0
                results = []
                print("New batch size: ", batch_size)

        return np.concatenate(results)

    @classmethod
    def construct(cls: Type["TextEmbeddingPipeline"], dataset: Dataset) -> "TextEmbeddingPipeline":
        cuda_mode = torch.cuda.is_available()
        feature_extractor, model = cls.get_model()
        if cuda_mode:
            model = model.to("cuda:0")
        embedding_transform = functools.partial(
            TextEmbeddingPipeline._embedding_transform,
            cuda_mode=cuda_mode,
            feature_extractor=feature_extractor,
            model=model,
        )

        ops = [("embedding", FunctionTransformer(embedding_transform))]
        return cls(ops)


# Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
class MiniLMEmbeddingPipeline(Pipeline, id="mini-lm", summary="MiniLM Embedding", modalities=[TextDatasetMixin]):
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


# Source: https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2
class AlbertSmallEmbeddingPipeline(
    Pipeline, id="albert-small-v2", summary="ALBERT Small", modalities=[TextDatasetMixin]
):
    """
    A pipeline that extracts sentence embeddings using a ALBERT Small model pre-trained on the 1B sentences.
    """

    @staticmethod
    def _embedding_transform(X: np.ndarray, model: SentenceTransformer) -> np.ndarray:
        embeddings = model.encode(list(X))
        return np.array(embeddings)

    @classmethod
    def construct(cls: Type["AlbertSmallEmbeddingPipeline"], dataset: Dataset) -> "AlbertSmallEmbeddingPipeline":
        # if dataset.X_train.ndim > 1:
        #     raise ValueError("The provided dataset features must have either 0 or 1 dimensions.")

        model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")
        embedding_transform = functools.partial(cls._embedding_transform, model=model)

        ops = [("embedding", FunctionTransformer(embedding_transform))]
        return cls(ops)


class FlattenPipeline(sklearn.pipeline.Pipeline):
    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return np.reshape(X, newshape=(X.shape[0], -1))

    def __init__(self):
        super().__init__([("flatten", FunctionTransformer(FlattenPipeline._flatten))])

import numpy as np
import torch

from numpy.typing import NDArray
from torchvision.transforms import v2 as transforms
from typing import Optional, Sequence, Union


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
OPENAI_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class TorchImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: NDArray,
        y: Optional[NDArray],
        transform: Optional[transforms.Transform] = None,
        device: Optional[str] = None,
    ):
        if X.ndim == 3:
            X = np.expand_dims(X, axis=X.ndim)  # If the image is grayscale, add a channel dimension.
        if X.shape[-1] == 1:
            X = np.tile(X, (1, 1, 1, 3))  # If the image is grayscale, replicate the channel dimension.
        if X.shape[-1] == 3:
            X = X.transpose((0, 3, 1, 2))  # If the channel is the last dimension, move it to the second dimension.
        if X.dtype != np.float32:
            X = X.astype(np.float32) / 255.0  # If the image is not in floating point format, convert it to float.
        self.X = X
        self.y = y
        self.transform = transform
        self.device = device

    def __getitem__(self, idx: Union[int, Sequence[int]]):
        X_items = torch.tensor(self.X[idx])
        y_items = None if self.y is None else torch.tensor(self.y[idx])

        if self.device is not None:
            X_items = X_items.to(self.device)
            y_items = None if y_items is None else y_items.to(self.device)

        if self.transform is not None:
            X_items = self.transform(X_items)

        if y_items is None:
            return X_items
        else:
            return {"pixel_values": X_items, "labels": y_items}

    def __getitems__(self, idx: Sequence[int]):
        if self.y is None:
            return list(self.__getitem__(idx))
        else:
            result = self.__getitem__(idx)
            return {"pixel_values": list(result["pixel_values"]), "labels": list(result["labels"])}

    def __len__(self):
        return self.X.shape[0]

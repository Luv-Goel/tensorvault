"""Dataset and DataLoader utilities."""

import numpy as np
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union


class Dataset:
    """Abstract dataset class."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample is a tuple of the corresponding elements from each tensor.
    """

    def __init__(self, *tensors: np.ndarray):
        assert len(tensors) > 0, "At least one tensor required"
        n = len(tensors[0])
        for t in tensors:
            assert len(t) == n, "All tensors must have same first dimension"
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        return tuple(t[idx] for t in self.tensors)


class DataLoader:
    """Iterates over a Dataset in batches.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffle indices each epoch.
        drop_last: If True, drop the last incomplete batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, ...]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        batch: List[Any] = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []

        if batch and not self.drop_last:
            yield self._collate(batch)

    def _collate(self, batch: List[Tuple]) -> Tuple[np.ndarray, ...]:
        """Collate list of samples into batched tensors."""
        n_tensors = len(batch[0])
        collated = []
        for i in range(n_tensors):
            arr = np.stack([item[i] for item in batch])
            collated.append(arr)
        return tuple(collated)

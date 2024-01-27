from typing import List

import torch.nn as nn
from torch.utils.data import Dataset


class Mapping:
    def __init__(self, values: List[str]):
        self.idx_to_value = values
        self.value_to_idx = {value: idx for idx, value in enumerate(values)}

    def __len__(self) -> int:
        return len(self.idx_to_value)

    def __getitem__(self, input: int | str) -> int | str:
        if isinstance(input, int):
            if input >= len(self):
                raise IndexError(f"Index {input} out of range")
            return self.idx_to_value[input]
        elif isinstance(input, str):
            if input not in self.value_to_idx:
                raise KeyError(f"Key {input} not found")
            return self.value_to_idx[input]
        else:
            raise TypeError("Input must be int or str")


class ApplyTransform(Dataset):
    """Apply transform to dataset"""

    def __init__(self, dataset: Dataset, transform: nn.Module):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        image, target = self.dataset[idx]
        image, target = self.transform(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.dataset)

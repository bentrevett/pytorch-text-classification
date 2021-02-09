import torch
from typing import Callable, List, Union


def sequential_transforms(*transforms: List[Callable]) -> Callable:
    def f(x):
        for transform in transforms:
            x = transform(x)
        return x
    return f


def to_longtensor(x: Union[int, List[int]]) -> torch.LongTensor:
    return torch.tensor(x).long()

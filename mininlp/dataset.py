import datasets
import torch
from typing import Callable, Optional, Tuple


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.arrow_dataset.Dataset, text_transforms: Optional[Callable] = None,
                 label_transforms: Optional[Callable] = None):
        self.data = data
        self.text_transforms = text_transforms
        self.label_transforms = label_transforms

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        text = self.data[i]['text']
        label = self.data[i]['label']
        text = text if self.text_transforms is None else self.text_transforms(text)
        label = label if self.label_transforms is None else self.label_transforms(label)
        return text, label

    def __len__(self) -> int:
        return len(self.data)

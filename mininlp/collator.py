"""Collator class."""

import torch
import torch.nn as nn
from typing import List, Tuple


class TextClassificationCollator:
    """
    Class to handle a collator, a function that converts batches of text and labels to tensors.

    A collator must be created with a `pad_idx`, which defines how the tokens should be padded.
    """

    def __init__(self, pad_idx: int, batch_first: bool = False):
        """Initialize the Collator object."""
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def collate(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.LongTensor,
                                                                                       torch.LongTensor]:
        """Collate a batch of text and labels by padding and converting to torch.LongTensors."""
        text, labels = zip(*batch)

        text = nn.utils.rnn.pad_sequence(text,
                                         padding_value=self.pad_idx,
                                         batch_first=self.batch_first)

        labels = torch.LongTensor(labels)

        return text, labels

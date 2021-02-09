"""Vocab class."""

import collections
import datasets
from .tokenizer import Tokenizer
from typing import Dict, List, Optional, Tuple


class Vocab:
    """
    Class to handle a vocabulary, a mapping between strings and a their corresponding integer values.

    Vocabulary must be created with a counter where each key is a token and each value is the number
    of times that tokens appears in the training dataset.
    """

    def __init__(self, counter: collections.Counter, min_freq: int = 1, max_size: Optional[int] = 30_000,
                 unk_token: Optional[str] = '<unk>', pad_token: Optional[str] = '<pad>',
                 special_tokens: List[str] = []):
        """Initialize the Vocab object and builds the vocabulary mappings."""
        assert min_freq >= 1, 'min_freq must be >= 1'
        assert len(special_tokens) == len(set(special_tokens)), 'special_tokens must be unique'
        assert all([(isinstance(s, str) for s in special_tokens)]), 'special_tokens must be List[str]'
        assert unk_token not in special_tokens, 'unk_token should not be in special_tokens'
        assert pad_token not in special_tokens, 'pad_token should not be in special_tokens'

        self.min_freq = min_freq
        self.max_size = max_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

        self._stoi, self._itos = self._create_vocab(counter, min_freq, max_size, unk_token, pad_token, special_tokens)

    def __len__(self):
        """Allow us to do len(Vocab) to get the length of the vocabulary."""
        return len(self._itos)

    def _create_vocab(self, counter: collections.Counter, min_freq: int, max_size: Optional[int],
                      unk_token: Optional[str], pad_token: Optional[str], special_tokens: List) -> Tuple[Dict[str, int],
                                                                                                         List[str]]:
        """
        Handle the actual vocabulary creation.

        Tokens that appear less than min_freq times are ignored
        Once the vocabulary reaches max size, no more tokens are added
        `unk_token` is the token used to replace tokens not in the vocabulary
        `pad_token` is used to pad sequences
        `special_tokens` are other tokens we want appended to the start of our vocabulary, i.e. start of sequence tokens
        """
        stoi = dict()

        if unk_token is not None:
            stoi[unk_token] = len(stoi)
        if pad_token is not None:
            stoi[pad_token] = len(stoi)
        for special_token in special_tokens:
            stoi[special_token] = len(stoi)

        max_size = max_size - len(stoi)

        for token, count in counter.most_common(max_size):
            if count >= min_freq:
                if token not in stoi:
                    stoi[token] = len(stoi)
            else:
                break

        assert len(stoi) > 0, 'Created vocabulary is empty!'

        itos = list(stoi.keys())

        return stoi, itos

    def stoi(self, token: str) -> int:
        """
        Convert a token (str) into its corresponding integer value from the vocabulary.

        If the token is not in the vocabulary, returns the integer value of the unk_token
        If unk_token is set to None, throws an error
        """
        if isinstance(token, list):
            return [self.stoi(t) for t in token]
        assert isinstance(token, str), f'Input to vocab.stoi should be str, got {type(token)}'

        if token in self._stoi:
            return self._stoi[token]
        else:
            assert self.unk_token is not None, f'token {token} is not in the vocab and unk_token = None!'
            return self._stoi[self.unk_token]

    def itos(self, integer: int) -> str:
        """
        Convert an integer into its corresponding token (str) from the vocabulary.

        If the integer value is outside of the vocabulary range, throws an error.
        """
        if isinstance(integer, list):
            return [self.itos(i) for i in integer]

        assert isinstance(integer, int), f'Input to vocab.itos should be an integer, got {type(integer)}'
        assert integer >= 0, f'Input to vocab.itos should be a non-negative, got {integer}'
        assert integer < len(self._itos), f'Input integer out of range, should be <{len(self._itos)}, got {integer}'

        return self._itos[integer]


def build_vocab_counter(data: datasets.arrow_dataset.Dataset, field: str, tokenizer: Tokenizer) -> collections.Counter:
    counter = collections.Counter()
    for example in data:
        text = example[field]
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    return counter

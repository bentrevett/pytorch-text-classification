"""Tokenizer class."""

from typing import List, Optional


class Tokenizer:
    """
    Class to handle a tokenizer, a function that converts strings to lists of strings.

    A tokenizer must be created with a `tokenize_fn`, which defines how the string should be converted to a list of
    strings.
    """

    def __init__(self, tokenize_fn: callable, lower: bool = False, max_length: int = 1_000_000,
                 sos_token: Optional[str] = None, eos_token: Optional[str] = None):
        """Initialize the Tokenzer object."""
        assert callable(tokenize_fn)

        self.tokenize_fn = tokenize_fn
        self.lower = lower
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length - sum([1 for t in [sos_token, eos_token] if t is not None])

    def tokenize(self, s: str) -> List[str]:
        """Tokenize a string."""
        assert isinstance(s, str), f'input to tokenize should be str, got {type(s)}'

        tokens = self.tokenize_fn(s)

        assert isinstance(tokens, list), f'`tokenize_fn` should return List, got {type(tokens)}'
        assert all([isinstance(token, str) for token in tokens]), '`tokenize_fn` should return List[str]'

        if self.lower:
            tokens = [token.lower() for token in tokens]

        tokens = tokens[:self.max_length]

        if self.sos_token is not None:
            tokens = [self.sos_token] + tokens

        if self.eos_token is not None:
            tokens = tokens + [self.eos_token]

        return tokens

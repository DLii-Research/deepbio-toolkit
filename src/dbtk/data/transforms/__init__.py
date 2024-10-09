from itertools import chain, repeat
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union

from ..vocabularies import Vocabulary
from ..._utils import export, static_vars

# Funtional ----------------------------------------------------------------------------------------

@export
def random_truncate(
    x,
    min_length: int,
    max_length: int,
    rng: np.random.Generator
) -> bytes:
    """
    Randomly truncate a DNA sequence string to the given min/max length.
    """
    length = rng.integers(min(min_length, len(x)), min(max_length, len(x)) + 1)
    try:
        start = rng.integers(0, len(x) - length + 1)
    except ValueError as e:
        print(len(x), min_length, max_length, length)
        raise e
    return x[start:start+length]


@export
@static_vars(translation=bytes.maketrans(b"ACGT", b"TGCA"))
def reverse_complement(sequence: bytes) -> bytes:
    """
    Randomly reverse complement a DNA sequence string.
    """
    return sequence.translate(reverse_complement.translation)[::-1]


# Object Oriented ---------------------------------------------------------------------------------

@export
class Compose:
    """
    Compose a series of transforms.
    """
    def __init__(self, transforms: Union[List[Callable], Dict[Any, List[Callable]]], collate_fn: Optional[Callable] = None):
        self.transforms = transforms
        self.collate_fn = collate_fn

    def transform(self, x, transforms):
        for transform in transforms:
            x = transform(x)
        return x

    def __call__(self, x):
        if not isinstance(self.transforms, dict):
            result = self.transform(x, self.transforms)
        elif isinstance(x, dict):
            result = {
                key: (self.transform(value, self.transforms[key]) if key in self.transforms else value)
                for key, value in x.items()
            }
        else:
            result = type(x)((
                self.transform(value, self.transforms[key]) if key in self.transforms else value
                for key, value in enumerate(x)
            ))
        if self.collate_fn is not None:
            result = self.collate_fn(result)
        return result


@export
class Map:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        if isinstance(x, dict):
            return {key: self.transform(value) for key, value in x.items()}
        return tuple(map(self.transform, x))


@export
class Repeat:
    """
    Perform the same transformation multiple times to the same input.
    """
    def __init__(self, n, transform: Callable):
        self.n = n
        self.transform = transform

    def __call__(self, x):
        return tuple(map(self.transform, repeat(x, self.n)))


@export
class RandomReverseComplement:
    """
    Randomly reverse complement a DNA sequence string.
    """
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def __call__(self, sequence):
        if self.rng.random() < 0.5:
            return sequence
        return reverse_complement(sequence)


@export
class Truncate:
    def __init__(self, length: int):
        self.length = length

    def __call__(self, x):
        return x[:self.length]


@export
class RandomTruncate:
    """
    Randomly truncate a DNA sequence string to the given min/max length.
    """
    def __init__(
        self,
        min_length: int,
        max_length: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.rng = rng or np.random.default_rng()

    def __call__(self, sequence):
        max_length = self.max_length if self.max_length is not None else len(sequence)
        return random_truncate(sequence, self.min_length, max_length, self.rng)


@export
class Pad:
    def __init__(self, length: int, value: Any):
        self.length = length
        self.value = value

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return F.pad(x, (0, self.length - x.shape[-1]), value=self.value)
        return type(x)(chain(x, repeat(self.value, self.length - len(x) - 1)))


@export
class RandomTokenMask:
    def __init__(
        self,
        ratio: float,
        contiguous: bool = False,
        rng: Optional[np.random.Generator] = None
    ):
        self.ratio = ratio
        self.contiguous = contiguous
        self.rng = rng or np.random.default_rng()
        self._indices = self._contiguous_indices if self.contiguous else self._sparse_indices

    def _contiguous_indices(self, length, mask_length):
        offset = self.rng.integers(length - mask_length + 1)
        return np.arange(offset, offset + mask_length)

    def _sparse_indices(self, length, mask_length):
        return self.rng.choice(length, mask_length, replace=False)

    def __call__(self, tokens):
        tokens = list(tokens)
        mask_length = int(self.ratio*len(tokens))
        masked = [None]*mask_length
        for i, j in enumerate(self._indices(len(tokens), mask_length)):
            masked[i] = tokens[j]
            tokens[j] = "[MASK]"
        return {
            "tokens": tokens,
            "masked_tokens": masked
        }

@export
class Tokenize():
    def __init__(self, tokenizer: Callable):
        self.tokenizer = tokenizer

    def __call__(self, string):
        return tuple(self.tokenizer(string))


@export
class ToTokenIds():
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, tokens):
        return tuple(self.vocabulary(tokens))


@export
class ToTensor():
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype)

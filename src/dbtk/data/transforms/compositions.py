import numpy as np
from typing import Callable, Optional
from . import Compose
from ..vocabularies import DnaVocabulary, Vocabulary
from ..._utils import export

@export
class RandomDnaSampleTransform(Compose):
    def __init__(
        self,
        min_length: Optional[int] = 1,
        max_length: Optional[int] = None,
        kmer: int = 1,
        kmer_stride: int = 1,
        vocabulary: Optional[Vocabulary] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length if max_length is not None else min_length
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.vocabulary = vocabulary if vocabulary is not None else DnaVocabulary(self.kmer)

        super().__init__(self.build())

    def build(self):
        from ..tokenizers import DnaTokenizer
        from . import Map, Pad, RandomReverseComplement, RandomGroupedTruncate, Tokenize, ToTokenIds, ToTensor
        transforms = [
            RandomReverseComplement(),
            Tokenize(DnaTokenizer(self.kmer, self.kmer_stride)),
            ToTokenIds(self.vocabulary),
            ToTensor(),
            Pad(
                int((self.max_length - self.kmer) // self.kmer_stride) + 1,
                self.vocabulary["[PAD]"]
            )
        ]
        return [
            RandomGroupedTruncate(self.min_length, self.max_length),
            Map(Compose(transforms))
        ]


@export
class RandomDnaSequenceTransform(Compose):
    def __init__(
        self,
        min_length: Optional[int] = 1,
        max_length: Optional[int] = None,
        kmer: int = 1,
        kmer_stride: int = 1,
        vocabulary: Optional[Vocabulary] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.vocabulary = vocabulary if vocabulary is not None else DnaVocabulary(self.kmer)

        super().__init__(self.build())

    def build(self):
        from ..tokenizers import DnaTokenizer
        from . import Pad, RandomReverseComplement, RandomTruncate, Tokenize, ToTokenIds, ToTensor
        transforms = [
            RandomTruncate(self.min_length, self.max_length),
            RandomReverseComplement(),
            Tokenize(DnaTokenizer(self.kmer, self.kmer_stride)),
            ToTokenIds(self.vocabulary),
            ToTensor()
        ]
        if self.max_length is not None:
            transforms.append(
                Pad(
                    int((self.max_length - self.kmer) // self.kmer_stride) + 1,
                    self.vocabulary["[PAD]"]
                )
            )
        return transforms

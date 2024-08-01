from . import bert
from .. import layers
from ...data import tokenizers, vocabularies
from ..._utils import export

@export
class DnaBertModel(bert.BertModel):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        kmer: int = 1,
        kmer_stride: int = 1
    ):
        super().__init__(
            transformer_encoder,
            tokenizer=tokenizers.DnaTokenizer(kmer, kmer_stride),
            vocabulary=bert.BertVocabulary(vocabularies.dna(kmer))
        )
        self.kmer = kmer
        self.kmer_stride = kmer_stride

@export
class DnaBertPretrainingModel(bert.BertPretrainingModel):
    ...

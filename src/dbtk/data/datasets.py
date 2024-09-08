from dnadb.fasta import FastaDb, FastaMappingDb
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional, Sequence, Union

from .interfaces import FastaFile

class SequenceDataset(Dataset):
    def __init__(self, sequences: Union[FastaFile, FastaDb, Path, str], transform: Optional[Callable] = None):
        super().__init__()
        if isinstance(sequences, (str, Path)):
            if Path(sequences).is_dir():
                sequences = FastaDb(sequences)
            else:
                sequences = FastaFile.open(sequences)
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        entry = self.sequences[index]
        if self.transform is not None:
            entry = self.transform(entry)
        return entry


class AmpliconSampleDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[FastaMappingDb],
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.samples = list(samples)
        self.transform = transform

    def __getitem__(self, index):
        x = self.samples[index]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.samples)

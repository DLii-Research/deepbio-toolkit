from dataclasses import dataclass
import io
import mmap
from pathlib import Path
import re
from typing import Union
from typing_extensions import Buffer

from ..._utils import export

@export
class Fasta:
    """
    An indexable memory-mapped interface for FASTA files.
    """
    @dataclass
    class Entry:
        _fasta_file: "Fasta"
        _id_start: int
        _id_end: int
        _sequence_start: int
        _sequence_end: int

        @property
        def id(self) -> str:
            return self._fasta_file._read(self._id_start, self._id_end)

        @property
        def metadata(self) -> str:
            return self._fasta_file._read(self._id_end+1, self._sequence_start-1)

        @property
        def sequence(self) -> str:
            return self._fasta_file._read(self._sequence_start, self._sequence_end)

        def __len__(self) -> int:
            return len(self.sequence)

        def __str__(self) -> str:
            return ">" + self.id + " " + self.metadata + '\n' + self.sequence

        def __repr__(self) -> str:
            return "Entry:\n" + str(self)

    @classmethod
    def open(cls, path: Union[Path, str]):
        with open(path, 'r+') as f:
            return cls(mmap.mmap(f.fileno(), 0))

    def __init__(self, data: Union[str, mmap.mmap]):
        self.data = data
        self.entries = []
        self.id_map = {}
        # Lazy reading
        self._length = None
        if isinstance(self.data, str):
            pattern = re.compile(r">[^>]+")
            self._read = lambda start, end: self.data[start:end]
        else:
            pattern = re.compile(br">[^>]+")
            self._read = lambda start, end: self.data[start:end].decode() # type: ignore
        self._reader = re.finditer(pattern, self.data) # type: ignore
        self._eof = False

    def __iter__(self):
        yield from self.entries
        while self._read_next_entry():
            yield self.entries[-1]

    def __getitem__(self, key):
        if not isinstance(key, int):
            while key not in self.id_map and self._read_next_entry():
                continue
            key = self.id_map[key]
        else:
            while len(self.entries) <= key and self._read_next_entry():
                continue
        return self.entries[key]

    def __len__(self):
        if self._length is None:
            pattern = ">" if isinstance(self.data, str) else b">"
            self._length = sum(1 for _ in re.finditer(pattern, self.data))
            if self._length == len(self.entries):
                self._clean_lazy_loading()
        return self._length

    def _read(self, start: int, end: int):
        return self.data[start:end].decode()

    def _read_next_entry(self):
        try:
            match = next(self._reader)
            group = match.group()
            header_end = group.find('\n')
            sequence_id_length = ((group.find(' ') + 1) or (header_end + 1)) - 1
            sequence_id_start = match.start() + 1
            sequence_id_end = match.start() + sequence_id_length
            sequence_start = match.start() + header_end + 1
            sequence_end = match.end()
            if group.endswith('\n'):
                sequence_end -= 1
            self.entries.append(self.Entry(self, sequence_id_start, sequence_id_end, sequence_start, sequence_end))
            self.id_map[group[1:sequence_id_length]] = len(self.id_map)
        except StopIteration:
            self._length = len(self.entries)
        if not self._eof and self._length == len(self.entries):
            self._eof = True
            self._clean_lazy_loading()
        return not self._eof

    def _clean_lazy_loading(self):
        self.__getitem__ = lambda k: self.entries[self.id_map[k] if isinstance(k, str) else k]

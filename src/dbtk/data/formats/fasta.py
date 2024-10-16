from dataclasses import dataclass
import mmap
from pathlib import Path
import re
from typing import Generator, Optional, Union

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
            return self._fasta_file.data[self._id_start:self._id_end].decode()

        @property
        def metadata(self) -> str:
            return self._fasta_file.data[self._id_end+1:self._sequence_start-1].decode()

        @property
        def sequence(self) -> str:
            return self._fasta_file.data[self._sequence_start:self._sequence_end].decode()

        def __len__(self) -> int:
            return len(self.sequence)

        def __str__(self) -> str:
            return ">" + self.id + " " + self.metadata + '\n' + self.sequence

        def __repr__(self) -> str:
            return "Entry:\n" + str(self)

    def __init__(self, path: Union[Path, str], madvise: Optional[int] = None):
        with open(path, "r+") as f:
            self.data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if madvise is not None:
            self.data.madvise(madvise)
        self.entries = []
        self.id_map = {}
        # Lazy reading
        self._length = None
        self._reader = re.finditer(br">(\S+).*\n(\S+)", self.data)
        self._eof = False
        self._closed = False

    def __iter__(self) -> Generator[Entry, None, None]:
        yield from self.entries
        while self._read_next_entry():
            yield self.entries[-1]

    def __getitem__(self, key) -> Entry:
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
            self._length = sum(1 for _ in re.finditer(b">", self.data))
            if self._length == len(self.entries):
                self._clean_lazy_loading()
        return self._length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if isinstance(self.data, str):
            return
        self._reader = None
        self.data.close()
        self._closed = True

    def _read_next_entry(self):
        try:
            match = next(self._reader) # type: ignore
            self.id_map[match.group(1).decode()] = len(self.id_map)
            self.entries.append(
                self.Entry(
                    self,
                    *match.span(1), # (seq id start, seq id end)
                    *match.span(2)  # (seq start, seq end)
                )
            )
        except StopIteration:
            self._length = len(self.entries)
        except TypeError as exception:
            if self._closed:
                exception = Exception("Cannot read from closed file")
            raise exception
        if not self._eof and self._length == len(self.entries):
            self._eof = True
            self._clean_lazy_loading()
        return not self._eof

    def _clean_lazy_loading(self):
        self.__getitem__ = lambda k: self.entries[self.id_map[k] if isinstance(k, str) else k] # type: ignore

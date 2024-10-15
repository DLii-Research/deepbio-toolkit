import abc
from bitarray import bitarray, decodetree
from bitarray.util import (
    huffman_code,
    serialize as bitarray_serialize,
    deserialize as bitarray_deserialize
)
from collections import Counter
from dataclasses import dataclass
import deflate
import io
import mmap
import numpy as np
from pathlib import Path
import pickle
from typing import Iterable, Literal, Optional, Union

from ..._utils import export

class Compression(abc.ABC):

    IDENTIFIER = np.uint8(0)

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data: str) -> "Compression":
        return NotImplemented

    @abc.abstractmethod
    def compress(self, data: str) -> bytes:
        return NotImplemented

    @abc.abstractmethod
    def decompress(self, data: bytes) -> str:
        return NotImplemented

    def serialize(self) -> bytes:
        return b""

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: bytes) -> "Compression":
        return NotImplemented

class NoCompression(Compression):

    IDENTIFIER = np.uint8(0)

    # Reference encode and decode directly to remove overhead
    compress = str.encode # type: ignore
    decompress = bytes.decode # type: ignore

    @classmethod
    def from_data(cls, data: str) -> "NoCompression":
        return cls()

    @classmethod
    def deserialize(cls, data: bytes) -> "NoCompression":
        return cls()

class HuffmanCompression(Compression):

    IDENTIFIER = np.uint8(1)

    @classmethod
    def from_data(cls, data: str) -> "HuffmanCompression":
        return cls(huffman_code(Counter(data))) # type: ignore

    def __init__(self, tree: dict):
        self.tree = tree
        self.decode_tree = decodetree(tree)

    def compress(self, data: str) -> bytes:
        encoded = bitarray()
        encoded.encode(self.tree, data)
        serialized = bitarray_serialize(encoded)
        return np.uint16(len(serialized)).tobytes() + serialized

    def decompress(self, data: bytes) -> str:
        length = np.uint16(data[:2])
        return "".join(bitarray_deserialize(data[2:length+2]).decode(self.decode_tree))

    def serialize(self) -> bytes:
        return pickle.dumps(self.tree)

    @classmethod
    def deserialize(cls, data: bytes) -> "HuffmanCompression":
        return cls(pickle.loads(data))

class DeflateCompression(Compression):

    IDENTIFIER = np.uint8(2)

    @classmethod
    def from_data(cls, data: str) -> "DeflateCompression":
        return cls()

    def compress(self, data: str) -> bytes:
        return np.uint16(len(data)).tobytes() + deflate.deflate_compress(data.encode())

    def decompress(self, data: bytes) -> str:
        length = np.frombuffer(data[:2], dtype=np.uint16)[0]
        return deflate.deflate_decompress(data[2:], length).decode()

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, data: bytes = b"") -> "DeflateCompression":
        return cls()

@export
class SequenceStore:

    VERSION = 0x00000001

    @dataclass
    class Entry:
        __slots__ = ["_store", "index"]
        _store: "SequenceStore"
        index: int

        @property
        def id(self) -> str:
            return self._store.id(self.index)

        @property
        def metadata(self) -> str:
            return self._store.metadata(self.index)

        @property
        def sequence(self) -> str:
            return self._store.sequence(self.index)

        def __len__(self) -> int:
            return len(self.sequence)

        def __str__(self) -> str:
            header = ""
            if self.id:
                header += f">{self.id}"
            if self.metadata:
                header += f" {self.metadata}"
            return header.strip() + "\n" + self.sequence

        def __repr__(self) -> str:
            return str(self)

    class Writer:
        """
        Write a sequence store to the given file or buffer.
        """
        def __init__(
            self,
            path_or_handle: Union[str, Path, io.FileIO],
            sequence_compression: Optional[Literal["deflate", "huffman"]] = "deflate",
            id_compression: Optional[Literal["deflate", "huffman"]] = "huffman",
            metadata_compression: Optional[Literal["deflate", "huffman"]] = "deflate",
            progress: Optional[bool] = False
        ):
            if isinstance(path_or_handle, (str, Path)):
                self.handle = open(path_or_handle, "wb")
            else:
                self.handle = path_or_handle
            self.sequence_compression = sequence_compression
            self.id_compression = id_compression
            self.metadata_compression = metadata_compression
            self.progress = progress
            self.sequences = []
            self.ids = []
            self.metadata = {}

        def write(self, sequence: str, id: Optional[str], metadata: Optional[str]):
            self.sequences.append(sequence)
            if id is not None:
                self.ids.append(id)
            elif len(self.ids) > 0:
                raise ValueError("ID must be provided for all sequences")
            if metadata is not None:
                self.metadata[len(self.ids) - 1] = metadata.strip()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

        def _compressor(self, compression: Optional[Literal["deflate", "huffman"]], data: Iterable[str]) -> Compression:
            if compression == "deflate":
                return DeflateCompression()
            elif compression == "huffman":
                return HuffmanCompression.from_data("".join(data))
            elif compression is not None:
                raise ValueError(f"Unknown compression type: {compression}")
            return NoCompression()

        def _compress(self, data, compressor: Compression):
            # In-place compression
            for i, d in (data.items() if isinstance(data, dict) else enumerate(data)):
                data[i] = compressor.compress(d)
            return data

        def _serialize_compressor(self, compressor: Compression) -> bytes:
            compressor_bytes = compressor.serialize()
            result = b""
            result += compressor.IDENTIFIER.tobytes()            # sequence compression type
            result += np.uint64(len(compressor_bytes)).tobytes() # sequence compression length
            result += compressor_bytes
            return result

        def close(self):
            # Compress DNA sequences
            sequences = self.sequences
            sequence_compressor = self._compressor(self.sequence_compression, sequences)
            sequences = self._compress(sequences, sequence_compressor)
            sequence_block_size = max(map(len, sequences))

            # Compress identifiers
            ids = self.ids
            id_compressor = self._compressor(self.id_compression, ids)
            ids = self._compress(ids, id_compressor)
            id_block_size = max(map(len, ids))

            # Compress metadata
            metadata = self.metadata
            metadata_compressor = self._compressor(self.metadata_compression, metadata.values())
            metadata = self._compress(metadata, metadata_compressor)
            metadata_length = sum(map(len, metadata.values()))
            metadata_index_bits = int(np.ceil(np.log2(metadata_length))) if metadata_length > 0 else 0

            # Sequence information
            header = b""
            header += np.uint32(SequenceStore.VERSION).tobytes()          # version number
            header += np.uint32(len(sequences)).tobytes()                 # number of sequences
            header += np.uint32(sequence_block_size).tobytes()            # sequence block size
            header += np.uint32(id_block_size).tobytes()                  # identifier block size
            header += np.uint8(metadata_index_bits).tobytes()             # number of bits for metadata index

            # Compression Information
            header += self._serialize_compressor(sequence_compressor)
            if len(ids) > 0:
                header += self._serialize_compressor(id_compressor)
            if len(metadata) > 0:
                header += self._serialize_compressor(metadata_compressor)

            # Write header
            self.handle.write(header)

            # Write sequences
            for sequence in sequences:
                self.handle.write(sequence)

            # Write identifiers
            for id in ids:
                self.handle.write(id)

            # Write metadata
            metadata_index_dtype = getattr(np, f"uint{metadata_index_bits}")
            metadata_ends = np.cumsum(
                [len(metadata[id]) if id in metadata else 0 for id in ids],
                dtype=metadata_index_dtype
            )
            self.handle.write(metadata_ends.tobytes())
            for value in metadata.values():
                self.handle.write(value)

            self.handle.close()

    def __init__(self, path: Union[str, Path], madvise: Optional[int] = None):
        with open(path, "r+") as f:
            self.data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if madvise is not None:
            self.data.madvise(madvise)
        self.version, self.length, self.sequence_block_size, self.id_block_size, self.metadata_index_bits \
            = np.frombuffer(self.data, count=5, dtype=np.uint32)

    def sequence(self, key):
        if isinstance(key)

    def __getitem__(self, index: int) -> Entry:
        return self.Entry(self, index)

    def __len__(self) -> int:
        return self.length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.data.close()

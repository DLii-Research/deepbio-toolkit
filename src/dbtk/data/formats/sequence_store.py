import abc
from bitarray import bitarray, decodetree
from bitarray.util import (
    huffman_code,
    serialize as bitarray_serialize,
    deserialize as bitarray_deserialize
)
from collections import Counter
import deflate
import mmap
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import cast, Iterable, List, Optional, Type, Union

from ..._utils import export

class Compression(abc.ABC):

    IDENTIFIER = np.uint8(0)

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data: List[str]) -> "Compression":
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

@export
class SequenceStore:

    VERSION = 0x00000001


    class NoCompression(Compression):

        IDENTIFIER = np.uint8(0)

        # Reference encode and decode directly to remove overhead

        @classmethod
        def from_data(cls, data: List[str]) -> "SequenceStore.NoCompression":
            return cls()

        def compress(self, data: str) -> bytes:
            return data.encode()

        def decompress(self, data: bytes) -> str:
            return data.decode()

        @classmethod
        def deserialize(cls, data: bytes) -> "SequenceStore.NoCompression":
            return cls()

    class HuffmanCompression(Compression):

        IDENTIFIER = np.uint8(1)

        @classmethod
        def from_data(cls, data: List[str]) -> "SequenceStore.HuffmanCompression":
            return cls(huffman_code(Counter("".join(data)))) # type: ignore

        def __init__(self, tree: dict):
            self.tree = tree
            self.decode_tree = decodetree(tree)

        def compress(self, data: str) -> bytes:
            encoded = bitarray()
            encoded.encode(self.tree, data)
            serialized = bitarray_serialize(encoded)
            compressed = np.uint16(len(serialized)).tobytes() + serialized
            return compressed

        def decompress(self, data: bytes) -> str:
            length = np.frombuffer(data, count=1, dtype=np.uint16)[0]
            return "".join(bitarray_deserialize(data[2:length+2]).decode(self.decode_tree))

        def serialize(self) -> bytes:
            return pickle.dumps(self.tree)

        @classmethod
        def deserialize(cls, data: bytes) -> "SequenceStore.HuffmanCompression":
            return cls(pickle.loads(data))

    class DeflateCompression(Compression):

        IDENTIFIER = np.uint8(2)

        @classmethod
        def from_data(cls, data: List[str]) -> "SequenceStore.DeflateCompression":
            return cls()

        def compress(self, data: str) -> bytes:
            return np.uint16(len(data)).tobytes() + deflate.deflate_compress(data.encode())

        def decompress(self, data: bytes) -> str:
            length = np.frombuffer(data[:2], dtype=np.uint16)[0]
            return deflate.deflate_decompress(data[2:], length).decode()

        def serialize(self) -> bytes:
            return b""

        @classmethod
        def deserialize(cls, data: bytes = b"") -> "SequenceStore.DeflateCompression":
            return cls()

    @staticmethod
    def create(
        sequences: Iterable[str],
        path: Union[str, Path],
        compression: Optional[Type[Compression]] = DeflateCompression,
        show_progress: Optional[bool] = False
    ):
        """
        Write a sequence store to the given file.
        """
        sequences = list(sequences)
        # Instantiate compressor
        if compression is None:
            compression = SequenceStore.NoCompression
        compressor = compression.from_data(sequences)

        # Compress DNA sequences
        compressed_sequences = cast(List[bytes], sequences)
        if show_progress:
            label = "Encoding" if isinstance(compressor, SequenceStore.NoCompression) else "Compressing"
            it = tqdm(sequences, total=len(sequences), desc=f"{label} sequences")
        else:
            it = sequences
        for i, s in enumerate(it):
            compressed_sequences[i] = compressor.compress(s)
        sequence_block_size = max(map(len, compressed_sequences))

        # Sequence information
        header = b""
        header += np.uint32(SequenceStore.VERSION).tobytes()     # version number
        header += np.uint32(len(compressed_sequences)).tobytes() # number of sequences
        header += np.uint32(sequence_block_size).tobytes()       # sequence block size

        compressor_bytes = compressor.serialize()
        header += compressor.IDENTIFIER.tobytes()            # sequence compression type
        header += np.uint64(len(compressor_bytes)).tobytes() # sequence compressor bytes length
        header += compressor_bytes                           # sequence compressor data

        with open(path, "wb") as handle:
            handle.write(header)
            if show_progress:
                compressed_sequences = tqdm(compressed_sequences, desc="Writing sequences")
            for sequence in compressed_sequences:
                handle.write(sequence + b"\x00"*(sequence_block_size - len(sequence)))

    def __init__(self, path: Union[str, Path], madvise: Optional[int] = None):
        with open(path, "r+") as f:
            self.data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if madvise is not None:
            self.data.madvise(madvise)
        self._load()

    def _load(self):
        self._offset=0
        self.version, self.length, self.sequence_block_size = np.frombuffer(
            self.data,
            count=3,
            offset=self._offset,
            dtype=np.uint32
        )
        self._offset += 3*4
        compressor_id = np.frombuffer(self.data, count=1, offset=self._offset, dtype=np.uint8)[0]
        self._offset += 1*1
        compressor_length = int(np.frombuffer(self.data, count=1, offset=self._offset, dtype=np.uint64)[0])
        self._offset += 1*8
        compressor_bytes = self.data[self._offset:self._offset+compressor_length]
        self.compressor = {
            compressor.IDENTIFIER: compressor
            for compressor in [
                SequenceStore.NoCompression,
                SequenceStore.HuffmanCompression,
                SequenceStore.DeflateCompression
            ]
        }[compressor_id].deserialize(compressor_bytes)
        self._offset += compressor_length

    def sequence(self, index: int) -> str:
        offset = self._offset + index*self.sequence_block_size
        return self.compressor.decompress(self.data[offset:offset+self.sequence_block_size])

    def __getitem__(self, index: int) -> str:
        return self.sequence(index)

    def __len__(self) -> int:
        return self.length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.data.close()

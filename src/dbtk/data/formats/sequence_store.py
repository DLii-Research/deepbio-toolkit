"""
The SequenceStore is a custom binary format for efficiently storing sequence data with fast read
access.

Mode 1: sequence store without samples/abundance information
Mode 2: sequence store with samples/abundance information

- Sequence IDs are optional. If none are provided, they will not be stored and will be generated
automatically upon access based on the sample name information. If provided, they must be unique
and must be present for all input sequences.

1. Write all sequences to the store.
"""
import bisect
from bitarray import bitarray, decodetree
from bitarray.util import deserialize, huffman_code, serialize
from collections import Counter
from itertools import chain
import mmap
import numpy as np
from pathlib import Path
import pickle
import shelve
from tqdm import tqdm
from typing import Dict, Iterator, List, Optional, overload, Tuple, Union
from ..._utils import export

def encode_string(string: str, huffman_codes: Dict[str, bitarray]) -> bytes:
    """
    Encode a string using huffman codes.

    Args:
        string (str): The string to encode
        huffman_codes (Dict[str, bitarray]): The huffman codes

    Returns:
        bytes: The encoded string
    """
    encoded = bitarray()
    encoded.encode(huffman_codes, string)
    return serialize(encoded) + b'\xff'

def decode_string(encoded: bytes, decode_tree: decodetree) -> str:
    """
    Decode a huffman encoded string.

    Args:
        encoded (bytes): The encoded string
        huffman_codes (Dict[str, bitarray]): The huffman codes

    Returns:
        str: The decoded string
    """
    decoded = deserialize(encoded[:-1])
    return "".join(decoded.decode(decode_tree))

def uint_size(n: int) -> int:
    """
    Calculate the number of bits required to store an unsigned integer.

    Args:
        n (int): The integer to store

    Returns:
        int: The unsigned integer size in bits
    """
    return min(64, max(int(np.ceil(np.log2(n))), 8))

def uint(bit_count: int) -> np.dtype:
    """
    Create a numpy dtype for an unsigned integer with a specific number of bits.

    Args:
        bit_count (int): The number of bits for the unsigned integer

    Returns:
        np.dtype: The numpy dtype for the unsigned integer
    """
    return getattr(np, f"uint{bit_count}")


class EncodedStringView:
    def __init__(self, view: np.ndarray, huffman_codes: dict[str, bitarray]):
        self.view = view
        self.huffman_codes = huffman_codes
        self.decode_tree = decodetree(huffman_codes)

    def __len__(self) -> int:
        return len(self.view)

    def __contains__(self, s: str) -> bool:
        encoded = encode_string(s, self.huffman_codes)
        return any(encoded == x for x in self.view)

    @overload
    def __getitem__(self, i: int) -> str: ...
    @overload
    def __getitem__(self, i: Union[slice, np.ndarray] = slice(None)) -> List[str]: ...
    def __getitem__(self, i: Union[int, slice, np.ndarray] = slice(None)) -> Union[str, List[str]]:
        if isinstance(i, int):
            return decode_string(self.view[i], self.decode_tree)
        return [decode_string(s, self.decode_tree) for s in self.view[i]]

    def __iter__(self) -> Iterator[str]:
        return (decode_string(s, self.decode_tree) for s in self.view)


@export
class SequenceStore:
    """
    A class for storing sequences in a custom binary format for efficient read access.
    """
    class SequenceWriter:
        def __init__(
            self,
            path: Union[str, Path],
            sequence_id_bucket_ratio: float = 0.1,
            show_progress: bool = False,
        ):
            """
            Create a new SequenceWriter.

            Args:
                path (Union[str, Path]): The path to the sequence store file
                sequence_id_bucket_ratio (float, optional): The bucket/key ratio for the sequence IDs hash table . Defaults to 0.1.
                show_progress (bool, optional): Show progress during writing. Defaults to False.
            """
            self.path = path
            self.sequence_id_bucket_ratio = sequence_id_bucket_ratio
            self.show_progress = show_progress
            # sequence -> sequence_id
            self.sequences: Dict[str, Tuple[str, int]] = {}
            self.has_sequence_ids = False

        def write(self, sequence: str, sequence_id: Optional[str] = None) -> int:
            """
            Write a sequence to the store.

            Args:
                sequence (str): The sequence to write
                sequence_id (Optional[str], optional): The corresponding sequence identifier. Defaults to None.

            Returns:
                int: The index of the inserted sequence
            """
            if sequence in self.sequences:
                return self.sequences[sequence][1]
            if sequence_id is not None:
                self.has_sequence_ids = True
            else:
                assert not self.has_sequence_ids
            sequence_id = sequence_id if sequence_id is not None else ""
            self.sequences[sequence] = (sequence_id, len(self.sequences))
            return self.sequences[sequence][1]

        def finish(self):
            """
            Write the sequences to the store on disk.
            """
            # Encode the sequences
            sequence_huffman_codes = huffman_code(Counter(chain.from_iterable(
                tqdm(
                    self.sequences,
                    desc="Analyzing sequences",
                    disable=(not self.show_progress)
                )
            ))) # type: ignore
            sequences = np.array([
                encode_string(s, sequence_huffman_codes)
                for s in tqdm(
                    self.sequences,
                    desc="Encoding sequences",
                    disable=(not self.show_progress)
                )
            ])

            sequence_ids = None
            n_sequence_id_buckets = 0
            sequence_id_block_size = 0
            if self.has_sequence_ids:
                # Create a sequence ID hash table
                n_sequence_id_buckets = int(self.sequence_id_bucket_ratio*len(sequences))
                sequence_id_huffman_codes = huffman_code(Counter(chain.from_iterable(
                    map(lambda x: x[0], tqdm(
                        self.sequences.values(),
                        desc="Analyzing sequence IDs",
                        disable=(not self.show_progress)
                    ))
                ))) # type: ignore
                sequence_ids = [[] for _ in range(n_sequence_id_buckets)]
                for sequence_id, i in (
                    tqdm(
                        self.sequences.values(),
                        desc="Encoding sequence IDs",
                        disable=(not self.show_progress)
                    )
                ):
                    bucket_index = hash(sequence_id) % n_sequence_id_buckets
                    sequence_id = encode_string(sequence_id, sequence_id_huffman_codes)
                    bisect.insort(sequence_ids[bucket_index], (sequence_id, i))
                    sequence_id_block_size = max(sequence_id_block_size, len(sequence_id))

                # Hash table information
                sequence_id_bucket_lengths = np.array(list(map(len, sequence_ids)), dtype=uint(uint_size(sequence_id_block_size)))
                sequence_id_bucket_offsets = np.zeros(len(sequence_ids), dtype=uint(uint_size(len(sequences))))
                np.cumsum(sequence_id_bucket_lengths[:-1], out=sequence_id_bucket_offsets[1:])

                # Flatten the hash table and sort the sequences based on the squence ID order
                self.sorting = np.array([i for bucket in sequence_ids for _, i in bucket])
                sequences = sequences[self.sorting]
                sequence_ids = np.array([sequence_id for bucket in sequence_ids for sequence_id, _ in bucket])

            # Write the store to disk
            with open(self.path, "wb") as f:
                # Write the header
                # [uint64] number of sequences
                f.write(np.uint64(len(sequences)).tobytes())
                # [uint32] sequence block size
                f.write(np.uint32(int(str(sequences.dtype)[2:])).tobytes())
                sequence_huffman_codes_bytes = pickle.dumps(sequence_huffman_codes)
                # [uint32] sequence huffman codes length
                f.write(np.uint32(len(sequence_huffman_codes_bytes)).tobytes())
                # [bytes] sequence huffman codes
                f.write(sequence_huffman_codes_bytes)
                # [uint64] sequence ID hash table length
                f.write(np.uint64(n_sequence_id_buckets).tobytes())
                if self.has_sequence_ids:
                    # [uint32] sequence ID block size
                    f.write(np.uint32(sequence_id_block_size).tobytes())
                    sequence_id_huffman_codes_bytes = pickle.dumps(sequence_id_huffman_codes)
                    # [uint32] sequence ID huffman codes length
                    f.write(np.uint32(len(sequence_id_huffman_codes_bytes)).tobytes())
                    # [bytes] sequence ID huffman codes
                    f.write(sequence_id_huffman_codes_bytes)
                # Write the body
                f.write(sequences.tobytes())
                if self.has_sequence_ids:
                    assert sequence_ids is not None
                    # [bytes] sequence IDs
                    f.write(sequence_ids.tobytes())
                    # [uint32] sequence ID bucket lengths
                    f.write(sequence_id_bucket_lengths.tobytes())
                    # [uint32] sequence ID bucket offsets
                    f.write(sequence_id_bucket_offsets.tobytes())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.finish()


    class SampleWriter:
        def __init__(self, path: Union[str, Path]):
            pass

        def write(self, sample: str, abundance: float):
            pass

        def finish(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.finish()


    def __init__(self, path: Union[str, Path], madvise: Optional[int] = None):
        """
        Create a new SequenceStore.

        Args:
            path (Union[str, Path]): The path to the sequence store file
            mode (int, optional): The mode of the sequence store. Defaults to 1.
        """
        self.path = path
        self.file_handle = open(self.path, "r+b")
        self.mmap = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        if madvise is not None:
            self.mmap.madvise(madvise)

        self.load()

        if not self.has_sequence_ids:
            def raise_value_error(*args, **kwargs) -> int:
                raise ValueError("Sequence IDs are not stored in this SequenceStore")
            self.lookup = raise_value_error

    def load(self):
        """
        Load the sequence store from disk.
        """
        # Read the header
        # [uint64] number of sequences
        self._n_sequences = np.frombuffer(self.mmap.read(8), dtype=np.uint64)[0]
        # [uint32] sequence block size
        self._sequence_block_size = np.frombuffer(self.mmap.read(4), dtype=np.uint32)[0]
        # [uint32] sequence huffman codes length
        sequence_huffman_codes_length = np.frombuffer(self.mmap.read(4), dtype=np.uint32)[0]
        # [bytes] sequence huffman codes
        self._sequence_huffman_codes = pickle.loads(self.mmap.read(sequence_huffman_codes_length))
        self._sequence_decode_tree = decodetree(self._sequence_huffman_codes)
        # [uint64] sequence ID hash table length
        self._n_sequence_id_buckets = np.frombuffer(self.mmap.read(8), dtype=np.uint64)[0]
        if self._n_sequence_id_buckets > 0:
            # [uint32] sequence ID block size
            self._sequence_id_block_size = np.frombuffer(self.mmap.read(4), dtype=np.uint32)[0]
            # [uint32] sequence ID huffman codes length
            sequence_id_huffman_codes_length = np.frombuffer(self.mmap.read(4), dtype=np.uint32)[0]
            # [bytes] sequence ID huffman codes
            self._sequence_id_huffman_codes = pickle.loads(self.mmap.read(sequence_id_huffman_codes_length))
            self._sequence_id_decode_tree = decodetree(self._sequence_id_huffman_codes)
        # Read the body
        offset = self.mmap.tell()
        self.mmap.seek(0)
        # [bytes] Memory-mapped view of sequences
        self.sequences = EncodedStringView(
            np.frombuffer(
                self.mmap,
                offset=offset,
                count=self._n_sequences,
                dtype=f"|S{self._sequence_block_size}"),
            self._sequence_huffman_codes
        )
        offset += self.sequences.view.nbytes

        if self._n_sequence_id_buckets > 0:
            # [bytes] Memory-mapped view of sequence IDs
            self.sequence_ids = EncodedStringView(
                np.frombuffer(
                    self.mmap,
                    offset=offset,
                    count=self._n_sequences,
                    dtype=f"|S{self._sequence_id_block_size}"),
                self._sequence_id_huffman_codes
            )
            offset += self.sequence_ids.view.nbytes

            # [uint] sequence ID bucket lengths
            self._sequence_id_bucket_lengths = np.frombuffer(
                self.mmap,
                offset=offset,
                count=self._n_sequence_id_buckets,
                dtype=uint(uint_size(self._sequence_id_block_size)))
            offset += self._sequence_id_bucket_lengths.nbytes

            # [uint] sequence ID bucket offsets
            self._sequence_id_bucket_offsets = np.frombuffer(
                self.mmap,
                offset=offset,
                count=self._n_sequence_id_buckets,
                dtype=uint(uint_size(self._n_sequences)))
            offset += self._sequence_id_bucket_offsets.nbytes

    def close(self):
        """
        Close the sequence store.
        """
        if self.has_sequence_ids:
            del self.sequence_ids
            del self._sequence_id_bucket_lengths
            del self._sequence_id_bucket_offsets
        del self.sequences
        self.mmap.close()
        self.file_handle.close()

    @property
    def has_sequence_ids(self) -> bool:
        """
        Check if the sequence store has sequence IDs.

        Returns:
            bool: True if the sequence store has sequence IDs
        """
        return self._n_sequence_id_buckets > 0

    def lookup(self, sequence_id: str) -> int:
        """
        Look up the index of a sequence ID.

        Args:
            sequence_id (str): The sequence ID

        Raises:
            KeyError: If the sequence ID is not found
            ValueError: If the store does not store sequence IDs

        Returns:
            int: The index of the sequence ID
        """
        if not self.has_sequence_ids:
            raise ValueError("Sequence IDs are not stored in this SequenceStore")
        bucket_index = int(hash(sequence_id) % self._n_sequence_id_buckets)
        bucket_start = self._sequence_id_bucket_offsets[bucket_index]
        bucket_end = bucket_start + self._sequence_id_bucket_lengths[bucket_index]
        encoded_sequence_id = encode_string(sequence_id, self._sequence_id_huffman_codes)
        i = int(bucket_start + np.searchsorted(self.sequence_ids.view[bucket_start:bucket_end], encoded_sequence_id))
        if i >= bucket_end or self.sequence_ids.view[i] != encoded_sequence_id:
            raise KeyError(f"Sequence ID '{sequence_id}' not found")
        return i

    def sequence(self, identifier: str) -> str:
        """
        Get a sequence by sequence ID.

        Args:
            identifier (str): The sequence ID

        Raises:
            KeyError: If the sequence ID is not found
            ValueError: If the store does not store sequence IDs

        Returns:
            str: The sequence
        """
        return self.sequences[self.lookup(identifier)]

    def __len__(self) -> int:
        """
        Get the number of sequences in the store.

        Returns:
            int: The number of sequences
        """
        return self._n_sequences

    @overload
    def __getitem__(self, i: Union[str, int]) -> str:
        """
        Get a sequence by index or sequence ID.

        Args:
            i (Union[str, int]): The index or sequence ID

        Returns:
            str: The sequence
        """
        ...
    @overload
    def __getitem__(self, i: Union[slice, np.ndarray]) -> List[str]:
        """
        Get an iterator over sequences by slice or indices

        Args:
            i (Union[slice, np.ndarray]): The slice or indices

        Yields:
            Generator[str, None, None]: The sequence
        """
        ...
    def __getitem__(self, i: Union[str, int, slice, np.ndarray]) -> Union[str, List[str]]:
        if isinstance(i, str):
            return self.sequence(i)
        return self.sequences[i]

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all sequences in the store.

        Yields:
            Generator[str, None, None]: The sequence
        """
        return iter(self.sequences) # type: ignore

from bitarray import bitarray, decodetree
from bitarray.util import deserialize, serialize
import mmap
import numpy as np
import re
from tqdm import tqdm

class SequenceDb:
    """
    A compact sequence storage container using hamming encodings to compress
    DNA sequences, and memory-mapping for fast reads.
    """

    huffman_codes = {
        'N': bitarray('000'),
        'C': bitarray('001'),
        'T': bitarray('01'),
        'A': bitarray('10'),
        'G': bitarray('11')
    }
    decode_tree = decodetree(huffman_codes)

    @staticmethod
    def _encode_sequence(sequence):
        encoded = bitarray()
        encoded.encode(SequenceDb.huffman_codes, re.sub(r"[^ACTGN]", "N", sequence))
        return serialize(encoded)

    @staticmethod
    def _decode_sequence(sequence):
        return deserialize(sequence).decode(SequenceDb.decode_tree)

    @staticmethod
    def create(path, sequences, progress: bool = False):
        with open(path, "wb") as f:
            encoded_sequences = list(map(SequenceDb._encode_sequence, sequences))
            n = np.uint32(len(encoded_sequences))
            block_size = np.uint32(2 + max(map(len, encoded_sequences)))
            header = n.tobytes() + block_size.tobytes()
            f.write(n.tobytes())
            f.write(block_size.tobytes())
            if progress:
                encoded_sequences = tqdm(encoded_sequences)
            for sequence in encoded_sequences:
                length = len(sequence).to_bytes(2, "big", signed=False)
                entry = length + sequence + b'\x00'*(block_size - 2 - len(sequence))
                f.write(entry)

    def __init__(self, path):
        with open(path, "r+b") as f:
            self.mm = mmap.mmap(f.fileno(), 0)
        self.length, self.block_size = np.frombuffer(self.mm, count=2, dtype=np.uint32)

    def __getitem__(self, index):
        index = 8 + self.block_size*index
        length = int.from_bytes(self.mm[index:index+2], "big")
        return self._decode_sequence(self.mm[index+2:index+2+length])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return self.length
import numpy as np
import tempfile
import unittest
import unittest.mock

from dbtk.data.formats import Fasta, SequenceStore

def generate_dna_sequence(length: int, rng: np.random.Generator):
    return "".join(rng.choice(list("ACGTN"), length))

class TestFasta(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_sequences = 10
        self.identifiers = [f"seq{i}" for i in range(self.n_sequences)]
        self.sequences = [generate_dna_sequence(30, self.rng) for _ in range(self.n_sequences)]
        self.metadata = [f"Metadata {i}" for i in range(self.n_sequences)]
        # Write temporary FASAT file
        self.directory = tempfile.TemporaryDirectory()
        with open(self.directory.name + "/test.fasta", "w") as f:
            f.write("\n".join([
                f">{i} {m}\n{s}"
                for i, s, m in zip(self.identifiers, self.sequences, self.metadata)
            ]))
        self.fasta = Fasta(self.directory.name + "/test.fasta")

    def tearDown(self):
        self.fasta.close()
        self.directory.cleanup()

    def test_lazy_length(self):
        # Ensure stored length is 0
        self.assertIsNone(self.fasta._length)
        self.assertEqual(len(self.fasta), self.n_sequences)

    def test_lazy_entries(self):
        self.assertEqual(len(self.fasta.entries), 0)
        self.assertEqual(len(self.fasta.id_map), 0)
        entry = self.fasta[0]
        self.assertIs(entry, self.fasta.entries[0])
        self.assertEqual(len(self.fasta.entries), 1)
        self.assertEqual(len(self.fasta.id_map), 1)

    def test_lazy_cleanup(self):
        method_id = id(self.fasta.__getitem__)
        list(self.fasta.entries)
        self.assertNotEqual(method_id, id(self.fasta.__getitem__))

    def test_index_lookup(self):
        for i, identifier in enumerate(self.identifiers):
            entry = self.fasta[i]
            self.assertEqual(entry.id, identifier)

    def test_identifier_lookup(self):
        for identifier in self.identifiers:
            entry = self.fasta[identifier]
            self.assertEqual(entry.id, identifier)

    def test_entry_values(self):
        for i, (identifier, sequence, metadata) in enumerate(zip(self.identifiers, self.sequences, self.metadata)):
            entry = self.fasta[i]
            self.assertEqual(entry.id, identifier)
            self.assertEqual(entry.sequence, sequence)
            self.assertEqual(entry.metadata, metadata)

    def test_iteration(self):
        for i, entry in enumerate(self.fasta):
            self.assertEqual(entry.id, self.identifiers[i])
        self.assertTrue(self.fasta._eof)


class TestSequenceStoreDeflateCompression(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_sequences = 10
        self.sequences = [generate_dna_sequence(30, self.rng) for _ in range(self.n_sequences)]
        # Write temporary sequence store
        self.compression = SequenceStore.DeflateCompression.from_data(self.sequences)

    def test_compression(self):
        for sequence in self.sequences:
            compressed = self.compression.compress(sequence)
            decompressed = self.compression.decompress(compressed)
            self.assertEqual(decompressed, sequence)


class TestSequenceStoreHuffmanCompression(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_sequences = 10
        self.sequences = [generate_dna_sequence(30, self.rng) for _ in range(self.n_sequences)]
        # Write temporary sequence store
        self.compression = SequenceStore.HuffmanCompression.from_data(self.sequences)

    def test_compression(self):
        for sequence in self.sequences:
            compressed = self.compression.compress(sequence)
            decompressed = self.compression.decompress(compressed)
            self.assertEqual(decompressed, sequence)


class TestSequenceStore(unittest.TestCase):
    def compression_method(self):
        return SequenceStore.HuffmanCompression

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_sequences = 10
        self.sequences = [generate_dna_sequence(30, self.rng) for _ in range(self.n_sequences)]
        # Write temporary sequence store
        self.directory = tempfile.TemporaryDirectory()
        SequenceStore.create(
            self.sequences,
            self.directory.name + "/test.seq",
            compression=self.compression_method()
        )
        self.sequence_store = SequenceStore(self.directory.name + "/test.seq")

    def tearDown(self):
        self.sequence_store.close()
        self.directory.cleanup()

    def test_version(self):
        self.assertEqual(self.sequence_store.version, SequenceStore.VERSION)

    def test_length(self):
        self.assertEqual(len(self.sequence_store), self.n_sequences)

    def test_compressor(self):
        compression_method = self.compression_method()
        if compression_method is None:
            compression_method = SequenceStore.NoCompression
        self.assertIsInstance(self.sequence_store.compressor, compression_method)

    def test_get_sequence(self):
        for i, sequence in enumerate(self.sequences):
            self.assertEqual(self.sequence_store[i], sequence)

class TestSequenceStoreDeflate(TestSequenceStore):
    def compression_method(self):
        return SequenceStore.DeflateCompression

class TestSequenceStoreHuffman(TestSequenceStore):
    def compression_method(self):
        return SequenceStore.HuffmanCompression

if __name__ == "__main__":
    unittest.main()

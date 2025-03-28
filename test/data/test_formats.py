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

class TestSequenceStore(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_sequences = 10
        self.sequences = [generate_dna_sequence(self.rng.integers(10, 30), self.rng) for _ in range(self.n_sequences)]
        self.directory = tempfile.TemporaryDirectory()
        with SequenceStore.SequenceWriter(self.directory.name + "/test.seq") as writer:
            for sequence in self.sequences:
                writer.write(sequence)
        self.store = SequenceStore(self.directory.name + "/test.seq")

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_length(self):
        self.assertEqual(len(self.store), self.n_sequences)
        self.assertEqual(len(self.store.sequences), self.n_sequences)

    def test_has_all_sequences(self):
        self.assertTrue(all(sequence in self.sequences for sequence in self.store))


class TestSequenceStoreWithoutSequenceIds(TestSequenceStore):
    def test_has_sequence_ids(self):
        self.assertFalse(self.store.has_sequence_ids)

    def test_number_of_sequence_id_buckets(self):
        self.assertEqual(self.store._n_sequence_id_buckets, 0)

    def test_get_sequence(self):
        for i, sequence in enumerate(self.sequences):
            self.assertEqual(self.store[i], sequence)

    def test_get_sequences_from_slice(self):
        # try a random slice
        self.assertEqual(self.store[3:7], self.sequences[3:7])


class TestSequenceStoreWithSequenceIds(TestSequenceStore):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_sequences = 10
        self.identifiers = [f"seq{i}" for i in range(self.n_sequences)]
        self.sequences = [generate_dna_sequence(self.rng.integers(10, 30), self.rng) for _ in range(self.n_sequences)]
        self.directory = tempfile.TemporaryDirectory()
        with SequenceStore.SequenceWriter(self.directory.name + "/test.seq") as writer:
            for sequence, identifier in zip(self.sequences, self.identifiers):
                writer.write(sequence, identifier)
        self.store = SequenceStore(self.directory.name + "/test.seq")

    def test_has_sequence_ids(self):
        self.assertTrue(self.store.has_sequence_ids)

    def test_has_all_sequence_ids(self):
        self.assertTrue(all(identifier in self.identifiers for identifier in self.store.sequence_ids))

    def test_number_of_sequence_id_buckets(self):
        self.assertGreater(self.store._n_sequence_id_buckets, 0)

    def test_get_sequence(self):
        for sequence, identifier in zip(self.sequences, self.identifiers):
            i = self.store.lookup(identifier)
            self.assertEqual(self.store[i], sequence)

    def test_get_sequence_from_identifier(self):
        for sequence, identifier in zip(self.sequences, self.identifiers):
            self.assertEqual(self.store[identifier], sequence)

    def test_get_sequence_ids_from_slice(self):
        self.assertTrue(all(identifier in self.identifiers for identifier in self.store.sequence_ids[3:7]))

if __name__ == "__main__":
    unittest.main()

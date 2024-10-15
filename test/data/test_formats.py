import numpy as np
import tempfile
import unittest
import unittest.mock

from dbtk.data.formats import Fasta

def generate_dna_sequence(length: int, rng: np.random.Generator):
    return "".join(rng.choice(list("ACGT"), length))

def generate_rna_sequence(length: int, rng: np.random.Generator):
    return "".join(rng.choice(list("ACGU"), length))

def generate_iupac_sequence(length: int, rng: np.random.Generator):
    return "".join(rng.choice(list("ACGTURYSWKMBDHVN"), length))

def generate_protein_sequence(length: int, rng: np.random.Generator):
    return "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), length))

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

if __name__ == "__main__":
    unittest.main()

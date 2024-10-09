from itertools import product

def dna(kmer: int = 1):
    return list(map(str, map("".join, product("ACGT", repeat=kmer))))

class Vocabulary:
    def __init__(self, words):
        self._word_to_index = {}
        self._index_to_word = []
        self.update(["[PAD]", "[UNK]", *words])

    def add(self, word):
        if word in self._word_to_index:
            return
        self._word_to_index[word] = len(self._index_to_word)
        self._index_to_word.append(word)

    def update(self, words):
        for word in words:
            self.add(word)

    def __call__(self, words):
        return map(self.__getitem__, words)

    def __getitem__(self, key):
        return self._word_to_index.get(key, self._word_to_index["[UNK]"])

    def __len__(self):
        return len(self._index_to_word)


class DnaVocabulary(Vocabulary):
    def __init__(self, kmer: int = 1):
        super().__init__(dna(kmer))
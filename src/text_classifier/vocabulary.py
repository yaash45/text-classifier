from typing import Iterable

import numpy as np


class Vocabulary(dict[str, int]):
    """
    The corpus of words seen by the classifier and
    a permanent index value associated with it.
    """

    _count: int = 0

    def register(self, words: Iterable[str]):
        """
        Adds a collection of words to the current Vocabulary

        Args:
            words: a collection of words to index and record


        Example:
            ```
            words = ["test", "word", "woman", "mystery"]

            v = Vocabulary()

            v.register(words)

            print(v) # {"test" : 0, "word": 1, "woman": 2, "mystery": 3}
            ```
        """
        for word in words:
            if word not in self:
                self[word] = self._count
                self._count += 1

    def index_of(self, word: str) -> int:
        """
        Obtains the index of a given word in the vocabulary

        Args:
            word: the index of this word is queried from this Vocabulary

        Returns:
            -1 if the word doesn't exist in the vocabulary, a non-zero
            integer representing the index of the word otherwise

        Example:
            ```
            words = ["test", "word", "woman", "mystery"]

            # prep the vocabulary
            v = Vocabulary()
            v.register(words)

            # query indices for words
            assert v.index_of("woman") == 2
            assert v.index_of("lady") == -1
            ```
        """
        return self.get(word, -1)


class WordBag(dict[str, int]):
    """
    This is a Bag-of-Words data representation in the form
    of a dictionary where the key is a word, and the corresponding
    value its recorded frequency of occurence in a document


    Example:
    ```
    words = ["test", "word", "woman", "mystery", "test"]

    bag = WordBag(words)

    print(bag) # {"test": 2, "word": 1, "woman": 1, "mystery": 1}
    ```
    """

    def __init__(self, words: Iterable[str]):
        for word in words:
            if word in self:
                self[word] += 1
            else:
                self[word] = 1


def vectorize(bag: WordBag, vocab: Vocabulary) -> np.ndarray:
    """
    Convert a WordBag into a vector representation.

    This function takes a WordBag and a Vocabulary as input, and
    returns a numpy array where the index of the array corresponds
    to the index of the word in the vocabulary, and the value at
    that index is the frequency of the word in the WordBag.

    Args:
        bag: A WordBag containing the words to be converted
        vocab: A Vocabulary containing the mapping of words to indices

    Returns:
        A numpy array representing the vectorized WordBag
    """
    vector: np.ndarray = np.zeros(len(vocab), dtype=int)

    for word, freq in bag.items():
        index = vocab.index_of(word)

        if index < 0:
            continue

        vector[index] = freq

    return vector

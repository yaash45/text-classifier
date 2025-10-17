from pytest import mark

from text_classifier.features import Vocabulary, WordBag, vectorize


def test_vocabulary_registration_no_repetition():
    words = ["test", "this", "vocab"]
    v = Vocabulary()
    v.register(words)

    # verify the size of the vocabulary
    assert len(v) == 3

    # verify each item's expected index
    assert v.index_of("test") == 0
    assert v.word_at(0) == "test"

    assert v.index_of("this") == 1
    assert v.word_at(1) == "this"

    assert v.index_of("vocab") == 2
    assert v.word_at(2) == "vocab"

    # non-existent vocabulary item
    assert v.index_of("gabagool") == -1
    assert v.word_at(10) is None


def test_vocabulary_registration_with_repetition():
    words = ["test", "this", "vocab", "test"]
    v = Vocabulary()
    v.register(words)

    # verify the size of the vocabulary
    assert len(v) == 3  # do not store duplicates

    # verify each item's expected index
    assert v.index_of("test") == 0
    assert v.word_at(0) == "test"

    assert v.index_of("this") == 1
    assert v.word_at(1) == "this"

    assert v.index_of("vocab") == 2
    assert v.word_at(2) == "vocab"

    assert v.index_of("test") == 0  # check that the index is consistent

    # non-existent vocabulary item
    assert v.index_of("gabagool") == -1
    assert v.word_at(10) is None


def test_word_bag():
    words = ["word", "word", "test", "word", "scan", "gabagool"]

    bag = WordBag(words)

    assert dict(bag) == {
        "word": 3,
        "test": 1,
        "scan": 1,
        "gabagool": 1,
    }


@mark.parametrize(
    "words,expected",
    [
        (
            # some repeated elements
            [
                "test",
                "this",
                "new",
                "function",
                "test",
                "this",
                "very",
                "well",
                "well",
                "test",
            ],
            [3, 2, 1, 1, 1, 2],
        ),
        (
            # one word repeated multiple times
            ["test", "test", "test", "test"],
            [4],
        ),
        (
            # one occurence of each word
            ["a", "b", "c", "d", "e"],
            [1, 1, 1, 1, 1],
        ),
        (
            # empty words
            [],
            [],
        ),
    ],
)
def test_vectorize(words: list[str], expected: list[int]):
    v = Vocabulary()
    v.register(words)

    bag = WordBag(words)

    result = vectorize(bag, v)
    assert list(result) == expected

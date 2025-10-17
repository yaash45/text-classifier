from text_classifier.features import Vocabulary


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

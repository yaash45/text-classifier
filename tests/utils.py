from spacy.lang.en.stop_words import STOP_WORDS


def filter_stop_words(words: list[str]) -> list[str]:
    """
    Helper function to filter out stop words from a list
    of given words. If stop words are ever updated in the
    library, we will have the most up-to-date expectations
    in our tests if we use this function.

    Args:
        words: a list of words to filter

    Returns:
        a list of words where none of the items belong to the
        STOP_WORDS set.
    """
    return [i for i in words if i not in STOP_WORDS]

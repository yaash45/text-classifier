import tempfile
from unittest.mock import patch

from pytest import mark

from text_classifier.word_reader import WordReader

from .utils import filter_stop_words

TEST_DATA_SET: list[tuple[str, list[str]]] = [
    (
        "hello my name is Tony Soprano, How ya doin",
        [
            "hello",
            "my",
            "name",
            "is",
            "tony",
            "soprano",
            "how",
            "ya",
            "doin",
        ],
    ),
    (
        "those kids are EXTREMELY unruly. 1, 2, 3, and 4, times they were told to stop.",
        [
            "those",
            "kids",
            "are",
            "extremely",
            "unruly",
            "and",
            "times",
            "they",
            "were",
            "told",
            "to",
            "stop",
        ],
    ),
    (
        "# # # whoa... # # # whoa... ### get out!!!",
        [
            "whoa",
            "whoa",
            "get",
            "out",
        ],
    ),
    (
        "this is not very nice of ya. You're in the wrong place pal",
        [
            "this",
            "is",
            "not",
            "very",
            "nice",
            "of",
            "ya",
            "you",
            "'re",
            "in",
            "the",
            "wrong",
            "place",
            "pal",
        ],
    ),
    (
        """wow. \n\n 
        You're really not going to bring me my Gabagool? Madone!\t \n \n   # 
        What's the matter with you??""",
        [
            "wow",
            "you",
            "'re",
            "really",
            "not",
            "going",
            "to",
            "bring",
            "me",
            "my",
            "gabagool",
            "madone",
            "what",
            "'s",
            "the",
            "matter",
            "with",
            "you",
        ],
    ),
]


@mark.parametrize(
    "text,expected",
    TEST_DATA_SET,
)
def test_parse_words(text: str, expected: list[str]):
    """
    Test the core word parsing functionality of the WordReader class
    """

    reader = WordReader()

    # filtering out the stop words in the test ensures that we conform to
    # the latest set provided by the library
    assert reader.parse_words(text) == filter_stop_words(expected)


@mark.parametrize(
    "text,expected",
    TEST_DATA_SET,
)
def test_read_file_tokens(text: str, expected: list[str]):
    """
    Test that words can be read from a file containing text
    """

    reader = WordReader()

    with tempfile.NamedTemporaryFile() as temp:
        # Setup by writing some text into the temp file
        temp.write(bytes(text, "utf-8"))
        temp.flush()

        # Parse the written text from the file into tokens
        words = list(reader.read_file_tokens(temp.name))
        assert words == filter_stop_words(expected)


@mark.parametrize(
    "text,expected",
    TEST_DATA_SET,
)
def test_read_user_input_tokens(text: str, expected: list[str]):
    """
    Test that word tokens can be parsed through user input
    """

    reader = WordReader()

    with patch("builtins.input", return_value=text):
        words = list(reader.read_user_input_tokens("please enter some text:"))
        assert words == filter_stop_words(expected)

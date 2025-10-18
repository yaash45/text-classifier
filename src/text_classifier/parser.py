"""
This module containts the routines used to parse text
blobs into word tokens. These blobs can come from several
different sources: individual python strings, file paths,
and user input (via the builtins.input function)
"""

from typing import Iterator


def parse_words(text: str) -> list[str]:
    """
    Parse words out of a string of text, splitting
    them into individual word tokens that do not contain
    stop words.

    Args:
        text: input string to tokenize and filter into words

    Returns:
        a list of string tokens that represent the words in
        the input text string
    """
    # lazy import the natural language model module
    from .nlp import nlp_model

    tokens = nlp_model(text)

    return [
        t.text.lower()
        for t in tokens
        if not t.is_punct and not t.is_space and not t.is_stop and not t.is_digit
    ]


def read_file_words(src: str) -> Iterator[str]:
    """
    Given a file path, read and yield the word tokens
    contained within the file.

    The output is filtered and sanitized to exclude invalid
    tokens like punctuation, etc.


    Args:
        src: file path to read word tokens from

    Yields:
        One word at a time, as parsed from the file path `src`
    """
    with open(src, "r+") as f:
        for line in f.readlines():
            for word in parse_words(line):
                yield word


def read_user_input_words(prompt: str) -> Iterator[str]:
    """
    Read the user's input string and yield word tokens
    contained within the input text.

    Args:
        prompt: the string to display as a prompt to get the user's
            input

    Yields:
        One word at a time, as parsed from the user's input string
    """
    input_text = input(prompt)

    for word in parse_words(input_text):
        yield word

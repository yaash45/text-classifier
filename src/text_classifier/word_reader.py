from typing import Iterator

import spacy


class WordReader:
    """
    Class to read words from various sources, such as user input in the shell,
    or a file path. This class is responsible for reading strings of text, and
    parsing them into word tokens.

    The class can be used as follows:

    ```
    reader = WordReader()

    words_iterator = reader.read_user_input_tokens("Please enter some text: ")

    print(list(words_iterator))

    ```
    """

    _nlp = spacy.load("en_core_web_sm")

    def parse_words(self, text: str) -> list[str]:
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

        tokens = self._nlp(text)

        return [
            t.text.lower()
            for t in tokens
            if not t.is_punct and not t.is_space and not t.is_stop and not t.is_digit
        ]

    def read_file_tokens(self, src: str) -> Iterator[str]:
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
                for word in self.parse_words(line):
                    yield word

    def read_user_input_tokens(self, prompt: str) -> Iterator[str]:
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

        for word in self.parse_words(input_text):
            yield word

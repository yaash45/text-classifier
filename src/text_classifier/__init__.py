from .word_reader import WordReader


def main() -> None:
    print("Hello from text-classifier!")

    reader = WordReader()
    print("please enter some input:")
    text = "hello this is some cool    \n\n \t text that's going to COMPLETELY wreak havoc on the tokenizerrr!!"

    result = reader.parse_words(text)

    print(result)

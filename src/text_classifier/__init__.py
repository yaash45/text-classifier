from .word_reader import WordReader


def main() -> None:
    print("Hello from text-classifier!")

    reader = WordReader()

    result = list(reader.read_user_input_tokens("Please enter some text: "))

    print(result)

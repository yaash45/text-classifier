from .features import Vocabulary, WordBag, vectorize
from .parser import parse_words


def main() -> None:
    print("Hello from text-classifier!\n")

    docs: list[str] = [
        "this is Tony Soprano, the mafia don of New Jersey. Gabagool is his favourite food. Bring him some Gabagool from Satrialles in New Jersey",
        "Sylvio: ay Ton, Christopher came in earlier... said he was carrying a.. lighter bag",
        "Tony: !!!!! WHAT?? STOP BREAKING MY ....",
        "see you tomorrow boss",
        "boss, boss, boss, boss.... Christopher, Tony, and Sylvio are going to take care of the gabagool! $100",
    ]

    vocab = Vocabulary()
    bags: list[WordBag] = []

    # register all the seen words with the vocabulary
    for doc in docs:
        words = parse_words(doc)
        vocab.register(words)
        bag = WordBag(words)
        bags.append(bag)

    print(f"vocab = {vocab}\n")

    for i, bag in enumerate(bags):
        print(f"bag [{i}] = {bag}")
        vector = vectorize(bag, vocab)
        print(f"         - {vector}\n")

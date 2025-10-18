import argparse

from .features import Vocabulary, WordBag, vectorize
from .logger import configure_logger, get_logger
from .parser import parse_words


def parse_args():
    """
    Parse command line arguments into an argparse.Namespace object.

    Current arguments parsed: `verbose`

    Returns:
        argparse.Namespace: containing the parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run the text-classifier app.")

    # The magic: `-v` can be repeated to increase verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv, -vvv for more detail)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logger(args.verbose)

    logger = get_logger(__name__)

    logger.info("Hello from text-classifier!\n")

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

    logger.debug(f"vocab = {vocab}\n")

    for i, bag in enumerate(bags):
        logger.info(f"bag [{i}] = {bag}")
        vector = vectorize(bag, vocab)
        logger.debug(f"{vector}\n")

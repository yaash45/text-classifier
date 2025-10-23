import argparse

from rich.pretty import pretty_repr

from .classifier import Classifier
from .logger import configure_logger, get_logger


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

    docs: list[tuple[str, str]] = [
        (
            "this is Tony Soprano, the mafia don of New Jersey. Gabagool is his favourite food. Bring him some Gabagool from Satrialles in New Jersey",
            "positive",
        ),
        (
            "Sylvio: ay Ton, Christopher came in earlier... said he was carrying a.. lighter bag",
            "negative",
        ),
        (
            "Tony: !!!!! WHAT?? STOP BREAKING MY ....",
            "negative",
        ),
        (
            "see you tomorrow boss",
            "positive",
        ),
        (
            "boss, boss, boss, boss.... Christopher, Tony, and Sylvio are going to take care of the gabagool! $100",
            "neutral",
        ),
        (
            "Tony Soprano is one grade A pain in the wrong place. He is the boss of the new jersey mafia, but he is a terrible angry human individual.",
            "negative",
        ),
        (
            "this is not good at all Tony... Sylvio and Christopher took care of Junior, but you have to make sure Carmine is happy",
            "negative",
        ),
    ]

    c = Classifier()

    c.train(docs)

    logger.info(f"priors = {pretty_repr(c.priors)}\n")
    logger.info(f"likelihoods = {pretty_repr(c.word_likelihoods_per_category)}\n")
    logger.info(f"categories = {pretty_repr(c.categories)}\n")

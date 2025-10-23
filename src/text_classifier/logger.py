import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(color_system="256")


def configure_logger(verbosity: int):
    """
    Configure the root logger of the application dynamically.

    The root logger will be configured to log at either the
    DEBUG or INFO level, depending on the verbosity parameter.
    """
    if verbosity >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Update the root logger’s level dynamically
    logging.getLogger().setLevel(level)

    # Configure logging once, globally
    logging.basicConfig(
        level=level,
        format="[%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name.

    Args:
        name: The name of the logger to return.

    Returns:
        A logger with the given name.
    """
    return logging.getLogger(name)
    return logging.getLogger(name)

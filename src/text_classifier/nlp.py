"""
This module contains the natural language processing model
used by the rest of the text classifier.

It is contained within this separate module to serve as a
pythonic 'singleton', as it will only be loaded once.
"""

from time import perf_counter

import spacy

from .logger import get_logger

logger = get_logger(__name__)

import_start = perf_counter()

nlp_model = spacy.load("en_core_web_sm")

import_end = perf_counter()

logger.debug(
    f"[yellow bold italic]NLP model loading complete. Time elapsed: {import_end - import_start:.2f} seconds[/]"
)

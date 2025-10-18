"""
This module contains the natural language processing model
used by the rest of the text classifier.

It is contained within this separate module to serve as a
pythonic 'singleton', as it will only be loaded once.
"""

import spacy

nlp_model = spacy.load("en_core_web_sm")

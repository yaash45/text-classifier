from .features import Vocabulary
from .logger import get_logger
from .parser import parse_words

logger = get_logger(__name__)


class Classifier:
    def __init__(self):
        # store model vocabulary
        self._vocab = Vocabulary()

        # word frequencies-per-category
        self._word_freq_per_category: dict[str, dict[str, int]] = {}

        # store category priors, aka P(category)
        self._priors: dict[str, float] = {}

        # store total word count per category
        self._total_words_per_category: dict[str, int] = {}

        # store word likelihoods-per-category, aka P(word|category)
        self._likelihoods: dict[str, dict[str, float]] = {}

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocab

    @property
    def priors(self) -> dict[str, float]:
        return self._priors

    @property
    def word_likelihoods_per_category(self) -> dict[str, dict[str, float]]:
        return self._likelihoods

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self._priors.keys())

    def total_words_for_category(self, category: str) -> int:
        return self._total_words_per_category.get(category.lower(), -1)

    def _build_category_word_counts(self, data: list[tuple[str, str]]):
        for word, category in data:
            if category in self._word_freq_per_category:
                cur_word_count = self._word_freq_per_category[category].get(word, 0)
                self._word_freq_per_category[category][word] = cur_word_count + 1
            else:
                self._word_freq_per_category.setdefault(category, {word: 1})

    def train(self, dataset: list[tuple[str, str]]):
        total_doc_count = len(dataset)
        docs_per_category: dict[str, int] = {}

        # build vocabulary and per-category word counts
        for doc, label in dataset:
            # clean the label in case the dataset is inconsistent
            label = label.lower()

            # count number of docs-per-category to compute priors
            if label in docs_per_category:
                docs_per_category[label] += 1
            else:
                docs_per_category[label] = 1

            words = parse_words(doc)

            self._vocab.register(words)

            # count occurrences of a word in a given category
            self._build_category_word_counts([(w, label) for w in words])

        # Calculate category priors
        for cat, category_doc_count in docs_per_category.items():
            self._priors[cat] = category_doc_count / total_doc_count

        # Calculate word likelihoods
        self._total_words_per_category: dict[str, int] = {
            category: len(word_count_map)
            for category, word_count_map in self._word_freq_per_category.items()
        }

        for category, word_count_map in self._word_freq_per_category.items():
            category_total = self._total_words_per_category[category]
            for word, freq in word_count_map.items():
                if category not in self._likelihoods:
                    self._likelihoods[category] = {}

                self._likelihoods[category][word] = freq / category_total

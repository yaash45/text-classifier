from .features import Vocabulary
from .logger import get_logger
from .parser import parse_words

logger = get_logger(__name__)


class WordLikelihood(dict[str, float]):
    """
    Stores smoothed word likelihoods for a vocabulary.

    Uses Laplace/Lidstone smoothing: unseen words are assigned
    k / (total_words + k * vocab_size).

    Attributes:
        total_words (int): The total number of words in the category.

        vocab_size (int): The size of the vocabulary.

        k (float): The smoothing parameter for Laplace smoothing. Defaults to 1.0.
    """

    def __init__(self, total_words: int, vocab_size: int, k: float = 1.0):
        super().__init__()
        self.total_words = total_words
        self.vocab_size = vocab_size
        self.k = k

    def __getitem__(self, key: str) -> float:
        if key in self:
            return super().__getitem__(key)
        else:
            # lazily calculate unseen word probability on demand
            return self.k / (self.total_words + (self.k * self.vocab_size))


class Classifier:
    """
    A simple Naive Bayes classifier for text classification.

    This class is responsible for training on a dataset of labeled documents,
    and subsequently classifying new, unseen documents into their respective categories.

    Attributes:
        vocabulary (Vocabulary): The corpus of words seen by the classifier and
            a permanent index value associated with it.

        priors (dict[str, float]): Category priors, aka P(category).

        total_words_per_category (dict[str, int]): Total word count per category.

        word_likelihoods_per_category (dict[str, dict[str, float]]): Word likelihoods-per-category, aka P(word|category).
    """

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
        self._likelihoods: dict[str, WordLikelihood] = {}

    @property
    def vocabulary(self) -> Vocabulary:
        """
        The Vocabulary object associated with this classifier.

        This Vocabulary object is used to keep track of all the words
        seen in the training data, and to map these words to unique
        indices.

        Returns:
            Vocabulary: the classifier's Vocabulary object
        """
        return self._vocab

    @property
    def priors(self) -> dict[str, float]:
        """
        A dictionary mapping category labels to their respective priors.

        The prior of a category is the probability of observing that category
        given no other information. This is calculated as the number of
        documents in the training set belonging to that category divided
        by the total number of documents in the training set.

        Returns:
            dict[str, float]: a dictionary mapping category labels to their
                respective priors
        """
        return self._priors

    @property
    def word_likelihoods_per_category(self) -> dict[str, WordLikelihood]:
        """
        A dictionary mapping category labels to their respective word likelihoods.

        The word likelihood of a word given a category is the probability of observing
        that word given that category. This is calculated as the frequency of the word
        in the category divided by the total number of words in the category.

        Note that the probabilities are calculated with Laplace smoothing.

        Returns:
            dict[str, dict[str, float]]: a dictionary mapping category labels to their
                respective word likelihoods
        """
        return self._likelihoods

    @property
    def categories(self) -> tuple[str, ...]:
        """
        Returns a tuple of category labels associated with this classifier.

        These are the labels used when training the classifier, and are used
        to compute the priors and word likelihoods associated with each
        category.

        Returns:
            tuple[str, ...]: a tuple of category labels associated with this classifier
        """
        return tuple(self._priors.keys())

    def total_words_for_category(self, category: str) -> int:
        """
        Returns the total number of words in the given category.

        Args:
            category (str): the category to query for the total word count

        Returns:
            int: the total number of words in the given category
        """
        return self._total_words_per_category.get(category.lower(), 0)

    def _build_category_word_counts(self, data: list[tuple[str, str]]):
        """
        Builds per-category word counts given a list of word-category pairs.

        Args:
            data (list[tuple[str, str]]): A list of tuples where the first element
                is a word and the second element is the category the word belongs to.

        Populates the classifiers `_word_freq_per_category` attribute with the
        category-word counts.
        """
        for word, category in data:
            if category in self._word_freq_per_category:
                cur_word_count = self._word_freq_per_category[category].get(word, 0)
                self._word_freq_per_category[category][word] = cur_word_count + 1
            else:
                self._word_freq_per_category.setdefault(category, {word: 1})

    def train(
        self,
        dataset: list[tuple[str, str]],
        k: float = 1.0,
    ):
        """
        Train a classifier using a given dataset.

        Args:
            dataset (list[tuple[str, str]]): A list of tuples where the first element
                is a document and the second element is the category the document belongs to.
            k (float): The smoothing parameter for Laplace smoothing. Defaults to 1.0.
        """
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
            category: sum(word_count_map.values())
            for category, word_count_map in self._word_freq_per_category.items()
        }

        for category, word_count_map in self._word_freq_per_category.items():
            category_total = self._total_words_per_category[category]
            for word, freq in word_count_map.items():
                if category not in self._likelihoods:
                    self._likelihoods[category] = WordLikelihood(
                        k=k, vocab_size=len(self._vocab), total_words=category_total
                    )

                self._likelihoods[category][word] = (
                    self._calculate_smoothed_word_likelihood(
                        freq,
                        category_total,
                        k,
                    )
                )

    def _calculate_smoothed_word_likelihood(
        self,
        word_freq: int,
        category_total: int,
        k: float = 1.0,
    ) -> float:
        """
        Computes the Laplace smoothed probability of a word given its frequency in a category.

        Args:
            word_freq (int): the frequency of the word in the category
            category_total (int): the total number of words in the category
            k (float, optional): the smoothing parameter. Defaults to 1.0.

        Returns:
            float: the Laplace smoothed probability of the word in the category
        """
        numerator = word_freq + k
        denominator = category_total + (k * len(self._vocab))

        return numerator / denominator

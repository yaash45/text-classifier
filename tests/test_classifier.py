from pytest import approx

from text_classifier.classifier import Classifier


def test_classifier_training():
    test_dataset = [
        ("love my cat", "positive"),
        ("love my dog", "positive"),
        ("hate my cat", "negative"),
    ]

    c = Classifier()

    # Train with Laplace Smoothing (alpha = 1)
    c.train(test_dataset, k=1)

    # check if vocabulary is built accurately
    assert len(c.vocabulary) == 4  # 'my' is a stop word and is removed
    assert sorted(list(c.vocabulary.keys())) == sorted(["love", "cat", "dog", "hate"])

    # check word counts per category
    assert c.total_words_for_category("positive") == 4
    assert c.total_words_for_category("negative") == 2
    assert c.total_words_for_category("neutral") == 0  # non-existent category

    # check priors
    assert approx(c.priors["positive"], 0.01) == 2 / 3
    assert approx(c.priors["negative"], 0.01) == 1 / 3

    # check likelihoods
    likelihoods = c.word_likelihoods_per_category

    assert approx(likelihoods["positive"]["love"], 0.01) == 3 / 8
    assert approx(likelihoods["positive"]["cat"], 0.01) == 2 / 8
    assert approx(likelihoods["positive"]["dog"], 0.01) == 2 / 8
    assert approx(likelihoods["positive"]["hate"], 0.01) == 1 / 8

    assert approx(likelihoods["negative"]["love"], 0.01) == 1 / 6
    assert approx(likelihoods["negative"]["cat"], 0.01) == 2 / 6
    assert approx(likelihoods["negative"]["dog"], 0.01) == 1 / 6
    assert approx(likelihoods["negative"]["hate"], 0.01) == 2 / 6

    # unseen words should have a non-zero probability returned
    unseen_word = "blah"
    assert approx(likelihoods["positive"][unseen_word], 0.01) == 1 / 8
    assert approx(likelihoods["negative"][unseen_word], 0.01) == 1 / 6

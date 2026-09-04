import pytest

from sampling import softmax, apply_temperature

def test_softmax_probabilities_sum_to_one():
    logits = [4.0, 2.0, 1.0]
    probabilities = softmax(logits)

    assert sum(probabilities) == pytest.approx(1.0)

def test_softmax_keeps_highest_most_likely():
    logits = [4.0, 2.0, 1.0]
    probabilities = softmax(logits)

    assert probabilities[0] > probabilities[1]
    assert probabilities[1] > probabilities[2]

def test_lower_temperature_makes_distribution_sharper():
    logits = [4.0, 2.0, 1.0]
    low_temp_logits = apply_temperature(logits, 0.5)
    high_temp_logits = apply_temperature(logits, 2.0)

    low_temp_probabs = softmax(low_temp_logits)
    high_temp_probabs = softmax(high_temp_logits)

    assert low_temp_probabs > high_temp_probabs

def test_temperature_must_be_positive():
    logits = [4.0, 2.0, 1.0]
    with pytest.raises(ValueError):
        apply_temperature(logits, 0)
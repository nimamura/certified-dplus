"""Exact existence tests for error-free 1-D data (Weidelt 1986; Yee & Paulson 1988)."""

import random
from fractions import Fraction as F
from itertools import combinations

import pytest

from certified_dplus.exact1d import classify, is_consistent, is_regular, weidelt_minors
from data1d import LARSEN_CONSISTENT_TRIPLES, exact_response, larsen_exact


def test_single_frequency_needs_positive_g_and_h():
    assert is_regular([F(2)], [(F(3), F(-1))])
    assert not is_consistent([F(2)], [(F(-3), F(-1))])
    assert not is_consistent([F(2)], [(F(3), F(1))])
    assert weidelt_minors([F(2)], [(F(3), F(-1))]) == ([F(1, 2)], [F(3)])


def test_model_with_many_poles_is_regular():
    omega = [F(1, 3), F(1), F(5, 2)]
    c = exact_response(omega, F(1, 7), [F(2), F(1, 2), F(3), F(1)], [F(0), F(1, 5), F(2), F(9)])
    assert is_regular(omega, c)
    assert is_consistent(omega, c)
    assert classify(omega, c) == "regular"


def test_model_with_few_poles_is_degenerate_but_consistent():
    # one pole, a0 = 0, sampled at three frequencies: fewer than 2M positive constants
    omega = [F(1, 3), F(1), F(5, 2)]
    c = exact_response(omega, F(0), [F(2)], [F(1, 5)])
    assert not is_regular(omega, c)
    assert is_consistent(omega, c)
    assert classify(omega, c) == "degenerate"


def test_perturbed_degenerate_data_is_inconsistent():
    omega = [F(1, 3), F(1), F(5, 2)]
    c = exact_response(omega, F(0), [F(2)], [F(1, 5)])
    c[1] = (c[1][0] * F(101, 100), c[1][1])
    assert classify(omega, c) == "inconsistent"


def test_two_frequency_minors_match_weidelt_27():
    rng = random.Random(1)
    seen = set()
    for _ in range(300):
        w1 = F(rng.randint(1, 50), 10)
        w2 = w1 + F(rng.randint(1, 50), 10)
        g1, h1, g2, h2 = (F(rng.randint(1, 400), 7) for _ in range(4))
        c = [(g1, -h1), (g2, -h2)]
        dre, dim = g2 - g1, -(h2 - h1)
        ere, eim = w2 * g2 - w1 * g1, -(w2 * h2 - w1 * h1)
        dw2 = (w2 - w1) ** 2
        expected = (
            (h2 / w2 - h1 / w1) / (w2 - w1) < 0  # (27a)
            and (g2 - g1) / (w2 - w1) < 0  # (27b)
            and (dre**2 + dim**2) / dw2 < h1 * h2 / (w1 * w2)  # (27c)
            and (ere**2 + eim**2) / dw2 < g1 * g2  # (27d)
        )
        assert is_regular([w1, w2], c) == expected
        seen.add(expected)
    assert seen == {True, False}


def test_larsen_pairs():
    bad = [p for p in combinations(range(1, 8), 2) if not is_consistent(*larsen_exact(p))]
    assert bad == [(6, 7)]


def test_larsen_triples():
    good = {t for t in combinations(range(1, 8), 3) if is_consistent(*larsen_exact(t))}
    assert good == LARSEN_CONSISTENT_TRIPLES


def test_larsen_quadruples_and_full_set_fail():
    assert not any(is_consistent(*larsen_exact(q)) for q in combinations(range(1, 8), 4))
    assert classify(*larsen_exact(range(1, 8))) == "inconsistent"


def test_weidelt_and_yee_paulson_agree_on_all_larsen_subsets():
    for size in range(1, 8):
        for sub in combinations(range(1, 8), size):
            omega, c = larsen_exact(sub)
            # no subset of these data is exactly degenerate
            assert is_regular(omega, c) == is_consistent(omega, c), sub


def test_inputs_must_be_distinct_positive_frequencies():
    with pytest.raises(ValueError):
        is_consistent([F(1), F(1)], [(F(1), F(-1)), (F(1), F(-1))])
    with pytest.raises(ValueError):
        is_consistent([F(0)], [(F(1), F(-1))])


def test_accepts_floats_and_complex():
    assert is_regular([0.5, 2.0], [3.0 - 2.0j, 1.5 - 1.25j]) in (True, False)

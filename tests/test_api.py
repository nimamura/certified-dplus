"""Tests for the top-level interface."""

from fractions import Fraction

import numpy as np
import pytest

from certified_dplus import ConsistencyResult, consistency_test
from certified_dplus.forward1d import c_layered
from data1d import larsen_si

OMEGA = np.logspace(-4, 0, 12)
LAYERED = np.array([c_layered(w, [0.01, 1.0], [2e3, 1e3], 0.05) for w in OMEGA])


def test_exact_layered_response_is_consistent():
    res = consistency_test(OMEGA, LAYERED, 0.01 * np.abs(LAYERED))
    assert isinstance(res, ConsistencyResult)
    assert res.chi2_upper < 1e-10
    assert res.chi2_lower <= res.chi2_upper
    assert res.rms_upper < 1e-5


def test_bounds_bracket_the_minimum_for_real_data():
    omega, c, s = larsen_si(range(1, 8))
    res = consistency_test(omega, c, s)
    assert isinstance(res.chi2_lower, Fraction) and isinstance(res.chi2_upper, Fraction)
    assert float(res.chi2_lower) == pytest.approx(7.41227233, rel=1e-8)
    assert 0 < res.chi2_upper - res.chi2_lower < 1e-9
    assert res.rms_lower == pytest.approx((7.41227233 / 14) ** 0.5, rel=1e-7)


def test_certificate_can_be_rechecked_independently():
    from certified_dplus.certificate import lower_bound

    omega, c, s = larsen_si(range(1, 8))
    res = consistency_test(omega, c, s)
    assert lower_bound(omega, c, s, res.certificate) == res.chi2_lower


def test_relative_errors_are_the_default_weights():
    a = consistency_test(OMEGA, LAYERED * (1 + 0.05 * np.sin(np.arange(12))))
    b = consistency_test(OMEGA, LAYERED * (1 + 0.05 * np.sin(np.arange(12))), sigma=None)
    assert a.chi2_lower == b.chi2_lower > 0

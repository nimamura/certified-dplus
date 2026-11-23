"""Tests for the exact dual certificate of the D+ misfit.

Lemma: if Re(conj(y_j) / (lam + i omega_j)) / s_j summed over j is <= 0 for all
lam >= 0 and sum_j Re(y_j) / s_j <= 0, then every 1-D model has
chi^2 >= 2 Re sum conj(y_j) d_j - sum |y_j|^2 with d_j = c_j / s_j.
"""

from fractions import Fraction as F
from itertools import combinations

import numpy as np
import pytest

from certified_dplus.certificate import CertificateError, certify, chi2_exact, lower_bound
from certified_dplus.dplus import fit_dplus
from certified_dplus.exact1d import is_consistent
from data1d import LARSEN_CONSISTENT_TRIPLES, exact_response, larsen_exact, larsen_si
from test_dplus import noisy_layered_data

OMEGA = [F(1, 2), F(2)]
C = [(F(3), F(-2)), (F(1), F(-2))]
S = [F(1), F(1, 2)]


def test_zero_vector_is_a_trivial_certificate():
    assert lower_bound(OMEGA, C, S, [(F(0), F(0))] * 2) == 0


def test_interior_vector_gives_the_stated_bound():
    # y = (-1, +1) is strictly feasible; bound = 2 y.d - |y|^2 with d = c / s
    y = [(F(-1), F(1)), (F(-1), F(1))]
    d = [(F(3), F(-2)), (F(2), F(-4))]
    expected = 2 * sum(-dr + di for dr, di in d) - 4
    assert lower_bound(OMEGA, C, S, y) == expected


@pytest.mark.parametrize(
    "y",
    [
        [(F(1), F(0)), (F(0), F(0))],  # violates the a0 column (and large lambda)
        [(F(0), F(-1)), (F(0), F(0))],  # violates at lambda = 0
        [(F(-1), F(1)), (F(8), F(1))],  # negative at lambda = 0, positive for large lambda
    ],
)
def test_infeasible_vectors_are_rejected(y):
    with pytest.raises(CertificateError):
        lower_bound(OMEGA, C, S, y)


def test_touching_zero_is_accepted_but_crossing_is_not():
    # single frequency omega = 1, s = 1: g(lam) = (yR lam - yI) / (lam^2 + 1)
    assert lower_bound([F(1)], [(F(1), F(-1))], [F(1)], [(F(0), F(0))]) == 0
    assert lower_bound([F(1)], [(F(1), F(-1))], [F(1)], [(F(-1), F(0))]) == 2 * F(-1) - 1
    with pytest.raises(CertificateError):
        lower_bound([F(1)], [(F(1), F(-1))], [F(1)], [(F(-1), F(-1))])


def test_chi2_exact_matches_definition():
    a0, a, lam = F(1, 3), [F(2), F(1)], [F(0), F(3, 2)]
    model = exact_response(OMEGA, a0, a, lam)
    expected = sum(
        ((cr - mr) ** 2 + (ci - mi) ** 2) / s**2
        for (cr, ci), (mr, mi), s in zip(C, model, S, strict=True)
    )
    assert chi2_exact(OMEGA, C, S, a0, a, lam) == expected


def test_chi2_exact_rejects_non_dplus_parameters():
    with pytest.raises(ValueError):
        chi2_exact(OMEGA, C, S, F(-1), [F(1)], [F(1)])
    with pytest.raises(ValueError):
        chi2_exact(OMEGA, C, S, F(0), [F(-1)], [F(1)])
    with pytest.raises(ValueError):
        chi2_exact(OMEGA, C, S, F(0), [F(1)], [F(-1)])


def test_soundness_against_random_models():
    omega, c, s = noisy_layered_data(n=8, seed=2)
    fit = fit_dplus(omega, c, s)
    y, bound = certify(omega, c, s, fit.residual)
    assert bound > 0
    rng = np.random.default_rng(4)
    for _ in range(200):
        k = rng.integers(1, 6)
        value = chi2_exact(
            omega, c, s, 10 ** rng.uniform(0, 5), 10 ** rng.uniform(0, 6, k), rng.uniform(0, 3, k)
        )
        assert value >= bound


def test_certified_gap_below_one_percent_for_noisy_data():
    omega, c, s = noisy_layered_data()  # 50 frequencies, 5 % noise
    fit = fit_dplus(omega, c, s)
    y, lower = certify(omega, c, s, fit.residual)
    upper = chi2_exact(omega, c, s, fit.model.a0, fit.model.a, fit.model.lam)
    assert lower == lower_bound(omega, c, s, y)  # independent re-check of the stored vector
    assert 0 < lower <= upper
    assert float(upper) == pytest.approx(fit.chi2, rel=1e-9)
    assert float((upper - lower) / upper) < 0.01


def test_full_larsen_set_has_certified_positive_misfit():
    omega, c, s = larsen_si(range(1, 8))
    fit = fit_dplus(omega, c, s)
    _, lower = certify(omega, c, s, fit.residual)
    upper = chi2_exact(omega, c, s, fit.model.a0, fit.model.a, fit.model.lam)
    assert 0 < lower <= upper
    assert float((upper - lower) / upper) < 0.01


def test_certificate_agrees_with_matrix_conditions_on_larsen_subsets():
    """Inconsistent subsets get a certified chi^2 > 0; consistent ones are fitted to ~0."""
    subsets = [s for k in (2, 3, 4) for s in combinations(range(1, 8), k)]
    assert len(subsets) == 21 + 35 + 35
    n_bad = 0
    for sub in subsets:
        consistent = is_consistent(*larsen_exact(sub))
        if len(sub) == 3:
            assert consistent == (sub in LARSEN_CONSISTENT_TRIPLES)
        omega, c, s = larsen_si(sub)
        fit = fit_dplus(omega, c, s)
        _, lower = certify(omega, c, s, fit.residual)
        if consistent:
            assert fit.chi2 < 1e-12, sub
            assert lower <= 0, sub
        else:
            n_bad += 1
            assert lower > 0, sub
            assert float(lower) == pytest.approx(fit.chi2, rel=0.01), sub
    assert n_bad == 1 + 18 + 35


def test_descartes_bisection_agrees_with_sturm():
    """The fast root test and the Sturm reference agree on random square-free polynomials."""
    import random

    from flint import fmpq_poly

    from certified_dplus.certificate import _has_no_positive_root, _sturm_count_positive_roots

    rng = random.Random(7)
    x = fmpq_poly([0, 1])
    seen = set()
    for _ in range(300):
        p = fmpq_poly([1])
        roots = set()
        for _ in range(rng.randint(1, 4)):  # distinct rational real roots, some positive
            roots.add(F(rng.randint(-40, 60), rng.choice([1, 3, 7, 1000])))
        roots.discard(F(0))
        for r in roots:
            p *= x - fmpq_poly([r.numerator]) / r.denominator
        for _ in range(rng.randint(0, 3)):  # complex pairs, possibly very close to the axis
            re, im = F(rng.randint(-5, 50), 7), F(1, rng.choice([1, 10, 10**6, 10**12]))
            p *= x * x - 2 * x * fmpq_poly([re.numerator]) / re.denominator + fmpq_poly(
                [(re * re + im * im).numerator]
            ) / (re * re + im * im).denominator
        expected = _sturm_count_positive_roots(p) == 0
        assert expected == (not any(r > 0 for r in roots))
        assert _has_no_positive_root(p) == expected
        seen.add(expected)
    assert seen == {True, False}

"""Tests for the D+ fit, partial-fraction <-> sheet conversion and exact interpolants."""

import numpy as np
import pytest

from certified_dplus.conventions import MU0
from certified_dplus.dplus import (
    PartialFractions,
    fit_dplus,
    interpolate_model_one,
    interpolate_model_two,
    kkt_function,
    to_sheets,
)
from certified_dplus.forward1d import c_layered, c_sheets
from data1d import V3_DATA, V3_MODEL_ONE, V3_MODEL_TWO, V3_OMEGA

SHEETS = dict(depth=[2.0e3, 3.0e4, 1.5e5], conductance=[300.0, 2.0e3, 5.0e4])


def noisy_layered_data(n=50, rel=0.05, seed=0):
    omega = 2 * np.pi / np.logspace(0, 4, n)
    c = c_layered(omega, [0.01, 0.5, 0.002], [3e3, 2e3, 6e4], 0.3)
    rng = np.random.default_rng(seed)
    s = rel * np.abs(c)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * s
    return omega, c + noise, s


# --- partial fractions and sheets -------------------------------------------------


def test_partial_fraction_response():
    pf = PartialFractions(a0=100.0, a=np.array([2.0, 5.0]), lam=np.array([0.0, 0.3]))
    w = np.array([0.1, 1.0])
    expected = 100.0 + 2.0 / (1j * w) + 5.0 / (0.3 + 1j * w)
    assert pf.response(w) == pytest.approx(expected, rel=1e-14)


def test_single_sheet_over_insulator_to_sheets():
    d, tau = 3.0e3, 500.0
    m = to_sheets(PartialFractions(a0=d, a=np.array([1 / (MU0 * tau)]), lam=np.array([0.0])))
    assert m.depth == pytest.approx([d]) and m.conductance == pytest.approx([tau])
    assert m.perfect_conductor_depth is None


def test_surface_sheet_over_perfect_conductor_to_sheets():
    tau, big_d = 800.0, 4.0e4
    pf = PartialFractions(
        a0=0.0, a=np.array([1 / (MU0 * tau)]), lam=np.array([1 / (MU0 * tau * big_d)])
    )
    m = to_sheets(pf)
    assert m.depth == pytest.approx([0.0], abs=1e-9)
    assert m.conductance == pytest.approx([tau])
    assert m.perfect_conductor_depth == pytest.approx(big_d)


@pytest.mark.parametrize("with_zero_pole", [False, True])
@pytest.mark.parametrize("a0", [0.0, 750.0])
def test_to_sheets_round_trip(with_zero_pole, a0):
    rng = np.random.default_rng(3)
    lam = np.sort(10 ** rng.uniform(-4, 1, 5))
    if with_zero_pole:
        lam[0] = 0.0
    pf = PartialFractions(a0=a0, a=10 ** rng.uniform(1, 3, 5), lam=lam)
    m = to_sheets(pf)
    assert np.all(np.diff(m.depth) > 0) and np.all(np.asarray(m.conductance) > 0)
    assert (m.perfect_conductor_depth is None) == with_zero_pole
    w = np.logspace(-5, 2, 23)
    c = c_sheets(w, m.depth, m.conductance, perfect_conductor_depth=m.perfect_conductor_depth)
    assert c == pytest.approx(pf.response(w), rel=1e-10)


# --- V3: Weidelt's two extremal interpolants ----------------------------------------


def test_v3_model_one_from_data():
    pf = interpolate_model_one(V3_OMEGA, V3_DATA)
    assert pf.a0 == 0.0 and np.all(pf.lam > 0) and np.all(pf.a > 0)
    assert pf.response(V3_OMEGA) == pytest.approx(V3_DATA, rel=1e-10)
    m = to_sheets(pf)
    assert m.depth == pytest.approx(V3_MODEL_ONE["depth"], rel=1e-5, abs=1.0)
    assert m.conductance == pytest.approx(V3_MODEL_ONE["conductance"], rel=1e-5)
    assert m.perfect_conductor_depth == pytest.approx(V3_MODEL_ONE["pc"], rel=1e-5)


def test_v3_model_two_from_data():
    pf = interpolate_model_two(V3_OMEGA, V3_DATA)
    assert pf.a0 > 0 and pf.lam.min() == 0.0 and np.all(pf.a > 0)
    assert pf.response(V3_OMEGA) == pytest.approx(V3_DATA, rel=1e-10)
    m = to_sheets(pf)
    assert m.depth == pytest.approx(V3_MODEL_TWO["depth"], rel=1e-5)
    assert m.conductance == pytest.approx(V3_MODEL_TWO["conductance"], rel=1e-5)
    assert m.perfect_conductor_depth is None


# --- D+ fit -----------------------------------------------------------------------


@pytest.mark.parametrize("pc", [None, 4.0e5])
def test_fit_of_exact_dplus_data_is_essentially_zero(pc):
    omega = 2 * np.pi / np.logspace(1, 5, 20)
    c = c_sheets(omega, **SHEETS, perfect_conductor_depth=pc)
    s = 0.01 * np.abs(c)
    fit = fit_dplus(omega, c, s)
    assert fit.chi2 < 1e-12
    assert fit.model.response(omega) == pytest.approx(c, rel=1e-7)


def test_fit_of_exact_layered_data_is_essentially_zero():
    omega = 2 * np.pi / np.logspace(0, 4, 30)
    c = c_layered(omega, [0.01, 0.5, 0.002], [3e3, 2e3, 6e4], 0.3)
    fit = fit_dplus(omega, c, 0.01 * np.abs(c))
    assert fit.chi2 < 1e-10


def test_fit_of_noisy_data_satisfies_kkt_and_is_a_dplus_model():
    omega, c, s = noisy_layered_data()
    fit = fit_dplus(omega, c, s)
    pf = fit.model
    assert pf.a0 >= 0 and np.all(pf.a > 0) and np.all(pf.lam >= 0)
    assert pf.a.size <= 2 * omega.size
    res = (c - pf.response(omega)) / s
    assert fit.residual == pytest.approx(res, rel=1e-12, abs=1e-12)
    assert fit.chi2 == pytest.approx(np.sum(np.abs(res) ** 2), rel=1e-12)
    assert 10.0 < fit.chi2 < 200.0  # 100 real data, 5 % noise
    # optimality: g(lambda) = A(lambda)^T r <= 0 everywhere, and for the a0 column
    lam = np.concatenate([[0.0], np.logspace(-9, 6, 4000)])
    g = kkt_function(omega, s, fit.residual, lam)
    scale = np.abs(kkt_function(omega, s, np.abs(fit.residual), lam))
    assert np.all(g <= 1e-7 * scale)
    assert np.sum(fit.residual.real / s) <= 1e-7 * np.sum(np.abs(fit.residual.real) / s)
    # the fitted model is realisable as sheets
    m = to_sheets(pf)
    assert np.all(np.asarray(m.conductance) > 0)


def test_fit_beats_any_other_dplus_model():
    omega, c, s = noisy_layered_data(n=12, seed=5)
    fit = fit_dplus(omega, c, s)
    rng = np.random.default_rng(11)
    for _ in range(50):
        trial = PartialFractions(
            a0=fit.model.a0 * rng.uniform(0.5, 1.5),
            a=fit.model.a * rng.uniform(0.8, 1.2, fit.model.a.size),
            lam=fit.model.lam * rng.uniform(0.8, 1.2, fit.model.lam.size),
        )
        chi2 = np.sum(np.abs((c - trial.response(omega)) / s) ** 2)
        assert chi2 >= fit.chi2 * (1 - 1e-9)


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_dplus([1.0, 2.0], [1 - 1j], [1.0, 1.0])
    with pytest.raises(ValueError):
        fit_dplus([1.0], [1 - 1j], [0.0])
    with pytest.raises(ValueError):
        fit_dplus([-1.0], [1 - 1j], [1.0])

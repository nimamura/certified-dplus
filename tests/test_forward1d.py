"""Tests for the 1-D forward problem.

References are computed with mpmath at 50 digits from closed forms written
independently of the recursion used in the implementation (exponential /
reflection-coefficient forms and explicit continued fractions). The gate G0
requires relative error <= 1e-10.
"""

import math

import mpmath as mp
import numpy as np
import pytest

from certified_dplus import conventions as cv
from certified_dplus.forward1d import c_halfspace, c_layered, c_sheets

mp.mp.dps = 50
RTOL = 1e-10
MU0_MP = 4 * mp.pi * mp.mpf(10) ** -7

# periods from 1 ms to ~3 h
OMEGAS = 2 * math.pi / np.logspace(-3, 4, 29)


def rel_err(value, ref):
    return float(abs(mp.mpc(complex(value)) - ref) / abs(ref))


def k_mp(omega, sigma):
    return mp.sqrt(1j * mp.mpf(float(omega)) * MU0_MP * mp.mpf(sigma))


# --- V1: uniform half-space -------------------------------------------------


@pytest.mark.parametrize("sigma", [1e-4, 1e-2, 1.0, 10.0])
def test_halfspace_matches_analytic(sigma):
    c = c_halfspace(OMEGAS, sigma)
    for w, ci in zip(OMEGAS, c, strict=True):
        assert rel_err(ci, 1 / k_mp(w, sigma)) < RTOL
    assert cv.phase_deg(c) == pytest.approx(np.full(OMEGAS.size, 45.0), abs=1e-9)


def test_layered_with_no_layers_is_halfspace():
    assert c_layered(OMEGAS, [], [], 0.03) == pytest.approx(c_halfspace(OMEGAS, 0.03), rel=1e-14)


def test_scalar_omega_returns_scalar_complex():
    c = c_halfspace(1.0, 0.01)
    assert np.ndim(c) == 0


# --- two-layer analytic solution ---------------------------------------------


def two_layer_ref(omega, s1, h, s2):
    """c = (1/k1) (1 + r e^{-2 k1 h}) / (1 - r e^{-2 k1 h}), r = (k1 - k2)/(k1 + k2)."""
    k1, k2 = k_mp(omega, s1), k_mp(omega, s2)
    r = (k1 - k2) / (k1 + k2)
    e = mp.exp(-2 * k1 * mp.mpf(h))
    return (1 + r * e) / (1 - r * e) / k1


@pytest.mark.parametrize(
    ("s1", "h", "s2"),
    [
        (0.01, 1000.0, 1.0),  # resistive over conductive
        (1.0, 1000.0, 0.01),  # conductive over resistive
        (1e-4, 30000.0, 10.0),  # extreme contrast of the tentative range
        (10.0, 50.0, 1e-4),
        (0.1, 1.0, 0.1001),  # nearly uniform, thin layer
        (3.3, 4000.0, 0.01),  # deep ocean over resistive lithosphere
    ],
)
def test_two_layer_matches_analytic(s1, h, s2):
    c = c_layered(OMEGAS, [s1], [h], s2)
    worst = max(
        rel_err(ci, two_layer_ref(w, s1, h, s2)) for w, ci in zip(OMEGAS, c, strict=True)
    )
    assert worst < RTOL


def test_layer_over_perfect_conductor_and_insulator():
    s1, h = 0.05, 2500.0
    c_pc = c_layered(OMEGAS, [s1], [h], math.inf)
    c_ins = c_layered(OMEGAS, [s1], [h], 0.0)
    for w, a, b in zip(OMEGAS, c_pc, c_ins, strict=True):
        k = k_mp(w, s1)
        assert rel_err(a, mp.tanh(k * h) / k) < RTOL
        assert rel_err(b, mp.coth(k * h) / k) < RTOL


def test_insulating_layer_shifts_c_by_its_thickness():
    # sigma = 0 layer: c_top = c_bottom + h
    c = c_layered(OMEGAS, [0.0], [1234.5], 0.2)
    assert c == pytest.approx(c_halfspace(OMEGAS, 0.2) + 1234.5, rel=1e-13)


def test_three_layer_matches_independent_transfer_matrix():
    sig, thick, base = [0.02, 2.0, 0.001], [800.0, 300.0, 20000.0], 0.5

    def ref(omega):
        # propagate (E, E') upward with 2x2 matrices, starting from E = 1, E' = -k below
        kb = k_mp(omega, base)
        vec = mp.matrix([1, -kb])
        for s, h in zip(reversed(sig), reversed(thick), strict=True):
            k = k_mp(omega, s)
            m = mp.matrix(
                [[mp.cosh(k * h), -mp.sinh(k * h) / k], [-k * mp.sinh(k * h), mp.cosh(k * h)]]
            )
            vec = m * vec
        return -vec[0] / vec[1]

    c = c_layered(OMEGAS, sig, thick, base)
    assert max(rel_err(ci, ref(w)) for w, ci in zip(OMEGAS, c, strict=True)) < RTOL


def test_split_layer_is_invariant():
    a = c_layered(OMEGAS, [0.3], [700.0], 0.004)
    b = c_layered(OMEGAS, [0.3, 0.3, 0.3], [100.0, 250.0, 350.0], 0.004)
    assert a == pytest.approx(b, rel=1e-13)


def test_no_overflow_for_electrically_thick_layer():
    # k h ~ 6e3: cosh would overflow, tanh form must not
    c = c_layered(2 * math.pi * 1e4, [10.0], [1e4], 1e-4)
    assert np.isfinite(c)
    assert rel_err(c, 1 / k_mp(2 * math.pi * 1e4, 10.0)) < RTOL


def test_layered_response_is_physical():
    c = c_layered(OMEGAS, [0.01, 1.0, 1e-3], [500.0, 2000.0, 1e4], 0.1)
    assert np.all(c.real > 0) and np.all(c.imag < 0)
    phase = cv.phase_deg(c)
    assert np.all((phase > 0) & (phase < 90))


def test_layered_rejects_bad_input():
    with pytest.raises(ValueError):
        c_layered(1.0, [0.1, 0.2], [100.0], 0.1)  # length mismatch
    with pytest.raises(ValueError):
        c_layered(1.0, [-0.1], [100.0], 0.1)  # negative conductivity
    with pytest.raises(ValueError):
        c_layered(1.0, [0.1], [-5.0], 0.1)  # negative thickness
    with pytest.raises(ValueError):
        c_layered(-1.0, [0.1], [5.0], 0.1)  # omega must be positive


# --- thin-sheet stacks (D+ models) --------------------------------------------


def sheets_ref(omega, depths, taus, pc_depth=None):
    """Explicit continued fraction, bottom up, at 50 digits."""
    w = mp.mpf(float(omega))
    if pc_depth is None:
        inv_c = mp.mpf(0)  # insulator below: 1/c = 0
        z = mp.mpf(depths[-1])
    else:
        z = mp.mpf(pc_depth)
        inv_c = None  # c = 0 at the perfect conductor
    for d, tau in zip(reversed(depths), reversed(taus), strict=True):
        gap = z - mp.mpf(d)
        if inv_c is None:
            c = gap
        elif inv_c == 0:
            c = mp.inf
        else:
            c = 1 / inv_c + gap
        inv_c = (0 if c == mp.inf else 1 / c) + 1j * w * MU0_MP * mp.mpf(tau)
        z = mp.mpf(d)
    return 1 / inv_c + z


def test_single_sheet_over_insulator():
    # c = d + 1/(i omega mu0 tau)
    d, tau = 3000.0, 500.0
    c = c_sheets(OMEGAS, [d], [tau])
    for w, ci in zip(OMEGAS, c, strict=True):
        assert rel_err(ci, d + 1 / (1j * mp.mpf(float(w)) * MU0_MP * tau)) < RTOL


def test_sheet_stack_matches_continued_fraction():
    depths, taus = [0.0, 1.5e3, 4.0e4, 2.5e5], [20.0, 800.0, 3.0e3, 1.0e5]
    for pc in (None, 6.0e5):
        c = c_sheets(OMEGAS, depths, taus, perfect_conductor_depth=pc)
        worst = max(
            rel_err(ci, sheets_ref(w, depths, taus, pc)) for w, ci in zip(OMEGAS, c, strict=True)
        )
        assert worst < RTOL


def test_sheet_is_limit_of_thin_layer():
    tau, d = 2000.0, 5000.0
    w = 2 * math.pi / 100.0
    h = 1e-3
    thin = c_layered(w, [0.0, tau / h], [d, h], 0.0)
    assert thin == pytest.approx(c_sheets(w, [d], [tau]), rel=1e-6)


def test_sheets_reject_bad_input():
    with pytest.raises(ValueError):
        c_sheets(1.0, [100.0, 50.0], [1.0, 1.0])  # depths not increasing
    with pytest.raises(ValueError):
        c_sheets(1.0, [100.0], [-1.0])  # negative conductance
    with pytest.raises(ValueError):
        c_sheets(1.0, [100.0], [1.0], perfect_conductor_depth=50.0)  # conductor above sheet
    with pytest.raises(ValueError):
        c_sheets(1.0, [-1.0], [1.0])  # sheet above the surface


# --- V3: two-frequency extremal models of Weidelt (1985) ------------------------
# Values as printed in Weidelt (1985); given to 6
# digits, data to 3 digits, so the comparison tolerance is loose.

V3_OMEGA = cv.omega_from_cpd(np.array([1.0, 4.0]))
V3_DATA = np.array([575.0 - 260.0j, 290.0 - 275.0j]) * 1e3


def test_w85_model_one_reproduces_data():
    c = c_sheets(
        V3_OMEGA, [0.0, 523.675e3], [4.09466e3, 3.77586e4], perfect_conductor_depth=783.023e3
    )
    assert c == pytest.approx(V3_DATA, rel=2e-5)


def test_w85_model_two_reproduces_data():
    c = c_sheets(V3_OMEGA, [105.947e3, 699.673e3], [6.66233e3, 1.01835e5])
    assert c == pytest.approx(V3_DATA, rel=2e-5)

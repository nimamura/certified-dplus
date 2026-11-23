"""Tests for the conventions module (docs/conventions.md)."""

import math

import numpy as np
import pytest

from certified_dplus import conventions as cv


def test_mu0_is_classical_defined_value():
    assert cv.MU0 == 4e-7 * math.pi


def test_omega_from_period_and_frequency():
    assert cv.omega_from_period(2.0) == pytest.approx(math.pi, rel=1e-15)
    assert cv.omega_from_frequency(0.5) == pytest.approx(math.pi, rel=1e-15)
    # cycles per day, used by W85 / W95
    assert cv.omega_from_cpd(1.0) == pytest.approx(2 * math.pi / 86400.0, rel=1e-15)


def test_impedance_is_i_omega_mu0_c():
    omega, c = 0.3, 1200.0 - 700.0j
    assert cv.impedance(omega, c) == pytest.approx(1j * omega * cv.MU0 * c, rel=1e-15)


def test_halfspace_c_gives_true_resistivity_and_45_degrees():
    omega, sigma = 2 * math.pi / 300.0, 0.02
    c = 1.0 / np.sqrt(1j * omega * cv.MU0 * sigma)
    assert c.real > 0 and c.imag < 0  # c = g - ih
    assert cv.apparent_resistivity(omega, c) == pytest.approx(1.0 / sigma, rel=1e-13)
    assert cv.apparent_conductivity(omega, c) == pytest.approx(sigma, rel=1e-13)
    assert cv.phase_deg(c) == pytest.approx(45.0, abs=1e-12)


def test_phase_limits():
    # c real (thin sheet over insulator limit is c -> -i*h, perfect conductor at depth is c real)
    assert cv.phase_deg(500.0 + 0.0j) == pytest.approx(90.0, abs=1e-12)
    assert cv.phase_deg(0.0 - 500.0j) == pytest.approx(0.0, abs=1e-12)


def test_helpers_vectorise():
    omega = np.array([0.1, 1.0, 10.0])
    c = 1.0 / np.sqrt(1j * omega * cv.MU0 * 0.5)
    assert cv.apparent_resistivity(omega, c) == pytest.approx(np.full(3, 2.0), rel=1e-13)
    assert cv.phase_deg(c) == pytest.approx(np.full(3, 45.0), abs=1e-12)

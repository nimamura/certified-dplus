"""Sign, unit and time-dependence conventions. Single source: docs/conventions.md.

SI units, time dependence exp(+i omega t), z downward with the surface at z = 0.
Response c(omega) = E_x / (i omega B_y) = -E(0)/E'(0) in metres, c = g - i h.
"""

import math

import numpy as np

# Classical defined value of the vacuum permeability [H/m].
MU0 = 4e-7 * math.pi

SECONDS_PER_DAY = 86400.0


def omega_from_period(period):
    """Angular frequency [rad/s] from period [s]."""
    return 2.0 * math.pi / np.asarray(period, dtype=float)


def omega_from_frequency(frequency):
    """Angular frequency [rad/s] from frequency [Hz]."""
    return 2.0 * math.pi * np.asarray(frequency, dtype=float)


def omega_from_cpd(cycles_per_day):
    """Angular frequency [rad/s] from cycles per day."""
    return omega_from_frequency(np.asarray(cycles_per_day, dtype=float) / SECONDS_PER_DAY)


def impedance(omega, c):
    """Impedance Z = E_x / H_y = i omega mu0 c [Ohm]."""
    return 1j * np.asarray(omega) * MU0 * np.asarray(c)


def apparent_resistivity(omega, c):
    """rho_a = omega mu0 |c|^2 [Ohm m]."""
    return np.asarray(omega) * MU0 * np.abs(c) ** 2


def apparent_conductivity(omega, c):
    """sigma_a = 1 / (omega mu0 |c|^2) [S/m]."""
    return 1.0 / apparent_resistivity(omega, c)


def phase_deg(c):
    """Impedance phase arg Z = 90 deg + arg c, in degrees (45 for a half-space)."""
    return 90.0 + np.degrees(np.angle(c))

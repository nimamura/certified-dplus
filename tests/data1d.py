"""Shared 1-D test data transcribed from the cited papers."""

from fractions import Fraction

import numpy as np

from certified_dplus import conventions as cv

# Yee & Paulson (1988) Table 1: Larsen (1975) Hawaii responses.
# omega [1/ks], Re c [km], Im c [km], s [km]; kept as decimal strings so that
# the exact tests see exactly the printed numbers.
LARSEN = [
    ("0.03636", "332.8", "-176.0", "2.75"),
    ("0.1091", "236.5", "-145.8", "2.75"),
    ("0.1818", "190.9", "-140.3", "2.20"),
    ("0.2545", "159.5", "-132.0", "1.57"),
    ("0.3272", "136.9", "-120.7", "1.83"),
    ("0.4000", "119.0", "-109.0", "1.75"),
    ("0.4727", "103.7", "-98.80", "1.69"),
]

# Consistent triples listed on p.289-290 (1-based indices); all other triples fail.
LARSEN_CONSISTENT_TRIPLES = (
    {(1, 2, k) for k in (3, 4, 5, 6, 7)}
    | {(1, 3, k) for k in (4, 5, 6, 7)}
    | {(1, 4, k) for k in (5, 6, 7)}
    | {(1, 5, 6)}
    | {(2, 3, k) for k in (4, 5, 6, 7)}
)


def larsen_exact(indices):
    """(omega, c) as exact rationals in the printed units for 1-based indices."""
    rows = [LARSEN[i - 1] for i in indices]
    omega = [Fraction(r[0]) for r in rows]
    c = [(Fraction(r[1]), Fraction(r[2])) for r in rows]
    return omega, c


def larsen_si(indices):
    """(omega [rad/s], c [m], s [m]) as float arrays for 1-based indices."""
    rows = [LARSEN[i - 1] for i in indices]
    omega = np.array([float(r[0]) for r in rows]) * 1e-3
    c = np.array([complex(float(r[1]), float(r[2])) for r in rows]) * 1e3
    s = np.array([float(r[3]) for r in rows]) * 1e3
    return omega, c, s


# Weidelt (1985) two-frequency example (target V3).
V3_OMEGA = cv.omega_from_cpd(np.array([1.0, 4.0]))
V3_DATA = np.array([575.0 - 260.0j, 290.0 - 275.0j]) * 1e3
V3_MODEL_ONE = dict(depth=[0.0, 523.675e3], conductance=[4.09466e3, 3.77586e4], pc=783.023e3)
V3_MODEL_TWO = dict(depth=[105.947e3, 699.673e3], conductance=[6.66233e3, 1.01835e5], pc=None)


def exact_response(omega, a0, a, lam):
    """Exact rational (re, im) of a0 + sum a_n / (lam_n + i omega) for rational inputs."""
    out = []
    for w in omega:
        re, im = Fraction(a0), Fraction(0)
        for an, ln in zip(a, lam, strict=True):
            den = ln * ln + w * w
            re += an * ln / den
            im -= an * w / den
        out.append((re, im))
    return out

"""1-D consistency test for magnetotelluric responses with a verified misfit bound.

    result = consistency_test(omega, c, sigma)

fits the best 1-D response (Parker's D+) and returns two exact rational numbers
that bracket the smallest achievable chi^2: an upper value from an explicit
model and a lower value from a dual certificate checked in rational arithmetic.
Conventions (e^{+i omega t}, SI, c = E / (i omega B)): docs/conventions.md.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from certified_dplus.certificate import certify, chi2_exact
from certified_dplus.dplus import DPlusFit, fit_dplus

__version__ = "0.1.0"
__all__ = ["ConsistencyResult", "consistency_test"]


@dataclass(frozen=True)
class ConsistencyResult:
    """Bracket of the smallest chi^2 over all 1-D responses, for the data as stored."""

    chi2_lower: Fraction  # proven: no 1-D response has a smaller chi^2
    chi2_upper: Fraction  # chi^2 of the D+ model in `fit`, evaluated exactly
    n: int  # number of frequencies (2 n real data)
    fit: DPlusFit
    certificate: list  # dual vector y as (re, im) Fractions; recheck with certificate.lower_bound

    @property
    def rms_lower(self):
        """sqrt(chi2_lower / 2n): rms misfit per real datum, lower side."""
        return math.sqrt(self.chi2_lower / (2 * self.n))

    @property
    def rms_upper(self):
        return math.sqrt(self.chi2_upper / (2 * self.n))


def consistency_test(omega, c, sigma=None):
    """Smallest misfit of any 1-D conductivity profile to the responses c(omega).

    omega: angular frequencies [rad/s], positive and distinct.
    c:     complex responses [m], c = E / (i omega B) = Z / (i omega mu0), Im c < 0.
    sigma: standard error of c at each frequency [m] (one value for the real and
           the imaginary part); None means equal relative errors, sigma = |c|.

    chi^2 = sum_j |c_j - c_1D(omega_j)|^2 / sigma_j^2.
    """
    w = np.asarray(omega, dtype=float)
    data = np.asarray(c, dtype=complex)
    s = np.abs(data) if sigma is None else np.asarray(sigma, dtype=float)
    fit = fit_dplus(w, data, s)
    y, lower = certify(w, data, s, fit.residual)
    upper = chi2_exact(w, data, s, fit.model.a0, fit.model.a, fit.model.lam)
    return ConsistencyResult(max(lower, Fraction(0)), upper, w.size, fit, y)

"""1-D magnetotelluric forward problem for layered models and thin-sheet stacks.

The response c = -E/E' is continuous in z and is carried upward from the
bottom as a projective pair (n, d) with c = n / d, so that c = infinity
(insulating basement) and c = 0 (perfect conductor) need no special cases.

    uniform layer (sigma, h), k^2 = i omega mu0 sigma, T = tanh(k h) / k:
        n_top = n + T d
        d_top = d + k^2 T n
    thin sheet of conductance tau:
        d_top = d + i omega mu0 tau n

Conventions: docs/conventions.md.
"""

import math

import numpy as np

from certified_dplus.conventions import MU0

# Below this |k h| the series for tanh(kh)/k is used (also covers sigma = 0).
_SMALL_KH = 1e-4


def _as_omega(omega):
    w = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("omega must be positive and finite")
    return w


def _tanh_over_k(k, h):
    """tanh(k h) / k, finite and accurate as k -> 0 (value h)."""
    kh = k * h
    small = np.abs(kh) < _SMALL_KH
    safe_k = np.where(small, 1.0, k)
    kh2 = kh * kh
    series = h * (1.0 - kh2 / 3.0 + 2.0 * kh2 * kh2 / 15.0)
    return np.where(small, series, np.tanh(kh) / safe_k)


def _basement_pair(w, sigma):
    """(n, d) of a uniform half-space; sigma = 0 is an insulator, inf a perfect conductor."""
    if sigma < 0 or math.isnan(sigma):
        raise ValueError("basement conductivity must be >= 0")
    one = np.ones_like(w, dtype=complex)
    if sigma == 0:
        return one, 0.0 * one
    if math.isinf(sigma):
        return 0.0 * one, one
    return one, np.sqrt(1j * w * MU0 * sigma)


def _through_layer(w, n, d, sigma, h):
    k2 = 1j * w * MU0 * sigma
    t = _tanh_over_k(np.sqrt(k2), h)
    return n + t * d, d + k2 * t * n


def _ratio(n, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(d == 0, np.inf, n / np.where(d == 0, 1.0, d))
    return c[()] if c.ndim == 0 else c


def c_halfspace(omega, sigma):
    """Response of a uniform half-space, c = 1 / sqrt(i omega mu0 sigma) [m]."""
    return c_layered(omega, [], [], sigma)


def c_layered(omega, sigma, thickness, basement_sigma):
    """Surface response c(omega) [m] of uniform layers over a uniform half-space.

    sigma, thickness: conductivities [S/m] and thicknesses [m] of the layers, top first.
    basement_sigma: conductivity of the terminating half-space; 0 for an insulator,
    math.inf for a perfect conductor.
    """
    w = _as_omega(omega)
    sig = np.asarray(sigma, dtype=float)
    thick = np.asarray(thickness, dtype=float)
    if sig.ndim != 1 or sig.shape != thick.shape:
        raise ValueError("sigma and thickness must be 1-D sequences of equal length")
    if np.any(sig < 0) or not np.all(np.isfinite(sig)):
        raise ValueError("layer conductivities must be finite and >= 0")
    if np.any(thick < 0) or not np.all(np.isfinite(thick)):
        raise ValueError("layer thicknesses must be finite and >= 0")

    n, d = _basement_pair(w, float(basement_sigma))
    for s, h in zip(sig[::-1], thick[::-1], strict=True):
        n, d = _through_layer(w, n, d, s, h)
    return _ratio(n, d)


def c_sheets(omega, depth, conductance, perfect_conductor_depth=None):
    """Surface response c(omega) [m] of thin sheets separated by insulators (a D+ model).

    depth: sheet depths [m], strictly increasing, >= 0.
    conductance: sheet conductances [S], >= 0.
    perfect_conductor_depth: depth of a terminating perfect conductor, or None for an
    insulator below the deepest sheet.
    """
    w = _as_omega(omega)
    z = np.asarray(depth, dtype=float)
    tau = np.asarray(conductance, dtype=float)
    if z.ndim != 1 or z.shape != tau.shape:
        raise ValueError("depth and conductance must be 1-D sequences of equal length")
    if np.any(z < 0) or np.any(np.diff(z) <= 0) or not np.all(np.isfinite(z)):
        raise ValueError("sheet depths must be finite, >= 0 and strictly increasing")
    if np.any(tau < 0) or not np.all(np.isfinite(tau)):
        raise ValueError("sheet conductances must be finite and >= 0")

    if perfect_conductor_depth is None:
        n, d = _basement_pair(w, 0.0)
        below = z[-1] if z.size else 0.0
    else:
        below = float(perfect_conductor_depth)
        if z.size and below <= z[-1]:
            raise ValueError("perfect conductor must lie below the deepest sheet")
        if below < 0:
            raise ValueError("perfect conductor depth must be >= 0")
        n, d = _basement_pair(w, math.inf)

    for zi, ti in zip(z[::-1], tau[::-1], strict=True):
        n = n + (below - zi) * d  # insulating gap
        d = d + 1j * w * MU0 * ti * n  # sheet
        below = zi
    n = n + below * d  # insulator between the surface and the top sheet
    return _ratio(n, d)

"""D+ models: best fit (upper side of min chi^2), sheet conversion, exact interpolants.

A D+ response is c(omega) = a0 + sum_n a_n / (lam_n + i omega) with a0 >= 0,
a_n > 0, lam_n >= 0 (Parker 1980, eq. 13). The misfit is

    chi^2 = sum_j |c_j - c(omega_j)|^2 / s_j^2 .

The fit is a non-negative least-squares problem on a grid of poles that is
refined around the active poles and where the optimality function g(lam) is
positive, followed by a local polish. Its optimality is not taken on trust:
`certificate.certify` turns the residual into an exactly verified lower bound.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from flint import fmpq, fmpq_poly
from scipy.optimize import least_squares, nnls

from certified_dplus.conventions import MU0


@dataclass(frozen=True)
class PartialFractions:
    """c(omega) = a0 + sum_n a[n] / (lam[n] + i omega); a0 [m], a [m/s], lam [1/s]."""

    a0: float
    a: np.ndarray
    lam: np.ndarray

    def response(self, omega):
        w = np.asarray(omega, dtype=float)
        terms = self.a / (self.lam + 1j * w[..., None])
        return self.a0 + terms.sum(axis=-1)


@dataclass(frozen=True)
class SheetModel:
    """Thin sheets in insulators; perfect conductor at the given depth or insulator below."""

    depth: list
    conductance: list
    perfect_conductor_depth: float | None


@dataclass(frozen=True)
class DPlusFit:
    model: PartialFractions
    chi2: float
    residual: np.ndarray  # (c - c_model) / s, complex


# --- partial fractions -> sheets ---------------------------------------------------


def _q(x):
    f = Fraction(float(x))
    return fmpq(f.numerator, f.denominator)


def to_sheets(pf):
    """Sheet depths and conductances of a D+ response (continued fraction, Parker 1980 eq. 12).

    The expansion is carried out in exact rational arithmetic on the stored
    floats, so it is stable for any number of poles.
    """
    lam = [_q(x) for x in pf.lam]
    amp = [_q(x) for x in pf.a]
    if not lam:
        raise ValueError("a D+ model needs at least one pole")
    x = fmpq_poly([0, 1])
    q = fmpq_poly([1])
    for ln in lam:
        q *= x + ln
    p = fmpq_poly([0])
    for an, ln in zip(amp, lam, strict=True):
        p += an * (q // (x + ln))

    depth = Fraction(float(pf.a0))
    depths, taus = [], []
    while True:
        alpha = q.leading_coefficient() / p.leading_coefficient()  # mu0 * tau
        depths.append(float(depth))
        taus.append(float(Fraction(int(alpha.p), int(alpha.q))) / MU0)
        q_below = q - alpha * x * p
        if q_below.is_zero():
            return SheetModel(depths, taus, None)
        if q_below.degree() != p.degree():
            raise ValueError("not a D+ response (continued fraction breaks down)")
        gap = p.leading_coefficient() / q_below.leading_coefficient()
        if gap <= 0 or alpha <= 0:
            raise ValueError("not a D+ response (non-positive sheet parameter)")
        depth += Fraction(int(gap.p), int(gap.q))
        p_below = p - gap * q_below
        if p_below.is_zero():
            return SheetModel(depths, taus, float(depth))
        p, q = p_below, q_below


# --- exact interpolants of regular data (Weidelt 1986, eqs. 9-10) -----------------------


def _interpolate(omega, c, zero_pole):
    w = np.asarray(omega, dtype=float)
    data = np.asarray(c, dtype=complex)
    m = w.size
    w0, c0 = np.exp(np.mean(np.log(w))), np.mean(np.abs(data))
    s = 1j * w / w0
    d = data / c0
    # c = p / q with q monic of degree m; model II has q(0) = 0 and deg p = m
    n_p = m + 1 if zero_pole else m
    q_powers = range(1, m) if zero_pole else range(m)
    cols = [s**k for k in range(n_p)] + [-d * s**k for k in q_powers]
    mat = np.array(cols).T
    rhs = d * s**m
    sol = np.linalg.solve(
        np.vstack([mat.real, mat.imag]), np.concatenate([rhs.real, rhs.imag])
    )
    p = sol[:n_p]
    q = np.zeros(m + 1)
    q[m] = 1.0
    q[list(q_powers)] = sol[n_p:]
    a0 = p[m] if zero_pole else 0.0
    roots = np.roots(q[::-1])
    if zero_pole:
        roots = roots[np.argsort(np.abs(roots))]
        roots[0] = 0.0
    p_poly = np.polynomial.Polynomial(p) - a0 * np.polynomial.Polynomial(q)
    dq = np.polynomial.Polynomial(q).deriv()
    amp = p_poly(roots) / dq(roots)
    ok = np.all(np.abs(roots.imag) < 1e-9 * (1 + np.abs(roots))) and np.all(
        np.abs(amp.imag) < 1e-9 * np.abs(amp)
    )
    lam, amp = -roots.real, amp.real
    if not ok or a0 < 0 or np.any(lam < 0) or np.any(amp <= 0):
        raise ValueError("data have no regular 1-D interpretation")
    order = np.argsort(lam)
    return PartialFractions(float(a0 * c0), amp[order] * c0 * w0, lam[order] * w0)


def interpolate_model_one(omega, c):
    """Exact interpolant with a0 = 0 and M positive poles (surface sheet, perfect conductor)."""
    return _interpolate(omega, c, zero_pole=False)


def interpolate_model_two(omega, c):
    """Exact interpolant with a0 > 0 and a pole at lam = 0 (buried sheets, insulator below)."""
    return _interpolate(omega, c, zero_pole=True)


# --- D+ fit ------------------------------------------------------------------------------


def kkt_function(omega, sigma, residual, lam):
    """g(lam) = sum_j Re( conj(r_j) / (lam + i omega_j) ) / s_j; <= 0 at an optimal fit."""
    w = np.asarray(omega, dtype=float)
    r = np.asarray(residual, dtype=complex) / np.asarray(sigma, dtype=float)
    lam = np.asarray(lam, dtype=float)
    kern = 1.0 / (lam[..., None] + 1j * w)
    return (np.conj(r) * kern).real.sum(axis=-1)


def _columns(w, s, lam):
    """Real design matrix for [a0, poles...]; rows are Re then Im, weighted by 1/s."""
    kern = 1.0 / (lam[None, :] + 1j * w[:, None]) / s[:, None]
    const = np.concatenate([1.0 / s, np.zeros_like(s)])[:, None]
    return np.hstack([const, np.vstack([kern.real, kern.imag])])


def _solve(w, s, d, lam):
    mat = _columns(w, s, lam)
    norms = np.linalg.norm(mat, axis=0)
    x, _ = nnls(mat / norms, d, maxiter=50 * mat.shape[1])
    x = x / norms
    return x[0], x[1:]


def _chi2(w, s, c, a0, amp, lam):
    model = a0 + (amp / (lam + 1j * w[:, None])).sum(axis=1)
    res = (c - model) / s
    return float(np.sum(np.abs(res) ** 2)), res


def _merge(amp, lam, tol):
    """Merge positive poles closer than a relative distance tol (amplitude-weighted)."""
    order = np.argsort(lam)
    amp, lam = amp[order], lam[order]
    out_a, out_l = [amp[0]], [lam[0]]
    for a, ln in zip(amp[1:], lam[1:], strict=True):
        if out_l[-1] > 0 and ln <= out_l[-1] * (1 + tol):
            total = out_a[-1] + a
            out_l[-1] = np.exp((out_a[-1] * np.log(out_l[-1]) + a * np.log(ln)) / total)
            out_a[-1] = total
        else:
            out_a.append(a)
            out_l.append(ln)
    return np.array(out_a), np.array(out_l)


def _polish(w, s, c, a0, amp, lam):
    """Local least-squares polish of all parameters of the active set."""
    pos = lam > 0
    scale_a, scale_l = amp.copy(), np.where(pos, lam, 1.0)
    c_scale = float(np.mean(np.abs(c)))

    def unpack(x):
        a = x[1 : 1 + amp.size] * scale_a
        ln = np.where(pos, x[1 + amp.size :] * scale_l, 0.0)
        return x[0] * c_scale, a, ln

    def fun(x):
        b0, a, ln = unpack(x)
        model = b0 + (a / (ln + 1j * w[:, None])).sum(axis=1)
        res = (c - model) / s
        return np.concatenate([res.real, res.imag])

    x0 = np.concatenate([[a0 / c_scale], np.ones(amp.size), np.where(pos, 1.0, 0.0)])
    lower = np.zeros_like(x0)
    upper = np.full_like(x0, np.inf)
    upper[1 + amp.size :][~pos] = 1e-300  # poles at zero stay at zero
    try:
        sol = least_squares(
            fun, x0, bounds=(lower, upper), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200
        )
    except ValueError:
        return a0, amp, lam
    return unpack(sol.x)


def fit_dplus(omega, c, sigma, per_decade=30, refinements=12):
    """Best-fitting D+ model in the weighted 2-norm.

    omega [rad/s], c [m] complex, sigma [m] one standard error per frequency.
    """
    w = np.asarray(omega, dtype=float)
    data = np.asarray(c, dtype=complex)
    s = np.asarray(sigma, dtype=float)
    if not (w.ndim == 1 and w.shape == data.shape == s.shape and w.size):
        raise ValueError("omega, c and sigma must be 1-D arrays of equal length")
    if np.any(w <= 0) or np.unique(w).size != w.size:
        raise ValueError("frequencies must be positive and distinct")
    if np.any(s <= 0) or not np.all(np.isfinite(s)):
        raise ValueError("standard errors must be positive")
    d = np.concatenate([data.real / s, data.imag / s])

    lo, hi = np.log10(w.min()) - 4.0, np.log10(w.max()) + 4.0
    n_grid = int(round((hi - lo) * per_decade)) + 1
    grid = np.logspace(lo, hi, n_grid)
    dense = np.logspace(lo, hi, 4 * n_grid)
    step = (hi - lo) / (n_grid - 1)  # decades

    lam = np.concatenate([[0.0], grid])
    best = None
    for _ in range(refinements):
        a0, amp = _solve(w, s, d, lam)
        active = amp > 0
        chi2, res = _chi2(w, s, data, a0, amp[active], lam[active])
        if best is None or chi2 < best[0]:
            best = (chi2, a0, amp[active], lam[active])
        # refine around active poles and wherever the optimality condition fails
        step /= 3.0
        offsets = 10.0 ** (step * np.arange(-3, 4))
        pos = lam[active & (lam > 0)]
        new = [np.array([0.0]), pos, (pos[:, None] * offsets[None, :]).ravel()]
        g = kkt_function(w, s, res, dense)
        peaks = np.flatnonzero((g > 0) & (g >= np.roll(g, 1)) & (g >= np.roll(g, -1)))
        new.append(dense[peaks])
        lam = np.unique(np.concatenate(new))

    chi2, a0, amp, lam = best
    if amp.size == 0:  # data best fitted by a constant: add a negligible far pole
        amp, lam = np.array([np.finfo(float).tiny]), np.array([grid[-1]])
    # tidy up: merge duplicates, polish, and keep whichever version fits best
    candidates = [(a0, amp, lam)]
    for tol in (1e-6, 1e-3):
        m_amp, m_lam = _merge(amp, lam, tol)
        b0, b_amp = _solve(w, s, d, m_lam)
        keep = b_amp > 0
        if keep.any():
            candidates.append((b0, b_amp[keep], m_lam[keep]))
            candidates.append(_polish(w, s, data, b0, b_amp[keep], m_lam[keep]))
    scored = []
    for b0, b_amp, b_lam in candidates:
        if b0 < 0 or np.any(b_amp <= 0) or np.any(b_lam < 0) or not np.all(np.isfinite(b_lam)):
            continue
        scored.append((_chi2(w, s, data, b0, b_amp, b_lam)[0], b0, b_amp, b_lam))
    chi2, a0, amp, lam = min(scored, key=lambda t: t[0])
    order = np.argsort(lam)
    model = PartialFractions(float(a0), amp[order], lam[order])
    res = (data - model.response(w)) / s
    return DPlusFit(model, float(np.sum(np.abs(res) ** 2)), res)

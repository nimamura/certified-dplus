"""Exact dual certificates for the smallest 1-D misfit (lower side of min chi^2).

Data c_j at frequencies omega_j with standard errors s_j; d_j = c_j / s_j.
Every 1-D response has the form c(omega) = a0 + int da(lam) / (lam + i omega)
with a0 >= 0 and a non-negative measure on lam >= 0.

Lemma. Let y be a complex vector with

    g(lam) = sum_j Re( conj(y_j) / (lam + i omega_j) ) / s_j <= 0   for all lam >= 0,
    g_inf  = sum_j Re(y_j) / s_j <= 0.

Then every 1-D model satisfies chi^2 >= 2 Re sum_j conj(y_j) d_j - sum_j |y_j|^2.

The sign condition is a polynomial inequality on [0, inf) and is decided here
in rational arithmetic (square-free factorisation, then Descartes' rule of
signs with bisection). Floats
are converted exactly. The verifier `lower_bound` shares no code with the
fitting routines; derivation in docs/certificate.md.
"""

from fractions import Fraction
from numbers import Complex

from flint import fmpq, fmpq_poly, fmpz_poly


class CertificateError(Exception):
    """The vector is not a valid dual certificate."""


# --- exact input handling ---------------------------------------------------------


def _frac(x):
    return x if isinstance(x, Fraction) else Fraction(x)


def _pairs(values):
    out = []
    for v in values:
        if isinstance(v, Complex) and not isinstance(v, Fraction | int | float):
            out.append((Fraction(v.real), Fraction(v.imag)))
        else:
            re, im = v
            out.append((_frac(re), _frac(im)))
    return out


def _parse(omega, c, sigma):
    w = [_frac(x) for x in omega]
    s = [_frac(x) for x in sigma]
    data = _pairs(c)
    if not (len(w) == len(s) == len(data)) or not w:
        raise ValueError("omega, c and sigma must be non-empty and of equal length")
    if any(x <= 0 for x in w) or len(set(w)) != len(w):
        raise ValueError("frequencies must be positive and distinct")
    if any(x <= 0 for x in s):
        raise ValueError("standard errors must be positive")
    return w, data, s


def _q(x):
    return fmpq(x.numerator, x.denominator)


# --- sign of a rational polynomial on [0, inf) ------------------------------------------


def _sign(x):
    return (x > 0) - (x < 0)


def _primitive(p):
    """Positive rescaling to integer coefficients with unit content (limits growth)."""
    num = p.numer()
    content = abs(num.content())
    return fmpq_poly(num) / content if content else p


def _sturm_count_positive_roots(p):
    """Number of distinct roots in (0, inf) of a polynomial with p(0) != 0.

    Reference implementation (slow for high degree); used to cross-check
    `_has_no_positive_root` in the tests.
    """
    seq = [_primitive(p), _primitive(p.derivative())]
    while not seq[-1].is_zero():
        rem = seq[-2] % seq[-1]
        seq.append(_primitive(-rem))
    seq.pop()

    def changes(signs):
        signs = [s for s in signs if s]
        return sum(a != b for a, b in zip(signs, signs[1:], strict=False))

    at_zero = changes([_sign(q.coeffs()[0]) for q in seq])
    at_inf = changes([_sign(q.leading_coefficient()) for q in seq])
    return at_zero - at_inf


_X_PLUS_1 = fmpz_poly([1, 1])


def _variations(f):
    signs = [_sign(a) for a in f.coeffs() if a != 0]
    return sum(a != b for a, b in zip(signs, signs[1:], strict=False))


def _no_root_in_unit_interval(f):
    """True iff the square-free integer polynomial f has no root in (0, 1).

    Descartes' rule of signs with bisection (Collins-Akritas): the number of
    sign variations of (x+1)^n f(1/(x+1)) bounds the number of roots in (0, 1)
    and is exact when it is 0 or 1.
    """
    stack = [f]
    while stack:
        f = stack.pop()
        v = _variations(fmpz_poly(f.coeffs()[::-1])(_X_PLUS_1))
        if v == 0:
            continue
        if v == 1:
            return False
        n = f.degree()
        left = fmpz_poly([a * 2 ** (n - k) for k, a in enumerate(f.coeffs())])  # 2^n f(x/2)
        if left(1) == 0:  # root at the midpoint
            return False
        stack.append(left)
        stack.append(left(_X_PLUS_1))  # 2^n f((x+1)/2)
    return True


def _has_no_positive_root(p):
    """True iff the square-free polynomial p, with p(0) != 0, has no root in (0, inf)."""
    f = p.numer()
    if f(1) == 0:
        return False
    return _no_root_in_unit_interval(f) and _no_root_in_unit_interval(
        fmpz_poly(f.coeffs()[::-1])  # x^n f(1/x): roots in (1, inf) -> (0, 1)
    )


def _nonpositive_on_half_line(p):
    """True iff p(lam) <= 0 for every lam >= 0."""
    if p.is_zero():
        return True
    if p.leading_coefficient() > 0:
        return False
    coeffs = p.coeffs()
    low = next(i for i, a in enumerate(coeffs) if a != 0)
    if coeffs[low] > 0:  # sign just to the right of lam = 0
        return False
    r = fmpq_poly(coeffs[low:])  # p = lam^low * r, r(0) < 0
    # r changes sign exactly at its roots of odd multiplicity
    _, factors = r.factor_squarefree()
    odd = fmpq_poly([1])
    for f, mult in factors:
        if mult % 2:
            odd *= f
    if odd.degree() < 1:
        return True
    return _has_no_positive_root(odd)


def _dual_polynomial(w, s, y):
    """g(lam) * prod_k (lam^2 + omega_k^2) as an exact polynomial in lam."""
    quad = [fmpq_poly([_q(x * x), 0, 1]) for x in w]
    full = fmpq_poly([1])
    for f in quad:
        full *= f
    p = fmpq_poly([0])
    for wj, sj, (yr, yi), f in zip(w, s, y, quad, strict=True):
        num = fmpq_poly([_q(-yi * wj / sj), _q(yr / sj)])
        p += num * (full // f)
    return p


# --- public API ---------------------------------------------------------------------


def lower_bound(omega, c, sigma, y):
    """Verify the certificate y and return the proven lower bound on chi^2 (a Fraction).

    Raises CertificateError if y violates the sign condition anywhere on lam >= 0.
    """
    w, data, s = _parse(omega, c, sigma)
    yv = _pairs(y)
    if len(yv) != len(w):
        raise ValueError("y must have one entry per frequency")
    if sum(yr / sj for (yr, _), sj in zip(yv, s, strict=True)) > 0:
        raise CertificateError("constant-term condition violated: sum Re(y_j)/s_j > 0")
    if not _nonpositive_on_half_line(_dual_polynomial(w, s, yv)):
        raise CertificateError("g(lambda) > 0 for some lambda >= 0")
    bound = Fraction(0)
    for (cr, ci), sj, (yr, yi) in zip(data, s, yv, strict=True):
        bound += 2 * (yr * cr + yi * ci) / sj - (yr * yr + yi * yi)
    return bound


def chi2_exact(omega, c, sigma, a0, a, lam):
    """Exact chi^2 of the D+ model a0 + sum a_n / (lam_n + i omega) (upper side)."""
    w, data, s = _parse(omega, c, sigma)
    a0 = _frac(a0)
    amp = [_frac(x) for x in a]
    poles = [_frac(x) for x in lam]
    if len(amp) != len(poles):
        raise ValueError("a and lam must have equal length")
    if a0 < 0 or any(x <= 0 for x in amp) or any(x < 0 for x in poles):
        raise ValueError("not a D+ model: need a0 >= 0, a_n > 0, lam_n >= 0")
    total = Fraction(0)
    for wj, (cr, ci), sj in zip(w, data, s, strict=True):
        re, im = a0, Fraction(0)
        for an, ln in zip(amp, poles, strict=True):
            den = ln * ln + wj * wj
            re += an * ln / den
            im -= an * wj / den
        total += ((cr - re) ** 2 + (ci - im) ** 2) / (sj * sj)
    return total


def certify(omega, c, sigma, residual, exponents=range(-15, 1)):
    """Turn a numerical residual r = (c - c_fit) / s into a verified certificate.

    The residual of an optimal fit satisfies the sign condition only up to
    rounding, so it is pushed into the interior: y = r + eps * y0 with
    y0_j = (-1 + i) / |d_j|, which is strictly feasible because every 1-D
    response has Re c > 0 and Im c < 0. The smallest eps on a geometric ladder
    that passes exact verification is used.

    Returns (y, bound) with y a list of (re, im) Fractions; falls back to the
    trivial certificate y = 0, bound = 0.
    """
    w, data, s = _parse(omega, c, sigma)
    r = _pairs(residual)
    zero = [(Fraction(0), Fraction(0))] * len(w)
    norm = float(sum(a * a + b * b for a, b in r)) ** 0.5
    if norm == 0.0:
        return zero, Fraction(0)
    interior = []
    for (cr, ci), sj in zip(data, s, strict=True):
        mag = Fraction(float(cr * cr + ci * ci) ** 0.5 / float(sj))
        if mag == 0:
            mag = Fraction(1)
        interior.append((-1 / mag, 1 / mag))
    for k in exponents:
        eps = Fraction(norm * 10.0**k)
        y = [(a + eps * p, b + eps * q) for (a, b), (p, q) in zip(r, interior, strict=True)]
        try:
            return y, lower_bound(w, data, s, y)
        except CertificateError:
            continue
    return zero, Fraction(0)

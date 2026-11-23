"""Exact existence tests for error-free 1-D magnetotelluric data.

Given c_j = g_j - i h_j at distinct positive frequencies omega_j, decide in
rational arithmetic whether some 1-D conductivity profile reproduces the data.

* Weidelt (1986), eqs. (11)-(14): the data have a *regular* 1-D interpretation
  iff the 2M leading principal minors of two M x M Hankel matrices are positive.
* Yee & Paulson (1988), eq. (12): a 1-D interpretation exists iff two N x N
  Hermitian Pick matrices are non-negative definite.

Floats are converted exactly (every float is a rational number), so the answer
refers to the numbers as stored. This module depends only on the standard library.
"""

from fractions import Fraction
from numbers import Complex

REGULAR = "regular"
DEGENERATE = "degenerate"
INCONSISTENT = "inconsistent"


def _frac(x):
    return x if isinstance(x, Fraction) else Fraction(x)


def _parse(omega, c):
    w = [_frac(x) for x in omega]
    data = []
    for x in c:
        if isinstance(x, Complex) and not isinstance(x, Fraction | int | float):
            data.append((Fraction(x.real), Fraction(x.imag)))
        else:
            re, im = x
            data.append((_frac(re), _frac(im)))
    if len(w) != len(data) or not w:
        raise ValueError("omega and c must be non-empty and of equal length")
    if any(x <= 0 for x in w) or len(set(w)) != len(w):
        raise ValueError("frequencies must be positive and distinct")
    return w, data


def _det(matrix):
    """Determinant by fraction-exact Gaussian elimination."""
    m = [row[:] for row in matrix]
    n = len(m)
    det = Fraction(1)
    for k in range(n):
        pivot = next((r for r in range(k, n) if m[r][k] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            m[k], m[pivot] = m[pivot], m[k]
            det = -det
        det *= m[k][k]
        for r in range(k + 1, n):
            f = m[r][k] / m[k][k]
            if f:
                for col in range(k, n):
                    m[r][col] -= f * m[k][col]
    return det


def weidelt_minors(omega, c):
    """Hankel minors Delta_k(0) and Delta_k(1), k = 1..M, of Weidelt (1986) eq. (13)."""
    w, data = _parse(omega, c)
    m = len(w)
    w2 = [x * x for x in w]
    beta = [Fraction(0)] * (2 * m)
    for j in range(m):
        denom = Fraction(1)
        for other in range(m):
            if other != j:
                denom *= w2[other] - w2[j]
        g, h = data[j][0], -data[j][1]
        for k in range(m):
            alpha = (-w2[j]) ** k / denom  # eq. (11)
            beta[2 * k] += h / w[j] * alpha  # eq. (12a)
            beta[2 * k + 1] += g * alpha  # eq. (12b)
    out = []
    for i in (0, 1):
        out.append(
            [_det([[beta[i + r + s] for s in range(k)] for r in range(k)]) for k in range(1, m + 1)]
        )
    return out[0], out[1]


def is_regular(omega, c):
    """True iff the data admit a regular 1-D interpretation (Weidelt 1986, eq. 14)."""
    d0, d1 = weidelt_minors(omega, c)
    return all(x > 0 for x in d0) and all(x > 0 for x in d1)


# --- Hermitian matrices over Q(i), entries stored as (re, im) ------------------------


def _cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def pick_matrices(omega, c):
    """The two Pick matrices of Yee & Paulson (1988), eq. (12), as lists of (re, im)."""
    w, data = _parse(omega, c)
    n = len(w)
    q = [[None] * n for _ in range(n)]
    qhat = [[None] * n for _ in range(n)]
    for j in range(n):
        for k in range(n):
            s = w[j] + w[k]
            (gj, ij), (gk, ik) = data[j], data[k]
            # i (c_j - conj(c_k)) / (w_j + w_k)
            diff = (gj - gk, ij + ik)
            q[j][k] = (-diff[1] / s, diff[0] / s)
            # (w_j c_j + w_k conj(c_k)) / (w_j + w_k)
            qhat[j][k] = ((w[j] * gj + w[k] * gk) / s, (w[j] * ij - w[k] * ik) / s)
    return q, qhat


def is_positive_semidefinite(matrix):
    """Exact test of a Hermitian matrix over Q(i) for non-negative definiteness."""
    m = [row[:] for row in matrix]
    active = list(range(len(m)))
    while active:
        diag = {i: m[i][i][0] for i in active}
        if any(v < 0 for v in diag.values()):
            return False
        pivot = next((i for i in active if diag[i] > 0), None)
        if pivot is None:
            # all remaining diagonal entries vanish: the block must vanish entirely
            return all(m[i][j] == (0, 0) for i in active for j in active)
        active.remove(pivot)
        p = diag[pivot]
        for i in active:
            for j in active:
                # Schur complement: m_ij - m_ip conj(m_jp) / p
                a, b = m[i][pivot], m[j][pivot]
                prod = _cmul(a, (b[0], -b[1]))
                m[i][j] = (m[i][j][0] - prod[0] / p, m[i][j][1] - prod[1] / p)
    return True


def is_consistent(omega, c):
    """True iff some 1-D model reproduces the data exactly (Yee & Paulson 1988, eq. 12)."""
    q, qhat = pick_matrices(omega, c)
    return is_positive_semidefinite(q) and is_positive_semidefinite(qhat)


def classify(omega, c):
    """'regular', 'degenerate' (consistent, on the boundary) or 'inconsistent'."""
    if is_regular(omega, c):
        return REGULAR
    return DEGENERATE if is_consistent(omega, c) else INCONSISTENT

# The dual certificate

Implementation: `src/certified_dplus/certificate.py`. The derivation below is by
hand; it is not formalised.

## Setting

Data: responses c_j at distinct positive frequencies omega_j (j = 1..N) with
standard errors s_j > 0; d_j = c_j / s_j.

Model responses: c(omega) = a0 + int_{[0,inf)} da(lambda) / (lambda + i omega),
a0 >= 0, a a non-negative measure. Every 1-D conductivity profile has a
response of this form (Weidelt 1972; Parker 1980; Yee & Paulson 1988, eq. 6).
This representation theorem is taken from the literature.

Misfit: chi^2[c] = sum_j |c_j - c(omega_j)|^2 / s_j^2.

For a complex vector y define

```
g_y(lambda) = sum_j Re( conj(y_j) / (lambda + i omega_j) ) / s_j
            = sum_j ( Re y_j * lambda - Im y_j * omega_j ) / ( s_j (lambda^2 + omega_j^2) )
g_y(inf)    = sum_j Re y_j / s_j
```

## Lemma

If (i) g_y(lambda) <= 0 for all lambda >= 0 and (ii) g_y(inf) <= 0, then every
response c of the form above satisfies

```
chi^2[c] >= L(y) := 2 Re sum_j conj(y_j) d_j - sum_j |y_j|^2 .
```

## Proof

Let m_j = c(omega_j) / s_j and r_j = d_j - m_j, so chi^2 = sum |r_j|^2.

1. sum |r_j - y_j|^2 >= 0 gives sum |r_j|^2 >= 2 Re sum conj(y_j) r_j - sum |y_j|^2.
2. Re sum conj(y_j) r_j = Re sum conj(y_j) d_j - Re sum conj(y_j) m_j.
3. Re sum conj(y_j) m_j = a0 g_y(inf) + int g_y(lambda) da(lambda) <= 0 by (i), (ii),
   a0 >= 0 and da >= 0. (The sum is finite, so it commutes with the integral.)
4. Combine 1-3. ∎

The lemma holds for any y that satisfies (i) and (ii), optimal or not. A vector
obtained numerically may therefore be rounded or perturbed; if it passes the
check, the bound is valid.

## Tightness

For the residual r* of the best fit, the optimality (KKT) conditions of the
convex problem give g_{r*} <= 0 with equality at the poles of the optimal model,
hence L(r*) = chi^2_min. This condition is known: for the equivalent problem of
least-squares fitting of Stieltjes functions see Grabovsky (2022, Theorem 2.6),
who attributes it to Caprini (1980). The implementation does not rely on
tightness; it verifies (i) and (ii) for the vector it is given.

## Exact check of (i)

Multiplying by prod_k (lambda^2 + omega_k^2) > 0 turns (i) into a sign condition
on a polynomial P of degree at most 2N - 1 with rational coefficients (floats
are rationals). The check:

1. P identically zero: pass.
2. Leading coefficient positive: fail.
3. Write P = lambda^m R with R(0) != 0. R(0) > 0: fail.
4. Square-free factorisation of R; let Q be the product of the factors of odd
   multiplicity. R changes sign only at roots of Q.
5. Pass iff Q has no root in (0, inf), decided by Descartes' rule of signs with
   bisection (Collins-Akritas); only the cases "0 variations: no root" and
   "1 variation: exactly one root" are used.

Step 5 is cross-checked in the tests against a Sturm-sequence implementation.

## Constructing a certificate

The numerical residual r violates (i) slightly because g touches zero at the
poles. It is moved into the interior: y = r + eps * y0 with
y0_j = (-1 + i) / |d_j|, for which g_{y0} < 0. The smallest eps on the ladder
||r|| * 10^k (k = -15, ..., 0) that passes the exact check is used; if none
passes, y = 0 and the bound is 0.

## Scope of the claim

When `lower_bound` returns L: for the stored numbers (omega_j, c_j, s_j), every
response with a spectral representation has chi^2 >= L. This rests on the
derivation above, the representation theorem, and the correctness of the
verifier and of FLINT's rational arithmetic.

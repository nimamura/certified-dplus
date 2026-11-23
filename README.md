# certified-dplus

A 1-D consistency test for magnetotelluric (MT) and geomagnetic-induction
responses, with a machine-checked bound.

Given responses at a finite set of frequencies with standard errors, the
package answers: **what is the smallest chi-squared misfit that any 1-D
conductivity profile can achieve?** It returns two exact rational numbers that
bracket the answer:

- an **upper value**: the misfit of an explicit best-fitting model (Parker's D+,
  a stack of thin conducting sheets), evaluated in rational arithmetic;
- a **lower value**: a bound that holds for *every* 1-D profile, obtained from a
  dual certificate whose validity is checked in rational arithmetic, not in
  floating point.

If the two agree, the minimum is known to that accuracy for the data as stored
in the computer. In our tests they agree to between 1e-13 and 2e-4 in relative
terms.

## Why

D+ (Parker 1980; Parker & Whaler 1981) is the standard test of whether MT data
are compatible with a 1-D Earth. A numerical optimiser can stop short of the
minimum without saying so: for the seven Hawaiian responses of Larsen (1975) as
tabulated by Yee & Paulson (1988), Parker's `dplus.f` with its default settings
returns chi-squared = 7.4240 (and prints a warning), whereas the minimum lies
between 7.41227233069 and 7.41227233072. A verified lower bound removes the
doubt: a value below it is impossible for any 1-D model.

## Install

```
pip install git+https://github.com/OWNER/certified-dplus
```

Requires Python 3.11 or later, NumPy, SciPy and python-flint.

## Use

```python
import numpy as np
from certified_dplus import consistency_test
from certified_dplus.dplus import to_sheets

# Larsen (1975), Hawaii; table 1 of Yee & Paulson (1988), converted to SI
omega = np.array([3.636e-05, 1.091e-04, 1.818e-04, 2.545e-04, 3.272e-04, 4.000e-04, 4.727e-04])
c = np.array([332800 - 176000j, 236500 - 145800j, 190900 - 140300j, 159500 - 132000j,
              136900 - 120700j, 119000 - 109000j, 103700 - 98800j])   # metres
sigma = np.array([2750.0, 2750.0, 2200.0, 1570.0, 1830.0, 1750.0, 1690.0])

res = consistency_test(omega, c, sigma)
print(float(res.chi2_lower))   # 7.412272330694978  (proven lower bound)
print(float(res.chi2_upper))   # 7.412272330721188  (achieved by res.fit.model)
print(res.rms_lower)           # 0.7276...          (per real datum, 14 data)
print(to_sheets(res.fit.model))  # sheet depths [m] and conductances [S]
```

`c` is the response c = E / (i omega B) = Z / (i omega mu0) in metres, with time
dependence e^{+i omega t} (so Im c < 0). For data in the e^{-i omega t}
convention, pass the complex conjugate. Helper conversions are in
`certified_dplus.conventions`; details in [docs/conventions.md](docs/conventions.md).

The certificate can be re-checked on its own, without the fitting code:

```python
from certified_dplus.certificate import lower_bound
lower_bound(omega, c, sigma, res.certificate) == res.chi2_lower   # True
```

## What is in the package

| Module | Content |
| --- | --- |
| `dplus` | D+ fit by non-negative least squares on a pole grid with refinement; conversion to a sheet model |
| `certificate` | Verification of a dual certificate and exact evaluation of chi-squared (python-flint only) |
| `exact1d` | Existence conditions for error-free data: Hankel minors (Weidelt 1986), Pick matrices (Yee & Paulson 1988), standard library only |
| `forward1d` | Responses of layered and thin-sheet models |
| `conventions` | Units, signs, impedance and apparent-resistivity conversions |

## What is and is not proven

- The lower bound rests on (a) a short lemma, derived by hand in
  [docs/certificate.md](docs/certificate.md); (b) the spectral representation of
  1-D responses, c(omega) = a0 + int da(lambda) / (lambda + i omega) with
  a0 >= 0 and a non-negative measure (Weidelt 1972; Parker 1980), taken from the
  literature; and (c) the correctness of the verifier (about 150 lines) and of
  FLINT's rational arithmetic. Neither (a) nor (b) is formalised in a proof
  assistant.
- Bounds refer to the floating-point numbers handed to the function, converted
  exactly to rationals.
- The optimality condition behind the certificate is not ours: for the
  equivalent problem of fitting Stieltjes functions it is given by Grabovsky
  (2022, Theorem 2.6), who attributes it to Caprini (1980).
- The chi-squared used here weights the complex residual with one standard
  error per frequency. Misfits defined on apparent resistivity and phase are
  not covered.

## Tests

```
uv sync
uv run pytest -q
```

The tests reproduce the consistent pairs, triples and quadruples of Yee &
Paulson (1988, tables 2 and 3), the two extremal two-frequency models of
Weidelt (1985), and analytic layered responses to 1e-15.

## References

- Caprini, I. (1980). General method of using positivity in analytic continuations. Rev. Roumaine Phys. 25, 731–740.
- Grabovsky, Y. (2022). Reconstructing Stieltjes functions from their approximate values: a search for a needle in a haystack. SIAM J. Appl. Math. 82, 1135–1166. doi:10.1137/21M1392279
- Parker, R. L. (1980). The inverse problem of electromagnetic induction: existence and construction of solutions based on incomplete data. J. Geophys. Res. 85, 4421–4428. doi:10.1029/JB085iB08p04421
- Parker, R. L. & Whaler, K. A. (1981). Numerical methods for establishing solutions to the inverse problem of electromagnetic induction. J. Geophys. Res. 86, 9574–9584. doi:10.1029/JB086iB10p09574
- Weidelt, P. (1972). The inverse problem of geomagnetic induction. Z. Geophys. 38, 257–289.
- Weidelt, P. (1986). Discrete frequency inequalities for magnetotelluric impedances of one-dimensional conductors. J. Geophys. 59, 171–176.
- Yee, E. & Paulson, K. V. (1988). Necessary and sufficient conditions for the existence of a solution to the one-dimensional magnetotelluric inverse problem. Geophys. J. Int. 93, 279–293. doi:10.1111/j.1365-246X.1988.tb02002.x

## Licence

MIT.

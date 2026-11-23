# Conventions

| Item | Convention |
| --- | --- |
| Units | SI: length m, conductivity S/m, conductance S, angular frequency rad/s |
| Time dependence | e^{+i omega t}, omega > 0 |
| Coordinates | right-handed, z downwards, surface at z = 0, insulating air above |
| mu0 | 4 pi x 10^-7 H/m (the Earth is non-magnetic) |
| Displacement currents | neglected |

The horizontal electric field obeys E'' = i omega mu0 sigma(z) E, and

```
c(omega) = E_x / (i omega B_y) = -E(0) / E'(0)     [m]
c = g - i h,  g >= 0,  h >= 0
Z = E_x / H_y = i omega mu0 c                      [ohm]
rho_a = omega mu0 |c|^2,   phase = arg Z = 90 deg + arg c
```

A uniform half-space has c = 1 / sqrt(i omega mu0 sigma) and phase 45 deg.

D+ representation: c(omega) = a0 + sum_n a_n / (lambda_n + i omega) with
a0 >= 0 [m], a_n > 0 [m/s], lambda_n >= 0 [1/s]. a0 is the depth of the
shallowest sheet.

Data in the e^{-i omega t} convention: take the complex conjugate of c (or Z).

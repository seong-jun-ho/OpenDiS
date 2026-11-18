# -*- coding: utf-8 -*-
"""Appendix C of the DDM paper: the stress of a straight dislocation.

    P. Cuong Nguyen, N. Aragon, I. Ryu,
    "Defect dynamics modeling of mesoscale plasticity",
    Int. J. Solids Struct. 307 (2025) 113132, Appendix C.

These are what the coupled model is compared against in Section 3.2
(Fig. 6).  Three solutions per character:

    infinite     the classical Volterra field, no surface        (C1, C4)
    cylinder     a cylinder of radius R, with the image stress   (C2, C5)
    nonsingular  Cai et al. (2006), a spread core of radius a    (C3, C6)

The dislocation line is along z through the origin.  A screw has b along
z; an edge has b along x and slips on y = 0.  x, y are measured from the
line, in the same length unit as b; mu and b set the stress unit.

Run this file to check the formulas against each other:

    python3 analytic_fields.py

    (11/18/2026, junho)
"""
import sys

import numpy as np

__all__ = ['screw_infinite', 'screw_cylinder', 'screw_nonsingular',
           'edge_infinite', 'edge_cylinder', 'edge_nonsingular', 'VOIGT']

VOIGT = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']


def _zeros(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def _pack(xx, yy, zz, yz, xz, xy):
    """Stack the six components the way VOIGT names them."""
    return np.stack(np.broadcast_arrays(xx, yy, zz, yz, xz, xy), axis=-1)


# ----------------------------------------------------------------- screw
def screw_infinite(x, y, mu, b):
    """C1: screw dislocation in an infinite medium."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x * x + y * y
    k = mu * b / (2.0 * np.pi)
    z = _zeros(x)
    return _pack(z, z, z, k * x / r2, -k * y / r2, z)


def screw_cylinder(x, y, mu, b, R):
    """C2: screw dislocation along the axis of a cylinder of radius R.

    The 1/R^2 terms adjust for the image torque (Eshelby, 1953); the ends
    are taken as constrained, so there is no image twist.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x * x + y * y
    k = mu * b / (2.0 * np.pi)
    z = _zeros(x)
    return _pack(z, z, z,
                 k * (x / r2 - 2.0 * x / R ** 2),
                 k * (2.0 * y / R ** 2 - y / r2),
                 z)


def screw_nonsingular(x, y, mu, b, a):
    """C3: screw dislocation with a spread core of radius a (Cai 2006)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ra2 = x * x + y * y + a * a
    k = mu * b / (2.0 * np.pi)
    f = 1.0 + a * a / ra2
    z = _zeros(x)
    return _pack(z, z, z, k * x / ra2 * f, -k * y / ra2 * f, z)


# ------------------------------------------------------------------ edge
def edge_infinite(x, y, mu, b, nu):
    """C4: edge dislocation in an infinite medium."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x * x + y * y
    k = mu * b / (2.0 * np.pi * (1.0 - nu))
    z = _zeros(x)
    return _pack(-k * y * (3.0 * x * x + y * y) / r2 ** 2,
                 k * y * (x * x - y * y) / r2 ** 2,
                 -2.0 * nu * k * y / r2,
                 z, z,
                 k * x * (x * x - y * y) / r2 ** 2)


def edge_cylinder(x, y, mu, b, nu, R):
    """C5: edge dislocation along the axis of a cylinder of radius R.

    NOTE on sigma_xy.  Appendix C prints four lines for this solution and
    the third repeats sigma_xx verbatim, so the paper as published has no
    sigma_xy here.  The term below is not a guess: equilibrium fixes it.
    Writing the correction as sigma' = sigma_infinite + D, the printed
    lines give Dxx = 3 k y / R^2 and Dyy = k y / R^2 with
    k = mu b / (2 pi (1 - nu)).  Then

        dDxx/dx + dDxy/dy = 0  ->  Dxy does not depend on y
        dDxy/dx + dDyy/dy = 0  ->  dDxy/dx = -k / R^2

    so Dxy = -k x / R^2, up to a constant taken as zero.  selftest checks
    that this keeps the field in equilibrium and that sigma_zz stays
    nu (sigma_xx + sigma_yy).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x * x + y * y
    k = mu * b / (2.0 * np.pi * (1.0 - nu))
    z = _zeros(x)
    sxx = k * (3.0 * y / R ** 2 - y / r2 * (1.0 + 2.0 * x * x / r2))
    syy = k * (y / R ** 2 - y / r2 * (1.0 - 2.0 * x * x / r2))
    szz = 2.0 * nu * k * (2.0 * y / R ** 2 - y / r2)
    sxy = k * x * (x * x - y * y) / r2 ** 2 - k * x / R ** 2
    return _pack(sxx, syy, szz, z, z, sxy)


def edge_nonsingular(x, y, mu, b, nu, a):
    """C6: edge dislocation with a spread core of radius a (Cai 2006)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ra2 = x * x + y * y + a * a
    k = mu * b / (2.0 * np.pi * (1.0 - nu))
    z = _zeros(x)
    sxx = -k * y / ra2 * (1.0 + 2.0 * (x * x + a * a) / ra2)
    syy = k * y / ra2 * (1.0 - 2.0 * (y * y + a * a) / ra2)
    szz = -2.0 * nu * k * y / ra2 * (1.0 + a * a / ra2)
    sxy = k * x / ra2 * (1.0 - 2.0 * y * y / ra2)
    return _pack(sxx, syy, szz, z, z, sxy)


# ------------------------------------------------------------- self test
def _grid(n=41, lo=-300.0, hi=300.0, hole=40.0):
    """Points away from the core, where every form is finite."""
    t = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(t, t)
    keep = (X * X + Y * Y) > hole * hole
    return X[keep], Y[keep]


def _close(name, A, B, tol, results):
    d = np.abs(A - B)
    s = np.abs(B).max()
    rel = d.max() / s if s > 0 else d.max()
    ok = rel < tol
    print('  %-52s %s  (%.2e)' % (name, 'ok' if ok else 'FAILED', rel))
    results.append(ok)


def _divergence(f, x, y, h=1e-3):
    """Numerical div(sigma) in the xy plane; a static field has it zero."""
    def s(xx, yy):
        return f(xx, yy)
    sxx_x = (s(x + h, y)[..., 0] - s(x - h, y)[..., 0]) / (2 * h)
    sxy_y = (s(x, y + h)[..., 5] - s(x, y - h)[..., 5]) / (2 * h)
    sxy_x = (s(x + h, y)[..., 5] - s(x - h, y)[..., 5]) / (2 * h)
    syy_y = (s(x, y + h)[..., 1] - s(x, y - h)[..., 1]) / (2 * h)
    return sxx_x + sxy_y, sxy_x + syy_y


def main():
    mu, b, nu = 50.0e9, 1.0, 0.3
    x, y = _grid()
    ok = []

    print('limits')
    _close('C2 cylinder -> C1 infinite as R -> inf',
           screw_cylinder(x, y, mu, b, 1.0e9), screw_infinite(x, y, mu, b),
           1e-8, ok)
    _close('C3 nonsingular -> C1 infinite as a -> 0',
           screw_nonsingular(x, y, mu, b, 1.0e-6), screw_infinite(x, y, mu, b),
           1e-8, ok)
    _close('C5 cylinder -> C4 infinite as R -> inf',
           edge_cylinder(x, y, mu, b, nu, 1.0e9),
           edge_infinite(x, y, mu, b, nu), 1e-8, ok)
    _close('C6 nonsingular -> C4 infinite as a -> 0',
           edge_nonsingular(x, y, mu, b, nu, 1.0e-6),
           edge_infinite(x, y, mu, b, nu), 1e-8, ok)

    print('plane strain: sigma_zz = nu (sigma_xx + sigma_yy)')
    for name, f in (('C4 edge infinite', lambda: edge_infinite(x, y, mu, b, nu)),
                    ('C5 edge cylinder', lambda: edge_cylinder(x, y, mu, b, nu, 1200.0)),
                    ('C6 edge nonsingular', lambda: edge_nonsingular(x, y, mu, b, nu, 5.0))):
        S = f()
        _close(name, S[..., 2], nu * (S[..., 0] + S[..., 1]), 1e-10, ok)

    print('equilibrium: div(sigma) = 0')
    for name, f in (
            ('C4 edge infinite',
             lambda xx, yy: edge_infinite(xx, yy, mu, b, nu)),
            ('C5 edge cylinder  (checks the reconstructed sigma_xy)',
             lambda xx, yy: edge_cylinder(xx, yy, mu, b, nu, 1200.0)),
            ('C6 edge nonsingular',
             lambda xx, yy: edge_nonsingular(xx, yy, mu, b, nu, 5.0))):
        d1, d2 = _divergence(f, x, y)
        scale = np.abs(f(x, y)).max() / 40.0      # per unit length
        r = max(np.abs(d1).max(), np.abs(d2).max()) / scale
        good = r < 1e-3
        print('  %-52s %s  (%.2e)' % (name, 'ok' if good else 'FAILED', r))
        ok.append(good)

    print('')
    print('PASS' if all(ok) else 'FAIL')
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())

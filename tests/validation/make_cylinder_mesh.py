# -*- coding: utf-8 -*-
"""A hexahedral cylinder mesh for the Section 3.2 and 3.3 test cases.

The paper meshes a cylinder with C3D8R elements -- R = 1200 b and 50 b
elements for the stress field (Fig. 6), R = 3900 b and 100 b elements for
the interaction force (Fig. 7).  A polar mesh cannot do that: it has a
singular line on the axis.  So the cross-section is an O-grid -- a square
block in the middle, four blocks fanning out to the circle -- which is all
hexahedra and has no degenerate element anywhere.  The disk is then
extruded along z.

The axis is z, the cylinder runs from -L/2 to +L/2, and lengths are in b,
matching the DD side.

Writes an ABAQUS input deck with

    *Node, *Element type=C3D8R
    Nset  ALL, ZMIN, ZMAX, LATERAL
    Elset ALL

Run it to see the mesh it would make and check it:

    python make_cylinder_mesh.py --R 1200 --L 2400 --h 50 --out cyl.inp
    python make_cylinder_mesh.py --selftest

    (11/18/2026, junho)
"""
import argparse
import sys

import numpy as np


def disk(R, h, n_sq=None, core=0.5):
    """O-grid of a disk of radius R: (xy, quads).

    core is the half-width of the middle square as a fraction of R.  n_sq
    is the number of cells along one side of that square; left alone it is
    chosen so the cells on the outer circle are about h across.
    """
    c = core * R
    if n_sq is None:
        # circumference / 4 per side, one cell of h each
        n_sq = max(2, int(round(2.0 * np.pi * R / (4.0 * h))))
    n_r = max(1, int(round((R - c) / h)))

    # --- the middle square, (n_sq+1)^2 nodes
    t = np.linspace(-c, c, n_sq + 1)
    X, Y = np.meshgrid(t, t, indexing='ij')
    xy = [np.stack([X.ravel(), Y.ravel()], axis=1)]

    def sq_id(i, j):
        return i * (n_sq + 1) + j

    quads = [[sq_id(i, j), sq_id(i + 1, j), sq_id(i + 1, j + 1), sq_id(i, j + 1)]
             for i in range(n_sq) for j in range(n_sq)]

    # --- the square's boundary, walked once counterclockwise
    ring = ([sq_id(i, 0) for i in range(n_sq, 0, -1)]          # y = -c
            + [sq_id(0, j) for j in range(0, n_sq)]            # x = -c
            + [sq_id(i, n_sq) for i in range(0, n_sq)]         # y = +c
            + [sq_id(n_sq, j) for j in range(n_sq, 0, -1)])    # x = +c
    m = len(ring)                                              # = 4 n_sq

    inner = xy[0][ring]
    ang = np.arctan2(inner[:, 1], inner[:, 0])
    outer = np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)

    # --- the rings, from the square out to the circle
    base = (n_sq + 1) ** 2
    prev = list(ring)
    for k in range(1, n_r + 1):
        f = float(k) / n_r
        xy.append(inner + f * (outer - inner))
        cur = [base + (k - 1) * m + i for i in range(m)]
        for i in range(m):
            j = (i + 1) % m
            quads.append([prev[i], prev[j], cur[j], cur[i]])
        prev = cur

    return np.vstack(xy), np.asarray(quads, dtype=int), n_sq, n_r


def quad_area(p):
    """Signed area of a quad, so the winding can be checked."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def build(R, L, h, n_sq=None, core=0.5):
    xy, quads, n_sq, n_r = disk(R, h, n_sq, core)

    # every quad counterclockwise, so the extruded hexahedron is positive
    for q in range(len(quads)):
        if quad_area(xy[quads[q]]) < 0:
            quads[q] = quads[q][::-1]

    n_z = max(1, int(round(L / h)))
    z = np.linspace(-0.5 * L, 0.5 * L, n_z + 1)
    nd = len(xy)

    nodes = np.empty((nd * (n_z + 1), 3))
    for k in range(n_z + 1):
        nodes[k * nd:(k + 1) * nd, 0:2] = xy
        nodes[k * nd:(k + 1) * nd, 2] = z[k]

    elems = np.empty((len(quads) * n_z, 8), dtype=int)
    e = 0
    for k in range(n_z):
        lo, hi = k * nd, (k + 1) * nd
        for q in quads:
            elems[e, 0:4] = q + lo
            elems[e, 4:8] = q + hi
            e += 1

    return nodes, elems, dict(n_sq=n_sq, n_r=n_r, n_z=n_z, nd=nd,
                              per_disk=len(quads))


def hex_volume(p):
    """Volume of a hexahedron by splitting it into six tetrahedra."""
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6),
            (3, 4, 6, 7), (1, 4, 5, 6)]
    v = 0.0
    for a, b, c, d in tets:
        v += abs(np.dot(np.cross(p[b] - p[a], p[c] - p[a]), p[d] - p[a])) / 6.0
    return v


def write_inp(path, nodes, elems, R, L, tol=1e-6):
    zlo, zhi = -0.5 * L, 0.5 * L
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    zmin = np.where(np.abs(nodes[:, 2] - zlo) < tol * L + 1e-9)[0] + 1
    zmax = np.where(np.abs(nodes[:, 2] - zhi) < tol * L + 1e-9)[0] + 1
    lat = np.where(np.abs(r - R) < 1e-6 * R)[0] + 1

    def nset(f, name, ids):
        f.write('*Nset, nset=%s\n' % name)
        for i in range(0, len(ids), 16):
            f.write(', '.join('%d' % v for v in ids[i:i + 16]) + '\n')

    with open(path, 'w') as f:
        f.write('*Heading\n')
        f.write('** cylinder R=%g L=%g, %d nodes, %d C3D8R elements\n'
                % (R, L, len(nodes), len(elems)))
        f.write('** made by make_cylinder_mesh.py (11/18/2026, junho)\n')
        f.write('*Node\n')
        for i, p in enumerate(nodes):
            f.write('%d, %.8e, %.8e, %.8e\n' % (i + 1, p[0], p[1], p[2]))
        f.write('*Element, type=C3D8R\n')
        for i, c in enumerate(elems):
            f.write('%d, ' % (i + 1) + ', '.join('%d' % (v + 1) for v in c) + '\n')
        f.write('*Nset, nset=ALL, generate\n1, %d, 1\n' % len(nodes))
        f.write('*Elset, elset=ALL, generate\n1, %d, 1\n' % len(elems))
        nset(f, 'ZMIN', zmin)
        nset(f, 'ZMAX', zmax)
        nset(f, 'LATERAL', lat)
    return len(zmin), len(zmax), len(lat)


def report(nodes, elems, R, L, info):
    vols = np.array([hex_volume(nodes[c]) for c in elems])
    exact = np.pi * R * R * L
    edges = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4)):
        edges.append(np.linalg.norm(nodes[elems[:, a]] - nodes[elems[:, b]],
                                    axis=1))
    edges = np.concatenate(edges)
    print('  square %d x %d cells, %d rings, %d layers'
          % (info['n_sq'], info['n_sq'], info['n_r'], info['n_z']))
    print('  %d nodes, %d elements (%d per disk)'
          % (len(nodes), len(elems), info['per_disk']))
    print('  element edges  min %.1f  median %.1f  max %.1f b'
          % (edges.min(), np.median(edges), edges.max()))
    print('  volume %.6e vs pi R^2 L %.6e  ->  %.4f %% low'
          % (vols.sum(), exact, 100.0 * (1.0 - vols.sum() / exact)))
    print('  smallest element %.3e b^3, any non-positive: %s'
          % (vols.min(), 'YES' if vols.min() <= 0 else 'no'))
    return vols, exact


def selftest():
    ok = []
    for R, L, h in ((100.0, 200.0, 10.0), (1200.0, 2400.0, 50.0)):
        print('R = %g, L = %g, h = %g' % (R, L, h))
        nodes, elems, info = build(R, L, h)
        vols, exact = report(nodes, elems, R, L, info)

        # a straight hexahedral mesh cuts the corners off the circle, so it
        # is always a little small; how little says the mesh is fine enough
        short = 1.0 - vols.sum() / exact
        ok.append(('volume within 1 %% of pi R^2 L', short < 0.01, short))
        ok.append(('every element has positive volume', vols.min() > 0,
                   vols.min()))
        # every node inside the circle
        rr = np.hypot(nodes[:, 0], nodes[:, 1]).max()
        ok.append(('no node outside the radius', rr <= R * (1 + 1e-9), rr))
        # each node used by at least one element
        used = np.unique(elems)
        ok.append(('no orphan node', len(used) == len(nodes),
                   len(nodes) - len(used)))
        print('')

    print('checks')
    good = True
    for name, passed, val in ok:
        print('  %-42s %s  (%s)' % (name, 'ok' if passed else 'FAILED', val))
        good &= bool(passed)
    print('')
    print('PASS' if good else 'FAIL')
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=float, default=1200.0)
    ap.add_argument('--L', type=float, default=2400.0)
    ap.add_argument('--h', type=float, default=50.0)
    ap.add_argument('--n-sq', type=int, default=None)
    ap.add_argument('--core', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    nodes, elems, info = build(a.R, a.L, a.h, a.n_sq, a.core)
    report(nodes, elems, a.R, a.L, info)
    if a.out:
        n1, n2, n3 = write_inp(a.out, nodes, elems, a.R, a.L)
        print('  wrote %s  (ZMIN %d, ZMAX %d, LATERAL %d nodes)'
              % (a.out, n1, n2, n3))
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""One nucleation event: is the plastic strain we hand to ABAQUS its own area?

A nucleated loop is a regular hexagon of circumradius R, and its nodes are
born with their old position at the centre of the loop.  Every one of the six
segments therefore sweeps the triangle (centre, r_i, r_j) on the step it
appears, and the six triangles are the hexagon:

    A = 6 * (1/2) R^2 sin(60 deg) = (3 sqrt(3) / 2) R^2

Nothing in that is a modelling choice -- it is the area of a hexagon.  So the
plastic strain the loop contributes has to be

    dEp = A |b| / V * sym(nhat (x) bhat)

and this script reads it back out of the file ABAQUS was given.

Usage:  python3 proof_nucleation.py [run.log] [output_dir]

    (11/18/2026, junho)
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R_LOOP = 200.0            # Nucleation.FullLoop hardcodes this
L_BOX = 1000.0            # test_nucleation_one.py: Lbox
VOIGT = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']


def hexagon_area(R):
    return 1.5 * np.sqrt(3.0) * R * R


def find_nucleation_step(logpath):
    """Return (istep, centre) of the step the loop was inserted on.

    step_nucleate runs before step_print_info, so the "step = N" that follows
    the insertion is the step the loop belongs to -- not the one before it.
    """
    ins_re = re.compile(r'Nucleation loop inserted!\s*center=\[([^\]]*)\]')
    step_re = re.compile(r'^step = (\d+)')
    centre = None
    with open(logpath) as f:
        for line in f:
            if centre is None:
                m = ins_re.search(line)
                if m:
                    centre = np.array([float(v) for v in m.group(1).split()])
                continue
            m = step_re.match(line)
            if m:
                return int(m.group(1)), centre
    return None, None


def read_dis_segs(path):
    """header (n, n_final, t_step, EffSlip) and rows as a (N, 23) array."""
    with open(path) as f:
        head = f.readline().split()
    rows = np.loadtxt(path, skiprows=1, ndmin=2)
    return (int(head[0]), int(head[1]), float(head[2]), float(head[3])), rows


def swept(rows):
    """Sn per row, with the same formula write_Dis_Segs used."""
    r1_old, r2_old = rows[:, 2:5], rows[:, 5:8]
    r1_new, r2_new = rows[:, 8:11], rows[:, 11:14]
    return 0.5 * np.cross(r2_new - r1_old, r1_new - r2_old)


def burgers_from_dyad(dyad, Sn):
    """dyad = nhat (x) b, so divide out the largest component of nhat."""
    n = np.linalg.norm(Sn)
    if n == 0.0:
        return None
    nhat = Sn / n
    i = int(np.argmax(np.abs(nhat)))
    return dyad.reshape(3, 3)[i] / nhat[i]


def voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[1, 2], T[0, 2], T[0, 1]])


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'proof',
                                                             'run_one_nuc.log')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE,
                                                             'output_one_nuc')

    istep, centre = find_nucleation_step(log)
    if istep is None:
        sys.exit('no "Nucleation loop inserted" in %s -- nothing nucleated' % log)
    print('nucleated on step %d, loop centre %s'
          % (istep, np.array2string(centre, precision=1)))

    path = os.path.join(out, 'Dis_Segs_%d' % istep)
    (nseg, _, t_step, eff_slip), rows = read_dis_segs(path)
    print('%s: %d segments, t_step %.3e, EffSlip %.1f b'
          % (os.path.basename(path), nseg, t_step, eff_slip))

    Sn = swept(rows)

    # The six nucleated segments are the ones whose two old ends are both at
    # the centre of the loop.  Nothing else in the network has that.
    at_centre = (np.linalg.norm(rows[:, 2:5] - centre, axis=1) < 1e-6) & \
                (np.linalg.norm(rows[:, 5:8] - centre, axis=1) < 1e-6)
    k = np.where(at_centre)[0]
    print('%d segments start from the loop centre' % k.size)
    if k.size == 0:
        sys.exit('found none -- did the old positions get registered?')

    tri = hexagon_area(R_LOOP) / 6.0
    print('')
    print('   seg        swept area [b^2]     hexagon/6      ratio')
    for i in k:
        a = np.linalg.norm(Sn[i])
        print('  %4d-%-4d   %16.4f  %13.4f  %9.6f'
              % (rows[i, 0], rows[i, 1], a, tri, a / tri))

    total = np.linalg.norm(Sn[k].sum(axis=0))
    A = hexagon_area(R_LOOP)
    print('')
    print('  sum of the six      %16.4f' % total)
    print('  (3 sqrt3 / 2) R^2   %16.4f   with R = %.0f b' % (A, R_LOOP))
    print('  ratio               %16.9f   (%.2e off)'
          % (total / A, abs(total / A - 1.0)))

    # the plastic strain the loop contributes
    V = L_BOX ** 3
    b = burgers_from_dyad(rows[k[0], 14:23], Sn[k[0]])
    P = np.outer(Sn[k].sum(axis=0), b) / V
    dep_loop = voigt(0.5 * (P + P.T))

    nhat = Sn[k].sum(axis=0) / total
    P_ana = np.outer(A * nhat, b) / V
    dep_ana = voigt(0.5 * (P_ana + P_ana.T))

    print('')
    print('  b = %s   |b| = %.6f' % (np.array2string(b, precision=6), np.linalg.norm(b)))
    print('  V = %.3e b^3' % V)
    print('')
    print('  component      from Dis_Segs        A |b| / V * sym      ratio')
    for j, name in enumerate(VOIGT):
        r = dep_loop[j] / dep_ana[j] if abs(dep_ana[j]) > 1e-30 else float('nan')
        print('      %2s        %16.9e  %16.9e  %9.6f'
              % (name, dep_loop[j], dep_ana[j], r))

    # Second path.  step_nucleate runs after step_update_response, so the loop
    # is NOT in the dEp the DD side accumulated for the load update -- it
    # reaches ABAQUS through the coordinates in Dis_Segs instead.  So what has
    # to match dEp_pydis.csv is everything in the file except the loop.
    csv = os.path.join(out, 'dEp_pydis.csv')
    if os.path.exists(csv):
        want = None
        with open(csv) as f:
            f.readline()
            for line in f:
                p = line.split(',')
                if int(p[0]) == istep:
                    want = np.array([float(v) for v in p[1:7]])
        if want is not None:
            bs = np.array([burgers_from_dyad(rows[i, 14:23], Sn[i])
                           if np.linalg.norm(Sn[i]) > 0 else np.zeros(3)
                           for i in range(len(rows))])
            rest = np.setdiff1d(np.arange(len(rows)), k)
            Prest = np.einsum('ni,nj->ij', Sn[rest], bs[rest]) / V
            dep_rest = voigt(0.5 * (Prest + Prest.T))
            print('')
            print('  the rest of the network on this step, two paths:')
            print('  (the loop itself is not in dEp_pydis -- step_nucleate runs')
            print('   after step_update_response, so it only goes out in Dis_Segs)')
            print('  component    from Dis_Segs       dEp_pydis.csv      ratio')
            for j, name in enumerate(VOIGT):
                r = dep_rest[j] / want[j] if abs(want[j]) > 1e-30 else float('nan')
                print('      %2s      %16.9e  %16.9e  %9.6f'
                      % (name, dep_rest[j], want[j], r))

    off = abs(total / A - 1.0)
    print('')
    print('VERDICT: swept area is the hexagon to %.2e' % off)
    return 0 if off < 1e-6 else 1


if __name__ == '__main__':
    sys.exit(main())

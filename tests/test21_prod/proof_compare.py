# -*- coding: utf-8 -*-
"""Is the stress we get back from ABAQUS real?

Elasticity is linear, so doubling the load must double the stress.
`proof_linear.sh` runs the same deck twice with load 1000 and 2000 and keeps
the final FEMSTRESS of each run; this script divides one by the other.

The loading direction is DOF 2 (y), so sigma_yy is the component that must
come out at 2.000.  The other components do NOT have to -- see below.

    (11/18/2026, junho)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['xx', 'yy', 'zz', 'yz', 'zx', 'xy']
LOAD_DIR = 1          # index into COLS: yy


def read_femstress(path):
    """FEMSTRESS rows: id1 id2 Sxx Syy Szz Syz Szx Sxy T."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            rows.append(((p[0], p[1]), [float(v) for v in p[2:8]]))
    return rows


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'proof')
    a = read_femstress(os.path.join(d, 'femstress_x1.txt'))   # load 1000
    b = read_femstress(os.path.join(d, 'femstress_x2.txt'))   # load 2000

    if len(a) != len(b):
        sys.exit('segment count differs: %d vs %d -- the two runs did not '
                 'reach the same step' % (len(a), len(b)))

    print('segment            ' + ''.join('%9s' % c for c in COLS))
    ratios = []
    for (k1, s1), (k2, s2) in zip(a, b):
        if k1 != k2:
            sys.exit('segment ids differ: %s vs %s' % (k1, k2))
        r = [(y / x if abs(x) > 0.0 else float('nan')) for x, y in zip(s1, s2)]
        ratios.append(r[LOAD_DIR])
        print('%s-%s  ' % k1 + ''.join('%9.4f' % v for v in r))

    n = len(ratios)
    mean = sum(ratios) / n
    worst = max(abs(v - 2.0) for v in ratios) / 2.0 * 100.0
    print('')
    print('sigma_%s : mean %.4f over %d segments, worst deviation %.2f %%'
          % (COLS[LOAD_DIR], mean, n, worst))
    print('')
    print('The other components scatter, and that is the point: the total is')
    print('    sigma = sigma_applied (scales with load) + sigma_dislocation (does not),')
    print('so only the loading direction, where the applied part dominates, lands on 2.')
    print('A hand-written constant file would have given 1.00 everywhere.')

    return 0 if worst < 1.0 else 1


if __name__ == '__main__':
    sys.exit(main())

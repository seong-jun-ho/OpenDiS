# -*- coding: utf-8 -*-
"""Does the FEM -> DD node feedback do the four things MATLAB does?

rn_check is the half of the handshake that comes back from ABAQUS, and it
is easy to get the sense of it backwards, because the important case is a
node that has NO row at all.  So drive apply_rn_check with a file written
by hand, where the right answer is known, instead of waiting for a coupled
run to produce the situation.

The reference is generate_postprocessing_data12_cleanup_vec.m:206-221.

    python3 selftest_rn_check.py          (no ABAQUS, no MPI, a second)

    (11/18/2026, junho)
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ['../../python', '../../lib', '../../core/pydis/python',
          '../../extensions/eig_fem']:
    q = os.path.abspath(os.path.join(HERE, p))
    if q not in sys.path:
        sys.path.append(q)

from framework.disnet_manager import DisNetManager          # noqa: E402
from pydis import DisNet, Cell                              # noqa: E402
from eig_fem_dis.prod_protocol import ProdProtocol          # noqa: E402
from eig_fem_dis.simulate.sim_eig_fem_dis import SimulationDriver  # noqa: E402

FREE, SURF, PIN = 0, 6, 7


class Shim(object):
    """Just enough of the driver to call the one method under test."""
    def __init__(self, pp, surface_pin=False):
        self.prod_protocol = pp
        self.surface_pin = surface_pin

    apply_rn_check = SimulationDriver.apply_rn_check


def build(constraints):
    """A closed loop of len(constraints) nodes, at x = 0, 100, 200, ...

    Closed, because DisNet refuses a chain with dangling ends (the Burgers
    vector would not sum to zero at them), and at least three nodes,
    because a two-node loop would be the same edge twice.  So every case
    below carries a spare node that the assertions ignore.
    """
    n = len(constraints)
    assert n >= 3, "DisNet needs at least three nodes in a closed loop"
    rn = np.array([[100.0 * i, 0.0, 0.0, c] for i, c in enumerate(constraints)])
    links = np.array([[i, (i + 1) % n, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                      for i in range(n)])
    cell = Cell(h=1000.0 * np.eye(3), is_periodic=[False] * 3)
    return DisNetManager(DisNet(cell=cell, rn=rn, links=links))


def write_rn_check(folder, rows, repeat=3):
    """Write it the way the VUMAT does: one block per ABAQUS increment."""
    with open(os.path.join(folder, 'rn_check0001.txt'), 'w') as f:
        for _ in range(repeat):
            for i, flag, xyz in rows:
                f.write('%06d %5d %15.8E %15.8E %15.8E\n'
                        % (i, flag, xyz[0], xyz[1], xyz[2]))


def state_of(DM):
    d = DM.export_data()['nodes']
    return (np.asarray(d['constraints']).ravel().tolist(),
            np.asarray(d['positions']))


def check(name, got, want):
    ok = got == want
    print('  %-46s %s' % (name, 'ok' if ok else 'FAILED  %s != %s'
                          % (got, want)))
    return ok


def main():
    folder = tempfile.mkdtemp(prefix='rncheck_')
    pp = ProdProtocol(folder, num_cpus=1, verbose=False)
    allok = True

    # ---------------------------------------------------------------- 1
    # free -> left the body, free -> pinned by the FEM, free -> stays free
    DM = build([FREE, FREE, FREE])
    write_rn_check(folder, [
        (0, FREE, (0.0, 0.0, 0.0)),      # placed, echoes 0  -> stays free
        (2, PIN, (0.0, 0.0, 0.0)),       # placed, echoes 7  -> pinned
        # node 1 has no row at all       -> it left the body -> surface
    ])
    Shim(pp).apply_rn_check(DM, {})
    c, _ = state_of(DM)
    allok &= check('free node the FEM placed stays free', c[0], FREE)
    allok &= check('free node with no row becomes a surface node', c[1], SURF)
    allok &= check('free node the FEM echoes 7 for is pinned', c[2], PIN)

    # ---------------------------------------------------------------- 2
    # a surface node is moved onto the projection point, and only then
    DM = build([SURF, SURF, FREE])
    write_rn_check(folder, [
        (0, SURF, (11.0, 22.0, 33.0)),   # projection -> move there
        (1, SURF, (0.0, 0.0, 0.0)),      # no projection -> leave it alone
        (2, FREE, (0.0, 0.0, 0.0)),      # spare, placed and free
    ])
    Shim(pp).apply_rn_check(DM, {})
    c, R = state_of(DM)
    allok &= check('surface node moves onto the projection',
                   R[0].tolist(), [11.0, 22.0, 33.0])
    allok &= check('surface node without a projection does not move',
                   R[1].tolist(), [100.0, 0.0, 0.0])
    allok &= check('projecting does not change the flag', c[0], SURF)

    # ---------------------------------------------------------------- 3
    # a pinned node the FEM cannot place becomes a surface node
    DM = build([PIN, PIN, FREE])
    write_rn_check(folder, [(0, PIN, (0.0, 0.0, 0.0)),
                            (2, FREE, (0.0, 0.0, 0.0))])
    Shim(pp).apply_rn_check(DM, {})
    c, _ = state_of(DM)
    allok &= check('pinned node the FEM placed stays pinned', c[0], PIN)
    allok &= check('pinned node with no row becomes a surface node',
                   c[1], SURF)

    # ---------------------------------------------------------------- 4
    # the file is emptied, so the next cycle does not read this one again
    write_rn_check(folder, [(0, FREE, (0.0, 0.0, 0.0))])
    pp.read_rn_check()
    size = os.path.getsize(os.path.join(folder, 'rn_check0001.txt'))
    allok &= check('rn_check is emptied after it is read', size, 0)

    # ---------------------------------------------------------------- 5
    # surface_pin turns every surface node into a pinned one
    DM = build([FREE, SURF, FREE])
    write_rn_check(folder, [(1, SURF, (0.0, 0.0, 0.0)),
                            (2, FREE, (0.0, 0.0, 0.0))])
    Shim(pp, surface_pin=True).apply_rn_check(DM, {})
    c, _ = state_of(DM)
    allok &= check('surface_pin: node that left the body is pinned',
                   c[0], PIN)
    allok &= check('surface_pin: existing surface node is pinned', c[1], PIN)

    print('')
    print('PASS' if allok else 'FAIL')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())

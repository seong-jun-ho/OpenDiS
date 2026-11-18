# -*- coding: utf-8 -*-
"""Does the surface cleanup do the six things the MATLAB driver does?

Reference: generate_postprocessing_data12_cleanup_vec.m:341-470.

  1  a two-armed node between two surface nodes becomes a surface node
  2  a segment with surface nodes at both ends is deleted
  3  a segment whose midpoint is in no element, one arm or fewer at each
     end, is deleted as debris
  4  a one-armed surface node whose arm midpoint is outside moves there
  5  a node with more than six arms, and its neighbours, are pinned
  6  a node left with no arms is deleted

Each rule is driven from a network and a link table written by hand, so
the right answer is known without running anything.

    python3 selftest_surface_cleanup.py       (no ABAQUS, a second)

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

from framework.disnet_manager import DisNetManager                  # noqa
from pydis import DisNet, Cell                                      # noqa
from eig_fem_dis.prod_protocol import ProdProtocol                  # noqa
from eig_fem_dis.simulate.sim_eig_fem_dis import SimulationDriver   # noqa

FREE, SURF, PIN = 0, 6, 7
B = np.array([1.0, 0.0, 0.0])
N = np.array([0.0, 0.0, 1.0])


class Shim(object):
    """Just enough of the driver to call the one method under test."""
    def __init__(self, pp):
        self.prod_protocol = pp
        self.fem_node_feedback = True
    surface_cleanup = SimulationDriver.surface_cleanup


def loop(constraints):
    """A closed loop of len(constraints) nodes at x = 0, 100, 200, ..."""
    n = len(constraints)
    rn = np.array([[100.0 * i, 0.0, 0.0, c] for i, c in enumerate(constraints)])
    links = np.array([[i, (i + 1) % n, B[0], B[1], B[2], N[0], N[1], N[2]]
                      for i in range(n)])
    cell = Cell(h=10000.0 * np.eye(3), is_periodic=[False] * 3)
    return DisNetManager(DisNet(cell=cell, rn=rn, links=links))


def hub_of_triangles(k=4):
    """One node shared by k triangles, so it carries 2k arms."""
    rn = [[0.0, 0.0, 0.0, FREE]]
    links = []
    for t in range(k):
        a, b = 1 + 2 * t, 2 + 2 * t
        rn.append([100.0 * (t + 1), 100.0, 0.0, FREE])
        rn.append([100.0 * (t + 1), 200.0, 0.0, FREE])
        links += [[0, a, B[0], B[1], B[2], N[0], N[1], N[2]],
                  [a, b, B[0], B[1], B[2], N[0], N[1], N[2]],
                  [b, 0, B[0], B[1], B[2], N[0], N[1], N[2]]]
    cell = Cell(h=10000.0 * np.eye(3), is_periodic=[False] * 3)
    return DisNetManager(DisNet(cell=cell, rn=np.array(rn),
                                links=np.array(links)))


def write_links(folder, rows):
    """Output_link0001.txt: n1 n2 element.  A pair that is absent is
    exactly what 'the midpoint is in no element' looks like."""
    with open(os.path.join(folder, 'Output_link0001.txt'), 'w') as f:
        for a, b, e in rows:
            f.write('%06d %06d %08d\n' % (a, b, e))


def state_of(DM):
    d = DM.export_data()
    return (np.asarray(d['nodes']['constraints']).ravel().tolist(),
            np.asarray(d['nodes']['positions']).reshape(-1, 3),
            len(np.asarray(d['segs']['nodeids']).reshape(-1, 2)))


def check(name, got, want, results):
    ok = got == want
    print('  %-50s %s%s' % (name, 'ok' if ok else 'FAILED',
                            '' if ok else '  %s != %s' % (got, want)))
    results.append(ok)


def all_pairs(DM, elem):
    """Every segment in the element table, so rules 3 and 4 stay asleep."""
    ids = np.asarray(DM.export_data()['segs']['nodeids'], int).reshape(-1, 2)
    return [(int(a), int(b), elem) for a, b in ids]


def main():
    folder = tempfile.mkdtemp(prefix='cleanup_')
    pp = ProdProtocol(folder, num_cpus=1, verbose=False)
    ok = []

    # ---------------------------------------------------------------- 2, 6
    DM = loop([SURF, SURF, FREE, FREE, FREE])
    write_links(folder, all_pairs(DM, 7))
    Shim(pp).surface_cleanup(DM, {'istep': 1})
    c, R, nseg = state_of(DM)
    ok.append(True)
    check('segment between two surface nodes is deleted', nseg, 4, ok)
    check('nothing is orphaned yet, so no node is dropped', len(c), 5, ok)

    # a whole loop of surface nodes: every arm goes, every node follows
    DM = loop([SURF] * 5)
    write_links(folder, all_pairs(DM, 7))
    Shim(pp).surface_cleanup(DM, {'istep': 1})
    c, R, nseg = state_of(DM)
    check('all-surface loop: every segment deleted', nseg, 0, ok)
    check('all-surface loop: every node deleted', len(c), 0, ok)

    # ------------------------------------------------------------------- 1
    # node 1 has two arms and both its neighbours are surface nodes
    DM = loop([SURF, FREE, SURF, FREE, FREE])
    write_links(folder, all_pairs(DM, 7))
    Shim(pp).surface_cleanup(DM, {'istep': 5})
    c, R, nseg = state_of(DM)
    check('the flag walks onto the node between two of them',
          c[1], SURF, ok)
    check('a node with only one surface neighbour is left alone',
          c[3], FREE, ok)

    # and not before step 2, as MATLAB gates it
    DM = loop([SURF, FREE, SURF, FREE, FREE])
    write_links(folder, all_pairs(DM, 7))
    Shim(pp).surface_cleanup(DM, {'istep': 1})
    c, R, nseg = state_of(DM)
    check('the flag does not walk on the first steps', c[1], FREE, ok)

    # ------------------------------------------------------------------- 3
    # Leave a single segment standing on its own -- one arm at each end --
    # and keep it out of the element table.  That is debris.
    DM2 = loop([FREE, FREE, FREE, FREE, FREE])
    G = DM2.get_disnet(DisNet)
    tags = [tuple(t) for t in
            np.asarray(DM2.export_data()['nodes']['tags'], int)]
    for a, b in ((1, 2), (2, 3), (3, 4), (4, 0)):
        G._remove_edge(tags[a], tags[b])
    for i in (2, 3, 4):
        G._remove_node(tags[i])
    before = state_of(DM2)[2]
    write_links(folder, [(3, 4, 7)])               # anything but (0, 1)
    Shim(pp).surface_cleanup(DM2, {'istep': 1})
    c, R, nseg = state_of(DM2)
    check('a lone segment in no element is debris', (before, nseg), (1, 0), ok)
    check('and its two nodes go with it', len(c), 0, ok)

    # a segment the table DOES know about is left alone
    DM2 = loop([FREE, FREE, FREE, FREE, FREE])
    G = DM2.get_disnet(DisNet)
    tags = [tuple(t) for t in
            np.asarray(DM2.export_data()['nodes']['tags'], int)]
    for a, b in ((1, 2), (2, 3), (3, 4), (4, 0)):
        G._remove_edge(tags[a], tags[b])
    for i in (2, 3, 4):
        G._remove_node(tags[i])
    write_links(folder, [(0, 1, 7)])
    Shim(pp).surface_cleanup(DM2, {'istep': 1})
    check('a lone segment the FEM placed is kept', state_of(DM2)[2], 1, ok)

    # ------------------------------------------------------------------- 4
    DM = loop([SURF, FREE, FREE, FREE, FREE])
    G = DM.get_disnet(DisNet)
    tags = [tuple(t) for t in
            np.asarray(DM.export_data()['nodes']['tags'], int)]
    G._remove_edge(tags[4], tags[0])               # node 0 keeps one arm
    pairs = all_pairs(DM, 7)
    write_links(folder, [p for p in pairs
                         if sorted(p[:2]) != sorted([0, 1])])
    Shim(pp).surface_cleanup(DM, {'istep': 1})
    c, R, nseg = state_of(DM)
    check('a lone surface node moves to its arm midpoint',
          [round(v, 6) for v in R[0]], [50.0, 0.0, 0.0], ok)

    # ------------------------------------------------------------------- 5
    DM = hub_of_triangles(4)                        # the hub carries 8 arms
    write_links(folder, all_pairs(DM, 7))
    Shim(pp).surface_cleanup(DM, {'istep': 1})
    c, R, nseg = state_of(DM)
    check('a node with more than six arms is pinned', c[0], PIN, ok)
    check('and so are its neighbours', c[1], PIN, ok)

    print('')
    print('PASS' if all(ok) else 'FAIL')
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())

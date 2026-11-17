# -*- coding: utf-8 -*-
"""Adapter so PyDiS can drive the *production* ParaDiS VUMAT.

The production VUMAT (VUMAT_DDM_multiple_DDcycle.for) was written to talk to
the MATLAB driver.  This module makes PyDiS play MATLAB's part, without
touching the VUMAT.  Three things differ from our own test VUMAT:

  1. rn_matrix.txt   the VUMAT wants the node list every exchange
                     (VUMAT:1886-1893)
  2. flag protocol   ABAQUSworking%04d.flag  <->  Matlabworking%04d.flag
                     (DDM_Linux_v0.m:156, 192, 379)
  3. stress output   the VUMAT writes ELEMENT stress to Output_stress%04d.txt
                     (VUMAT:487, 490-492); our calforce reads SEGMENT stress
                     from FEMSTRESS, so we map element -> segment here.

Nothing here runs unless the driver is told to use it, so the existing
test21 path is unaffected.
"""
import os
import time

import numpy as np

class ProdProtocol:
    """Speak the production VUMAT's file and flag protocol."""

    def __init__(self, foldername_ABAQUS, num_cpus=1, poll=0.25, verbose=True,
                 stress_scale=1.0e6):
        """stress_scale converts the VUMAT's stress unit to the DD side's.

        The production deck works in b and MPa (*Density 1.573e-21,
        E = 79000), because the VUMAT assumes burgers = 1.0.  calforce
        expects Pa, so the default multiplies by 1e6.  Set it to 1.0 if the
        deck is already in Pa.
        """
        self.dir = foldername_ABAQUS
        self.num_cpus = int(num_cpus)
        self.poll = poll
        self.verbose = verbose
        self.stress_scale = float(stress_scale)
        self._elem_ids = None
        self._elem_lo = None
        self._elem_hi = None
        self._elem_c = None

    # ------------------------------------------------------------ helpers
    def _p(self, name):
        return os.path.join(self.dir, name)

    def _flag(self, head, rank):
        return self._p('%s%04d.flag' % (head, rank + 1))

    def _say(self, msg):
        if self.verbose:
            print('[PROD] %s' % msg)

    # ------------------------------------------------------------ 1. flags
    def init_flags(self):
        """Hand the first turn to ABAQUS (MATLAB does this before launching it)."""
        for r in range(self.num_cpus):
            # clear any stale Matlabworking from an earlier run
            m = self._flag('Matlabworking', r)
            if os.path.exists(m):
                os.remove(m)
            with open(self._flag('ABAQUSworking', r), 'w') as f:
                f.write('')
        self._say('created ABAQUSworking flags for %d rank(s)' % self.num_cpus)

    def wait_for_dd_turn(self, timeout=3600.0):
        """Block until every ABAQUSworking flag is gone.

        The VUMAT renames ABAQUSworking%04d.flag -> Matlabworking%04d.flag when
        it has written its output and wants the DD side to run.
        """
        t0 = time.time()
        n = 0
        while True:
            left = [r for r in range(self.num_cpus)
                    if os.path.exists(self._flag('ABAQUSworking', r))]
            if not left:
                self._say('DD turn (waited %.1f s)' % (time.time() - t0))
                return True
            dead = self._abaqus_died()
            if dead:
                raise RuntimeError('ABAQUS stopped after %.0f s: %s'
                                   % (time.time() - t0, dead))
            if time.time() - t0 > timeout:
                self._say('TIMEOUT after %.0f s; still held: %s' % (timeout, left))
                return False
            n += 1
            if n % 200 == 0:
                self._say('waiting for ABAQUS, %.0f s' % (time.time() - t0))
            time.sleep(self.poll)

    def _abaqus_died(self):
        """First fatal line in ABAQUS's own files, or None.

        Without this a dead ABAQUS looks exactly like a slow one and we wait
        out the whole timeout - 41 minutes on 2026-11-17.
        """
        marks = ('exited with error', 'Abaqus/Analysis exited',
                 'THE ANALYSIS HAS NOT BEEN COMPLETED', '***ERROR')
        for ext in ('.sta', '.log', '.msg', '.dat'):
            for name in os.listdir(self.dir):
                if not name.endswith(ext):
                    continue
                try:
                    with open(os.path.join(self.dir, name),
                              errors='replace') as f:
                        txt = f.read()
                except OSError:
                    continue
                for mk in marks:
                    i = txt.find(mk)
                    if i >= 0:
                        line = txt[i:txt.find('\n', i) if '\n' in txt[i:]
                                   else len(txt)]
                        return '%s: %s' % (name, line.strip()[:160])
        return None

    def release_to_abaqus(self):
        """Rename Matlabworking -> ABAQUSworking: the DD step is done."""
        for r in range(self.num_cpus):
            m, a = self._flag('Matlabworking', r), self._flag('ABAQUSworking', r)
            if os.path.exists(m):
                os.replace(m, a)
            else:
                # VUMAT never got to rename it; create directly so we don't deadlock
                with open(a, 'w') as f:
                    f.write('')
                self._say('rank %d had no Matlabworking flag; created ABAQUSworking'
                          % (r + 1))
        self._say('released to ABAQUS')

    # ------------------------------------------------------- 2. rn_matrix
    def write_rn_matrix(self, DM, state=None):
        """rn_matrix.txt - VUMAT:1886-1893.

            line 1 : num_rn
            rows   : id  flag  n_neighbours  x y z        (id starts from 0)

        flag follows ParaDiS: 0 free, 6 surface, 7 pinned.  PyDiS's
        DisNode.Constraints uses the same numbers for 6 and 7.
        """
        data = DM.export_data()
        nodes = data.get('nodes', {})
        R = np.asarray(nodes.get('positions'))
        tags = np.asarray(nodes.get('tags'))
        constraints = nodes.get('constraints', None)

        # neighbour count per node, from the segment list
        segs = data.get('segs', {})
        stags = np.asarray(segs.get('nodeids'))
        nnbr = np.zeros(len(R), dtype=int)
        if stags.size:
            for a, b in stags:
                nnbr[int(a)] += 1
                nnbr[int(b)] += 1

        path = self._p('rn_matrix.txt')
        with open(path, 'w') as f:
            f.write('%d\n' % len(R))
            for i in range(len(R)):
                if constraints is None:
                    flag = 0
                else:
                    flag = int(np.asarray(constraints).ravel()[i])
                f.write('%d %d %d %.8e %.8e %.8e\n'
                        % (i, flag, nnbr[i], R[i, 0], R[i, 1], R[i, 2]))
        self._say('rn_matrix.txt: %d nodes' % len(R))
        return path

    # --------------------------------------------------- 3. stress mapping
    def _load_mesh(self):
        """Element boxes from OriginalCoordNodes.txt (id + 8 nodes x 3)."""
        if self._elem_ids is not None:
            return
        d = np.loadtxt(self._p('OriginalCoordNodes.txt'), ndmin=2)
        self._elem_ids = d[:, 0].astype(int)
        xyz = d[:, 1:25].reshape(len(d), 8, 3)
        self._elem_lo = xyz.min(axis=1)
        self._elem_hi = xyz.max(axis=1)
        self._elem_c = xyz.mean(axis=1)
        self._say('mesh: %d elements' % len(d))

    def read_element_stress(self, truncate=True):
        """Output_stress%04d.txt -> {elem_id: [xx, yy, zz, yz, xz, xy, T]}.

        The VUMAT writes STATENEW(21,22,23,25,26,24) then TEMPNEW
        (VUMAT:490-492), i.e. xx, yy, zz, yz, zx, xy, T - zx is xz, so the
        stress order already matches what FEMSTRESS wants.

        Follows the production reader
        (generate_postprocessing_data12_cleanup_vec.m:32-72):

          * if the file holds more rows than there are elements, ABAQUS
            wrote twice this cycle; keep the FIRST half
          * empty the file afterwards, so the next cycle reads only its own
            output.  Without this the file grows without bound - it was
            2.2 MB after 20 steps here.
        """
        self._load_mesh()
        out = {}
        found = []
        for r in range(self.num_cpus):
            p = self._p('Output_stress%04d.txt' % (r + 1))
            if not os.path.exists(p):
                continue
            found.append(os.path.basename(p))
            d = np.loadtxt(p, ndmin=2)
            if d.size:
                nel = len(self._elem_ids) if self._elem_ids is not None else 0
                if nel and len(d) > nel:
                    d = d[:len(d) // 2]
                for row in d:
                    out[int(row[0])] = row[1:8].astype(float)
            if truncate:
                open(p, 'w').close()
        self._say('element stress: %d elements from %s'
                  % (len(out), ', '.join(found) if found else '(no file)'))
        return out

    def read_link_table(self):
        """(n1, n2) -> element, as the VUMAT itself assigned it.

        This is what the production MATLAB uses - it does not search
        geometrically at all, it looks the pair up in the table the VUMAT
        writes (generate_postprocessing_data12_cleanup_vec.m:298-336,
        VUMAT:1313-1330).

        Worth preferring over our own search: when a segment midpoint sits
        exactly on an element face, both neighbours pass a bounding-box
        test and the two codes break the tie differently.  Measured on
        test21_prod: 2 of 5 segments landed on a face and we picked the
        neighbour the VUMAT had not used (345 vs 445, 365 vs 465).  Taking
        the VUMAT's own answer also guarantees we read the stress of the
        element it deposited the eigenstrain into.

        The file is appended every exchange, so the LAST row for a pair is
        the current one.
        """
        tbl = {}
        for r in range(self.num_cpus):
            p = self._p('Output_link%04d.txt' % (r + 1))
            if not os.path.exists(p):
                continue
            for ln in open(p):
                f = ln.split()
                if len(f) >= 3:
                    try:
                        tbl[(int(f[0]), int(f[1]))] = int(f[2])
                    except ValueError:
                        pass
        return tbl

    def write_FEMSTRESS(self, DM, elem_stress, path='FEMSTRESS'):
        """Map element stress onto segments and write what calforce reads.

            n1 n2 s_xx s_yy s_zz s_yz s_xz s_xy flag
        """
        self._load_mesh()
        data = DM.export_data()
        # node ids here are ARRAY INDICES, the same numbering rn_matrix.txt uses
        ids = np.asarray(data['segs']['nodeids'], dtype=int).reshape(-1, 2)
        pos = np.asarray(data['nodes']['positions'], dtype=float).reshape(-1, 3)
        mid = 0.5 * (pos[ids[:, 0]] + pos[ids[:, 1]])

        link = self.read_link_table()
        nmiss = 0
        nsearch = 0
        with open(path, 'w') as f:
            for i in range(len(mid)):
                # the VUMAT's own assignment first; our search only as a fallback
                e = link.get((int(ids[i, 0]), int(ids[i, 1])))
                if e is None:
                    e = link.get((int(ids[i, 1]), int(ids[i, 0])))
                if e is None:
                    e = self._element_of(mid[i])
                    nsearch += 1
                s = elem_stress.get(e)
                if s is None:
                    # production skips a segment with no element rather than
                    # writing zeros for it
                    # (generate_postprocessing_data12_cleanup_vec.m:505, 525)
                    nmiss += 1
                    continue
                sig = s[:6] * self.stress_scale
                temp = s[6] if len(s) > 6 else 0.0
                # columns: n1 n2 | sigma(6) | temperature
                # the last column is TEMPNEW, not a flag - production writes
                # it when enabled_temp_disp is on (:507-515)
                f.write('%06d %06d ' % (ids[i, 0], ids[i, 1])
                        + ' '.join('%.8e' % v for v in sig)
                        + ' %.4e\n' % temp)
        self._say('FEMSTRESS: %d segments, %d from the VUMAT link table, '
                  '%d by our own search (%d had no element stress)'
                  % (len(mid), len(mid) - nsearch, nsearch, nmiss))
        return path

    def _element_of(self, p):
        """Element whose box contains p; nearest centroid if none does."""
        inside = np.all((p >= self._elem_lo - 1e-9) &
                        (p <= self._elem_hi + 1e-9), axis=1)
        idx = np.flatnonzero(inside)
        if idx.size:
            return int(self._elem_ids[idx[0]])
        j = int(np.argmin(np.sum((self._elem_c - p) ** 2, axis=1)))
        return int(self._elem_ids[j])

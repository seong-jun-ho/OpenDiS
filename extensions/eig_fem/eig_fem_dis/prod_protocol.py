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
                 stress_scale=1.0e6, snapshot_dir=None):
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
        # Set to a directory to keep one cycle's exchange files as they were,
        # before read_element_stress empties them.  For documenting the
        # interface; leave it None in a normal run.
        self.snapshot_dir = snapshot_dir
        self._snapped = False
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
            if self.snapshot_dir and not self._snapped and d.size:
                self._snapshot(r)
            if truncate:
                open(p, 'w').close()
        self._say('element stress: %d elements from %s'
                  % (len(out), ', '.join(found) if found else '(no file)'))
        return out

    def _snapshot(self, rank):
        """Copy this cycle's exchange files aside, before they are emptied."""
        import shutil
        os.makedirs(self.snapshot_dir, exist_ok=True)
        names = ['Output_stress%04d.txt' % (rank + 1),
                 'Output_link%04d.txt' % (rank + 1),
                 'Surface_stress%04d.txt' % (rank + 1),
                 'rn_check%04d.txt' % (rank + 1),
                 'Data_segs.txt', 'rn_matrix.txt', 'FEM_deltaU.txt',
                 'surface_elements.txt', 'OriginalCoordNodes.txt']
        kept = []
        for n in names:
            src = self._p(n)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.snapshot_dir, n))
                kept.append(n)
        self._snapped = True
        self._say('snapshot -> %s (%s)' % (self.snapshot_dir, ', '.join(kept)))

    def read_rn_check(self, truncate=True):
        """What the FEM says back about each dislocation node.

        The VUMAT appends one block per ABAQUS increment to
        rn_check%04d.txt (VUMAT_DDM_multiple_DDcycle.for:1538-1651), so by
        the time the turn comes back the file holds many blocks.  Rows are

            id  flag  x y z

        where id indexes rn_matrix.txt, flag is the constraint the FEM
        echoes back, and x y z is either 0 0 0 (nothing to do) or the point
        a surface node should be moved to.

        A node the FEM could not place in any element has NO row at all.
        That absence is the signal; see apply_rn_check.

        MATLAB sorts the rows and lets the first one per id win
        (generate_postprocessing_data12_cleanup_vec.m:189-209).  Do the
        same, so both sides pick the same row when a node appears more
        than once with different contents -- which happens, because the
        file carries one block per increment.

        (11/18/2026, junho)
        """
        rows = []
        for r in range(self.num_cpus):
            p = self._p('rn_check%04d.txt' % (r + 1))
            if not os.path.exists(p):
                continue
            try:
                d = np.loadtxt(p, ndmin=2)
            except ValueError:
                d = np.empty((0, 5))
            if d.size:
                rows.append(d.reshape(-1, 5))
            if truncate:
                open(p, 'w').close()
        if not rows:
            self._say('rn_check: nothing to read')
            return {}

        d = np.vstack(rows)
        d = d[np.lexsort(tuple(d[:, k] for k in range(d.shape[1] - 1, -1, -1)))]
        out = {}
        for row in d:
            i = int(row[0])
            if i not in out:
                out[i] = (int(row[1]), row[2:5].copy())
        self._say('rn_check: %d rows -> %d nodes' % (len(d), len(out)))
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

    def write_SurfaceStress(self, path='SurfaceStress'):
        """Surface_stress%04d.txt -> SurfaceStress, what Nucleation reads.

        The VUMAT writes 7 columns, elem_id plus the six stress components
        (VUMAT:497-500).  make_nuc_sites wants ten:

            id  x y z  S11 S22 S33 S23 S13 S12

        so put the element centroid in, from OriginalCoordNodes.txt, and
        scale the stress to Pa.  The component order already lines up:
        xx,yy,zz,yz,zx,xy is S11,S22,S33,S23,S13,S12 (zx is xz).

        Appended every exchange like Output_stress, so keep the last row
        per element and empty the file afterwards.
        """
        self._load_mesh()
        cen = {int(e): self._elem_c[i] for i, e in enumerate(self._elem_ids)}
        rows = {}
        for r in range(self.num_cpus):
            p = self._p('Surface_stress%04d.txt' % (r + 1))
            if not os.path.exists(p):
                continue
            d = np.loadtxt(p, ndmin=2)
            if d.size:
                for row in d:
                    rows[int(row[0])] = row[1:7].astype(float)
            open(p, 'w').close()

        if not rows:
            # nothing from ABAQUS yet; leave the file alone so the caller
            # can tell "not written" from "written empty"
            self._say("SurfaceStress: nothing to write yet")
            return None

        n = 0
        with open(path, 'w') as f:
            for e in sorted(rows):
                c = cen.get(e)
                if c is None:
                    continue
                s = rows[e] * self.stress_scale
                f.write('%d %.6e %.6e %.6e ' % (e, c[0], c[1], c[2])
                        + ' '.join('%.6e' % v for v in s) + '\n')
                n += 1
        self._say('SurfaceStress: %d sites' % n)
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

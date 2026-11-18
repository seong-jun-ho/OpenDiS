"""@package docstring
SimulationDriver: class for simulating dislocation network with boundary value problem (BVP)

Provide simulation functions based on other utlitity classes
"""
import sys, os
import subprocess
pydis_paths = ['../../../../python', '../../../../lib', '../../../../core/pydis/python']
[sys.path.append(os.path.abspath(path)) for path in pydis_paths if not path in sys.path]

import numpy as np
import pickle
from pydis.disnet import DisNet
from pydis.calforce.calforce_disnet import CalForce
from pydis.mobility.mobility_disnet import MobilityLaw
from pydis.simulate.sim_disnet import SimulateNetwork
from pydis.timeint.timeint_disnet import TimeIntegration
from pydis.visualize.vis_disnet import VisualizeNetwork
from framework.disnet_manager import DisNetManager
from pydis.util.plastic_strain import (segment_positions, old_positions_for,
                                       swept_area_vector)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except ImportError:
    print('-----------------------------------------')
    print(' cannot import matplotlib or mpl_toolkits')
    print('-----------------------------------------')

class SimulationDriver(SimulateNetwork):
    """SimulationDriver: class for simulating dislocation network with boundary value problem (BVP)
           extends SimulateNetwork from PyDiS
    """
# (08/14/2025, kyeongmi) Add arguments remote_stress, remote_force
    def __init__(self, state: dict,
                 remote_stress=None, remote_force=None,
                 surface_mobility=None, surface_topology=None,
                 nucleation=None,
                 max_nuc_events=None,
                 prod_protocol=None,
                 **kwargs) -> None:
        super().__init__(state, **kwargs)

        self.remote_stress = remote_stress
        self.remote_force = remote_force
        self.surface_mobility = surface_mobility
        self.surface_topology = surface_topology
        self.nucleation = nucleation
        # (11/18/2026, junho) Stop nucleating after this many loops.  None is
        # unlimited, which is what a production run wants; a validation run
        # sets 1 so the plastic strain of a single loop can be read on its own.
        self.max_nuc_events = max_nuc_events
        self.nuc_events = 0
        # (11/17/2026, junho) Set this to drive the *production* ParaDiS VUMAT
        # (eig_fem_dis.prod_protocol.ProdProtocol).  None keeps the
        # original ABAQUS_pause/running flag path untouched.
        self.prod_protocol = prod_protocol
        
        # Launch abaqus
        # or put at the beginning of test case

    def step_begin(self, DM: DisNetManager, state: dict):
        """step_begin: invoked at the begining of each time step
           (08/14/2025, kyeongmi) skip the first step and call FEMRemoteStress,
                                  which gets eigenstrain field from OpenDiS and
                                  runs ABAQUS to calculate stress field. 
        """
        print("------step_begin-------")
        if self.prod_protocol is not None:
            # Production VUMAT path: Data_segs.txt and rn_matrix.txt were written
            # and the turn handed over at the end of the previous step, so here we
            # only wait for ABAQUS and turn its ELEMENT stress into the SEGMENT
            # stress that calforce reads.
            istep = state['istep']
            if istep > 0:
                pp = self.prod_protocol
                if not pp.wait_for_dd_turn():
                    raise RuntimeError("prod_protocol: ABAQUS did not hand back")
                pp.write_FEMSTRESS(DM, pp.read_element_stress())
                pp.write_SurfaceStress()
            return
        if self.remote_stress != None:
            istep = state['istep']
            if istep == 0: # skip the first step
                pass
            else: # starting from second step
            # should call FEMRemotStress(DM, state)
            # export eig strain increment (from previous step)
            # remove ABAQUS_pause.flag
            # call function in the remote_stress module
                self.calforce.FEMRemoteStress(DM, state)
                
        pass
    def step_nucleate(self, DM: DisNetManager, state: dict):
        """
        3 Def is called here 
        1) make_nuc_sites 
        2) compute_nucleation
        3) loopgenerate

        """
        # 0) nucleation import
        nu = self.nucleation
        if nu is None:
            print("[NUC] no nucleator attached; skip")
            return

        if (self.max_nuc_events is not None
                and self.nuc_events >= self.max_nuc_events):
            print("[NUC] %d loop(s) nucleated, cap is %d; skip"
                  % (self.nuc_events, self.max_nuc_events))
            return

        # (11/17/2026, junho) nothing has run on the FEM side yet on the first
        # step, so there is no stress to decide nucleation from.  Skip rather
        # than fail - production has the same gap on its first cycle.
        import os as _os
        _sf = _os.path.join(getattr(nu, "workdir", "."),
                            getattr(nu, "dir_femstress", "") or "",
                            getattr(nu, "stress_filename", "SurfaceStress"))
        if not _os.path.exists(_sf):
            print("[NUC] no %s yet (first step); skip"
                  % _os.path.basename(_sf))
            return

        # 1) call make_nuc_sites
        nu.make_nuc_sites()
        N = 0 if nu.ids is None else nu.ids.size
        # Check whether SurfaceStress is loaded or not 
        print(f"[NUC] sites loaded: N={N}") 

        # 2) call compute_nucleation
        nu.compute_nucleation(time_now=state.get("time", 0.0))
        totalR = 0.0 if nu.R is None or nu.R.size == 0 else float(nu.R[-1])
        maxP   = 0.0 if nu.P is None or nu.P.size == 0 else float(nu.P.max())
        maxRSS = 0.0 if nu.RSSmax is None or nu.RSSmax.size == 0 else float(np.max(np.abs(nu.RSSmax)))
        print(f"[NUC] totals: totalR={totalR:.3e}, maxP={maxP:.3e}, max|RSS|={maxRSS:.3e} Pa" )

        # 3) call compute_nucleation
        sel = np.where(nu.F == 1)[0]  
        if sel.size == 0:
            print("[NUC] no sites flagged by KMC; skip")
            return

        # 4) call loopgenerate
        for k in sel:
            # (11/17/2026, junho) Give the new loop nodes an old position.
            # save_old_nodes already ran in step_integrate, so a node born here
            # has no start-of-step position.  ParaDiS puts it at the centre of
            # the loop (src/LoopGenerate_FCC.cc:131-133), which makes the
            # per-segment swept triangles add up to the area of the loop: a
            # freshly nucleated loop contributes its own area to the plastic
            # strain of that step, once.
            G = DM.get_disnet(DisNet)
            before = set(tuple(t) for t in G.all_nodes_tags())

            nu.loopgenerate(DM, site_index=int(k), state=state)
            self.nuc_events += 1

            G = DM.get_disnet(DisNet)
            fresh = [t for t in (tuple(x) for x in G.all_nodes_tags())
                     if t not in before]
            oldpos = state.get('oldpos_dict')
            if oldpos is None:
                print("[NUC] no oldpos_dict - skipping the old position")
            elif fresh:
                c = np.array([float(nu.x[k]), float(nu.y[k]), float(nu.z[k])])
                for t in fresh:
                    oldpos[t] = c.copy()
                print("[NUC] %d new nodes: old position set to the loop centre %s"
                      % (len(fresh), np.array2string(c, precision=1)))
        

    def step_integrate(self, DM: DisNetManager, state: dict):
        """step_integrate: invoked for time-integration at each time step
        """
        # (11/17/2026, junho) uncommented save_old_nodes / plastic_strain
        self.save_old_nodes(DM, state)

        print("------step_integrate-------")
        state = self.calforce.NodeForce(DM, state)
        state = self.mobility.Mobility(DM, state)
        state = self.timeint.Update(DM, state)
        self.plastic_strain(DM, state)
        # calculate eigstrain field
        # call function in the remote_stress module

    def step_topological_operations(self, DM: DisNetManager, state: dict):
        """step_topological_operations: invoked for handling topological events at each time step
        """
        print("------step_topological_operations-------")
        if self.cross_slip is not None:
            self.cross_slip.Handle(DM, state)

        # The order of topology vs collision is opposite to ExaDiS
        if self.topology is not None:
            self.topology.Handle(DM, state)

        #ToDo: may need to write a combined Topology class for both bulk and surface nodes
        if self.surface_topology is not None:
            self.surface_topology.Handle(DM, state)

        if self.collision is not None:
            self.collision.HandleCol(DM, state)

        if self.remesh is not None:
            self.remesh.Remesh(DM, state)

    # (2025/08/07, Ill Ryu)
    # For data communication, we need to keep the same dislocation network in plastic strain and rn_links
    # It requires to write a files(Dis_Segs) after topology operations.
    # We are waiting for the update which could export old_position of nodes.
    def write_Dis_Segs(self, DM: DisNetManager, filename_Dis_Segs: str,
                       state: dict = None):
        """Write the segments in the format the VUMAT reads
        (VUMAT_DDM_multiple_DDcycle.for:1826-1829).

          line 1 : n_segment  n_segment_final  t_step  EffSlip
          rows   : id1 id2 | r1_old(3) r2_old(3) | r1_new(3) r2_new(3) | dyad(9)

        dyad = outer(nhat, b), nhat the unit normal of the swept
        quadrilateral - the same convention as ParaDiS src/WriteArms.cc.

        (11/17/2026, junho) old positions come from save_old_nodes.
        """
        cell = DM.cell
        data = DM.export_data()
        tags1, tags2, r1, r2, burgers = segment_positions(data)
        Nseg = len(burgers)

        oldpos = None if state is None else state.get('oldpos_dict')
        if oldpos is None:
            raise ValueError("write_Dis_Segs: no state['oldpos_dict'] - "
                             "is step_integrate calling save_old_nodes?")

        # (11/17/2026, junho) a node made by topology after save_old_nodes has
        # no start-of-step position; it swept nothing, so old = current.
        # Count them out loud - this is an approximation, not a detail.
        def _old_or_current(tags, now):
            out = np.empty((len(tags), 3), dtype=float)
            missing = 0
            for i, t in enumerate(tags):
                key = tuple(t)
                if key in oldpos:
                    out[i] = oldpos[key]
                else:
                    out[i] = now[i]
                    missing += 1
            return out, missing

        r1_old, m1 = _old_or_current(tags1, r1)
        r2_old, m2 = _old_or_current(tags2, r2)
        if m1 + m2:
            print("write_Dis_Segs: %d node slots had no start-of-step position "
                  "(created by topology after save_old_nodes); "
                  "treated as not having swept" % (m1 + m2))

        # same minimum-image order as segment_dyads
        r2 = np.asarray(cell.closest_image(Rref=r1, R=r2))
        r3 = np.asarray(cell.closest_image(Rref=r1, R=r1_old))
        r4 = np.asarray(cell.closest_image(Rref=r3, R=r2_old))

        Sn = swept_area_vector(r1, r2, r3, r4)
        nrm = np.linalg.norm(Sn, axis=1, keepdims=True)
        nhat = np.divide(Sn, nrm, out=np.zeros_like(Sn), where=(nrm > 0.0))
        dyad = nhat[:, :, None] * burgers[:, None, :]

        n_moved = int((nrm > 0.0).sum())
        print("write_Dis_Segs: %d segments, %d swept a non-zero area"
              % (Nseg, n_moved))

        t_step = float(state.get('dt', 0.0))
        eff_slip = float(state.get('EffSlip', 0.0))

        nid1 = np.asarray(tags1, dtype=int).reshape(-1, 2)[:, 1]
        nid2 = np.asarray(tags2, dtype=int).reshape(-1, 2)[:, 1]

        with open(filename_Dis_Segs, "w") as file:
            # nothing merges substeps yet, so the final block is all of it
            file.write("%d %d %.8e %.8e\n" % (Nseg, Nseg, t_step, eff_slip))
            for i in range(Nseg):
                vals = list(r3[i]) + list(r4[i]) + list(r1[i]) + list(r2[i]) \
                       + list(dyad[i].ravel())
                file.write("%d %d " % (nid1[i], nid2[i])
                           + " ".join("%.6e" % v for v in vals) + "\n")
        print("Dis_Segs file has been written")

    def read_Dis_Segs(self, filename_Dis_Segs: str):
        """read from 'Dis_Segs' file
        """
        # This is copied from disnet.py (get_segs_data_with_positions)
        data = np.loadtxt(filename_Dis_Segs, delimiter=' ', dtype=float)

        return data

    def change_surface_node_flag(self, DM: DisNetManager, state: dict):
        """ Change the flag of node outside the FEM domain
        """
        G = DM.get_disnet(DisNet)
        node1 = G.nodes((0,0))
        node1.constraint = 6 
        return state

    def step_write_files(self, DM: DisNetManager, state: dict):
        print("------step_write_files-------")
        if self.write_freq != None:
            istep = state['istep']
            if istep % self.write_freq == 0:
                DM.write_json(os.path.join(self.write_dir, f'disnet_{istep}.json'))
                if self.save_state:
                    with open(os.path.join(self.write_dir, f'state_{istep}.pickle'), 'wb') as file:
                        pickle.dump(state, file)
            # Here we call a new function to print out the data to Dis_Segs
            self.write_Dis_Segs(DM, os.path.join(self.write_dir,
                                f'Dis_Segs_{istep}'), state)
            # (11/17/2026, junho) Put the same thing where ABAQUS reads it: the
            # VUMAT reads Data_segs.txt in vexternaldb(StartIncrement) and takes
            # the eigenstrain out there.
            aba = state.get('foldername_ABAQUS', None)
            if aba:
                self.write_Dis_Segs(DM, os.path.join(aba, 'Data_segs.txt'), state)
                # (11/17/2026, junho) Production VUMAT also wants the node list, and
                # the turn is handed back by renaming the flag (MATLAB does the
                # same at DDM_Linux_v0.m:379).
                if self.prod_protocol is not None:
                    self.prod_protocol.write_rn_matrix(DM, state)
                    self.prod_protocol.release_to_abaqus()

            # (11/17/2026, junho) For checking: keep the dEp that PyDiS worked out
            # itself.  Against the value computed by hand from Dis_Segs it gives
            # two independent paths to the same plastic strain.
            dep = state.get('dEp', None)
            if dep is not None:
                p = os.path.join(self.write_dir, 'dEp_pydis.csv')
                if not os.path.exists(p):
                    with open(p, 'w') as f:
                        f.write('step,dEp_xx,dEp_yy,dEp_zz,dEp_yz,dEp_xz,dEp_xy\n')
                with open(p, 'a') as f:
                    f.write('%d,' % istep
                            + ','.join('%.10e' % v for v in np.asarray(dep).ravel())
                            + '\n')  # create with the step number

    #def my_function(self, DM: DisNetManager, state: dict):
    #    print("my_new_function is called")
    #    pass


    def step_end(self, DM: DisNetManager, state: dict):
        """(09/09/2025, kyeongmi) ABAQUS should be terminated at the end of the simulation.
        """
        foldername_ABAQUS = state.get("foldername_ABAQUS", None)
        jobname_head = state.get("jobname_head", None)
        ABAQUS_input_filename = state.get("ABAQUS_input_filename", None)
        num_cpus = state.get("num_cpus")
        ABAQUS_jobname = jobname_head + ABAQUS_input_filename
        istep = state.get("istep")
        #print(f"DEBUG: istep = {istep}, max_step = {self.max_step}, type(istep) = {type(istep)}")
        if istep == self.max_step -1:
            # (09/09/2025, kyeongmi) Creates ABAQUS_stop.flag
            ABAQUS_stop = f"{foldername_ABAQUS}/ABAQUS_stop.flag"
            with open(ABAQUS_stop, "w") as f:
                f.write("")
            print(f"[istep = {istep}] created {ABAQUS_stop} file, vexternaldb forces ABAQUS to quit ..")
            ABAQUS_pause = f"{foldername_ABAQUS}/ABAQUS_pause.flag"
            if os.path.exists(ABAQUS_pause):
                os.remove(ABAQUS_pause)
                print(f"    remove {ABAQUS_pause} file")
            else:
                print(f"    {ABAQUS_pause} does not exist.")
            ABAQUS_running = f"{foldername_ABAQUS}/ABAQUS_running.flag"
            if os.path.exists(ABAQUS_running):
                os.remove(ABAQUS_running)
                print(f"    remove {ABAQUS_running} file")
            else:
                print(f"    {ABAQUS_running} does not exist.")
        else:
            pass


    def step(self, DM: DisNetManager, state: dict):
        istep = state.get("istep", -1)


        self.step_begin(DM, state)


        self.step_integrate(DM, state)
        
        self.step_post_integrate(DM, state)
        

        self.step_topological_operations(DM, state)
        

        self.step_update_response(DM, state)
        self.step_nucleate(DM, state)   #  Apply nucleation-induced plastic strain just before writing files  
                                        # to ensure it is fully reflected in the output
        self.step_write_files(DM, state)
        self.step_print_info(DM, state)
        self.step_visualize(DM, state)

        self.step_end(DM, state)
        print(f"[DEBUG] step {istep} at END: #nodes = {DM.get_disnet(DisNet).num_nodes()}")

        return state
    


class Nucleation:
    MAT_TYPE_FCC = 1
    MAT_TYPE_BCC = 2
    # def __init__ replaces "param.cc" in C code 
    def __init__(self, 
                 workdir=None,
                 dir_femstress="",
                 stress_filename='SurfaceStress',
                 material_type=MAT_TYPE_FCC,
                 # thermal / physical
                 temperature_K: float = 298.15, 
                 v0: float = 1.0e13,                     # attempt frequency [1/s]
                 # activation constants
                 act_a=4.811799e+00,
                 act_b=-2.359345e+00,
                 act_c=4.742173e-03,
                 act_d=-2.457447e+00,
                 act_e=-1.330434e-01,
                 # thresholds / time
                 nuc_stress_min: float = 0.0,     
                 dt: float = 1e-8,
                 # SCF params
                 rng_seed: int = 8917346,
                 scf_mu: float = 1.0,
                 scf_sd: float = 0.05,
                 scf_min: float = 0.0,
                 scf_max: float = 9.0,
                 overrides: dict | None = None):
        o = overrides or {}

        # workdir or directory
        self.workdir = workdir or os.getcwd()
        self.dir_femstress = dir_femstress
        self.stress_filename = stress_filename

        self.material_type = material_type
        self.T = float(temperature_K)
        self.kB_eV = 8.6173324e-5 * self.T    # [eV]  
        self.v0 = float(v0)                   # [1/s]

        self.a, self.b, self.c, self.d, self.e = act_a, act_b, act_c, act_d, act_e
        self.nuc_stress_min = float(nuc_stress_min)
        self.dt = float(dt)

        # random 
        self.rng = np.random.default_rng(int(rng_seed))

        # SCF params 
        self.scf_mu  = float(scf_mu)
        self.scf_sd  = float(scf_sd)
        self.scf_min = float(scf_min)
        self.scf_max = float(scf_max)

        # empty data
        self.ids = None          # (N,) int
        self.x = self.y = self.z = None
        self.S11 = self.S22 = self.S33 = None
        self.S23 = self.S13 = self.S12 = None
        self.SCF = None          # (N,)

        # Initialize kMC buffer
        self.RSSmax = None       # (N,)
        self.SlipID = None       # (N,)
        self.P = None            # (N,)
        self.R = None            # (N,)
        self.F = None            # (N,)

        self.time_now = 0.0
        # self.nuc_freq = 1.0e-12
 
        


    def make_nuc_sites(self):
        """
        Read SurfaceStress , Generate SCF ~ N (1, 0.05^2)  in 0<SCF<9
        """
        path = os.path.join(self.workdir, self.dir_femstress, self.stress_filename)
        data = np.loadtxt(path, ndmin=2)  # N, 10 

        if data.shape[1] != 10:
            raise ValueError(f"SurfaceStress must have 10 columns, got {data.shape[1]}")

        # initialization(Element id, center xyz, stress tensor 6)
        self.ids = data[:, 0].astype(int)
        self.x, self.y, self.z = data[:, 1], data[:, 2], data[:, 3]
        self.S11, self.S22, self.S33 = data[:, 4], data[:, 5], data[:, 6]
        self.S23, self.S13, self.S12 = data[:, 7], data[:, 8], data[:, 9]

        N = data.shape[0]


        # Generate SCF : N(1.0 , 0.05^2) & 0<scf < 9
        mu, sd = self.scf_mu, self.scf_sd                    
        lo, hi = self.scf_min, self.scf_max                 
        scf = np.abs(self.rng.normal(mu, sd, size=N))             
        bad = (scf <= lo) | (scf >= hi)                          
        while np.any(bad):                                         
            scf[bad] = np.abs(self.rng.normal(mu, sd, size=bad.sum()))
            bad = (scf <= lo) | (scf >= hi)
        self.SCF = scf

        # Initialize buffer
        self.RSSmax = np.zeros(N, dtype=float)
        self.SlipID = np.zeros(N, dtype=int)
        self.P = np.zeros(N, dtype=float)
        self.R = np.zeros(N, dtype=float)
        self.F = np.zeros(N, dtype=int)
        
        


    def compute_nucleation(self, max_nuc_count=1, time_now = None, debug_test=False):
        "Same as calculate_nucleation_cc"
        "1) Calculate RSSmax /probability(p)/cumulative probability(r)"    
        "2) Initialize F"
        "3) Set F[idx]=1 for the site selected by KMC "
        

        # 0) 
        if time_now is not None:
            self.time_now = float(time_now)


        # 1) call compute_RSS_and_probability 
        self.compute_RSS_and_probability()
        if self.F is None:
            raise RuntimeError("must be called after make_nuc_sites().")

        # 2) Initialize F
        self.F[:] = 0
        # 3) determine site for nucleation, total R is obtained by compute_RSS_and_probaiblity
        totalR = float(self.R[-1]) if self.R.size else 0.0
        if totalR <= 0.0:
            print(f"[NUC] totalR=0 → no event this step") 
            return
        if totalR < 1.0:
            # if random number is outside of  totalR, No nucleation is generated in this step.
            if self.rng.random() >= totalR:
                return
        # determine site
        u = self.rng.random() * totalR
        u = min(u, np.nextafter(totalR, 0.0))  
        # This line is consistent with algorithm i[0]< u < i[1] 
        i = int(np.searchsorted(self.R, u, side="right"))   
        if 0 <= i < self.R.size:
            self.F[i] = 1               

    def compute_RSS_and_probability(self):
        """
        1) Construct the stress tensor for each site
        2) Calculate RSS for all slip systems, then select RSSmax and Slip ID
        3) Multiple SCF 
        4) Calculate Q and P -->  P = Δt * v0 * exp(-Q/kBT)
        5) R = cumsum(p)
        """
        # check the number of Stress tensor (if it is 6 or not)
        if any(v is None for v in [self.S11, self.S22, self.S33, self.S23, self.S13, self.S12, self.SCF]):
            raise RuntimeError("Call after make_nuc_sites().")
        N = self.S11.size
        if any(x.size != N for x in [self.S22, self.S33, self.S23, self.S13, self.S12, self.SCF]):
            raise ValueError("Stress/SCF array lengths do not match.")

        # Buffer
        if self.RSSmax is None or self.RSSmax.size != N:
            self.RSSmax = np.zeros(N, dtype=float)
            self.SlipID = np.zeros(N, dtype=int)
            self.P      = np.zeros(N, dtype=float)
            self.R      = np.zeros(N, dtype=float)
              


        slip_nb = self._slip_nb()  # [(n,b), ...]
        stress_min = float(self.nuc_stress_min)  # [Pa]
        

        # call data from make_nuc_site

        for k in range(N):
            # make SS symmetric stress tensor
            SS = np.array([[self.S11[k], self.S12[k], self.S13[k]],
                          [self.S12[k], self.S22[k], self.S23[k]],
                        [self.S13[k], self.S23[k], self.S33[k]]], dtype=float)
        
            # RSS_all = b · (SS · n) for all slip system(FCC or BCC) 
            RSS_all = np.fromiter((float(np.dot(b, SS.dot(n))) for (n, b) in slip_nb), dtype=float, count=len(slip_nb))
            idx = int(np.argmax(np.abs(RSS_all)))       # Returns the index of the maximum absolute value      
            RSSmax=RSS_all[idx] * self.SCF[k]  
            self.RSSmax[k] = RSSmax
            self.SlipID[k] = idx


            # if Rssmax is below stress_min , P = 0 during this step
            if abs(RSSmax) <= stress_min:
                self.P[k] = 0.0
                continue

            RSS_GPa = max(abs(RSSmax) / 1e9, 0.5)


            #Calculate Activation energy Q (BCC, FCC)
            if self.material_type == Nucleation.MAT_TYPE_BCC:
                Qk = self.a * ((1.0 - (SS1 / self.b)) ** self.c) * (1.0 - (self.T / self.d))
            else:  # FCC
                Qk = self.a * (RSS_GPa ** self.b) - self.c * self.T * (RSS_GPa ** self.d) + self.e

        
            # Calculatre probability P 
            self.P[k] = self.dt * self.v0 * np.exp(-Qk / self.kB_eV)

        # For example, the cumulative sum of (1,2,3,4,5,6) is (1,3,6,10,15,21), and then, the last term gives the total R
        self.R = np.cumsum(self.P)
            
                                 

    def loopgenerate(self, DM,  site_index: int, state: dict, generator_fn=None):
        """
        Create the actual loop for the selected site_index
        """
        k = int(site_index)

        # check F flag
        if self.F is None or k < 0 or k >= self.F.size:
            raise IndexError(f"[NUC] invalid site_index={k}")
        if self.F[k] != 1:
            print(f"[NUC] skip site {k}: F[{k}]={self.F[k]} != 1")
            return None     

        #st copy
        st = dict(state or {})  # copy
        return self.FullLoop(DM, st, site_index=k)

    def FullLoop(self, DM: DisNetManager, state: dict, site_index: int):    
        """
        center/loop R/maxseg --> generate circle loop
        state
        - 'center': (cx, cy, cz)  # element center
        - 'loopR': float          # loop radius
        - 'mexseg': float         # maximum segment length 
        - 'normal': (nx, ny, nz)  # slip plane normal (
       
        """
        # n,b from selected slip system
        k = int(site_index)
        slip_id = int(self.SlipID[k])
        n, b = self._slip_nb()[slip_id]          
        print(f"[NUC] site {k}: slip_id={slip_id}")
        print(f"[NUC] RSSmax={self.RSSmax[k]:.3e} Pa")
        print(f"[NUC] normal (n) = {n}")
        print(f"[NUC] burgers(b) = {b}")

        # copy state as st
        st = dict(state)
        st.setdefault("center", (float(self.x[k]), float(self.y[k]), float(self.z[k])))
        st.setdefault("loopR", 40.0)
        st.setdefault("normal", n)
        st.setdefault("burgers", b)

        # call 
        c = np.asarray(st["center"], float)
        #R = float(st["loopR"])
        R = float(200)
        nrm = np.asarray(st["normal"], float)
        bvec = np.asarray(st["burgers"], float)

        # Num node: int(2πR/mexseg), 5~10
        num_min = int(st.get("num_min", 5))
        num_max = int(st.get("num_max", 10))
        seg_len = float(st.get("seg_len", st.get("mexseg", 15.0)))
        est = int(2.0 * np.pi * R / max(seg_len, 1e-9))
        #nnode = max(5, min(num_max, max(num_min, est)))
        nnode = int(6)

        # lower bound
        if nnode < num_min:
            nnode = num_min

        # upper bound
        if nnode > num_max:
            nnode = num_max
    
        # plane basis
        def _unit(v):
            v = np.asarray(v, float); nrmv = np.linalg.norm(v)
            return v / (np.linalg.norm(v) or 1.0)
        def _plane_basis_from_normal(n):
            n = _unit(n)
            seed = np.array([1.0,0,0]) if abs(n[0]) < 0.9 else np.array([0,1.0,0])
            e1 = _unit(seed - np.dot(seed,n)*n)
            e2 = np.cross(n,e1)
            return e1,e2
        e1,e2 = _plane_basis_from_normal(nrm)

        # graph tag
        G = DM.get_disnet(DisNet)
        gid0, lid0 = G.get_new_tag(recycle=False)
        # node coordinate
        thetas = np.linspace(0.0, 2.0 * np.pi, nnode, endpoint=False)
        xyz = np.array([c + R * (np.cos(t) * e1 + np.sin(t) * e2) for t in thetas])
        constraints = np.zeros((nnode, 1), dtype=int)
        constraints[[0, -1], 0] = 7

        # rn : [gid, lid, x, y, z, constraint]
        tags = np.array([[gid0, lid0 + i] for i in range(nnode)], dtype=int)
        rn = np.hstack([tags, xyz, constraints])

        
        # links: [src_idx, dst_idx, bx, by, bz, nx, ny, nz]
        links_rows = []
        for i in range(nnode):
            j = (i + 1) % nnode
            links_rows.append([i, j, bvec[0], bvec[1], bvec[2], nrm[0], nrm[1], nrm[2]])
        links = np.array(links_rows, dtype=float)

        # insert in network
        G.add_nodes_segments_from_list(rn, links)
        print(f">>> Nucleation loop inserted! center={c}, R={R}, nnode={nnode}")

        
        return {"created_nodes": nnode, "center": c.tolist(), "radius": R}


    def _slip_nb(self):
        """Constructs the nbmatrix from the C code (each row is [n(3), b(3)])
        useLabfram= true --> rotate n and b with rotatematrix
        return: list of (n(3,), b(3,))"""
        if self.material_type== Nucleation.MAT_TYPE_BCC:
            # --- normals ---
            n1  = np.array([ 1/np.sqrt(2), 0.0,  1/np.sqrt(2)])
            n2  = np.array([ 0.0,  1/np.sqrt(2),  1/np.sqrt(2)])
            n3  = np.array([-1/np.sqrt(2), 0.0,  1/np.sqrt(2)])
            n4  = np.array([ 0.0, -1/np.sqrt(2),  1/np.sqrt(2)])
            n5  = np.array([ 1/np.sqrt(2), -1/np.sqrt(2), 0.0])
            n6  = np.array([ 1/np.sqrt(2),  1/np.sqrt(2), 0.0])
            n71 = np.array([ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)])
            n72 = np.array([ 1/np.sqrt(6), -2/np.sqrt(6),  1/np.sqrt(6)])
            n73 = np.array([-2/np.sqrt(6),  1/np.sqrt(6),  1/np.sqrt(6)])
            n81 = np.array([ 1/np.sqrt(6), -1/np.sqrt(6),  2/np.sqrt(6)])
            n82 = np.array([ 1/np.sqrt(6),  2/np.sqrt(6), -1/np.sqrt(6)])
            n83 = np.array([ 2/np.sqrt(6),  1/np.sqrt(6),  1/np.sqrt(6)])
            n91 = np.array([-1/np.sqrt(6),  1/np.sqrt(6),  2/np.sqrt(6)])
            n92 = np.array([ 1/np.sqrt(6),  2/np.sqrt(6),  1/np.sqrt(6)])
            n93 = np.array([ 2/np.sqrt(6),  1/np.sqrt(6), -1/np.sqrt(6)])
            n101= np.array([ 1/np.sqrt(6),  1/np.sqrt(6),  2/np.sqrt(6)])
            n102= np.array([-1/np.sqrt(6),  2/np.sqrt(6),  1/np.sqrt(6)])
            n103= np.array([ 2/np.sqrt(6), -1/np.sqrt(6),  1/np.sqrt(6)])
            # --- burgers ---
            b11 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)])
            b12 = np.array([-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b21 = np.array([ 1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)])
            b22 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)])
            b31 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b32 = np.array([ 1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)])
            b41 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b42 = np.array([-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b51 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b52 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)])
            b61 = np.array([ 1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)])
            b62 = np.array([-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b7  = np.array([-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)])
            b8  = np.array([-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            b9  = np.array([-1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)])
            b10 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)])

            normals = [
                n1,n1,n2,n2,n3,n3,n4,n4,n5,n5,n6,n6,
                n71,n72,n73,n81,n82,n83,n91,n92,n93,n101,n102,n103
            ]
            burgers = [
                b11,b12,b21,b22,b31,b32,b41,b42,b51,b52,b61,b62,
                b7, b7, b7, b8, b8, b8, b9, b9, b9, b10, b10, b10
            ]

        elif self.material_type == Nucleation.MAT_TYPE_FCC:
            # FCC: 12
            n1 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            n2 = np.array([-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)])
            n3 = np.array([ 1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)])
            n4 = np.array([ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)])
            b11= np.array([ 1/np.sqrt(2), -1/np.sqrt(2),  0.0])
            b12= np.array([ 1/np.sqrt(2),  0.0,         -1/np.sqrt(2)])
            b13= np.array([ 0.0,          1/np.sqrt(2), -1/np.sqrt(2)])
            b21= np.array([ 1/np.sqrt(2),  0.0,          1/np.sqrt(2)])
            b22= np.array([ 1/np.sqrt(2),  1/np.sqrt(2),  0.0])
            b23= np.array([ 0.0,         -1/np.sqrt(2),  1/np.sqrt(2)])
            b31= np.array([-1/np.sqrt(2), 0.0,           1/np.sqrt(2)])
            b32= np.array([ 0.0,          1/np.sqrt(2),  1/np.sqrt(2)])
            b33= np.array([ 1/np.sqrt(2),  1/np.sqrt(2),  0.0])
            b41= np.array([ 1/np.sqrt(2),  0.0,           1/np.sqrt(2)])
            b42= np.array([ 0.0,          1/np.sqrt(2),  1/np.sqrt(2)])
            b43= np.array([-1/np.sqrt(2),  1/np.sqrt(2),  0.0])

            normals = [n1,n1,n1, n2,n2,n2, n3,n3,n3, n4,n4,n4]
            burgers = [b11,b12,b13, b21,b22,b23, b31,b32,b33, b41,b42,b43]
        else:
            raise ValueError("Unknown materialType")
              
        if getattr(self, "use_lab_frame", False):
            R = np.asarray(getattr(self, "rot_matrix", np.eye(3)), dtype=float).reshape(3,3)
            normals = [R.dot(n) for n in normals]
            burgers = [R.dot(b) for b in burgers]

        return list(zip(normals, burgers))
   


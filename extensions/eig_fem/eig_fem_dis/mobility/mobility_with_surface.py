"""@package docstring
MobilityLaw: class for defining Mobility Laws with special treatment for surface nodes

Provide mobility law functions for all nodes including surface nodes
"""
import sys, os
pydis_paths = ['../../../../python', '../../../../lib', '../../../../core/pydis/python']
[sys.path.append(os.path.abspath(path)) for path in pydis_paths if not path in sys.path]

import numpy as np
from pydis.disnet import DisNet, DisNode, Tag
from framework.disnet_manager import DisNetManager
from framework.mobility_base import MobilityLaw_Base

from typing import Tuple

# (11/18/2026, junho) ParaDiS tests a node with
#     HAS_ANY_OF_CONSTRAINTS(constraint, FEM_DD_SURFACE_NODE)
# which is ((constraint & 6) != 0) -- include/Node.h:62,90.  Keep the same
# bit test, so the two codes call the same nodes surface nodes.  It is true
# for 6 and also for 7, the pinned nodes, which the bulk mobility already
# holds still, so including them changes nothing.
SURFACE_BITS = 6


class MobilityLaw(MobilityLaw_Base):
    """MobilityLaw: class for mobility laws of surface nodes
    """
    def __init__(self, state: dict={}, mobility_bulk=None,
                 surface_mobility: bool=True) -> None:
        self.mobility_bulk = mobility_bulk
        # (11/18/2026, junho) Hold surface nodes still, as ParaDiS does.
        # On is the only setting that matches ParaDiS; off is for showing
        # what the rule does.
        self.surface_mobility = surface_mobility
        
    def OneNodeMobility_Project(self, DM: DisNetManager, state: dict, tag: Tag,
                                f: np.array, update_state: bool=True) -> np.array:
        """A surface node does not move.

        ParaDiS means to give a surface node the average velocity of its
        unconstrained neighbours (src/MobilityLaw_FCC_0.cc:495-537), but
        that branch cannot run.  It counts neighbours with

            HAS_ANY_OF_CONSTRAINTS(nbr->constraint, UNCONSTRAINED)

        and UNCONSTRAINED is 0, so the test is ((c & 0) != 0), false for
        every node.  The count is always zero, the code falls into its
        "num <= 1" case, and the velocity is set to zero.  The Lr_max >=
        100 rule just below it can then never change anything either.

        So what every binary actually does is hold surface nodes still,
        and that is what is reproduced here.  Writing the averaging rule
        instead would NOT reproduce ParaDiS.

        (11/18/2026, junho)
        """
        return np.zeros(3)

    def Mobility(self, DM: DisNetManager, state: dict) -> dict:
        """Mobility: bulk velocities, then hold the surface nodes still."""
        state = self.mobility_bulk.Mobility(DM, state)
        if not self.surface_mobility:
            return state

        vel_dict = state.get("vel_dict")
        if vel_dict is None:
            print("Mobility: no vel_dict to correct")
            return state

        G = DM.get_disnet(DisNet)
        held = 0
        for tag in vel_dict:
            if int(G.nodes(tag).constraint) & SURFACE_BITS:
                vel_dict[tag] = self.OneNodeMobility_Project(
                    DM, state, tag, vel_dict[tag])
                held += 1
        state["vel_dict"] = vel_dict
        state = DisNet.convert_nodevel_dict_to_array(state)
        print("Mobility: %d surface/pinned node(s) held still" % held)
        return state

    def OneNodeMobility(self, DM: DisNetManager, state: dict, tag: Tag, f: np.array, update_state: bool=True) -> np.array:
        """OneNodeMobility: compute and return the mobility of one node specified by its tag
        """
        v = mobility_bulk.OneNodeMobility(DM, state, tag, f, update_state)
        #v = self.OneNodeMobility_Project(DM, state, tag, f, update_state)
        # update velocity dictionary if needed
        #if update_state:
        #    if "nodevels" in state and "nodeveltags" in state:
        #        nodeveltags = state["nodeveltags"]
        #        ind = np.where((nodeveltags[:,0]==tag[0])&(nodeveltags[:,1]==tag[1]))[0]
        #        if ind.size == 1:
        #            state["nodevels"][ind[0]] = v
        #        else:
        #            state["nodevels"] = np.vstack((state["nodevels"], v))
        #            state["nodeveltags"] = np.vstack((state["nodeveltags"], tag))
        #    else:
        #        state["nodevels"] = np.array([v])
        #        state["nodeveltags"] = np.array([tag])
        return v
    
        

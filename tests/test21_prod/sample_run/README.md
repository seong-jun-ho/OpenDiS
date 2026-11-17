# sample_run — one exchange of the coupled test, as it happened

(11/17/2026, junho)

A snapshot of the files DD and FEM pass each other in `tests/test21_prod`,
kept so the interface can be read without running anything.

| | |
|---|---|
| commit | `f05c0e3` |
| steps | 20 |
| wall time | 29.9 s |
| dislocation | one Frank-Read source, 5 segments |
| mesh | 1000 C3D8R elements, 488 on the surface |
| VUMAT | `extensions/eig_fem/eig_fem_dis/abaqus/VUMAT_DDM_multiple_DDcycle.for` (production, 1,911 lines) |

Taken from the first cycle whose stress had arrived, before
`ProdProtocol.read_element_stress` emptied it.

## DD -> FEM

| file | what it is |
|---|---|
| `Data_segs.txt` | the segments. Header `n_seg n_final t_step EffSlip`, then per row:<br>`id1 id2 \| r1_old(3) r2_old(3) \| r1_new(3) r2_new(3) \| dyad(9)`.<br>**old positions first**; dyad is the unit swept-face normal (x) b |
| `rn_matrix.txt` | the nodes. Header `num_rn`, then `id flag n_neighbours x y z` (id from 0).<br>flag follows ParaDiS: 0 free, 6 surface, 7 pinned |
| `surface_elements.txt` | `elem_id flag`, 6 if the element touches the free surface. Written once, from the mesh |
| `OriginalCoordNodes.txt` | `elem_id` + 8 nodes x 3 coordinates. Written once, from the mesh |
| `FEM_deltaU.txt` | `LoadType FEM_Du` |

## FEM -> DD

| file | what it is |
|---|---|
| `Output_stress0001.txt` | element stress from the VUMAT:<br>`elem_id sxx syy szz syz szx sxy T` (MPa; zx is xz) |
| `Output_link0001.txt` | `node1 node2 element` - which element the VUMAT put each segment in.<br>Read instead of searching, so we take the stress of the element the eigenstrain went into |
| `Surface_stress0001.txt` | surface sites for nucleation: `elem_id` + 6 stress components |
| `rn_check0001.txt` | `id status x y z`. The VUMAT projects a surface node onto the FEM surface and reports the new position; `0 0 0` means the node is inside.<br>**Nothing reads this yet** - see `MEMO_prod_vumat_port.md` section 7 |

## After the conversion

| file | what it is |
|---|---|
| `FEMSTRESS` | what `calforce` reads: `n1 n2` + 6 stress components (Pa) + temperature.<br>Element stress mapped onto segments through `Output_link`, scaled by 1e6 |
| `prod_log.txt` | the `[PROD]` lines of the run - the flag handover and the file counts each cycle |

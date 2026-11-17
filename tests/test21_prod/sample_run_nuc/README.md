# sample_run_nuc — the same exchange, with nucleation on

(11/17/2026, junho)

Companion to `../sample_run`, which has one Frank-Read source and nothing
else. Here nucleation is on, so loops appear and the network grows.

| | |
|---|---|
| steps | 30, 4.169085e+01 s |
| nucleation | **13 events**, six nodes each (a hexagonal loop) |
| nodes | 5 -> **79** |
| segments | 5 -> **80** |
| candidate sites | 488 (every surface element) |
| max RSS reached | 3.807e+09 Pa |
| VUMAT | production, `extensions/eig_fem/eig_fem_dis/abaqus/VUMAT_DDM_multiple_DDcycle.for` |

## The load is not the one you would use for physics

The prescribed displacement is **10000. b** on a 1000 b cube, raised from the
20 b of `../sample_run`. At 20 b the resolved shear stress only reaches
1.7 MPa and the nucleation rate comes out at 1e-277 - nothing can fire. The
load here is chosen to show the path working, not to model anything.

`ABAQUS/_inp.bak_load20` keeps the original deck.

## Nucleation is exponential in stress

From this run's log:

| max \|RSS\| (Pa) | totalR |
|---|---|
| 4.4e+08 | 6.9e-277 |
| 9.8e+08 | 1.9e-53 |
| 1.6e+09 | 1.7e-13 |
| 2.2e+09 | 2.5e-02 |
| 2.8e+09 | 7.7e+02 |
| 3.3e+09 | 1.4e+05 |

Seven times the stress, 10^282 times the rate. Which is why the stress
coming back from FEM has to be the real one.

## What the plastic strain does

```
step  0 :  xz only, 1.000e-06            one dislocation, simple shear
step 29 :  xx  1.08e-05   yy -1.57e-05   zz  4.92e-06
           yz  2.56e-06   xz  1.95e-05   xy  5.75e-07
```

All six components, once loops on several planes are present.

## Still missing: the G/10 cap

ParaDiS clamps the resolved shear stress at a tenth of the shear modulus
(`LoopGenerate.cc:559-562`). That is not ported, so RSS here runs up to
3.807e+09 Pa.

## Files

Same set as `../sample_run`, plus:

| file | what it is |
|---|---|
| `SurfaceStress` | the 10-column file the nucleation module reads: `id x y z S11 S22 S33 S23 S13 S12`, built from `Surface_stress0001.txt` by adding element centroids |
| `dEp_pydis.csv` | plastic strain per step, six components |
| `prod_log.txt` | the `[PROD]` and `[NUC]` lines |

A 16 s video of this run is in the handover folder as
`figures/test21_prod_nucleation.mp4` - not committed here, it is a binary.

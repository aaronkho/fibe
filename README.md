# fibe
FIxed Boundary Equilibrium (FIBE) solver for tokamak plasma geometries in pure Python

## Installation

This package is available on PyPI, so installation can simply be done with:

```
pip install fibe
```

However, if you wish to make modifications to the source code or use alpha versions, it is possible to install via:

```
git clone <this_repo>
pip install [--user] -e fibe
```

## Quick setup

Using the following lines in your script should allow for quick setup (parameters in <> are user-defined):

```
from fibe import FixedBoundaryEquilibrium
eq = FixedBoundaryEquilibrium()
eq.define_grid_and_boundary_with_mxh(
    nr=129,
    nz=129,
    rgeo=<R0_m>,
    zgeo=<Z0_m>,
    rminor=<Rmin_m>,
    kappa=<kappa>,
    cos_coeffs=[0.0, 0.0, 0.0],
    sin_coeffs=[0.0, np.arcsin(<delta>), -<zeta>],
)
eq.initialize_profiles_with_minimal_input(<P0_Pa>, <Ip_A>, <Bt_T>)
eq.initialize_psi()
eq.solve_psi()
```

To view the solution in script:

```
eq.plot_contour()
eq.plot_profiles()
```

To save the output in GEQDSK format:

```
eq.to_geqdsk(<geqdsk_filename>)
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`fibe` (FIxed Boundary Equilibrium) is a pure-Python fixed-boundary Grad-Shafranov (GS) equation
solver for tokamak plasma equilibria. Given a plasma boundary shape and pressure/current profiles,
it solves for the 2D poloidal flux map `psi(R,Z)` on a Cartesian (R,Z) finite-difference mesh, then
derives the usual equilibrium quantities (q-profile, flux-surface averages, safety factor, current
density, etc.). It reads/writes the standard G-EQDSK equilibrium file format and is meant to
interoperate with tools like `eqdsk` and `megpy`.

Everything is exposed through a single class, `fibe.FixedBoundaryEquilibrium`
(`src/fibe/core/classes.py`), which wraps the numerical routines in `src/fibe/core/math.py`. See
`README.md` for the canonical quick-start snippet (build a grid/boundary via MXH shaping, set
profiles, `initialize_psi()`, `solve_psi()`).

## Commands

Install (editable, with dev extras):
```
pip install -e ".[dev]"
```

Run the full test suite:
```
pytest
```

Run a single test file / test:
```
pytest tests/test_classes.py
pytest tests/test_classes.py::TestInitializationWithFP::test_psi_solver
```

Coverage:
```
pytest --cov=fibe
```

There is no lint/format tooling configured in this repo (no ruff/flake8/black config) — don't
assume one exists.

Console entry points (installed via `pyproject.toml`):
- `fibe_regrid_geqdsk` → `src/fibe/scripts/regrid_eqdsk.py` — reload a G-EQDSK, interpolate `psi`
  onto a new grid resolution, and re-converge.
- `fibe_q_with_bo` → `src/fibe/scripts/bayesian_optimization.py` — example of driving the F-profile
  shape with Optuna Bayesian optimization to hit target q-axis/q-edge values (requires `optuna`,
  which is an optional extra, not a hard dependency).

## Architecture / data flow

### Core object model

- `src/fibe/core/classes.py` — `FixedBoundaryEquilibrium`. All state lives in three dict-like
  attributes on the instance:
  - `self._data` — the main bag of arrays/scalars (grid vectors, `psi`, boundary/limiter contours,
    profiles, flux/current arrays, MXH shaping coefficients, solver diagnostics). Field names
    mirror G-EQDSK conventions (`rleft`, `rdim`, `zmid`, `zdim`, `simagx`, `sibdry`, `rmagx`,
    `zmagx`, `fpol`, `pres`, `ffprime`, `pprime`, `qpsi`, `cpasma`, `bcentr`, `rbdry`/`zbdry`,
    `rlim`/`zlim`, `jstar`, ...).
  - `self._fit` — spline fits (`tck` tuples) used to evaluate profiles/boundary continuously
    between grid points (e.g. `pres_fs`, `fpol_fs`, `qpsi_fs`, `jstar_fs`, `psi_rz`, boundary-gradient
    splines).
  - `self._fs` — dict keyed by psi level → traced flux-surface contour + its flux-surface-averaged
    geometric factors (used for q, current, and profile integrals).
  - Many mutating operations (`regrid`, COCOS conversion, etc.) stash a pre-mutation copy under
    `*_orig` keys in `self._data` so the previous state/grid can still be plotted or diffed
    (`plot_equilibrium_comparison`).

- `src/fibe/core/math.py` — stateless numerical routines called by `classes.py`. No solver state is
  held here; everything is passed in/out as plain arrays.

### Typical call sequence (see README.md for the canonical example)

1. **Grid + boundary**: `define_grid`/`define_grid_and_boundary_with_mxh` (or `define_boundary`,
   `define_boundary_with_mxh`, `define_wall`) — MXH (Miller Extended Harmonic) coefficients
   (`rgeo`, `zgeo`, `rminor`, `kappa`, `cos_coeffs`, `sin_coeffs`) are the canonical way to
   parameterize the plasma boundary shape from scratch, and are also computed back out of a solved
   equilibrium (`compute_mxh_parameters`) as a shape diagnostic.
2. **Profiles**: `initialize_profiles_with_minimal_input` or the individual
   `define_pressure_profile`/`define_f_profile`/`define_q_profile`/`define_plasma_current`/
   `define_toroidal_field`/`define_toroidal_current_density_profile` setters — each fits a bounded
   1D spline over normalized psi. `define_toroidal_current_density_profile` sets a target `jstar`
   (toroidal-current-density) profile instead of `fpol` directly; `initialize_current` then derives
   a consistent `fpol` from it (see "j*-driven initialization" below).
3. **Psi initialization**: `initialize_psi()` builds the irregular-boundary finite-difference
   stencil (`create_finite_difference_grid` → `math.generate_finite_difference_grid`), factorizes
   the sparse GS operator (`make_solver` → `scipy.sparse.linalg.factorized`), generates a seed
   `psi`, locates the magnetic axis, extends `psi` beyond the boundary, traces flux surfaces, and
   builds boundary splines. `initialize_current()` follows to seed the toroidal current.
4. **Solve**: three solvers of increasing self-consistency, all Picard iteration (direct sparse
   linear solve per step, not Newton) around the elliptic GS operator:
   - `solve_psi()` — fixed p(psi)/F(psi) shape. Takes `fixed_pressure=False` (default): if `True`,
     the Picard loop holds the pressure-driven current fixed to p(psi) and rescales only the
     F-driven current to match `cpasma` (with a fallback to the default uniform rescale when the
     F-driven current is too small to safely divide by).
   - `solve_psi_with_f_iteration()` — outer loop that refits F(psi) from the flux-surface-averaged
     F each iteration (using `fixed_pressure=True` internally, since holding p(psi) fixed while
     matching `cpasma` is its whole purpose).
   - `solve_psi_using_q_profile()` — outer loop that adjusts F(psi) to match a target q-profile.
5. **Post-processing**: `find_magnetic_axis`/`find_x_points`, `trace_flux_surfaces`,
   `recompute_pressure/f/q/phi_profile*`, `compute_flux_surface_averaged_jtor/jstar_profile`,
   `compute_mxh_parameters`, `regrid` (remap `psi` onto a new grid via bivariate spline),
   `check_psi_solution`.
6. **I/O / visualization**: `load_geqdsk`/`from_geqdsk`/`to_geqdsk` (COCOS handling lives in
   `utils/eqdsk.py`), `plot_contour`/`plot_profiles`/etc. (delegate to `utils/plotting.py`).

### `math.py` organization

- **Grid / finite-difference operator**: `generate_boundary_maps` (bit-flag classification of
  which neighbor of a boundary-adjacent cell is outside the plasma), `generate_finite_difference_grid`
  (builds the irregular-boundary FD stencil coefficients), `compute_finite_difference_matrix`
  (assembles the sparse pentadiagonal GS operator via `scipy.sparse.spdiags`).
- **GS solve step**: `compute_jtor`/`compute_psi` — one Picard iteration is essentially
  `psi_new = -solver(s5 * cur)` using the prefactored sparse LU solver.
- **Root-finding**: magnetic axis via quadratic Taylor expansion (`find_extrema_with_taylor_expansion`)
  and `megpy`'s null-point finder; x/o-points likewise via `megpy`. Older `scipy.optimize.root`
  on bivariate splines (`old_find_magnetic_axis`/`old_find_x_points` in `classes.py`) are kept
  intentionally as legacy fallback/comparison paths — not dead code to delete casually.
- **Contour tracing / flux-surface integrals**: `trace_contours_with_contourpy` (coarse, via
  `contourpy`), `trace_contour_with_splines` (fine, ray-based `brentq` root finding),
  `trace_contour_with_megpy`, `compute_flux_surface_quantities`/`compute_flux_surface_average_factors`,
  `compute_safety_factor_contour_integral`, `compute_jtor/jpar/jstar_contour_integral`.
  `compute_flux_surface_average_factors`'s outputs (`fs_vprime`, `fs_ir`, `fs_ir2`, ...) are *raw*
  line integrals (`sum(X * dl/Bp)`), not normalized flux-surface averages — divide by `fs_vprime`
  to get the average.
- **Boundary shape**: MXH coefficient conversion (`compute_mxh_coefficients_from_contours` /
  `compute_contours_from_mxh_coefficients`), boundary-segment and boundary-gradient splines,
  psi extension beyond the LCFS (`compute_psi_extension`). `shapely` is used for point-in-polygon
  containment tests, `pandas` for de-duplicating boundary/wall points.
- **Profile shaping / optimization**: parametric shape helpers (e.g. `weighted_exponential_shape`,
  `weighted_beta_shape`) and `optimize_ffprime` (`scipy.optimize.least_squares`, trust-region-
  reflective) for fitting FF' coefficients to axis value / target Ip / smooth shape simultaneously.

### Conventions worth knowing before editing

- Internally, `psi` is always stored with `simagx < sibdry` (axis flux less than boundary flux) —
  this is FiBE's fixed internal COCOS convention (COCOS=2). `insert_geqdsk_dict`/`initialize_psi`
  flip the sign of `psi`/`pprime`/`ffprime`/`qpsi` on load if the source G-EQDSK uses the opposite
  ordering, and `initialize_psi()` swaps `simagx`/`sibdry` (and flips `psi`) if the bootstrapped
  `psi_mult` normalization comes out negative, to preserve this ordering. COCOS detection/conversion
  utilities live in `utils/eqdsk.py` (`detect_cocos`/`convert_cocos`/`define_cocos`).
- `compute_jtor`/`compute_jpar` in `math.py` bake in an explicit sign flip "for COCOS convention" —
  easy to break silently if refactored without checking against a known G-EQDSK.
- Normalized psi (`xpsi`) is clipped to `>= 0` (negative numerical noise near the axis is zeroed),
  not clamped to 1 at the boundary.
- `_data['inout']` uses bit flags (`0b10`/`0b100`/`0b1000`/`0b10000`) marking which neighbor of an
  edge cell lies outside the plasma boundary; the irregular-boundary FD stencil coefficients
  (`s1..s5`, `a1,a2,b1,b2`) depend on this classification.
- `pprime`/`ffprime` as stored in `self._data` are derivatives with respect to *normalized* psi
  (`dP/dpsinorm`), not physical psi; `compute_ffprime_and_pprime_grid` is what rescales them by
  `1/(sibdry-simagx)` into the `dP/dpsi`/`dF/dpsi * F` units the GS operator (`compute_jtor`) and
  most flux-surface-integral helpers actually expect. Passing the raw stored values into a
  formula that wants physical-psi derivatives is an easy, silent unit-mismatch bug.
- G-EQDSK read/write supports three interchangeable backends (`utils/eqdsk.py`): the `eqdsk`
  package, `megpy`, and a self-contained pure-Python fallback (`read_geqdsk_file_fibe`/
  `write_geqdsk_file_fibe`) with no external dependency — useful if `eqdsk`/`megpy` aren't
  installed or behave unexpectedly.
- `utils/boundary.py` and `utils/profiles.py` are loose readers for externally supplied boundary/
  profile data (xarray/pandas/ascii), independent of the G-EQDSK path.

### j*-driven initialization (`define_toroidal_current_density_profile`)

An alternative to specifying `fpol` directly: define a target toroidal-current-density profile
`jstar(psinorm)` (a flux-surface average, `<Jtor/R>_fs/<1/R>_fs`, computed/verified by
`compute_jstar_contour_integral` — *not* the plain average `<Jtor>`), and let `initialize_current`
derive a consistent `fpol` from it via `recompute_f_from_toroidal_current_density`
(`compute_ffprime_from_jstar_pprime_and_contour` inverts the j*/p'/contour-geometry relationship
per flux surface). Deriving `fpol` this way needs the *true* physical psi span
(`sibdry - simagx`) to convert `pprime` into physical units, but that span is only known after
`cpasma` — itself derived from the `fpol` profile being computed — is known. `initialize_current`
resolves this with a small bounded fixed-point loop (guess a span, derive `fpol`, compute the
resulting `cpasma`, refine the span, repeat) before settling on a final `fpol`/`cpasma`.

This derivation uses the flux-surface geometry of whatever `psi` field currently exists (at
initialization, just the crude seed from `generate_initial_psi`). Once `fpol` is fit into a spline,
`solve_psi()`'s Picard iteration treats it as fixed and never re-derives it as the geometry
converges to the true equilibrium, so the achieved `jstar` profile (check via
`compute_flux_surface_averaged_jstar_profile`) can still drift from the target `jstar` after a full
solve, even though it matches almost exactly right after `initialize_psi()`.

## JAX autodiff scaffold (`src/fibe/jax/`, `jax-autodiff` branch, experimental)

Differentiates a converged GS solution w.r.t. profile shape / `cpasma` via implicit
differentiation, instead of unrolling the numpy Picard loop through autodiff. Requires the `jax`
extra (`pip install -e ".[jax]"`). Enables `jax_enable_x64` at import time — JAX's float32 default
silently caps agreement with the numpy solver (which converges to `erreq~1e-8`) at ~1e-7 relative
precision.

- `profiles.py` — hand-rolled JAX (Cox-de Boor recursion) evaluation of the cubic B-splines
  `core.math.generate_bounded_1d_spline` produces, validated against `scipy.interpolate.splev` to
  ~1e-10. Treats a fitted `(t, c, k)` as the interface — knots `t`/degree `k` frozen, coefficient
  vector `c` the differentiable leaf — rather than re-deriving `splrep`'s own (nonlinear, when
  smoothed) knot placement in JAX. `BSplineTCK` is a hand-registered pytree (not a `NamedTuple`)
  with `k` pinned as static aux data; a `NamedTuple` lets `jax.jit` trace `k`, which breaks the
  python-level slicing (`c[1:n]`, `range(k+1)`) that assumes it's concrete.
- `residual.py` — `gs_residual(psi, params, grid)` = `A @ psi + s5*cur(psi, theta)`, the same
  fixed-point equation `core.math.compute_psi`'s Picard step encodes
  (`psi_new = -solver(s5*cur)` ⟺ `A@psi_new + s5*cur(psi_old) = 0` at convergence). `A` is reused
  directly from `eq._data['matrix']` via `jax.experimental.sparse.BCOO`, not re-derived — only the
  nonlinear current term needs to be JAX-native. `GSGridConstants.grid_from_equilibrium(eq)`
  freezes grid/boundary geometry and the magnetic-axis grid index from an already-converged
  `FixedBoundaryEquilibrium`; `GSProfileParams.params_from_equilibrium(eq)` extracts the
  differentiable profile coefficients + `cpasma`. Only profile shape and `cpasma` are
  differentiable in this first pass — boundary-shape sensitivity would additionally require
  `A`/`s5`/the axis index themselves to become functions of theta.
  - `magnetic_axis_flux` ports `find_extrema_with_taylor_expansion`'s quadratic-Taylor axis-value
    formula, freezing only the `argmax` *grid index* (envelope theorem: the axis is a stationary
    point, so its location is insensitive to psi to first order) — the value itself stays a live,
    differentiable function of psi. `grid_from_equilibrium` must restrict that argmax search to
    `ijin` (interior points): `eq._data['psi']` has usually been through
    `extend_psi_beyond_boundary()`, which extrapolates *exterior* points to large values for
    plotting, and an unrestricted argmax grabs one of those instead of the true axis.
- `solve.py` — `solve_gs_implicit(params, grid, eq)` wires `gs_residual` into
  `jax.lax.custom_root`: the forward solve reuses `eq.solve_psi()` unchanged via
  `jax.pure_callback` (mutates `eq` in place — fine for the single-equilibrium case this targets,
  but will misbehave under `jax.vmap` over multiple parameter sets, which would all fight over the
  same mutable `eq`). The backward pass solves `J @ dpsi = y` (`J = ∂(gs_residual)/∂psi`) via a
  **dense** `jax.jacfwd` + `jnp.linalg.solve` — `jax.scipy.sparse.linalg.gmres` looks like the
  natural fit (`J` isn't symmetric, so CG is invalid) but reliably crashes with
  `NotImplementedError` when nested as `tangent_solve` inside `custom_root` (reproduced in a
  minimal case with no sparse ops at all — looks like a genuine `custom_root`/`custom_linear_solve`
  AD incompatibility in jax 0.10.2, not something specific to this codebase). The dense fallback is
  O(n²) memory, so it won't scale past the grid sizes tested here (tens of thousands of points) —
  revisit once the gmres/custom_root nesting issue is understood.

### Bug this surfaced in `classes.py` (not fixed there yet)

`solve_psi()`'s final `normalize_psi_to_original()` step affinely rescales `psi` to match
`simagx_orig`/`sibdry_orig` — the axis flux from whichever `solve_psi()` call happened to run
*first* — on every subsequent (non-`scratch`) call on the same object. Calling `solve_psi()` more
than once on one `FixedBoundaryEquilibrium` (e.g. after tweaking a profile) silently rescales the
new result back toward the *first* solve's scale, masking the true sensitivity to whatever
changed. `jax/solve.py`'s `_numpy_forward_solve` works around this by forcing `eq.scratch = True`
before every call (confirmed correct against an independent from-scratch numpy re-solve); any
other code path that reuses one equilibrium object across multiple `solve_psi()` calls should be
checked against this.

### Validation status (33×33 test grid)

- Forward solve matches direct `eq.solve_psi()` to ~1e-8.
- `d(psi)/d(pressure spline coefficient)` matches an independent from-scratch numpy finite
  difference to ~1e-7 relative error.
- `d(psi)/d(cpasma)` is off by ~17%, stable across FD step sizes spanning two orders of magnitude
  and unchanged by Newton-correcting `gs_residual` to machine-precision zero first (rules out
  truncation error and residual-nonzero-ness). A manual implicit-function-theorem calculation
  bypassing `custom_root` entirely reproduces the same ~17%-off value (rules out a `custom_root`
  wiring bug). Remaining suspect: `gs_residual`'s model of how `cpasma` enters through the
  `curscale` global renormalization — the one place a parameter feeds in through a domain-wide
  integral rather than pointwise. Not yet resolved — treat `cpasma` sensitivities from this module
  as directionally right but not quantitatively trustworthy until this is root-caused.

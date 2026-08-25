---
name: regrid-geqdsk
description: Reload a G-EQDSK and interpolate it onto a different (R,Z) grid resolution using fibe's fibe_regrid_geqdsk CLI, either as a pure spline upsample/downsample (--no-solve) or with a full Grad-Shafranov re-convergence. Use whenever asked to change an equilibrium's grid resolution, upsample/downsample a geqdsk, refine or coarsen a psi mesh, or fit a geqdsk's grid tightly to its own boundary contour.
---

# Regridding a G-EQDSK (`fibe_regrid_geqdsk`)

`fibe_regrid_geqdsk` (installed console command, `src/fibe/scripts/regrid_eqdsk.py`) loads a
G-EQDSK and calls `FixedBoundaryEquilibrium.regrid(nr, nz, ...)`, which is itself a *pure spline
interpolation* step: `psi` is re-evaluated on the new (R,Z) grid via a 2D bivariate spline, and
`pres`/`fpol`/`qpsi` via 1D splines onto the new grid's `psinorm` sampling.
`simagx`/`sibdry`/`rmagx`/`zmagx`/`cpasma`/`bcentr` are untouched. By default the script then also
calls `solve_psi` to re-converge the interpolated state against the actual Grad-Shafranov equation on
the new finite-difference operator (which can shift `psi` slightly from the pure interpolation,
since the interpolated profiles aren't exactly self-consistent with the new grid's own operator) —
pass `--no-solve` to skip that and keep the literal spline-interpolated fields as-is.

## Two distinct use cases — pick the right one

- **Just need a different resolution, fast, without re-deriving physics** (e.g. feeding a
  higher-resolution grid to a downstream tool that expects one, or a quick visual upsample): use
  `--no-solve`. No `--niter`/`--tol`/`--relax`/`--relaxj` involved at all (ignored if passed).
  Verified (real device G-EQDSK, 129×129 → 257×257): `cpasma`/`simagx`/`sibdry`/axis-value profiles
  come through exactly unchanged, and the interpolated `psi` evaluated back at the boundary contour
  lands cleanly on `sibdry` (a genuinely self-consistent map, just at a different resolution) — not
  a rough approximation.
- **Need the new resolution to be a genuinely re-converged equilibrium** (e.g. the *point* is resolution
  sensitivity of the actual GS solution, not just a resampled psi map): omit `--no-solve`. This
  actually re-solves, so it inherits the same convergence considerations as any `solve_psi` call —
  watch the printed `converged=`/`psi_error=` line.

## Running it

```
fibe_regrid_geqdsk equilibrium.geqdsk 257 257 --optimize --ofile regridded.geqdsk
fibe_regrid_geqdsk equilibrium.geqdsk 257 257 --no-solve --ofile upsampled.geqdsk
```

`ifile`, `nr`, `nz` are positional and required. `--ofile` defaults to
`<ifile's directory>/fibe_regridded_input.geqdsk` if omitted — **always pass `--ofile` explicitly**
when scripting this, since the script silently refuses to overwrite an existing output path (just
prints a message and does nothing) rather than clobbering it, and a stale default-named file from an
earlier run can silently make a later invocation look like a no-op. `--optimize` refits the grid's
`(rmin,rmax,zmin,zmax)` bounds tightly to the boundary (and wall, if present) contour instead of
keeping the original grid's bounds — usually what's wanted when changing resolution meaningfully,
less relevant for a same-bounds resolution bump.

## The `megpy` root-finder bug and `--new-method`

`regrid`'s own boundary retracing hits a real, confirmed `megpy` bug (a `scipy.optimize.fsolve`
shape mismatch inside its X-point finder) on realistically-shaped boundaries — **this triggers
regardless of `--no-solve`**, since `regrid` always retraces the boundary. The script defaults to
`old_method=True` (fibe's legacy `scipy.optimize.root`-based tracer, kept in `classes.py`
specifically for this) to route around it; pass `--new-method` to use the newer `megpy` tracer
instead (potentially more accurate boundary shape reconstruction when it doesn't hit the bug, e.g.
for simpler/more circular boundaries) if the default's boundary handling looks off for a specific
case. When *not* using `--no-solve`, `solve_psi`'s own separate post-loop axis refinement hits a
related `megpy` bug too, unconditionally skipped by the script (no flag to re-enable it) — the
resolved magnetic axis is only as refined as the last converged Picard iteration as a result,
negligible in practice. Full background on both bugs: this repo's `CLAUDE.md`, the
`fibe_regrid_geqdsk` entry-point description and the "Re-solving an already-loaded equilibrium..."
section (shared with `fibe_kinetic_resolve`, same underlying bug).

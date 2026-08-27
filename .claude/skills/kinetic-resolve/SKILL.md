---
name: kinetic-resolve
description: Re-solve a G-EQDSK fixed-boundary equilibrium's pressure profile against externally supplied kinetic profiles (ne/Te, or p(psin) directly) using fibe's fibe_kinetic_resolve CLI, while preserving the original equilibrium's total current profile (jstar) so the core doesn't develop an unphysical "current hole". Use whenever asked to re-solve/refine/refit an equilibrium with kinetic measurements, swap in a new pressure profile while keeping the current profile fixed, or diagnose a non-monotonic/folded flux surface after such a re-solve.
---

# Kinetic-constrained equilibrium re-solve (`fibe_kinetic_resolve`)

`fibe_kinetic_resolve` (installed console command, `src/fibe/scripts/resolve_with_kinetic_constraint.py`)
loads a G-EQDSK, replaces its pressure profile with an externally supplied one, and re-solves —
holding the boundary and total plasma current fixed, and F(psi) re-derived (not frozen) so only the
pressure-driven/F-driven *split* of the current changes. Two-stage internally: `derive_f_profile_
from_jstar_target` seeds F once (against the *original*, core-smoothed `jstar(psin)`, purely to
avoid an unphysical core "current hole" from the first Picard iteration), then `solve_psi_with_f_
iteration` takes over entirely, re-deriving F each outer iteration from the equilibrium's own
*actual* resolved current — so the target `jstar` is **not** preserved as ground truth throughout,
only used to avoid the hole. **Do not** re-solve by just calling `define_pressure_profile` +
`solve_psi` directly with F(psi) left untouched from the loaded G-EQDSK — that's the naive approach
this tool exists to replace, and it produces a real, visible "current hole": the core current
density dips/reverses unphysically wherever the new p'(psi) mismatches the original F(psi)'s
implicit assumptions, which shows up as a genuinely non-monotonic (folded) psi map near the axis,
not just a cosmetically-different profile.

Background, the real bugs this had to work around, and validation details all live in this
repo's `CLAUDE.md` under "Re-solving an already-loaded equilibrium against a new pressure profile
while preserving jstar" — read that if something here doesn't add up or the tool misbehaves in a new
way; don't re-derive the physics from scratch.

## Building the profiles file

Any format `fibe.utils.profiles.read_profiles_file` reads (xarray-openable netCDF, pandas HDF5,
whitespace-delimited ASCII — pick via `--profiles-interface`), with a `psin` (or `psinorm`/`xpsi`)
column plus either:
- `pres` (or `pressure`/`p`) — total kinetic pressure in **Pa**, directly, or
- `ne` (or `n_e`) in **m^-3** and `te` (or `t_e`) in **eV** — converted to pressure via
  `p = e*(n_e*T_e + ni_ratio*n_e * ti_ratio*T_e)` and `--ni-ratio`/`--ti-ratio` (both default 1.0 —
  override with real ion density/temperature ratios where known, e.g. accounting for impurity
  dilution or a measured Ti/Te ratio; don't leave them at the generic default and call it done if the
  caller actually has better numbers).

If the source data's flux coordinate is `rho_tor_norm` (or anything other than psin), that mapping
has to happen *before* this tool — it operates natively in `psin` (the coordinate the G-EQDSK/GS
equation itself use) and doesn't know how to derive a rho_tor_norm->psin conversion. Building it
generally needs the equilibrium's own q-profile-integrated toroidal flux
(`FixedBoundaryEquilibrium.recompute_phi_profile`, then `rho_tor_norm = sqrt(phi/phi[-1])`,
inverted); if the source is another consolidated dataset that already stores an equivalent mapping
(a poloidal-flux-normalized grid alongside a toroidal-flux value evaluated on it), prefer reusing
that over rederiving from scratch.

## Running it

```
fibe_kinetic_resolve \
    --geqdsk equilibrium.geqdsk \
    --profiles kinetic_profiles.nc \
    --ni-ratio 0.9 --ti-ratio 0.6 \
    --output resolved.geqdsk \
    --plot comparison.png
```

`--output` and `--geqdsk`/`--profiles` are required; `--plot` is optional but cheap and worth always
asking for — the comparison figure (pressure, q, j*, psi contours, psi_norm difference, all
original-vs-resolved) is the fastest way to eyeball the result before handing it back to whoever
asked for it. But don't rely on eyeballing the psi-contours panel alone to confirm "no current
hole" — call `FixedBoundaryEquilibrium.check_flux_surface_monotonicity()` on the resolved
equilibrium (see `CLAUDE.md`'s "Verifying 'no current hole' properly") for an actual check;
a naive visual check on a shaped/asymmetric plasma can miss a real defect or flag a false one.

`--nfiter`/`--errf`/`--relaxf` (defaults `10`/`1e-4`/`1.0`) control the outer `solve_psi_with_f_
iteration` loop (how many times F gets re-derived, its own convergence tolerance, and its own
damping) — separate from `--niter`/`--erreq`, which control each *inner* `solve_psi` call. Leave
these at their defaults unless the F-error specifically isn't converging (rare — the two-stage
workflow converges undamped on every real scenario checked so far).

Console output reports `converged=True/False` and the final `psi_error` — a `WARNING: did not
converge` on stderr means every entry in the internal relax schedule
(`1.0, 0.7, 0.5, 0.3, 0.15`, tried in order) failed to reach `--erreq` (default `1e-8`) within
`--niter` (default 300) iterations each. That schedule isn't CLI-exposed; if it's still not
converging, don't just crank `--niter` blindly — drop into Python and call
`fibe.scripts.resolve_with_kinetic_constraint.resolve_with_kinetic_profiles(...)` directly with a
custom, more finely-graded `relax_schedule` (e.g. extend it below 0.15), or investigate whether the
supplied pressure profile is implausibly different from the original (a huge, physically
questionable p' mismatch is exactly what makes the undamped Picard iteration oscillate instead of
converge).

## When to touch `--jstar-trust-from`

The original G-EQDSK's own `jstar(psin)` (what the resolved current profile targets) often carries a
sharp, unphysical near-axis spike from flux-surface-tracing noise — `build_core_smoothed_jstar_target`
replaces it below `--jstar-trust-from` (default `0.5`) with a quadratic-in-`psin` extrapolation,
built by ramping d(jstar)/d(psin) linearly from zero at the true axis up to the trusted region's own
local slope at the join point — continuous in both value and slope (C1) at the join by construction,
not an independent fit that merely approximates it. If the output plot's psi contours are still
folded/non-monotonic near the axis, or the "target" j* curve in the plot still shows a visible
near-axis spike, raise `--jstar-trust-from` (e.g. `0.7`) so more of the untrustworthy region gets
smoothed over.

## Known limitations (don't over-promise these to whoever asked)

- F(psi) is only *seeded* from the target `jstar`, not preserved as ground truth throughout — the
  two-stage workflow's own `solve_psi_with_f_iteration` stage re-derives F from the equilibrium's
  actual resolved current each outer iteration, so the resolved `jstar` need not match the target
  exactly (it isn't trying to). The total current (`cpasma`) itself is still preserved exactly.
- The resolved magnetic axis position, and `simagx`/`sibdry` (psi at the axis/boundary), come from
  the last converged Picard iteration, not a finer root-finder refinement (skipped due to an
  unrelated `megpy` bug) — negligible in practice (`psi.min()` differs from the written `simagx` by
  a relative ~1e-5 in cases checked) but worth knowing if someone asks about axis-position precision
  specifically, or needs `simagx`/`sibdry` to exactly match the true resolved psi extrema.

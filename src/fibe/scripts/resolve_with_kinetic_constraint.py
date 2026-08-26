"""Re-solves a fixed-boundary equilibrium's pressure profile from an
externally supplied kinetic profile (ne/Te, or a pressure profile p(psin)
directly), while holding the boundary shape, total plasma current, and
total current *profile* (jstar(psinorm), the flux-surface-averaged
Jtor/R) fixed to the original G-EQDSK -- i.e. F(psi) is re-derived (not
frozen) so that, combined with the new p'(psi), the resulting total
current density still matches the original reconstruction's; only the
*split* between the pressure-driven and F-driven contributions to the
current changes (`FixedBoundaryEquilibrium.derive_f_profile_from_jstar_
target`). This avoids the unphysical "current hole" that appears in the
core if F(psi) is instead left frozen while an independently-derived
p'(psi) (not tied to the original reconstruction) is swapped in.

Profiles file: read via `fibe.utils.profiles.read_profiles_file` (any of
the formats it understands -- xarray-openable netCDF, pandas HDF5, or
whitespace-delimited ASCII), with a 'psin' (or 'psinorm'/'xpsi') column
plus either:
  - 'pres' (or 'pressure'/'p') -- total kinetic pressure directly [Pa], or
  - 'ne' (or 'n_e') [m^-3] and 'te' (or 't_e') [eV] -- converted to
    pressure via `compute_pressure_from_kinetic_profiles` and
    --ni-ratio/--ti-ratio (n_i = ni_ratio*n_e, T_i = ti_ratio*T_e).

Known limitations, both confirmed against a real (negative-Ip/negative-Bt)
device G-EQDSK, not this script's own bugs:
- An environment-level bug in `megpy`'s null-point tracer (a
  `scipy.optimize.fsolve` shape mismatch inside `find_magnetic_axis`'s
  root-finder) crashes `solve_psi`'s post-Picard-loop axis refinement,
  independent of the pressure profile used. Skipped here (monkeypatched to
  a no-op on the instance) -- the Picard loop itself is unaffected, but the
  resolved magnetic axis is only as refined as the last converged Picard
  iteration, not the finer root-finder.
- Two more bugs in `megpy`'s contour tracer (hit by the flux-surface-
  average machinery `derive_f_profile_from_jstar_target`/`solve_psi`'s own
  q-profile recompute need) are patched at import time below, scoped to
  this process only -- see the comment there. Not applied inside `fibe`
  itself, since it isn't established whether they affect every installed
  `megpy` version or only the one this was found against; keeping the
  patch local here until that's known.

F(psi) is only *seeded* from the (optionally core-smoothed, see
`--jstar-trust-from`) target jstar via `derive_f_profile_from_jstar_target`
-- `solve_psi_with_f_iteration` then re-derives F each outer iteration from
the equilibrium's own resolved current, so the target is not preserved as
ground truth throughout, only used to avoid an unphysical core "current
hole" from the first iteration onward.

Usage:
    python3 -m fibe.scripts.resolve_with_kinetic_constraint \\
        --geqdsk equilibrium.geqdsk --profiles kinetic_profiles.nc \\
        --output resolved.geqdsk --plot comparison.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from .. import FixedBoundaryEquilibrium
from ..core.math import build_core_smoothed_jstar_target
from ..utils.profiles import read_profiles_file, compute_pressure_from_kinetic_profiles

import megpy.tracer as _megpy_tracer

# `megpy.tracer.contour` has (at least) two distinct bugs that surface when
# tracing near-axis flux surfaces (needed by
# `derive_f_profile_from_jstar_target`'s flux-surface-average machinery,
# and by `solve_psi`'s own post-solve q-profile recompute), both confirmed
# by reading megpy's own source, not just the traceback:
#  1. Its empty-contour branch (reached whenever a requested psi level has
#     zero intersections anywhere on the grid) slices a plain (x, y) tuple
#     with `[:, 0]` (`TypeError`) after referencing an undefined `radius`
#     (`NameError`) -- dead code that could never have worked.
#  2. `contour_minmax` (reached from the *non-empty* branch, for very
#     small/poorly-resolved near-axis contours) can index a `mask_out`
#     array out of bounds (`IndexError`) -- a separate bug, same failure
#     category (near-axis contours are small and easy to under-resolve).
# fibe's own caller (`trace_contour_with_megpy`, in `core/math.py`) already
# handles a *gracefully* empty contour (`{'contours': []}`) by skipping
# that level -- so the safe fix is just to stop `contour()` from crashing
# on either branch, not to change any fibe-side logic. Scoped to this
# script's own import (not applied globally inside fibe -- see module
# docstring).
_original_megpy_contour = _megpy_tracer.contour


def _patched_megpy_contour(*args, **kwargs):
    try:
        return _original_megpy_contour(*args, **kwargs)
    except (TypeError, NameError, IndexError):
        level = kwargs.get('level', args[3] if len(args) > 3 else None)
        return {'X': np.array([]), 'Y': np.array([]), 'theta_XY': np.array([]), 'level': level, 'contours': [], 'radius': 0.0}


_megpy_tracer.contour = _patched_megpy_contour

DEFAULT_RELAX_SCHEDULE = (1.0, 0.7, 0.5, 0.3, 0.15)


def compute_original_flux_surface_quantities(eq):
    """Traces flux surfaces on an as-loaded (never solve_psi'd) equilibrium
    and computes jstar from them -- the target-building input for
    `resolve_with_kinetic_profiles`.
    """
    eq.generate_psi_bivariate_spline()
    eq._fs = eq.trace_flux_surfaces()
    eq.compute_flux_surface_averaged_jstar_profile()


def resolve_with_kinetic_profiles(
    geqdsk_path,
    psin_p,
    p_new,
    jstar_trust_from=0.5,
    relax_schedule=DEFAULT_RELAX_SCHEDULE,
    niter=300,
    erreq=1.0e-8,
    nfiter=10,
    errf=1.0e-4,
    relaxf=1.0,
):
    """Loads `geqdsk_path`, replaces its pressure profile with (psin_p,
    p_new), and re-solves under a new F(psi) so the total current profile
    matches the original G-EQDSK's own (core-smoothed) jstar, holding the
    boundary and cpasma fixed. Two-stage approach (validated as strictly
    better than a one-shot F-derivation -- converges faster, undamped, and
    to a genuinely self-consistent current split rather than a uniformly-
    rescaled one):
      1. `FixedBoundaryEquilibrium.derive_f_profile_from_jstar_target`
         seeds F(psi) once, against the *original* (pre-solve) flux-surface
         geometry, purely to avoid an unphysical "current hole" forming in
         the core from the very first Picard iteration.
      2. `solve_psi_with_f_iteration` then takes over entirely: each outer
         iteration re-solves psi and re-derives F from the *actual* solved
         current (`_estimate_flux_surface_averaged_fpol`, self-consistency-
         driven), not from `jstar_target` again -- so the final F is
         consistent with the resolved geometry, not the pre-solve one.
         `jstar_target` is intentionally not preserved as ground truth
         throughout; it is only a better-than-frozen-F starting point, and
         the one hard physical requirement is that no current hole forms.

    Tries each relaxation factor in `relax_schedule` in turn (applied to
    both the F-iteration's own damping, via `relaxf`/`errf`/`nfiter` held
    fixed across the schedule, and each inner `solve_psi` call's own
    `relax`/`relaxj`) -- a substantially different p(psi) than the original
    reconstruction can make an un-damped Picard iteration oscillate in a
    limit cycle instead of converging -- and returns the first equilibrium
    that converges (or the last attempt, unconverged, with a warning, if
    none do). Confirmed on three real device G-EQDSKs (C-Mod shot
    1030516024, scenarios 54/9/197) to converge undamped
    (`relax=relaxj=1.0`) -- including scenario 197, which needed
    `relax=0.5` under the older one-shot approach.

    Returns (eq_resolved, eq_original, jstar_target, relax_used).
    """
    eq_orig = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
    compute_original_flux_surface_quantities(eq_orig)
    psin_grid = np.linspace(0.0, 1.0, eq_orig._data['nr'])
    jstar_target = build_core_smoothed_jstar_target(psin_grid, eq_orig._data['jstar'], trust_from=jstar_trust_from)

    last_eq = None
    for relax in relax_schedule:
        eq = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        eq.define_pressure_profile(p_new, psinorm=psin_p)
        eq.derive_f_profile_from_jstar_target(jstar_target, psinorm=psin_grid)  # seed F, avoids a core current hole
        eq.find_magnetic_axis = lambda: None  # see module docstring
        try:
            eq.solve_psi_with_f_iteration(
                nfiter=nfiter, errf=errf, relaxf=relaxf,
                nxiter=niter, erreq=erreq, relax=relax, relaxj=relax,
            )
        except Exception as exc:
            # An undamped (or insufficiently damped) Picard iteration can
            # diverge badly enough (psi -> NaN) that some *downstream*
            # geometry step -- not the Picard loop itself -- raises a raw
            # exception instead of solve_psi_with_f_iteration returning
            # normally with converged=False (confirmed: a real production
            # scenario hit generate_boundary_gradient_spline's IndexError
            # this way, from an all-NaN boundary gradient trace after F
            # blew up at relax=1.0). Treat any such exception the same as
            # a clean non-convergence -- log it and let the schedule try
            # the next, more damped relax value, rather than letting one
            # bad relax level abort the whole retry loop (and, one level
            # up, potentially an entire multi-scenario batch).
            print(
                f'WARNING: solve_psi_with_f_iteration raised {type(exc).__name__}: {exc} '
                f'at relax={relax} -- treating as non-convergence and trying the next relax value.',
                file=sys.stderr,
            )
            last_eq = eq
            continue
        eq.compute_flux_surface_averaged_jstar_profile()
        last_eq = eq
        if eq.converged:
            return eq, eq_orig, jstar_target, relax
    print(
        f'WARNING: did not converge with any relax in {relax_schedule} '
        f'(final psi_error={last_eq._data.get("psi_error", float("nan")):.3e}) -- using the last attempt anyway.',
        file=sys.stderr,
    )
    return last_eq, eq_orig, jstar_target, relax_schedule[-1]


def plot_comparison(eq_orig, eq_new, psin_p, p_new, jstar_target, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nr = eq_orig._data['nr']
    psin_grid = np.linspace(0.0, 1.0, nr)
    simagx_orig, sibdry_orig = float(eq_orig._data['simagx']), float(eq_orig._data['sibdry'])
    simagx_new, sibdry_new = float(eq_new._data['simagx']), float(eq_new._data['sibdry'])
    rmagx_orig, zmagx_orig = float(eq_orig._data['rmagx']), float(eq_orig._data['zmagx'])
    rmagx_new, zmagx_new = float(eq_new._data['rmagx']), float(eq_new._data['zmagx'])
    rbdry, zbdry = eq_orig._data['rbdry'], eq_orig._data['zbdry']
    rvec = eq_orig._data['rleft'] + np.linspace(0.0, 1.0, nr) * eq_orig._data['rdim']
    zvec = (eq_orig._data['zmid'] - 0.5 * eq_orig._data['zdim']) + np.linspace(0.0, 1.0, eq_orig._data['nz']) * eq_orig._data['zdim']

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes = axes.ravel()
    fig.suptitle('fibe: re-solve with externally supplied kinetic pressure')

    axes[0].plot(psin_grid, eq_orig._data['pres'] * 1e-3, c='r', label='original (geqdsk)')
    axes[0].plot(psin_p, p_new * 1e-3, c='g', ls=':', label='supplied (input)')
    axes[0].plot(psin_grid, eq_new._data['pres'] * 1e-3, c='b', ls='--', label='resolved')
    axes[0].set_xlabel('psin [-]')
    axes[0].set_ylabel('p [kPa]')
    axes[0].set_title('Pressure profile')
    axes[0].legend(loc='best', fontsize=8)

    axes[1].plot(psin_grid, eq_orig._data['qpsi'], c='r', label='original (geqdsk)')
    axes[1].plot(psin_grid, eq_new._data['qpsi'], c='b', ls='--', label='resolved')
    axes[1].set_xlabel('psin [-]')
    axes[1].set_ylabel('q [-]')
    axes[1].set_title('Safety factor profile')
    axes[1].legend(loc='best', fontsize=8)

    axes[2].plot(psin_grid, eq_orig._data['jstar'] * 1e-6, c='r', alpha=0.4, label='original (geqdsk, raw)')
    axes[2].plot(psin_grid, jstar_target * 1e-6, c='r', ls=':', label='target (smoothed)')
    axes[2].plot(psin_grid, eq_new._data['jstar'] * 1e-6, c='b', ls='--', label='resolved')
    axes[2].set_xlabel('psin [-]')
    axes[2].set_ylabel('j* = <Jtor/R>_fs / <1/R>_fs [MA/m^2]')
    axes[2].set_title('Flux-surface-averaged current density (j*)')
    axes[2].legend(loc='best', fontsize=8)

    # `psi` is extrapolated beyond the LCFS (extend_psi_beyond_boundary) so
    # it's well-defined on the whole rectangular grid -- not a physical flux
    # map, so the remaining panels are restricted to inside the boundary
    # and/or tightly cropped to the boundary's own bounding box.
    inside = eq_new._data['inout'].reshape(eq_new._data['nz'], eq_new._data['nr']) != 0
    rmargin = 0.05 * (np.nanmax(rbdry) - np.nanmin(rbdry))
    zmargin = 0.05 * (np.nanmax(zbdry) - np.nanmin(zbdry))

    lvec = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    rmesh, zmesh = np.meshgrid(rvec, zvec)
    levels_o = np.sort(lvec * (sibdry_orig - simagx_orig) + simagx_orig)
    axes[3].contour(rmesh, zmesh, eq_orig._data['psi'], levels=levels_o, colors='r', alpha=0.6, linewidths=1.0)
    levels_n = np.sort(lvec * (sibdry_new - simagx_new) + simagx_new)
    axes[3].contour(rmesh, zmesh, eq_new._data['psi'], levels=levels_n, colors='b', alpha=0.6, linewidths=1.0, linestyles='dashed')
    axes[3].plot(rbdry, zbdry, c='k', lw=1.5)
    axes[3].scatter([rmagx_orig], [zmagx_orig], marker='o', facecolors='none', edgecolors='r', label='O-point (orig)')
    axes[3].scatter([rmagx_new], [zmagx_new], marker='x', c='b', label='O-point (resolved)')
    axes[3].plot([], [], c='r', label='psi (orig)')
    axes[3].plot([], [], c='b', ls='--', label='psi (resolved)')
    axes[3].set_xlim(np.nanmin(rbdry) - rmargin, np.nanmax(rbdry) + rmargin)
    axes[3].set_ylim(np.nanmin(zbdry) - zmargin, np.nanmax(zbdry) + zmargin)
    axes[3].set_xlabel('R [m]')
    axes[3].set_ylabel('Z [m]')
    axes[3].set_aspect('equal')
    axes[3].legend(loc='best', fontsize=7)
    axes[3].set_title('psi contours (inside boundary)')

    xpsi_orig = (eq_orig._data['psi'] - simagx_orig) / (sibdry_orig - simagx_orig)
    xpsi_new = (eq_new._data['psi'] - simagx_new) / (sibdry_new - simagx_new)
    dxpsi = np.where(inside, xpsi_new - xpsi_orig, np.nan)
    vmax = float(np.nanmax(np.abs(dxpsi))) or 1.0
    im = axes[4].imshow(dxpsi, origin='lower', extent=(rvec[0], rvec[-1], zvec[0], zvec[-1]), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[4].plot(rbdry, zbdry, c='k', lw=1.0)
    axes[4].set_xlim(np.nanmin(rbdry) - rmargin, np.nanmax(rbdry) + rmargin)
    axes[4].set_ylim(np.nanmin(zbdry) - zmargin, np.nanmax(zbdry) + zmargin)
    axes[4].set_xlabel('R [m]')
    axes[4].set_ylabel('Z [m]')
    axes[4].set_aspect('equal')
    axes[4].set_title('psi_norm(resolved) - psi_norm(original), inside boundary')
    fig.colorbar(im, ax=axes[4], shrink=0.8)

    axes[5].axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--geqdsk', dest='geqdsk', type=str, required=True, help='Path to the input G-EQDSK file')
    parser.add_argument('--profiles', dest='profiles', type=str, required=True, help='Path to the externally-supplied profiles file (see module docstring for the expected columns)')
    parser.add_argument('--profiles-interface', dest='profiles_interface', type=str, default='xarray', choices=['xarray', 'pandas', 'ascii'], help='Backend to read --profiles with (see fibe.utils.profiles.read_profiles_file)')
    parser.add_argument('--ni-ratio', dest='ni_ratio', type=float, default=1.0, help='n_i = ni_ratio * n_e (only used if --profiles supplies ne/te rather than pres directly)')
    parser.add_argument('--ti-ratio', dest='ti_ratio', type=float, default=1.0, help='T_i = ti_ratio * T_e (only used if --profiles supplies ne/te rather than pres directly)')
    parser.add_argument('--jstar-trust-from', dest='jstar_trust_from', type=float, default=0.5, help='psin above which the original G-EQDSK jstar is trusted as-is; below it, a smooth (quadratic-in-psin, C1-continuous at the join) extrapolation replaces it (see build_core_smoothed_jstar_target)')
    parser.add_argument('--niter', dest='niter', type=int, default=300, help='Max Picard iterations per inner solve_psi call')
    parser.add_argument('--erreq', dest='erreq', type=float, default=1.0e-8, help='Convergence criterion on max relative psi error, per inner solve_psi call')
    parser.add_argument('--nfiter', dest='nfiter', type=int, default=10, help='Max outer F-iterations (solve_psi_with_f_iteration)')
    parser.add_argument('--errf', dest='errf', type=float, default=1.0e-4, help='Convergence criterion on max relative F error between outer F-iterations')
    parser.add_argument('--relaxf', dest='relaxf', type=float, default=1.0, help='Relaxation factor applied to F itself between outer F-iterations (1.0 = undamped)')
    parser.add_argument('--output', dest='output', type=str, required=True, help='Path to write the resolved G-EQDSK file to')
    parser.add_argument('--plot', dest='plot', type=str, default=None, help='Optional path to save a comparison plot (pressure/q/jstar/psi) to')
    return parser.parse_args()


def main():
    args = parse_args()
    profiles = read_profiles_file(args.profiles, interface=args.profiles_interface)
    if profiles.get('psinorm') is None:
        raise ValueError(f'{args.profiles} has no recognizable psin/psinorm/xpsi column.')
    psin_p = np.asarray(profiles['psinorm'], dtype=float)
    if 'pres' in profiles:
        p_new = np.asarray(profiles['pres'], dtype=float)
    elif 'ne' in profiles and 'te' in profiles:
        p_new = compute_pressure_from_kinetic_profiles(profiles['ne'], profiles['te'], ni_ratio=args.ni_ratio, ti_ratio=args.ti_ratio)
    else:
        raise ValueError(f"{args.profiles} has neither a 'pres' column nor both 'ne'/'te' columns.")

    eq_new, eq_orig, jstar_target, relax_used = resolve_with_kinetic_profiles(
        args.geqdsk, psin_p, p_new,
        jstar_trust_from=args.jstar_trust_from,
        niter=args.niter, erreq=args.erreq,
        nfiter=args.nfiter, errf=args.errf, relaxf=args.relaxf,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    eq_new.to_geqdsk(output)
    print(
        f'converged={eq_new.converged} (relax={relax_used}), psi_error={eq_new._data["psi_error"]:.3e} '
        f'-> {output}'
    )

    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(eq_orig, eq_new, psin_p, p_new, jstar_target, plot_path)
        print(f'-> {plot_path}')


if __name__ == '__main__':
    main()

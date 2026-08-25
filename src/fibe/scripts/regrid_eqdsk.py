"""Reloads a G-EQDSK, interpolates its psi map onto a new grid resolution
(optionally auto-fit to the boundary contour via --optimize), and
re-converges it -- or, with --no-solve, skips the re-convergence entirely
and just writes out the pure spline-upsampled/downsampled psi map (and
pres/fpol/qpsi, similarly spline-interpolated onto the new grid's
psinorm) as-is. FixedBoundaryEquilibrium.regrid() itself is exactly this
pure interpolation step (2D bivariate spline for psi, 1D splines for the
profiles) -- simagx/sibdry/rmagx/zmagx/cpasma/bcentr/the boundary shape
are untouched by it either way; solve_psi is a separate, subsequent step
that re-converges psi against the (interpolated, now slightly
inconsistent-with-the-finite-difference-operator-on-the-new-grid) profiles
via the Grad-Shafranov equation, which --no-solve skips.

Known limitation, confirmed against a real device G-EQDSK: an
environment-level bug in the installed `megpy`'s null-point tracer (a
`scipy.optimize.fsolve` shape mismatch, hit inside `find_x_points`/
`find_magnetic_axis`'s root-finders for realistically-shaped boundaries)
crashes both `regrid`'s own boundary-spline construction (regardless of
--no-solve -- regrid always retraces the boundary) and `solve_psi`'s
post-Picard-loop axis refinement (--no-solve sidesteps this half by
skipping solve_psi entirely). Worked around here, not fixed upstream:
`regrid_geqdsk` defaults to `old_method=True` (fibe's legacy
`scipy.optimize.root`-based X-point finder, kept intentionally in
`classes.py` for exactly this kind of situation -- pass --new-method to
use the newer `megpy`-based tracer instead, which may be more accurate
when it doesn't hit this bug), and, when actually solving, skips
`solve_psi`'s own axis refinement outright (monkeypatched to a no-op on
the instance) -- the Picard loop itself is unaffected, but the resolved
magnetic axis is only as refined as the last converged Picard iteration,
not the finer root-finder.
"""
import argparse
from pathlib import Path

from fibe import FixedBoundaryEquilibrium


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ifile', type=str, help='Path to input g-eqdsk file')
    parser.add_argument('nr', type=int, help='Number of grid points in R')
    parser.add_argument('nz', type=int, help='Number of grid points in Z')
    parser.add_argument('--niter', type=int, default=50, help='Maximum number of iterations for equilibrium solver')
    parser.add_argument('--tol', type=float, default=1.0e-8, help='Convergence criteria on psi error for equilibrium solver')
    parser.add_argument('--relax', type=float, default=1.0, help='Relaxation constant to smoothen psi stepping for stability')
    parser.add_argument('--relaxj', type=float, default=1.0, help='Relaxation constant to smoothen current stepping for stability')
    parser.add_argument('--ofile', type=str, default=None, help='Path for output g-eqdsk file')
    parser.add_argument('--optimize', default=False, action='store_true', help='Toggle on optimal grid dimensions to fit boundary contour')
    parser.add_argument('--new-method', dest='new_method', default=False, action='store_true', help='Use the newer megpy-based X-point tracer instead of the legacy scipy.optimize.root one')
    parser.add_argument('--no-solve', dest='no_solve', default=False, action='store_true', help='Skip solve_psi entirely, only spline-interpolate psi/pres/fpol/qpsi (--niter/--tol/--relax/--relaxj are ignored)')
    #parser.add_argument('--keep_psi_scale', default=False, action='store_true', help='Toggle on renormalization of psi solution to original psi scale')
    return parser.parse_args()


def regrid_geqdsk(ipath, nr, nz, niter, tol, relax, relaxj, optimize=False, old_method=True, solve=True):
    eq = FixedBoundaryEquilibrium.from_geqdsk(ipath)
    eq.regrid(nr, nz, optimal=optimize, old_method=old_method)
    if solve:
        eq.find_magnetic_axis = lambda: None  # see module docstring
        eq.solve_psi(niter, tol, relax, relaxj)
    return eq


def main():
    args = parse_command_line_arguments()
    ipath = Path(args.ifile)
    if ipath.is_file():
        eq = regrid_geqdsk(
            ipath, args.nr, args.nz, args.niter, args.tol, args.relax, args.relaxj,
            optimize=args.optimize, old_method=not args.new_method, solve=not args.no_solve,
        )
        opath = Path(args.ofile) if args.ofile is not None else ipath.resolve().parent / 'fibe_regridded_input.geqdsk'
        if not opath.exists():
            opath.parent.mkdir(parents=True, exist_ok=True)
            eq.to_geqdsk(opath)
            if args.no_solve:
                print(f'regridded (no solve) -> {opath}')
            else:
                print(f'converged={eq.converged}, psi_error={eq._data["psi_error"]:.3e} -> {opath}')
        else:
            print(f'{opath} already exists, not overwriting -- pass --ofile to write elsewhere.')
    else:
        print(f'{ipath} is not a file.')


if __name__ == '__main__':
    main()

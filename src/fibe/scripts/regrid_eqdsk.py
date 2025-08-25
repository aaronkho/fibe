import argparse
from fibe import FixedBoundaryEquilibrium


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('ifile', type=str, required=True, help='Path to input g-eqdsk file')
    parser.add_argument('nr', type=int, required=True, help='Number of grid points in R')
    parser.add_argument('nz', type=int, required=True, help='Number of grid points in Z')
    parser.add_argument('--niter', type=int, default=50, help='Maximum number of iterations for equilibrium solver')
    parser.add_argument('--tol', type=float, default=1.0e-8, help='Convergence criteria on psi error for equilibrium solver')
    parser.add_argument('--relax', type=float, defualt=1.0, help='Relaxation constant to smoothen psi stepping for stability')
    parser.add_argument('--relaxj', type=float, defualt=1.0, help='Relaxation constant to smoothen current stepping for stability')
    parser.add_argument('--ofile', type=str, default=None, help='Path for output g-eqdsk file')
    parser.add_arguemnt('--optimize', default=False, action='store_true', help='Toggle on optimal grid dimensions to fit boundary contour')
    #parser.add_argument('--keep_psi_scale', default=False, action='store_true', help='Toggle on renormalization of psi solution to original psi scale')
    return parser.parse_args()


def regrid_geqdsk(ipath, niter, tol, relax, relaxj):
    eq = FixedBoundaryEquilibrium.from_geqdsk(ipath)
    eq.regrid(args.nr, args.nz, optimal=args.optimize)
    eq.solve_psi(niter, tol, relax, relaxj)
    return eq


def main():
    args = parse_command_line_arguments()
    ipath = Path(args.ifile)
    if ipath.is_file():
        eq = regrid_geqdsk(ipath, args.niter, args.tol, args.relax, args.relaxj)
        opath = Path(args.ofile) if args.ofile is not None else ifile.resolve().parent / 'fibe_regridded_input.geqdsk'
        if not opath.exists():
            opath.parent.mkdir(parents=True, exist_ok=True)
            eq.to_geqdsk(opath)


if __name__ == '__main__':
    main()

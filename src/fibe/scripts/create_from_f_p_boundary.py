import argparse
from fibe import FixedBoundaryEquilibrium


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def compute_from_scratch_using_f_p_boundary():
    eq = FixedBoundaryEquilibrium()
    eq.define_grid(args.nr, args.nz, args.rmin, args.rmax, args.zmin, args.zmax)
    eq.define_boundary(args.rb, args.zb)
    eq.define_f_profile()
    eq.define_pressure_profile()
    eq.initialize_psi()
    eq.initialize_current()
    eq.solve_psi(args.niter, args.err, args.relax, args.relaxj)
    return eq


def main():
    args = parse_command_line_arguments()


if __name__ == '__main__':
    main()

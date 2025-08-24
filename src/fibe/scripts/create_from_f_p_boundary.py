import argparse
from fibe import FixedBoundaryEquilibrium


def main():
    eq = FixedBoundaryEquilibrium()
    eq.define_grid(args.nr, args.nz, args.rmin, args.rmax, args.zmin, args.zmax)
    eq.define_boundary(args.rb, args.zb)
    eq.setup()
    eq.init_psi()
    eq.run(args.niter, args.err, args.relax, args.relaxj)


if __name__ == '__main__':
    main()

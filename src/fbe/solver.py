import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
import numpy as np
import pandas as pd
from scipy.interpolate import splev, bisplev
from scipy.sparse.linalg import factorized
from scipy.optimize import root

from .math import (
    generate_bounded_1d_spline,
    generate_2d_spline,
    generate_optimal_grid,
    generate_boundary_maps,
    compute_grid_spacing,
    generate_finite_difference_grid,
    compute_jtor,
    compute_psi,
    compute_finite_difference_matrix,
    generate_initial_psi,
    compute_grad_psi_vector_from_2d_spline,
    order_contour_points_by_angle,
    generate_segments,
    generate_x_point_candidates,
    compute_intersection_from_line_segment_complex,
    compute_intersection_from_line_segment_coordinates,
    avoid_convex_curvature,
    generate_boundary_splines,
    find_extrema_with_taylor_expansion,
    compute_gradients_at_boundary,
    generate_boundary_gradient_spline,
    compute_psi_extension,
    compute_flux_surface_quantities,
    compute_safety_factor_contour_integral,
    trace_contours_with_contourpy,
    trace_contour_with_splines,
    compute_adjusted_contour_resolution,
)
from .eqdsk import (
    read_eqdsk_file,
    write_eqdsk_file,
    detect_cocos,
    convert_cocos
)

logger = logging.getLogger('fbe')
logger.setLevel(logging.INFO)


class FixedBoundaryEquilibrium():


    mu0 = 4.0e-7 * np.pi
    eqdsk_fields = [
        'nr',
        'nz',
        'rdim',
        'zdim', 
        'rcentr',
        'rleft',
        'zmid',
        'rmagx', 
        'zmagx',
        'simagx',
        'sibdry',
        'bcentr',
        'cpasma',
        'fpol',
        'pres',
        'ffprime',
        'pprime',
        'psi',
        'qpsi',
        'nbdry',
        'nlim',
        'rbdry',
        'zbdry',
        'rlim',
        'zlim',
        'gcase',
        'gid',
    ]


    def __init__(
        self,
        eqdsk=None,
    ):
        self._data = {}
        self._fit = {}
        self.solver = None
        self.error = None
        self.converged = None
        self._options = {
            'nxiter': 50,
            'erreq': 1.0e-8,
            'relax': 1.0,
            'relaxj': 1.0,
        }
        self._fs = None
        if isinstance(eqdsk, (str, Path)):
            self._data.update(read_eqdsk_file(eqdsk))
            self.enforce_boundary_duplicate_at_end()


    def save_original_data(self, fields):
        for key in fields:
            if f'{key}' in self._data and f'{key}_orig' not in self._data:
                self._data[f'{key}_orig'] = copy.deepcopy(self._data[f'{key}'])


    def save_original_fit(self, fields):
        for key in fields:
            if f'{key}' in self._fit and f'{key}_orig' not in self._fit:
                self._fit[f'{key}_orig'] = copy.deepcopy(self._fit[f'{key}'])


    def create_grid_basis_vectors(self):
        self.save_original_data(['rvec', 'zvec'])
        rmin = self._data['rleft']
        rmax = self._data['rleft'] + self._data['rdim']
        zmin = self._data['zmid'] - 0.5 * self._data['zdim']
        zmax = self._data['zmid'] + 0.5 * self._data['zdim']
        self._data['rvec'] = rmin + np.linspace(0.0, 1.0, self._data['nr']) * (rmax - rmin)
        self._data['zvec'] = zmin + np.linspace(0.0, 1.0, self._data['nz']) * (zmax - zmin)


    def create_grid_basis_meshes(self):
        self.create_grid_basis_vectors()
        self._data['rpsi'] = np.repeat(np.atleast_2d(self._data['rvec']), self._data['nz'], axis=0)
        self._data['zpsi'] = np.repeat(np.atleast_2d(self._data['zvec']).T, self._data['nr'], axis=1)


    def define_grid(self, nr, nz, rmin, rmax, zmin, zmax):
        '''Initialize rectangular grid. Use if no geqdsk is read.'''
        if 'nr' not in self._data:
            self._data['nr'] = int(nr)
        if 'nz' not in self._data:
            self._data['nz'] = int(nz)
        if 'rleft' not in self._data:
            self._data['rleft'] = float(rmin)
        if 'rdim' not in self._data:
            self._data['rdim'] = float(rmax - rmin)
        if 'zmid' not in self._data:
            self._data['zmid'] = float(zmax + zmin) / 2.0
        if 'zdim' not in self._data:
            self._data['zdim'] = float(zmax - zmin)


    def define_boundary(self, rbdry, zbdry):
        '''Initialize last-closed-flux-surface. Use if no geqdsk is read.'''
        if 'nbdry' not in self._data and 'rbdry' not in self._data and 'zbdry' not in self._data and len(rbdry) == len(zbdry):
            self._data['nbdry'] = len(rbdry)
            self._data['rbdry'] = copy.deepcopy(rbdry)
            self._data['zbdry'] = copy.deepcopy(zbdry)
            self.enforce_boundary_duplicate_at_end()


    def initialize_psi(self):
        '''Initialize psi. Use if no geqdsk is read.'''
        self.create_finite_difference_grid()
        self._data['psi'] = generate_initial_psi(self._data['nr'], self._data['nz'], self._data['rbdry'], self._data['zbdry'])
        self.find_magnetic_axis()
        ff, pp = self.compute_f_and_p_grid(self._data['xpsi'])
        self._data['cur'] = compute_jtor(
            self._data['inout'],
            self._data['rpsi'].ravel(),
            ff.ravel(),
            pp.ravel(),
            self._data['cur'],
            relax=1.0
        ) * np.sign(self._data['cpasma'])
        self._data['curscale'] = self._data['cpasma'] / (np.sum(self._data['cur']) * self._data['hrz'])
        self._data['cur'] *= self._data['curscale']


    def define_pressure_profile(self, pressure, psinorm=None, smooth=True):
        if isinstance(pressure, np.ndarray) and len(pressure) > 0:
            self.save_original_data(['pres', 'pprime'])
            #w = 100.0 / pressure if smooth else None
            #psin = np.linspace(0.0, 1.0, len(pressure))
            #if isinstance(psinorm, np.ndarray) and len(psinorm) == len(pressure):
            #    psin = psinorm.copy()
            #psin_mirror = -psin[::-1]
            #pressure_mirror = pressure[::-1]
            #w_mirror = w[::-1] if w is not None else None
            #if np.isclose(psin[0], psin_mirror[-1]):
            #    psin_mirror = psin_mirror[:-1]
            #    pressure_mirror = pressure_mirror[:-1]
            #    w_mirror = w_mirror[:-1] if w_mirror is not None else None
            #psin_fit = np.concatenate([psin_mirror, psin])
            #pressure_fit = np.concatenate([pressure_mirror, pressure])
            #w_fit = np.concatenate([w_mirror, w]) if w is not None else None
            #self._fit['pres_fs'] = {'tck': splrep(psin_fit, pressure_fit, w_fit, xb=-1.0, xe=1.0, k=3, quiet=1), 'bounds': (-1.0, 1.0)}
            self._fit['pres_fs'] = generate_bounded_1d_spline(pressure, xnorm=psinorm, symmetrical=True, smooth=smooth)
            self._data['pres'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'])
            self._data['pprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'], der=1)


    def define_f_profile(self, f, psinorm=None, smooth=True):
        if isinstance(f, np.ndarray) and len(f) > 0:
            self.save_original_data(['fpol', 'ffprime'])
            #w = 100.0 / f if smooth else None
            #psin = np.linspace(0.0, 1.0, len(f))
            #if isinstance(psinorm, np.ndarray) and len(psinorm) == len(f):
            #    psin = psinorm.copy()
            #psin_mirror = -psin[::-1]
            #f_mirror = f[::-1]
            #w_mirror = w[::-1] if w is not None else None
            #if np.isclose(psin[0], psin_mirror[-1]):
            #    psin_mirror = psin_mirror[:-1]
            #    f_mirror = f_mirror[:-1]
            #    w_mirror = w_mirror[:-1] if w_mirror is not None else None
            #psin_fit = np.concatenate([psin_mirror, psin])
            #f_fit = np.concatenate([f_mirror, f])
            #w_fit = np.concatenate([w_mirror, w]) if w is not None else None
            #self._fit['fpol_fs'] = {'tck': splrep(psin_fit, f_fit, w_fit, xb=-1.0, xe=1.0, k=3, quiet=1), 'bounds': (-1.0, 1.0)}
            self._fit['fpol_fs'] = generate_bounded_1d_spline(f, xnorm=psinorm, symmetrical=True, smooth=smooth)
            self._data['fpol'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'])
            self._data['ffprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'], der=1) * self._data['fpol']


    def compute_normalized_psi_map(self):
        self.save_original_data(['xpsi'])
        self._data['xpsi'] = np.abs((self._data['simagx'] - self._data['psi']) / (self._data['simagx'] - self._data['sibdry']))


    def generate_psi_bivariate_spline(self):
        self.save_original_fit(['psi_rz'])
        self.create_grid_basis_vectors()
        #rmin = self._data['rleft']
        #rmax = self._data['rleft'] + self._data['rdim']
        #zmin = self._data['zmid'] - 0.5 * self._data['zdim']
        #zmax = self._data['zmid'] + 0.5 * self._data['zdim']
        #rvec = rmin + np.linspace(0.0, 1.0, self._data['nr']) * (rmax - rmin)
        #zvec = zmin + np.linspace(0.0, 1.0, self._data['nz']) * (zmax - zmin)
        #psi = RectBivariateSpline(rvec, zvec, self._data['psi'].T)
        ##dpsidr = psi.partial_derivative(1, 0)
        ##dpsidz = psi.partial_derivative(0, 1)
        ##tr = psi.get_knots()[0]
        ##tz = psi.get_knots()[1]
        ##c = psi.get_coeffs()
        ##kr = psi.kx
        ##kz = psi.ky
        #tr, tz, c = psi.tck
        #kr, kz = psi.degrees
        #self._fit['psi_rz'] = {'tck': (tr, tz, c, kr, kz), 'bounds': (rmin, zmin, rmax, zmax)}
        self._fit['psi_rz'] = generate_2d_spline(self._data['rvec'], self._data['zvec'], self._data['psi'].T)


    def find_magnetic_axis(self):

        self.save_original_data(['simagx', 'rmagx', 'zmagx'])

        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()

        rmagx = self._data['rmagx'] if 'rmagx' in self._data else self._data['rleft'] + 0.5 * self._data['rdim']
        zmagx = self._data['zmagx'] if 'zmagx' in self._data else self._data['zmid']
        sol = root(lambda x: compute_grad_psi_vector_from_2d_spline(x, self._fit['psi_rz']['tck']), np.array([rmagx, zmagx]).flatten())
        if sol.success:
            r, z = sol.x
            self._data['rmagx'] = r
            self._data['zmagx'] = z
            self._data['simagx'] = bisplev(r, z, self._fit['psi_rz']['tck'])


    def find_x_points(self, sanitize=True):

        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        if 'rmagx' not in self._data or 'zmagx' not in self._data:
            self.find_magnetic_axis()

        hr = self._data['hr'] if 'hr' in self._data else self._data['rdim'] / float(self._data['nr'] - 1)
        hz = self._data['hz'] if 'hz' in self._data else self._data['zdim'] / float(self._data['nz'] - 1)
        xpoint_candidates = generate_x_point_candidates(
            self._data['rbdry'],
            self._data['zbdry'],
            self._data['rmagx'],
            self._data['zmagx'],
            self._fit['psi_rz']['tck'],
            0.03 * float(self._data['nr']) * hr,
            0.03 * float(self._data['nz']) * hz
        )

        xpoints = []
        for xpc in xpoint_candidates:
            sol = root(lambda x: compute_grad_psi_vector_from_2d_spline(x, self._fit['psi_rz']['tck']), xpc)
            if sol.success:
                r, z = sol.x
                xp = np.array([r, z])
                psixp = bisplev(r, z, self._fit['psi_rz']['tck'])
                dpsixp = np.abs((self._data['sibdry'] - psixp) / (self._data['sibdry'] - self._data['simagx']))
                if dpsixp < 0.001:
                    xpoints.append(xp)

        if sanitize:
            for i, xp in enumerate(xpoints):
                rbase = 0.5 * (xp[0] + np.nanmin(self._data['rbdry']))
                zbase = self._data['zmagx']
                rnewxp, znewxp = avoid_convex_curvature(
                    self._data['rbdry'],
                    self._data['zbdry'],
                    xp[0],
                    xp[-1],
                    self._data['rmagx'],
                    self._data['zmagx'],
                    rbase,
                    zbase
                )
                xpoints[i] = np.array([rnewxp, znewxp])

        self._data['xpoints'] = xpoints


    def create_boundary_splines(self, enforce_concave=False):

        self.save_original_fit(['lseg_abdry'])

        if 'xpoints' not in self._data:
            self.find_x_points()

        splines = generate_boundary_splines(
            self._data['rbdry'],
            self._data['zbdry'],
            self._data['rmagx'],
            self._data['zmagx'],
            self._data['xpoints'],
            enforce_concave=enforce_concave
        )
        if len(splines) > 0:
            self._fit['lseg_abdry'] = splines


    def refine_boundary_with_splines(self, nbdry=501):

        self.save_original_data(['nbdry', 'rbdry', 'zbdry'])

        if 'lseg_abdry' not in self._fit:
            self.create_boundary_splines()

        boundary = []
        for i, spline in enumerate(self._fit['lseg_abdry']):
            vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']
            npoints = int(np.rint(nbdry * (spline['bounds'][-1] - spline['bounds'][0]) / (2.0 * np.pi)))
            angle = np.linspace(spline['bounds'][0], spline['bounds'][-1], npoints)
            length = splev(angle, spline['tck'])
            vector = length * np.exp(1.0j * angle) + vmagx
            boundary.extend([v for v in vector])
        if len(boundary) > 0:  # May not be exactly the requested number of points
            self._data['nbdry'] = len(boundary)
            self._data['rbdry'] = np.array(boundary).flatten().real
            self._data['zbdry'] = np.array(boundary).flatten().imag
            self.enforce_boundary_duplicate_at_end()


    def create_finite_difference_grid(self):
        '''Setup the grid and compute the differences matrix.'''

        self.create_grid_basis_vectors()
        self._data.update(
            generate_finite_difference_grid(self._data['rvec'], self._data['zvec'], self._data['rbdry'], self._data['zbdry'])
        )


    def make_solver(self):
        if 'matrix' not in self._data:
            self.create_finite_difference_grid()
        self.solver = factorized(self._data['matrix'].tocsc())


    def find_magnetic_axis_from_grid(self):
        '''Compute magnetic axis location and psi value using second order differences'''
        self.save_original_data(['simagx', 'rmagx', 'zmagx'])
        rmagx, zmagx, simagx = find_extrema_with_taylor_expansion(self._data['rvec'], self._data['zvec'], copy.deepcopy(self._data['psi']))
        self._data['rmagx'] = rmagx
        self._data['zmagx'] = zmagx
        self._data['simagx'] = simagx


    def zero_psi_outside_boundary(self):
        self.save_original_data(['psi'])
        if 'ijout' not in self._data:
            self.create_finite_difference_grid()
        psi = copy.deepcopy(self._data['psi']).ravel()
        psi.put(self._data['ijout'], np.zeros((len(self._data['ijout']), ), dtype=float))
        self._data['psi'] = psi.reshape(self._data['nz'], self._data['nr'])


    def zero_magnetic_boundary(self):
        # Why is this here?
        self.save_original_data(['simagx', 'sibdry'])
        self._data['sibdry'] = 0.0


    def find_x_points_from_grid(self):
        self.save_original_fit(['gradr_bdry', 'gradz_bdry'])
        self.create_boundary_gradient_splines()
        abdry = np.linspace(0.0, 2.0 * np.pi, 5000)
        mag_grad_psi = splev(abdry, self._fit['gradr_bdry']['tck']) ** 2 + splev(abdry, self._fit['gradz_bdry']['tck']) ** 2
        axs = []
        for i, magnitude in enumerate(mag_grad_psi):
            if magnitude < 1.0e-2:
                axs.append(abdry[i])
        #self._data['theta_xpoint'] = np.array(axs)


    def regrid(
        self,
        nr=513,
        nz=513,
        rmin=None,
        rmax=None,
        zmin=None,
        zmax=None,
        optimal=False,
    ):
        '''Setup a new grid and map psi from an existing grid.'''

        self.save_original_data(['nr', 'nz', 'rleft', 'rdim', 'zmid', 'zdim', 'psi'])

        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        if self._data['nbdry'] < 301:
            self.refine_boundary_with_splines(nbdry=501)

        if rmin is None:
            rmin = self._data['rleft']
        if rmax is None:
            rmax = self._data['rleft'] + self._data['rdim']
        if zmin is None:
            zmin = self._data['zmid'] - 0.5 * self._data['zdim']
        if zmax is None:
            zmax = self._data['zmid'] + 0.5 * self._data['zdim']

        if optimal:
            rmin, rmax, zmin, zmax = generate_optimal_grid(nr, nz, self._data['rbdry'], self._data['zbdry'])
            self._data['rleft'] = rmin
            self._data['rdim'] = rmax - rmin
            self._data['zmid'] = (zmax + zmin) / 2.0
            self._data['zdim'] = zmax - zmin

        self._data['nr'] = nr
        self._data['nz'] = nz
        self.create_finite_difference_grid()
        self.make_solver()

        self._data['psi'] = bisplev(self._data['rvec'], self._data['zvec'], self._fit['psi_rz']['tck']).T
        self.recompute_pressure_profile()
        self.recompute_f_profile()
        self.recompute_q_profile()


    def compute_f_and_p_grid(self, psinorm):
        '''Function to compute current density from R and psiN'''
        ff = np.zeros(psinorm.shape)
        pp = np.zeros(psinorm.shape)
        if 'ffprime' in self._fit:
            ff = splev(psinorm, self._fit['ffprime']['tck'])
        else:
            ff = np.interp(psinorm, np.linspace(0.0, 1.0, self._data['fpol'].size), self._data['fpol'])
        if 'pprime' in self._fit:
            pp = splev(psinorm, self._fit['pprime']['tck'])
        else:
            pp = np.interp(psinorm, np.linspace(0.0, 1.0, self._data['pres'].size), self._data['pres'])
        return ff, pp


    def rescale_kinetic_profiles(self):
        if 'curscale' in self._data:
            self.save_original_data(['ffprime', 'pprime', 'fpol', 'pres'])
            if 'ffprime' in self._data:
                self._data['ffprime'] *= self._data['curscale']
            if 'pprime' in self._data:
                self._data['pprime'] *= self._data['curscale']
            if 'fpol' in self._data:
                self._data['fpol'] *= np.sqrt(self._data['curscale'])
            if 'pres' in self._data:
                self._data['pres'] *= self._data['curscale']


    def create_boundary_gradient_splines(self, tol=1.0e-6):
        if 'inout' not in self._data:
            self.create_finite_difference_grid()
        rgradr, zgradr, gradr, rgradz, zgradz, gradz = compute_gradients_at_boundary(
            self._data['rvec'],
            self._data['zvec'],
            copy.deepcopy(self._data['psi'].ravel()),
            self._data['inout'],
            self._data['ijedge'],
            self._data['a1'],
            self._data['a2'],
            self._data['b1'],
            self._data['b2'],
            tol=tol
        )
        self._fit['gradr_bdry'] = generate_boundary_gradient_spline(rgradr, zgradr, gradr, self._data['rmagx'], self._data['zmagx'], tol=tol)
        self._fit['gradz_bdry'] = generate_boundary_gradient_spline(rgradz, zgradz, gradz, self._data['rmagx'], self._data['zmagx'], tol=tol)


    def extend_psi_beyond_boundary(self):
        if 'gradr_bdry' not in self._fit or 'gradz_bdry' not in self._fit:
            self.create_boundary_gradient_splines()
        self._data['psi'] = compute_psi_extension(
            self._data['rvec'],
            self._data['zvec'],
            self._data['rbdry'],
            self._data['zbdry'],
            self._data['rmagx'],
            self._data['zmagx'],
            copy.deepcopy(self._data['psi']),
            self._data['ijout'],
            self._fit['gradr_bdry']['tck'],
            self._fit['gradz_bdry']['tck']
        )


    def trace_rough_flux_surfaces(self):
        psin = np.linspace(0.0, 1.0, self._data['nr'])
        psin[-1] = 0.9999
        psin = np.delete(psin, 0, axis=0)
        levels = psin * (self._data['sibdry'] - self._data['simagx']) + self._data['simagx']
        contours = trace_contours_with_contourpy(
            self._data['rvec'],
            self._data['zvec'],
            self._data['psi'],
            levels,
            self._data['rmagx'],
            self._data['zmagx']
        )
        return contours


    def trace_fine_flux_surfaces(self, maxpoints=51, minpoints=21):
        # Tracing fails in a sector, probably due to enforced angle convention when making lseg_abdry (solved maybe?)
        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        if 'fpol_fs' not in self._fit:
            self.recompute_f_profile()
        contours = self.trace_rough_flux_surfaces()
        psisign = np.sign(self._data['sibdry'] - self._data['simagx'])
        levels = np.sort(psisign * np.array(list(contours.keys())))
        #psi = RectBivariateSpline(self._data['rvec'], self._data['zvec'], self._data['psi'].T)
        #dpsidr = psi.partial_derivative(1, 0)
        #dpsidz = psi.partial_derivative(0, 1)
        fine_contours = {}
        fine_contours[float(self._data['simagx'])] = compute_flux_surface_quantities(
            0.0,
            np.array([self._data['rmagx']]),
            np.array([self._data['zmagx']]),
            self._fit['psi_rz']['tck'],
            self._fit['fpol_fs']['tck']
        )
        for level in levels:
            ll = psisign * level
            npoints = compute_adjusted_contour_resolution(
                self._data['rmagx'], 
                self._data['zmagx'], 
                self._data['rbdry'],
                self._data['zbdry'],
                contours[ll][:, 0],
                contours[ll][:, 0],
                maxpoints=maxpoints,
                minpoints=minpoints
            )
            rc, zc = trace_contour_with_splines(
                copy.deepcopy(self._data['psi']),
                ll,
                npoints,
                self._data['rmagx'],
                self._data['zmagx'],
                self._data['simagx'],
                self._data['sibdry'],
                self._fit['psi_rz'],
                self._fit['lseg_abdry'],
                resolution=251
            )
            if len(rc) > 3:
                psin = np.abs((ll - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx']))
                fine_contours[float(ll)] = compute_flux_surface_quantities(
                    psin,
                    rc,
                    zc,
                    self._fit['psi_rz']['tck'],
                    self._fit['fpol_fs']['tck']
                )
        return fine_contours


    def recompute_pressure_profile(self):
        self.define_pressure_profile(self._data['pres'], smooth=False)


    def recompute_f_profile(self):
        self.define_f_profile(self._data['fpol'], smooth=False)


    def recompute_q_profile(self, smooth=False):
        self.save_original_data(['qpsi'])
        psinorm = np.linspace(0.0, 1.0, len(self._data['qpsi']))
        #psin_mirror = -psin[::-1]
        #q_mirror = self._data['qpsi'][::-1]
        #w = 100.0 / self._data['qpsi'] if smooth else None
        #w_mirror = w[::-1] if w is not None else None
        #if np.isclose(psin[0], psin_mirror[-1]):
        #    psin_mirror = psin_mirror[:-1]
        #    q_mirror = q_mirror[:-1]
        #    w_mirror = w_mirror[:-1] if w_mirror is not None else None
        #psin_fit = np.concatenate([psin_mirror, psin])
        #q_fit = np.concatenate([q_mirror, self._data['qpsi']])
        #w_fit = np.concatenate([w_mirror, w]) if w is not None else None
        #self._fit['qpsi_fs'] = {'tck': splrep(psin_fit, q_fit, w_fit, xb=-1.0, xe=1.0, k=3, quiet=1), 'bounds': (-1.0, 1.0)}
        self._fit['qpsi_fs'] = generate_bounded_1d_spline(self._data['qpsi'], xnorm=psinorm, symmetrical=True, smooth=smooth)
        self._data['qpsi'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['qpsi_fs']['tck'])


    def recompute_q_profile_from_scratch(self):
        self.save_original_data(['qpsi'])
        if self._data['psi'][0, 0] == self._data['psi'][-1, -1] and self._data['psi'][0, -1] == self._data['psi'][-1, 0]:
            self.extend_psi_beyond_boundary()
        self._fs = self.trace_fine_flux_surfaces()
        qpsi = np.zeros((len(self._fs), ), dtype=float)
        for i, (level, contour) in enumerate(self._fs.items()):
            qpsi[i] = compute_safety_factor_contour_integral(contour)
        qpsi[0] = 2.0 * qpsi[1] - qpsi[2]  # Linear interpolation to axis
        self._data['qpsi'] = qpsi


    def renormalize_psi(self, simagx=None, sibdry=None):
        self.save_original_data(['simagx', 'sibdry', 'psi'])
        if 'psi' in self._data and 'simagx' in self._data and 'sibdry' in self._data and simagx is not None and sibdry is not None:
            self._data['psi'] = ((sibdry - simagx) * (self._data['psi'] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])) + simagx
            self._data['simagx'] = simagx
            self._data['sibdry'] = sibdry


    def solve_psi(
        self,
        nxiter=50,    # Max iterations in the equilibrium loop: recommend 50
        erreq=1.0e-8, # Convergence criteria in eq loop max(psiNew-psiOld)/max(psiNew) <= erreq: recommend 1.e-8
        relax=1.0,    # Relaxation parameter in psi correction in eq loop: recommend 1.0
        relaxj=1.0,   # Relaxation parameter in j correction in eq loop: recommend 1.0
    ):
        '''RUN THE EQ SOLVER'''

        if isinstance(nxiter, int):
            self._options['nxiter'] = abs(nxiter)
        if isinstance(erreq, float):
            self._options['erreq'] = erreq
        if isinstance(relaxj, float):
            self._options['relax'] = relax
        if isinstance(relaxj, float):
            self._options['relaxj'] = relaxj

        # INITIAL CURRENT PROFILE
        self.create_grid_basis_meshes()
        self.compute_normalized_psi_map()
        self.zero_psi_outside_boundary()
        self._data['cur'] = np.where(self._data['inout'] == 0, 0.0, self._data['cpasma'])
        for n in range(self._options['nxiter']):
            ff, pp = self.compute_f_and_p_grid(self._data['xpsi'])
            self._data['cur'] = compute_jtor(
                self._data['inout'],
                self._data['rpsi'].ravel(),
                ff.ravel(),
                pp.ravel(),
                self._data['cur'],
                relax=self._options['relaxj'] if n > 0 else 1.0
            ) * np.sign(self._data['cpasma'])
            self._data['curscale'] = self._data['cpasma'] / (np.sum(self._data['cur']) * self._data['hrz'])
            self._data['cur'] *= self._data['curscale']
            psi_new, psi_error = compute_psi(
                self.solver,
                self._data['s5'],
                self._data['cur'],
                copy.deepcopy(self._data['psi']).ravel(),
                relax=self._options['relax']
            )
            self._data['psi'] = psi_new.reshape(self._data['nz'], self._data['nr'])
            self.find_magnetic_axis_from_grid()
            self.zero_magnetic_boundary()
            self.compute_normalized_psi_map()
            #if self.nxiter < 0:
            #    print('max(psiNew-psiOld)/max(psiNew) = %8.2e'%(error))
            if psi_error <= self._options['erreq']: break
        self.rescale_kinetic_profiles()
        self.extend_psi_beyond_boundary()
        self.renormalize_psi()
        self.generate_psi_bivariate_spline()
        self.find_magnetic_axis()
        self.recompute_q_profile_from_scratch()

        self.error = psi_error
        if n + 1 == self._options['nxiter']:
            #print ('Failed to converge after %i iterations with error = %8.2e. Time = %6.1f S'%(abs_nxiter,error,t0))
            self.converged = False
        else:
            #print ('Converged after %i iterations with error = %8.2e. Time = %6.1f S'%(n+1,error,t0))
            self.converged = True

        if self.solver is not None:
            self._data['errsol'] = self.check_psi_solution()

        self._data['gcase'] = 'FBE'
        self._data['gid'] = 42


    def solve_psi_using_q_profile(
        self,
        nxqiter=50,
        errq=1.0e-3,
        nxiter=50,
        erreq=1.0e-8,
        relax=1.0,
        relaxj=1.0,
    ):
        # TODO: Compute F using q and fs, iterate until q convergence
        self.solve_psi(nxiter=nxiter, erreq=erreq, relax=relax, relaxj=relaxj)


    def check_psi_solution(self):
        '''Check accuracy of solution Delta*psi = mu0RJ'''
        # Compute Delta*psi and current density (force balance)
        ds = self._data['matrix'].dot(self._data['psi'].ravel())
        cur = self._data['s5'] * self._data['cur']
        curmax = np.abs(cur).max()
        errds  = np.abs(cur - ds).max() / curmax
        #print('max(-Delta*psi-mu0RJ)/max(mu0RJ) = %8.2e'%(errds))
        return errds


    def enforce_boundary_duplicate_at_end(self):
        if 'rbdry' in self._data and 'zbdry' in self._data:
            df = pd.DataFrame(data={'rbdry': self._data['rbdry'], 'zbdry': self._data['zbdry']}, index=pd.RangeIndex(self._data['nbdry']))
            df = df.drop_duplicates(subset=['rbdry', 'zbdry'], keep='first').reset_index(drop=True)
            rbdry = df['rbdry'].to_numpy()
            zbdry = df['zbdry'].to_numpy()
            self._data['rbdry'] = np.concatenate([rbdry, [rbdry[0]]])
            self._data['zbdry'] = np.concatenate([zbdry, [zbdry[0]]])
            self._data['nbdry'] = len(self._data['rbdry'])


    @classmethod
    def from_eqdsk(cls, path):
        return cls(eqdsk=path)


    def to_eqdsk(self, path, cocos=2):
        eqdsk = {k: v for k, v in self._data.items() if k in self.eqdsk_fields}
        eqdsk['gcase'] = 'FBE'
        eqdsk['gid'] = 42
        current_cocos = detect_cocos(eqdsk)
        eqdsk = convert_cocos(eqdsk, current_cocos, cocos)
        write_eqdsk_file(path, **eqdsk)


    #@classmethod
    #def from_contours(cls, contours):


    #@classmethod
    #def from_mxh_coefficients(cls, mxh_coeffs):
    #    mxh


    def plot_contour(self):
        if 'psi' in self._data:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)
            rmin = self._data['rleft']
            rmax = self._data['rleft'] + self._data['rdim']
            zmin = self._data['zmid'] - 0.5 * self._data['zdim']
            zmax = self._data['zmid'] + 0.5 * self._data['zdim']
            rvec = rmin + np.linspace(0.0, 1.0, self._data['nr']) * (rmax - rmin)
            zvec = zmin + np.linspace(0.0, 1.0, self._data['nz']) * (zmax - zmin)
            rmesh, zmesh = np.meshgrid(rvec, zvec)
            psidiff = self._data['sibdry'] - self._data['simagx']
            psisign = np.sign(psidiff)
            lvec = np.array([0.01, 0.04, 0.09, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.02, 1.05])
            levels = lvec * psidiff + self._data['simagx']
            ax.contour(rmesh, zmesh, psisign * self._data['psi'], levels=psisign * levels)
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary')
            if 'rlim' in self._data and 'zlim' in self._data:
                ax.plot(self._data['rlim'], self._data['zlim'], c='k', label='Limiter')
            if 'rmagx' in self._data and 'zmagx' in self._data:
                ax.scatter(self._data['rmagx'], self._data['zmagx'], marker='o', facecolors='none', edgecolors='r', label='O-points')
            if 'xpoints' in self._data:
                xparr = np.atleast_2d(self._data['xpoints'])
                ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points')
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            ax.legend(loc='best')
            fig.tight_layout()
            plt.show()
            plt.close(fig)


    def plot_grid_splitting(self):
        if 'inout' in self._data:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)
            rmin = np.nanmin(self._data['rvec'])
            rmax = np.nanmax(self._data['rvec'])
            zmin = np.nanmin(self._data['zvec'])
            zmax = np.nanmax(self._data['zvec'])
            rmesh = copy.deepcopy(self._data['rpsi']).ravel()
            zmesh = copy.deepcopy(self._data['zpsi']).ravel()
            mask = self._data['inout'] == 0
            ax.scatter(rmesh[~mask], zmesh[~mask], c='g', marker='.', s=0.1)
            ax.scatter(rmesh[mask], zmesh[mask], c='k', marker='x')
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary')
            if 'rlim' in self._data and 'zlim' in self._data:
                ax.plot(self._data['rlim'], self._data['zlim'], c='k', label='Limiter')
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            fig.tight_layout()
            plt.show()
            plt.close(fig)


    def plot_flux_surfaces(self):
        if self._fs is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)
            rmin = np.nanmin(self._data['rvec'])
            rmax = np.nanmax(self._data['rvec'])
            zmin = np.nanmin(self._data['zvec'])
            zmax = np.nanmax(self._data['zvec'])
            for level, contour in self._fs.items():
                ax.plot(contour['r'], contour['z'], c='b', label=f'{level:.3f}', alpha=0.5)
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary')
            if 'rlim' in self._data and 'zlim' in self._data:
                ax.plot(self._data['rlim'], self._data['zlim'], c='k', label='Limiter')
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            fig.tight_layout()
            plt.show()
            plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nr', type=int, required=True, help='Number of grid points in R')
    parser.add_argument('nz', type=int, required=True, help='Number of grid points in Z')
    parser.add_argument('niter', nargs='?', type=int, default=50, help='Maximum number of iterations for equilibrium solver')
    parser.add_argument('err', nargs='?', type=float, default=1.0e-8, help='Convergence criteria on psi error for equilibrium solver')
    parser.add_argument('relax', nargs='?', type=float, defualt=1.0, help='Relaxation constant to smoothen psi stepping for stability')
    parser.add_argument('relaxj', nargs='?', type=float, defualt=1.0, help='Relaxation constant to smoothen current stepping for stability')
    parser.add_argument('--ifile', type=str, default=None, help='Path to input g-eqdsk file')
    parser.add_argument('--ofile', type=str, default=None, help='Path for output g-eqdsk file')
    parser.add_arguemnt('--optimize', default=False, action='store_true', help='Toggle on optimal grid dimensions to fit boundary contour')
    parser.add_argument('--keep_psi_scale', default=False, action='store_true', help='Toggle on renormalization of psi solution to original psi scale')
    args = parser.parse_args()

    if args.ifile is None:
        # From scratch
        eq = FixedBoundaryEquilibrium()
        #eq.define_grid(args.nr, args.nz, args.rmin, args.rmax, args.zmin, args.zmax)
        #eq.define_boundary(args.rb, args.zb)
        #eq.setup()
        #eq.init_psi()
        #eq.run(args.niter, args.err, args.relax, args.relaxj)
    else:
        ipath = Path(args.ifile)
        if ipath.is_file():
            eq = FixedBoundaryEquilibrium.from_eqdsk(ipath)
            eq.regrid(args.nr, args.nz, optimal=args.optimize)
            eq.run(args.niter, args.err, args.relax, args.relaxj)
            if args.ofile is not None:
                opath = Path(args.ofile)
                if not opath.exists():
                    opath.parent.mkdir(parents=True, exist_ok=True)
                    eq.to_eqdsk(opath)


if __name__ == '__main__':
    main()

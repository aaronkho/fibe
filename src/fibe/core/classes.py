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
    compare_q,
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
    compute_f_from_safety_factor_and_contour,
    trace_contours_with_contourpy,
    trace_contour_with_splines,
    compute_adjusted_contour_resolution,
)
from ..utils.eqdsk import (
    read_geqdsk_file,
    write_geqdsk_file,
    detect_cocos,
    convert_cocos,
    contours_from_mxh_coefficients,
)

logger = logging.getLogger('fibe')
logger.setLevel(logging.INFO)


class FixedBoundaryEquilibrium():


    mu0 = 4.0e-7 * np.pi
    geqdsk_fields = [
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
        geqdsk=None,
    ):
        self._data = {}
        self._fit = {}
        self.solver = None
        self.psi_error = None
        self.q_error = None
        self.converged = None
        self._options = {
            'nxiter': 50,
            'erreq': 1.0e-8,
            'relax': 1.0,
            'relaxj': 1.0,
        }
        self._fs = None
        if isinstance(geqdsk, (str, Path)):
            self._data.update(read_geqdsk_file(geqdsk))
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


    def define_boundary_with_mxh(self, rgeo, zgeo, rminor, kappa, cos_coeffs, sin_coeffs, nbdry=201):
        if 'nbdry' not in self._data and 'rbdry' not in self._data and 'zbdry' not in self._data:
            theta = np.linspace(0.0, 2.0 * np.pi, nbdry) if isinstance(nbdry, int) else np.linspace(0.0, 2.0 * np.pi, 201)
            mxh = {
                'r0': np.array([rgeo]).flatten(),
                'z0': np.array([zgeo]).flatten(),
                'r': np.array([rminor]).flatten(),
                'kappa': np.array([kappa]).flatten(),
                'cos_coeffs': np.atleast_2d(np.array([cos_coeffs]).flatten()),
                'sin_coeffs': np.atleast_2d(np.array([sin_coeffs]).flatten()),
            }
            boundary = contours_from_mxh_coefficients(mxh, theta)
            self._data['nbdry'] = nbdry
            self._data['rbdry'] = copy.deepcopy(boundary['r'].flatten())
            self._data['zbdry'] = copy.deepcopy(boundary['z'].flatten())
            #self.enforce_boundary_duplicate_at_end()


    def initialize_psi(self):
        '''Initialize psi. Use if no geqdsk is read.'''
        self.create_finite_difference_grid()
        self.make_solver()
        self._data['psi'] = generate_initial_psi(self._data['rvec'], self._data['zvec'], self._data['rbdry'], self._data['zbdry'], self._data['ijin'])
        self._data['simagx'] = 1.0
        self._data['sibdry'] = 0.0
        self.find_magnetic_axis()
        self.compute_normalized_psi_map()
        self.create_boundary_splines()


    def initialize_current(self):
        self._data['curscale'] = 1.0
        ffp, pp = self.compute_ffprime_and_pprime_grid(self._data['xpsi'])
        self._data['cur'] = compute_jtor(
            self._data['inout'],
            self._data['rpsi'].ravel(),
            ffp.ravel(),
            pp.ravel()
        )
        self._data['cpasma'] = float(np.sum(self._data['cur']) * self._data['hrz'])


    def define_pressure_profile(self, pressure, psinorm=None, smooth=True):
        if isinstance(pressure, (list, tuple, np.ndarray)) and len(pressure) > 0:
            self.save_original_data(['pres', 'pprime'])
            pressure_new = np.array(pressure).flatten()
            self._fit['pres_fs'] = generate_bounded_1d_spline(pressure_new, xnorm=psinorm, symmetrical=True, smooth=smooth)
            self._data['pres'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'])
            self._data['pprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'], der=1)


    def define_f_profile(self, f, psinorm=None, smooth=True):
        if isinstance(f, (list, tuple, np.ndarray)) and len(f) > 0:
            self.save_original_data(['fpol', 'ffprime'])
            f_new = np.array(f).flatten()
            self._fit['fpol_fs'] = generate_bounded_1d_spline(f_new, xnorm=psinorm, symmetrical=True, smooth=smooth)
            self._data['fpol'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'])
            self._data['ffprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'], der=1) * self._data['fpol']


    def define_q_profile(self, q, psinorm=None, smooth=True):
        if isinstance(q, (list, tuple, np.ndarray)) and len(q) > 0:
            self.save_original_data(['qpsi'])
            q_new = np.array(q).flatten()
            self._fit['qpsi_fs'] = generate_bounded_1d_spline(qpsi, xnorm=psinorm, symmetrical=True, smooth=smooth)
            self._data['qpsi'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['qpsi_fs']['tck'])


    def define_current(self, cpasma):
        if isinstance(cpasma, (float, int)):
            if 'inout' not in self._data:
                self.create_finite_difference_grid()
            self._data['cpasma'] = float(cpasma)
            self._data['curscale'] = 1.0
            self._data['cur'] = np.where(self._data['inout'] == 0, 0.0, self._data['cpasma'] / (self._data['hrz'] * float(len(self._data['ijin']))))


    def define_f_and_pressure_profiles(self, f, pressure, psinorm=None, smooth=True):
        self.define_f_profile(f, psinorm=psinorm, smooth=smooth)
        self.define_pressure_profile(pressure, psinorm=psinorm, smooth=smooth)


    def define_pressure_and_q_profiles(self, pressure, q, ip, psinorm=None, smooth=True):
        self.define_pressure_profile(pressure, psinorm=psinorm, smooth=smooth)
        self.define_q_profile(q, psinorm=psinorm, smooth=smooth)
        self.define_current(ip)


    def compute_normalized_psi_map(self):
        self.save_original_data(['xpsi'])
        self._data['xpsi'] = (self._data['psi'] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
        self._data['xpsi'] = np.where(self._data['xpsi'] < 0.0, 0.0, self._data['xpsi'])


    def generate_psi_bivariate_spline(self):
        self.save_original_fit(['psi_rz'])
        self.create_grid_basis_vectors()
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
            self._data['rmagx'] = float(r)
            self._data['zmagx'] = float(z)
            self._data['simagx'] = float(bisplev(r, z, self._fit['psi_rz']['tck']))


    def find_x_points(self, sanitize=False):

        self.save_original_data(['xpoints'])

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
        self._data['rmagx'] = float(rmagx)
        self._data['zmagx'] = float(zmagx)
        self._data['simagx'] = float(simagx)


    def zero_psi_outside_boundary(self):
        self.save_original_data(['psi'])
        if 'ijout' not in self._data:
            self.create_finite_difference_grid()
        psi = copy.deepcopy(self._data['psi']).ravel()
        psi.put(self._data['ijout'], np.zeros((len(self._data['ijout']), ), dtype=float))
        self._data['psi'] = psi.reshape(self._data['nz'], self._data['nr'])


    def zero_magnetic_boundary(self):
        self.save_original_data(['simagx', 'sibdry'])
        self._data['sibdry'] = 0.0


    def find_x_points_from_grid(self):
        # Not currently used
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


    def compute_ffprime_and_pprime_grid(self, psinorm):
        dpsinorm_dpsi = 1.0 / (self._data['sibdry'] - self._data['simagx'])
        ffp = np.zeros_like(psinorm)
        pp = np.zeros_like(psinorm)
        if 'fpol_fs' in self._fit:
            ffp = splev(psinorm, self._fit['fpol_fs']['tck'], der=1) * splev(psinorm, self._fit['fpol_fs']['tck']) * dpsinorm_dpsi
        elif 'ffprime' in self._data:
            ffp = np.interp(psinorm, np.linspace(0.0, 1.0, self._data['ffprime'].size), self._data['ffprime']) * dpsinorm_dpsi
        if 'pres_fs' in self._fit:
            pp = splev(psinorm, self._fit['pres_fs']['tck'], der=1) * dpsinorm_dpsi
        elif 'pprime' in self._data:
            pp = np.interp(psinorm, np.linspace(0.0, 1.0, self._data['pprime'].size), self._data['pprime']) * dpsinorm_dpsi
        return ffp, pp


    def rescale_kinetic_profiles(self):
        if 'curscale' in self._data:
            self.save_original_data(['ffprime', 'pprime', 'fpol', 'pres'])
            if 'ffprime' in self._data:
                self._data['ffprime'] *= self._data['curscale']
            if 'pprime' in self._data:
                self._data['pprime'] *= self._data['curscale']
            if 'fpol' in self._data:
                self._data['fpol'] *= np.sign(self._data['curscale']) * np.sqrt(np.abs(self._data['curscale']))
            if 'pres' in self._data:
                self._data['pres'] *= self._data['curscale']
            # TODO: Rescale spline fits for these profiles too


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
        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        if 'fpol_fs' not in self._fit:
            self.recompute_f_profile()
        contours = self.trace_rough_flux_surfaces()
        psisign = np.sign(self._data['sibdry'] - self._data['simagx'])
        levels = np.sort(psisign * np.array(list(contours.keys())))
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
                self._fit['psi_rz']['tck'],
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
        self._fit['qpsi_fs'] = generate_bounded_1d_spline(self._data['qpsi'], xnorm=psinorm, symmetrical=True, smooth=smooth)
        self._data['qpsi'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['qpsi_fs']['tck'])


    def recompute_q_profile_from_scratch(self):
        self.save_original_data(['qpsi'])
        if self._data['psi'][0, 0] == self._data['psi'][-1, -1] and self._data['psi'][0, -1] == self._data['psi'][-1, 0]:
            self.extend_psi_beyond_boundary()
        self._fs = self.trace_fine_flux_surfaces()
        psinorm = np.zeros((len(self._fs), ), dtype=float)
        qpsi = np.zeros((len(self._fs), ), dtype=float)
        for i, (level, contour) in enumerate(self._fs.items()):
            psinorm[i] = (level - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
            qpsi[i] = compute_safety_factor_contour_integral(contour)
        qpsi[0] = 2.0 * qpsi[1] - qpsi[2]  # Linear interpolation to axis
        self._fit['qpsi_fs'] = generate_bounded_1d_spline(qpsi, xnorm=psinorm, symmetrical=True, smooth=False)
        self._data['qpsi'] = qpsi


    def renormalize_psi(self, simagx=None, sibdry=None):
        self.save_original_data(['simagx', 'sibdry', 'psi'])
        if 'psi' in self._data and 'simagx' in self._data and 'sibdry' in self._data and simagx is not None and sibdry is not None:
            self._data['psi'] = ((sibdry - simagx) * (self._data['psi'] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])) + simagx
            self._data['simagx'] = simagx
            self._data['sibdry'] = sibdry


    def normalize_psi_to_original(self):
        if 'simagx_orig' in self._data and 'sibdry_orig' in self._data:
            self.renormalize_psi(self._data['simagx_orig'], self._data['sibdry_orig'])


    def solve_psi(
        self,
        nxiter=100,   # Max iterations in the equilibrium loop: recommend 100
        erreq=1.0e-8, # Convergence criteria in eq loop max(psiNew-psiOld)/max(psiNew) <= erreq: recommend 1.e-8
        relax=1.0,    # Relaxation parameter in psi correction in eq loop: recommend 1.0
        relaxj=1.0,   # Relaxation parameter in j correction in eq loop: recommend 1.0
    ):
        '''RUN THE EQ SOLVER'''

        self.save_original_data(['gcase', 'gid', 'psi'])

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
        if 'cur' not in self._data:
            self.define_current(self._data['cpasma'])
        for n in range(self._options['nxiter']):
            ffp, pp = self.compute_ffprime_and_pprime_grid(self._data['xpsi'])
            self._data['cur'] = compute_jtor(
                self._data['inout'],
                self._data['rpsi'].ravel(),
                ffp.ravel(),
                pp.ravel(),
                self._data['cur'],
                relax=self._options['relaxj'] if n > 0 else 1.0
            )
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
            if psi_error <= self._options['erreq']: break
        #self.rescale_kinetic_profiles()
        #self.recompute_f_profile()
        #self.recompute_pressure_profile()
        self.extend_psi_beyond_boundary()
        self.normalize_psi_to_original()
        self.generate_psi_bivariate_spline()
        self.find_magnetic_axis()
        self.recompute_q_profile_from_scratch()

        self.psi_error = psi_error
        if n + 1 == self._options['nxiter']:
            logger.info(f'Failed to converge after {n + 1} iterations with maximum psi error of {psi_error:8.2e}')
            self.converged = False
        else:
            logger.info(f'Converged after {n + 1} iterations with maximum psi error of {psi_error:8.2e}')
            self.converged = True

        if self.solver is not None:
            self._data['errsol'] = self.check_psi_solution()

        self._data['gcase'] = 'FBE'
        self._data['gid'] = 0


    def solve_psi_using_q_profile(
        self,
        nxqiter=50,
        errq=1.0e-3,
        relaxq=1.0,
        nxiter=50,
        erreq=1.0e-8,
        relax=1.0,
        relaxj=1.0,
    ):

        self.save_original_data(['qpsi', 'fpol', 'ffprime'])
        self._data['qpsi_target'] = copy.deepcopy(self._data['qpsi'])

        if isinstance(nxqiter, int):
            self._options['nxqiter'] = abs(nxqiter)
        if isinstance(errq, float):
            self._options['errq'] = errq
        if isinstance(relaxq, float):
            self._options['relaxq'] = relaxq

        if 'cur' not in self._data:
            self.define_current(self._data['cpasma'])
        self.recompute_q_profile_from_scratch()
        for n in range(self._options['nxqiter']):
            q_old = copy.deepcopy(self._data['qpsi'])
            psinorm = np.linspace(0.0, 1.0, self._data['nr'])
            f = np.zeros_like(psinorm)
            for i, (level, contour) in enumerate(self._fs.items()):
                if level != self._data['simagx']:
                    f[i] = compute_f_from_safety_factor_and_contour(self._data['qpsi'][i], contour)
            #f *= np.sign(self._data['curscale']) * np.sqrt(np.abs(self._data['curscale']))
            self.define_f_profile(f[1:], psinorm=psinorm[1:], smooth=True)
            self.solve_psi(nxiter=nxiter, erreq=erreq, relax=relax, relaxj=relaxj)
            # TODO: Fix this error to modify F in the direction of q error reduction
            q_new, q_error = compare_q(
                self._data['qpsi'],
                q_old,
                self._data['qpsi_target'],
                relax=self._options['relaxq'],
            )
            self._data['qpsi'] = copy.deepcopy(q_new)
            if q_error <= self._options['errq']: break

        self.q_error = q_error
        if n + 1 == self._options['nxqiter']:
            logger.info(f'Failed to converge after {n + 1} iterations with maximum safety factor error of {q_error:8.2e}')
            self.converged = False
        else:
            logger.info(f'Converged after {n + 1} iterations with maximum safety factor error of {q_error:8.2e}')
            self.converged = True


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


    def load_geqdsk(self, path, clean=True):
        if isinstance(path, (str, Path)):
            if clean:
                self._data = {}
                self._fit = {}
                self.solver = None
                self.error = None
                self.converged = None
                self.fs = None
            self._data.update(read_geqdsk_file(path))
            self.enforce_boundary_duplicate_at_end()


    @classmethod
    def from_geqdsk(cls, path):
        return cls(geqdsk=path)


    def to_geqdsk(self, path, cocos=2):
        geqdsk = {k: v for k, v in self._data.items() if k in self.geqdsk_fields}
        geqdsk['gcase'] = 'FBE'
        geqdsk['gid'] = 0
        current_cocos = detect_cocos(geqdsk)
        geqdsk = convert_cocos(geqdsk, current_cocos, cocos)
        write_geqdsk_file(path, geqdsk)


    #@classmethod
    #def from_contours(cls, contours):


    #@classmethod
    #def from_mxh_coefficients(cls, mxh_coeffs):
    #    mxh


    def plot_contour(self, save=None):
        if 'rleft' in self._data and 'rdim' in self._data and 'zmid' in self._data and 'zdim' in self._data:
            lvec = np.array([0.01, 0.04, 0.09, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.02, 1.05])
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)
            rmin = self._data['rleft']
            rmax = self._data['rleft'] + self._data['rdim']
            zmin = self._data['zmid'] - 0.5 * self._data['zdim']
            zmax = self._data['zmid'] + 0.5 * self._data['zdim']
            rvec = rmin + np.linspace(0.0, 1.0, self._data['nr']) * (rmax - rmin)
            zvec = zmin + np.linspace(0.0, 1.0, self._data['nz']) * (zmax - zmin)
            if 'psi' in self._data:
                rmesh, zmesh = np.meshgrid(rvec, zvec)
                psidiff = self._data['sibdry'] - self._data['simagx']
                psisign = np.sign(psidiff)
                levels = lvec * psidiff + self._data['simagx']
                ax.contour(rmesh, zmesh, psisign * self._data['psi'], levels=psisign * levels)
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary')
            if 'rlim' in self._data and 'zlim' in self._data:
                ax.plot(self._data['rlim'], self._data['zlim'], c='k', label='Limiter')
            if 'rmagx' in self._data and 'zmagx' in self._data:
                ax.scatter(self._data['rmagx'], self._data['zmagx'], marker='o', facecolors='none', edgecolors='r', label='O-points')
            if 'xpoints' in self._data and len(self._data['xpoints']) > 0:
                xparr = np.atleast_2d(self._data['xpoints'])
                ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points')
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            ax.legend(loc='best')
            fig.tight_layout()
            if isinstance(save, (str, Path)):
                fig.savefig(save, dpi=100)
            plt.show()
            plt.close(fig)


    def plot_comparison_to_original(self, save=None):
        if 'psi' in self._data and 'psi_orig' in self._data:
            lvec = np.array([0.01, 0.04, 0.09, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.02, 1.05])
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)
            nr_new = self._data['nr']
            nz_new = self._data['nz']
            rleft_new = self._data['rleft']
            rdim_new = self._data['rdim']
            zmid_new = self._data['zmid']
            zdim_new = self._data['zdim']
            simagx_new = self._data['simagx']
            sibdry_new = self._data['sibdry']
            nr_old = self._data['nr_orig'] if 'nr_orig' in self._data else copy.deepcopy(nr_new)
            nz_old = self._data['nz_orig'] if 'nz_orig' in self._data else copy.deepcopy(nz_new)
            rleft_old = self._data['rleft_orig'] if 'rleft_orig' in self._data else copy.deepcopy(rleft_new)
            rdim_old = self._data['rdim_orig'] if 'rdim_orig' in self._data else copy.deepcopy(rdim_new)
            zmid_old = self._data['zmid_orig'] if 'zmid_orig' in self._data else copy.deepcopy(zmid_new)
            zdim_old = self._data['zdim_orig'] if 'zdim_orig' in self._data else copy.deepcopy(zdim_new)
            simagx_old = self._data['simagx_orig'] if 'simagx_orig' in self._data else copy.deepcopy(simagx_new)
            sibdry_old = self._data['sibdry_orig'] if 'sibdry_orig' in self._data else copy.deepcopy(sibdry_new)
            rmin_old = rleft_old
            rmax_old = rleft_old + rdim_old
            zmin_old = zmid_old - 0.5 * zdim_old
            zmax_old = zmid_old + 0.5 * zdim_old
            rvec_old = rmin_old + np.linspace(0.0, 1.0, nr_old) * (rmax_old - rmin_old)
            zvec_old = zmin_old + np.linspace(0.0, 1.0, nz_old) * (zmax_old - zmin_old)
            rmesh_old, zmesh_old = np.meshgrid(rvec_old, zvec_old)
            psidiff_old = sibdry_old - simagx_old
            psisign_old = np.sign(psidiff_old)
            levels_old = lvec * psidiff_old + simagx_old
            ax.contour(rmesh_old, zmesh_old, psisign_old * self._data['psi_orig'], levels=psisign_old * levels_old, colors='r', alpha=0.6)
            if 'rbdry_orig' in self._data and 'zbdry_orig' in self._data:
                ax.plot(self._data['rbdry_orig'], self._data['zbdry_orig'], c='r', label='Boundary (old)')
            elif 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary (old)')
            if 'rmagx_orig' in self._data and 'zmagx_orig' in self._data:
                ax.scatter(self._data['rmagx_orig'], self._data['zmagx_orig'], marker='o', facecolors='none', edgecolors='r', label='O-points (old)')
            elif 'rmagx' in self._data and 'zmagx' in self._data:
                ax.scatter(self._data['rmagx'], self._data['zmagx'], marker='o', facecolors='none', edgecolors='r', label='O-points (old)')
            if 'xpoints_orig' in self._data and len(self._data['xpoints_orig']) > 0:
                xparr = np.atleast_2d(self._data['xpoints_orig'])
                ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points (old)')
            #elif 'xpoints' in self._data and len(self._data['xpoints']) > 0:
            #    xparr = np.atleast_2d(self._data['xpoints'])
            #    ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points (old)')
            rmin_new = rleft_new
            rmax_new = rleft_new + rdim_new
            zmin_new = zmid_new - 0.5 * zdim_new
            zmax_new = zmid_new + 0.5 * zdim_new
            rvec_new = rmin_new + np.linspace(0.0, 1.0, nr_new) * (rmax_new - rmin_new)
            zvec_new = zmin_new + np.linspace(0.0, 1.0, nz_new) * (zmax_new - zmin_new)
            rmesh_new, zmesh_new = np.meshgrid(rvec_new, zvec_new)
            psidiff_new = sibdry_new - simagx_new
            psisign_new = np.sign(psidiff_new)
            levels_new = lvec * psidiff_new + simagx_new
            ax.contour(rmesh_new, zmesh_new, psisign_new * self._data['psi'], levels=psisign_new * levels_new, colors='b', alpha=0.6)
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='b', label='Boundary (new)')
            if 'rmagx' in self._data and 'zmagx' in self._data:
                ax.scatter(self._data['rmagx'], self._data['zmagx'], marker='o', facecolors='none', edgecolors='b', label='O-points (new)')
            if 'xpoints' in self._data and len(self._data['xpoints']) > 0:
                xparr = np.atleast_2d(self._data['xpoints'])
                ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='b', label='X-points (new)')
            rmin_plot = np.nanmin([rmin_old, rmin_new])
            rmax_plot = np.nanmax([rmax_old, rmax_new])
            zmin_plot = np.nanmin([zmin_old, zmin_new])
            zmax_plot = np.nanmax([zmax_old, zmax_new])
            ax.set_xlim(rmin_plot, rmax_plot)
            ax.set_ylim(zmin_plot, zmax_plot)
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            ax.legend(loc='best')
            fig.tight_layout()
            if isinstance(save, (str, Path)):
                fig.savefig(save, dpi=100)
            plt.show()
            plt.close(fig)


    def plot_grid_splitting(self, save=None):
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
            if isinstance(save, (str, Path)):
                fig.savefig(save, dpi=100)
            plt.show()
            plt.close(fig)


    def plot_flux_surfaces(self, save=None):
        if self._fs is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)
            rmin = np.nanmin(self._data['rvec'])
            rmax = np.nanmax(self._data['rvec'])
            zmin = np.nanmin(self._data['zvec'])
            zmax = np.nanmax(self._data['zvec'])
            for level, contour in self._fs.items():
                ax.plot(contour['r'], contour['z'], c='b', label=f'{level:.3f}', alpha=0.4)
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary')
            if 'rlim' in self._data and 'zlim' in self._data:
                ax.plot(self._data['rlim'], self._data['zlim'], c='k', label='Limiter')
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            fig.tight_layout()
            if isinstance(save, (str, Path)):
                fig.savefig(save, dpi=100)
            plt.show()
            plt.close(fig)


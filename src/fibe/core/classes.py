import copy
import logging
from pathlib import Path
from typing import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
import numpy as np
import pandas as pd
from scipy.interpolate import splev, bisplev
from scipy.integrate import cumulative_simpson
from scipy.sparse.linalg import factorized
from scipy.optimize import root
from shapely import Point, Polygon

from .math import (
    find_null_points,
    generate_bounded_1d_spline,
    generate_2d_spline,
    generate_optimal_grid,
    generate_finite_difference_grid,
    compute_jtor,
    compute_psi,
    generate_initial_psi,
    compute_grad_psi_vector_from_2d_spline,
    generate_x_point_candidates,
    avoid_convex_curvature,
    generate_boundary_splines,
    find_extrema_with_taylor_expansion,
    compute_gradients_at_boundary,
    generate_boundary_gradient_spline,
    compute_psi_extension,
    compute_flux_surface_quantities,
    compute_flux_surface_quantities_boundary,
    compute_safety_factor_contour_integral,
    compute_f_from_safety_factor_and_contour,
    compute_jtor_contour_integral,
    trace_contours_with_contourpy,
    trace_contour_with_splines,
    trace_contour_with_megpy,
    compute_adjusted_contour_resolution,
    compute_mxh_coefficients_from_contours,
    compute_contours_from_mxh_coefficients,
    check_fully_contained_contours,
)
from ..utils.eqdsk import (
    read_geqdsk_file,
    write_geqdsk_file,
    detect_cocos,
    convert_cocos,
)
from ..utils.plotting import (
    plot_equilibrium_contours,
    plot_equilibrium_heatmap,
    plot_equilibrium_comparison,
    plot_equilibrium_grid,
    plot_equilibrium_flux_surfaces,
    plot_equilibrium_profiles,
    plot_equilibrium_shaping_coefficients,
    plot_equilibrium_boundary_gradients,
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
        legacy_ip=False,
        bt_sign=None,
        ip_sign=None,
    ):
        self.scratch = True
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
            self.load_geqdsk(geqdsk, legacy_ip=legacy_ip, bt_sign=bt_sign, ip_sign=ip_sign)


    def save_original_data(self, fields, overwrite=False):
        for key in fields:
            if f'{key}' in self._data and (overwrite or f'{key}_orig' not in self._data):
                self._data[f'{key}_orig'] = copy.deepcopy(self._data[f'{key}'])


    def save_original_fit(self, fields, overwrite=False):
        for key in fields:
            if f'{key}' in self._fit and (overwrite or f'{key}_orig' not in self._fit):
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
            if 'rlim' in self._data and 'zlim' in self._data:
                if not check_fully_contained_contours(rbdry, zbdry, self._data['rlim'], self._data['zlim']):
                    logger.warning('The defined boundary is not fully contained within the wall contour.')
            self._data['nbdry'] = len(rbdry)
            self._data['rbdry'] = copy.deepcopy(rbdry)
            self._data['zbdry'] = copy.deepcopy(zbdry)
            self.enforce_boundary_duplicate_at_end()


    def define_boundary_with_mxh(self, rgeo, zgeo, rminor, kappa, cos_coeffs, sin_coeffs, nbdry=301):
        if 'nbdry' not in self._data and 'rbdry' not in self._data and 'zbdry' not in self._data:
            theta = np.linspace(0.0, 2.0 * np.pi, nbdry) if isinstance(nbdry, int) else np.linspace(0.0, 2.0 * np.pi, 301)
            mxh = {
                'r0': np.array([rgeo]).flatten(),
                'z0': np.array([zgeo]).flatten(),
                'r': np.array([rminor]).flatten(),
                'kappa': np.array([kappa]).flatten(),
                'cos_coeffs': np.atleast_2d(np.array([cos_coeffs]).flatten()),
                'sin_coeffs': np.atleast_2d(np.array([sin_coeffs]).flatten()),
            }
            boundary = compute_contours_from_mxh_coefficients(mxh, theta[:-1])
            rbdry = boundary['r'].flatten()
            zbdry = boundary['z'].flatten()
            if 'rlim' in self._data and 'zlim' in self._data:
                if not check_fully_contained_contours(rbdry, zbdry, self._data['rlim'], self._data['zlim']):
                    logger.warning('The defined boundary is not fully contained within the wall contour.')
            self._data['rbdry'] = copy.deepcopy(rbdry)
            self._data['zbdry'] = copy.deepcopy(zbdry)
            self._data['nbdry'] = len(self._data['rbdry'])
            self.enforce_boundary_duplicate_at_end()
            self._data['mxh_a'] = np.pi * np.power(mxh['r'], 2.0) / np.trapezoid(self._data['rbdry'], self._data['zbdry'])


    def define_wall(self, rwall, zwall):
        if 'nlim' not in self._data and 'rlim' not in self._data and 'zlim' not in self._data and len(rwall) == len(zwall):
            if 'rbdry' in self._data and 'zbdry' in self._data:
                if not check_fully_contained_contours(self._data['rbdry'], self._data['zbdry'], rwall, zwall):
                    logger.warning('The defined wall does not fully contain the boundary contour.')
            self._data['nlim'] = len(rwall)
            self._data['rlim'] = copy.deepcopy(rwall)
            self._data['zlim'] = copy.deepcopy(zwall)
            self.enforce_wall_duplicate_at_end()


    def define_grid_and_boundary_with_mxh(self, nr, nz, rgeo, zgeo, rminor, kappa, cos_coeffs, sin_coeffs, nbdry=301, rwall=None, zwall=None):
        self.define_boundary_with_mxh(rgeo, zgeo, rminor, kappa, cos_coeffs, sin_coeffs, nbdry=nbdry)
        if rwall is not None and zwall is not None:
            self.define_wall(rwall, zwall)
        rcont = self._data['rlim'] if 'rlim' in self._data and len(self._data['rlim']) > 0 else self._data['rbdry']
        zcont = self._data['zlim'] if 'zlim' in self._data and len(self._data['zlim']) > 0 else self._data['zbdry']
        rmin, rmax, zmin, zmax = generate_optimal_grid(nr, nz, rcont, zcont)
        self._data['nr'] = nr
        self._data['nz'] = nz
        self._data['rleft'] = rmin
        self._data['rdim'] = rmax - rmin
        self._data['zmid'] = (zmax + zmin) / 2.0
        self._data['zdim'] = zmax - zmin


    def initialize_profiles_with_minimal_input(self, pressure_axis, ip, bt, legacy_ip=False):
        if 'pres' not in self._data and 'nr' in self._data:
            #pressure_span = 0.9
            #pressure = np.exp(-np.power(np.linspace(0.0, 1.0, self._data['nr']) / 0.2, 2.0)) * pressure_span * pressure_axis + (1.0 - pressure_span) * pressure_axis
            # Cauchy distribution function has longer tails, better shape for generic initial pressure guess
            gamma = 0.3
            pressure = pressure_axis * (gamma ** 2) / (np.linspace(0.0, 1.0, self._data['nr']) ** 2 + gamma ** 2)
            self.define_pressure_profile(pressure)
            self._data['cpasma'] = ip
            self._data['bcentr'] = bt
            if legacy_ip:
                self._data['cpasma'] *= -1.0


    def initialize_psi(self):
        '''Initialize psi. Use if no geqdsk is read.'''
        self.scratch = True
        self.create_finite_difference_grid()
        self.make_solver()
        self._data['psi'] = generate_initial_psi(
            self._data['rvec'],
            self._data['zvec'],
            self._data['rbdry'],
            self._data['zbdry'],
            self._data['ijin']
        )
        self._data['simagx'] = -1.0
        self._data['sibdry'] = 0.0
        self.find_magnetic_axis_from_grid()
        if 'fpol' not in self._data and 'bcentr' in self._data:
            self.define_toroidal_field(self._data['bcentr'])
        if 'cpasma' not in self._data:
            self.compute_normalized_psi_map()
            self.initialize_current()
        psi_mult = 4.0e-7 * np.pi * -1.0 * float(self._data['cpasma']) * 0.5 * self._data['rdim']
        logger.debug(f'Psi initialization: {psi_mult}')
        self._data['psi'] *= psi_mult
        self._data['simagx'] *= psi_mult
        self._data['sibdry'] *= psi_mult
        self.old_find_magnetic_axis()
        self.save_original_data(['simagx', 'rmagx', 'zmagx', 'sibdry'], overwrite=True)
        self.extend_psi_beyond_boundary(initialization=True)
        self.compute_normalized_psi_map()
        self.create_boundary_splines()


    def initialize_current(self):
        if 'inout' not in self._data:
            self.create_finite_difference_grid()
        self._data['curscale'] = 1.0
        ffp, pp = self.compute_ffprime_and_pprime_grid(self._data['xpsi'])
        self._data['cur'] = np.where(self._data['inout'] == 0, 0.0, compute_jtor(self._data['rpsi'].ravel(), ffp.ravel(), pp.ravel()))
        self._data['cpasma'] = float(np.sum(self._data['cur']) * self._data['hrz'])


    def define_pressure_profile(self, pressure, psinorm=None, smooth=True, symmetrical=True):
        if isinstance(pressure, (list, tuple, np.ndarray)) and len(pressure) > 0:
            self.save_original_data(['pres', 'pprime'])
            pressure_new = np.array(pressure).flatten()
            self._fit['pres_fs'] = generate_bounded_1d_spline(pressure_new, xnorm=psinorm, symmetrical=symmetrical, smooth=smooth)
            self._data['pres'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'])
            self._data['pprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'], der=1)


    def define_f_profile(self, f, psinorm=None, smooth=True, symmetrical=True):
        if isinstance(f, (list, tuple, np.ndarray)) and len(f) > 0:
            self.save_original_data(['fpol', 'ffprime'])
            f_new = np.array(f).flatten()
            self._fit['fpol_fs'] = generate_bounded_1d_spline(f_new, xnorm=psinorm, symmetrical=symmetrical, smooth=smooth)
            self._data['fpol'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'])
            self._data['ffprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'], der=1) * self._data['fpol']


    def define_q_profile(self, q, psinorm=None, smooth=True, symmetrical=True):
        if isinstance(q, (list, tuple, np.ndarray)) and len(q) > 0:
            self.save_original_data(['qpsi'])
            q_new = np.array(q).flatten()
            self._fit['qpsi_fs'] = generate_bounded_1d_spline(q_new, xnorm=psinorm, symmetrical=symmetrical, smooth=smooth)
            self._data['qpsi'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['qpsi_fs']['tck'])


    def define_plasma_current(self, cpasma, legacy_ip=False):
        if isinstance(cpasma, (float, int)):
            if 'inout' not in self._data:
                self.create_finite_difference_grid()
            self._data['cpasma'] = float(cpasma)
            self._data['curscale'] = 1.0
            self._data['cur'] = np.where(self._data['inout'] == 0, 0.0, self._data['cpasma'] / (self._data['hrz'] * float(len(self._data['ijin']))))
            if legacy_ip:
                self._data['cpasma'] *= -1.0


    def define_toroidal_field(self, bcentr, rcentr=None):
        if isinstance(bcentr, (float, int)):
            if not isinstance(rcentr, (float, int)):
                if 'rmagx' in self._data:
                    rcentr = self._data['rmagx']
                else:
                    rcentr = self._data['rleft'] + 0.5 * self._data['rdim']
            self._data['rcentr'] = float(rcentr)
            self._data['bcentr'] = float(bcentr)
            if 'fpol' not in self._data:
                # Linear function for generic initial guess is sufficient for now, needs more testing...
                f_span = 0.005
                f_axis = self._data['rcentr'] * self._data['bcentr']
                f = np.linspace(f_axis, (1.0 - f_span) * f_axis, self._data['nr'])
                self.define_f_profile(f, smooth=True)


    def define_f_and_pressure_profiles(self, f, pressure, psinorm=None, smooth=True, symmetrical=True):
        self.define_f_profile(f, psinorm=psinorm, smooth=smooth, symmetrical=symmetrical)
        self.define_pressure_profile(pressure, psinorm=psinorm, smooth=smooth, symmetrical=symmetrical)


    def define_toroidal_field_and_pressure_profile(self, bt, pressure, psinorm=None, smooth=True):
        self.define_toroidal_field(bt)
        self.define_pressure_profile(pressure, psinorm=psinorm, smooth=smooth)


    def define_pressure_and_q_profiles(self, pressure, q, ip, bt, psinorm=None, smooth=True):
        self.define_pressure_profile(pressure, psinorm=psinorm, smooth=smooth)
        self.define_q_profile(q, psinorm=psinorm, smooth=smooth)
        self.define_plasma_current(ip)
        self.define_toroidal_field(bt)


    def compute_normalized_psi_map(self):
        self.save_original_data(['xpsi'])
        self._data['xpsi'] = (self._data['psi'] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
        self._data['xpsi'] = np.where(self._data['xpsi'] < 0.0, 0.0, self._data['xpsi'])


    def generate_psi_bivariate_spline(self, s=0):
        self.save_original_fit(['psi_rz'])
        self.create_grid_basis_vectors()
        self._fit['psi_rz'] = generate_2d_spline(self._data['rvec'], self._data['zvec'], self._data['psi'].T, s=0)


    def reset_plasma_current_sign(self, negative=False):
        if 'cpasma' in self._data:
            ip_sign = 1.0 if not negative else -1.0
            self._data['cpasma'] *= float(np.sign(self._data['cpasma']) * ip_sign)


    def reset_toroidal_field_sign(self, negative=False):
        if 'bcentr' in self._data:
            bt_sign = 1.0 if not negative else -1.0
            self._data['bcentr'] *= float(np.sign(self._data['bcentr']) * bt_sign)


    def old_find_magnetic_axis(self):
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
        else:
            logger.warning('Magnetic axis could not be found using the old method.')


    def old_find_x_points(self, sanitize=False):
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


    def find_magnetic_axis(self):
        self.save_original_data(['simagx', 'rmagx', 'zmagx'])
        if 'rvec' not in self._data or 'zvec' not in self._data:
            self.create_grid_basis_vectors()
        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        nulls = find_null_points(self._data['rvec'], self._data['zvec'], self._data['psi'], level=self._data['simagx'], atol=1.0e-3)
        for i, op in enumerate(nulls['o-points']):
            point_inside = Point([float(op[0]), float(op[1])])
            polygon = Polygon(np.vstack([np.atleast_2d(self._data['rbdry']), np.atleast_2d(self._data['zbdry'])]).T)
            if polygon.contains(point_inside):
                self._data['rmagx'] = float(op[0])
                self._data['zmagx'] = float(op[1])
                self._data['simagx'] = float(bisplev(self._data['rmagx'], self._data['zmagx'], self._fit['psi_rz']['tck']))
                break


    def find_x_points(self):
        if 'rvec' not in self._data or 'zvec' not in self._data:
            self.create_grid_basis_vectors()
        nulls = find_null_points(self._data['rvec'], self._data['zvec'], self._data['psi'], level=self._data['sibdry'], atol=1.0e-3)
        temp_xps = [np.array(xp) for xp in nulls['x-points']]
        vbdry = self._data['rbdry'] + 1.0j * self._data['zbdry']
        dthr = 2.0 * np.sqrt(np.mean(np.diff(self._data['rvec'])) ** 2.0 + np.mean(np.diff(self._data['zvec'])) ** 2.0)
        xpoints = []
        for xp in temp_xps:
            vmin = vbdry - np.array([xp[0] + 1.0j * xp[-1]])
            if np.nanmin(np.abs(vmin)) < dthr:
                xpoints.append(xp)
        self._data['xpoints'] = xpoints


    def refine_boundary_by_grid_trace(self):
        if 'rvec' not in self._data or 'zvec' not in self._data:
            self.create_grid_basis_vectors()
        self.save_original_data(['nbdry', 'rbdry', 'zbdry'])
        boundary = trace_contour_with_megpy(
            self._data['rvec'],
            self._data['zvec'],
            self._data['psi'],
            self._data['sibdry'],
            self._data['rmagx'],
            self._data['zmagx'],
            boundary=True
        )
        if 'r' in boundary and 'z' in boundary:
            self._data['rbdry'] = boundary['r']
            self._data['zbdry'] = boundary['z']
            self._data['nbdry'] = len(self._data['rbdry'])
            self.enforce_boundary_duplicate_at_end()


    def create_boundary_splines(self, enforce_concave=False, old_method=False):
        self.save_original_fit(['lseg_abdry'])
        if 'xpoints' not in self._data:
            if old_method:
                self.old_find_x_points()
            else:
                self.find_x_points()
        if self._data['nbdry'] < 51 and not old_method:
            self.refine_boundary_by_grid_trace()
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


    def refine_boundary_with_splines(self, nbdry=501, old_method=False):
        self.save_original_data(['nbdry', 'rbdry', 'zbdry'])
        if 'lseg_abdry' not in self._fit:
            self.create_boundary_splines(old_method=old_method)
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


    def regrid(
        self,
        nr=513,
        nz=513,
        rmin=None,
        rmax=None,
        zmin=None,
        zmax=None,
        optimal=False,
        smooth=False,
        symmetrical=True,
        old_method=False,
    ):
        '''Setup a new grid and map psi from an existing grid.'''

        self.save_original_data(['nr', 'nz', 'rleft', 'rdim', 'zmid', 'zdim', 'psi'])

        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        if self._data['nbdry'] < 201:
            self.refine_boundary_with_splines(nbdry=501, old_method=old_method)
        else:
            self.create_boundary_splines(old_method=old_method)

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
        if 'pres' in self._data:
            self.recompute_pressure_profile(smooth=smooth, symmetrical=symmetrical)
        if 'fpol' in self._data:
            self.recompute_f_profile(smooth=smooth, symmetrical=symmetrical)
        if 'qpsi' in self._data:
            self.recompute_q_profile(smooth=smooth)


    def compute_ffprime_and_pprime_grid(self, psinorm, internal_cutoff=0.01, no_fit=False):
        dpsinorm_dpsi = 1.0 / (self._data['sibdry'] - self._data['simagx'])
        ffp = np.zeros_like(psinorm)
        pp = np.zeros_like(psinorm)
        if not no_fit and 'fpol_fs' in self._fit:
            ffp_internal = splev(internal_cutoff, self._fit['fpol_fs']['tck'], der=1) * splev(internal_cutoff, self._fit['fpol_fs']['tck']) * dpsinorm_dpsi
            ffp = splev(psinorm, self._fit['fpol_fs']['tck'], der=1) * splev(psinorm, self._fit['fpol_fs']['tck']) * dpsinorm_dpsi
            ffp = np.where(psinorm < internal_cutoff, float(ffp_internal), ffp)
        elif 'ffprime' in self._data:
            ffp_internal = np.interp(internal_cutoff, np.linspace(0.0, 1.0, self._data['ffprime'].size), self._data['ffprime']) * dpsinorm_dpsi
            ffp = np.interp(psinorm, np.linspace(0.0, 1.0, self._data['ffprime'].size), self._data['ffprime']) * dpsinorm_dpsi
            ffp = np.where(psinorm < internal_cutoff, float(ffp_internal), ffp)
        if not no_fit and 'pres_fs' in self._fit:
            pp_internal = splev(internal_cutoff, self._fit['pres_fs']['tck'], der=1) * dpsinorm_dpsi
            pp = splev(psinorm, self._fit['pres_fs']['tck'], der=1) * dpsinorm_dpsi
            pp = np.where(psinorm < internal_cutoff, float(pp_internal), pp)
        elif 'pprime' in self._data:
            pp_internal = np.interp(internal_cutoff, np.linspace(0.0, 1.0, self._data['pprime'].size), self._data['pprime']) * dpsinorm_dpsi
            pp = np.interp(psinorm, np.linspace(0.0, 1.0, self._data['pprime'].size), self._data['pprime']) * dpsinorm_dpsi
            pp = np.where(psinorm < internal_cutoff, float(pp_internal), pp)
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


    def create_boundary_gradient_splines(self, tol=1.0e-6, smooth=False, initialization=False):
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
        self._data['agradr'] = np.angle(rgradr + 1.0j * zgradr - self._data['rmagx'] - 1.0j * self._data['zmagx'])
        self._data['agradz'] = np.angle(rgradz + 1.0j * zgradz - self._data['rmagx'] - 1.0j * self._data['zmagx'])
        self._data['gradr'] = gradr
        self._data['gradz'] = gradz
        s = 0
        if smooth:
            s = len(gradr) + np.sqrt(2 * len(gradz)) if initialization else 0.5 * (len(gradr) + len(gradz)) * 0.01
        #s = 5 * (len(gradr) + len(gradz)) if smooth else 0
        #if not self.scratch:
        #    ridx = np.nanargmin(np.abs(self._data['rvec'] - self._data['rmagx']))
        #    zidx = np.nanargmin(np.abs(self._data['zvec'] - self._data['zmagx']))
        #    nvec = (self._data['psi'][zidx, ridx:] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
        #    nidx = np.where(nvec < 1.0)[0][-1]
        #    if nidx == (len(self._data['rvec']) - 1):
        #        nidx -= 1
        #    norm = np.diff(self._data['psi'][zidx, nidx:])[0] / self._data['hr']
        #    dpsi = np.abs(self._data['sibdry'] - self._data['simagx'])
        #    rcap = np.nanmax([4.0 * dpsi / self._data['hr'], 2.0 * np.nanmin(np.abs(gradr))])
        #    zcap = np.nanmax([4.0 * dpsi / self._data['hz'], 2.0 * np.nanmin(np.abs(gradz))])
        #    gradr_mask = (np.abs(gradr) < rcap)
        #    gradz_mask = (np.abs(gradz) < zcap)
        #    rgradr = rgradr[gradr_mask]
        #    zgradr = zgradr[gradr_mask]
        #    gradr = gradr[gradr_mask]
        #    rgradz = rgradz[gradz_mask]
        #    zgradz = zgradz[gradz_mask]
        #    gradz = gradz[gradz_mask]
        #    s = int(10.0 * (np.nanmax(np.abs(gradr)) + np.nanmax(np.abs(gradz)))) if smooth else 0
        #    sfunc = generate_boundary_gradient_spline_with_windows
        self._fit['gradr_bdry'] = generate_boundary_gradient_spline(rgradr, zgradr, gradr, self._data['rmagx'], self._data['zmagx'], s=s)
        self._fit['gradz_bdry'] = generate_boundary_gradient_spline(rgradz, zgradz, gradz, self._data['rmagx'], self._data['zmagx'], s=s)


    def extend_psi_beyond_boundary(self, initialization=False):
        if 'gradr_bdry' not in self._fit or 'gradz_bdry' not in self._fit:
            self.create_boundary_gradient_splines(smooth=True, initialization=initialization)
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
        self.generate_psi_bivariate_spline()


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
        dpsi = self._data['sibdry'] - self._data['simagx']
        levels = np.sort(np.sign(dpsi) * np.array(list(contours.keys())))
        fine_contours = {}
        fine_contours[float(self._data['simagx'])] = compute_flux_surface_quantities(
            0.0,
            np.array([self._data['rmagx']]),
            np.array([self._data['zmagx']]),
            self._fit['psi_rz']['tck'],
            self._fit['fpol_fs']['tck']
        )
        for level in levels:
            ll = np.sign(dpsi) * level
            npoints = compute_adjusted_contour_resolution(
                self._data['rmagx'], 
                self._data['zmagx'], 
                self._data['rbdry'],
                self._data['zbdry'],
                contours[ll][:, 0],
                contours[ll][:, 1],
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
                psin = np.abs((ll - self._data['simagx']) / dpsi)
                fine_contours[float(ll)] = compute_flux_surface_quantities(
                    psin,
                    rc,
                    zc,
                    self._fit['psi_rz']['tck'],
                    self._fit['fpol_fs']['tck']
                )
        return fine_contours


    def trace_flux_surfaces(self):
        psinorm = np.linspace(0.0, 1.0, self._data['nr'])
        contours = {}
        contours[float(self._data['simagx'])] = compute_flux_surface_quantities(
            psinorm[0],
            np.array([self._data['rmagx']]),
            np.array([self._data['zmagx']]),
            self._fit['psi_rz']['tck'] if 'psi_rz' in self._fit else None,
            self._fit['fpol_fs']['tck'] if 'fpol_fs' in self._fit else None
        )
        for ll in psinorm[1:-1]:
            level = ll * (self._data['sibdry'] - self._data['simagx']) + self._data['simagx']
            contour = trace_contour_with_megpy(
                self._data['rvec'],
                self._data['zvec'],
                self._data['psi'],
                level,
                self._data['rmagx'],
                self._data['zmagx'],
                boundary=False
            )
            if 'r' in contour and 'z' in contour:
                contours[float(level)] = compute_flux_surface_quantities(
                    ll,
                    contour['r'],
                    contour['z'],
                    self._fit['psi_rz']['tck'] if 'psi_rz' in self._fit else None,
                    self._fit['fpol_fs']['tck'] if 'fpol_fs' in self._fit else None
                )
        contours[float(self._data['sibdry'])] = compute_flux_surface_quantities(
            psinorm[-1],
            self._data['rbdry'],
            self._data['zbdry'],
            self._fit['psi_rz']['tck'] if 'psi_rz' in self._fit else None,
            self._fit['fpol_fs']['tck'] if 'fpol_fs' in self._fit else None
        )
        #contours[float(self._data['sibdry'])] = compute_flux_surface_quantities_boundary(
        #    psinorm[-1],
        #    self._data['rbdry'],
        #    self._data['zbdry'],
        #    self._data['rmagx'],
        #    self._data['zmagx'],
        #    self._fit['gradr_bdry']['tck'] if 'gradr_bdry' in self._fit else None,
        #    self._fit['gradz_bdry']['tck'] if 'gradz_bdry' in self._fit else None,
        #    self._fit['fpol_fs']['tck'] if 'fpol_fs' in self._fit else None
        #)
        return contours


    def recompute_pressure_profile(self, smooth=False, symmetrical=True):
        self.define_pressure_profile(self._data['pres'], smooth=smooth, symmetrical=symmetrical)


    def recompute_f_profile(self, smooth=False, symmetrical=True):
        self.define_f_profile(self._data['fpol'], smooth=smooth, symmetrical=symmetrical)


    def recompute_f_profile_from_scratch(self):
        self.save_original_data(['fpol', 'ffprime'])
        self.save_original_fit(['fpol_fs'])
        if self._data['psi'][0, 0] == self._data['psi'][-1, -1] and self._data['psi'][0, -1] == self._data['psi'][-1, 0]:
            self.extend_psi_beyond_boundary()
        self._fs = self.trace_flux_surfaces()
        psinorm = np.zeros((len(self._fs), ), dtype=float)
        fpol = np.zeros_like(psinorm)
        for i, (level, contour) in enumerate(self._fs.items()):
            if level != self._data['simagx']:
                psinorm[i] = (level - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
                fpol[i] = np.sign(self._data['cpasma']) * compute_f_from_safety_factor_and_contour(self._data['qpsi'][i], contour)
        #fpol *= np.sign(self._data['curscale']) * np.sqrt(np.abs(self._data['curscale']))
        self.define_f_profile(fpol[1:], psinorm=psinorm[1:], smooth=False)


    def recompute_q_profile(self, smooth=False):
        self.save_original_data(['qpsi'])
        self.save_original_fit(['qpsi_fs'])
        psinorm = np.linspace(0.0, 1.0, len(self._data['qpsi']))
        self._fit['qpsi_fs'] = generate_bounded_1d_spline(self._data['qpsi'], xnorm=psinorm, symmetrical=True, smooth=smooth)
        self._data['qpsi'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['qpsi_fs']['tck'])
        self.recompute_phi_profile(smooth=smooth)


    def recompute_q_profile_from_scratch(self, approximate_lcfs=False):
        self.save_original_data(['qpsi'])
        self.save_original_fit(['qpsi_fs'])
        if self._data['psi'][0, 0] == self._data['psi'][-1, -1] and self._data['psi'][0, -1] == self._data['psi'][-1, 0]:
            self.extend_psi_beyond_boundary()
        self._fs = self.trace_flux_surfaces()
        psinorm = np.zeros((len(self._fs), ), dtype=float)
        qpsi = np.zeros_like(psinorm)
        for i, (level, contour) in enumerate(self._fs.items()):
            #current_inside = float(np.abs(self._data['cpasma'])) if level == float(self._data['sibdry']) else None
            psinorm[i] = (level - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
            qpsi[i] = np.sign(self._data['cpasma']) * compute_safety_factor_contour_integral(contour, current_inside=None)
        qpsi[0] = 2.0 * qpsi[1] - qpsi[2]  # Linear interpolation to axis
        if approximate_lcfs:
            qpsi[-1] = qpsi[-2] + 2.0 * (qpsi[-2] - qpsi[-3])  # Linear interpolation to separatrix with increased slope
        self._fit['qpsi_fs'] = generate_bounded_1d_spline(qpsi, xnorm=psinorm, symmetrical=True, smooth=False)
        self._data['qpsi'] = qpsi
        self.recompute_phi_profile(smooth=False)


    def recompute_phi_profile(self, smooth=False):
        self.save_original_data(['phi'])
        self.save_original_fit(['phi_fs'])
        if 'simagx' not in self._data or 'sibdry' not in self._data:
            self.find_magnetic_axis()
        if 'qpsi' in self._data:
            dpsi = self._data['sibdry'] - self._data['simagx']
            psi = np.linspace(0.0, 1.0, self._data['nr']) * dpsi + self._data['simagx']
            self._data['phi'] = cumulative_simpson(np.sign(dpsi) * self._data['qpsi'], x=np.sign(dpsi) * psi, initial=0.0)


    def renormalize_psi(self, simagx=None, sibdry=None):
        self.save_original_data(['simagx', 'sibdry', 'psi'])
        if 'psi' in self._data and 'simagx' in self._data and 'sibdry' in self._data and simagx is not None and sibdry is not None:
            self._data['psi'] = ((sibdry - simagx) * (self._data['psi'] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])) + simagx
            self._data['simagx'] = simagx
            self._data['sibdry'] = sibdry


    def normalize_psi_to_original(self):
        if not self.scratch and 'simagx_orig' in self._data and 'sibdry_orig' in self._data:
            self.renormalize_psi(self._data['simagx_orig'], self._data['sibdry_orig'])
        else:
            self.save_original_data(['simagx', 'sibdry'], overwrite=True)
            self.scratch = False


    def _update_current(self, current_new, relax=1.0):
        if relax > 0.0 and relax < 1.0:
            current_new = self._data['cur'] + relax * (current_new - self._data['cur'])
        self._data['curscale'] = self._data['cpasma'] / (np.sum(current_new) * self._data['hrz'])
        self._data['cur'] = self._data['curscale'] * current_new


    def _update_psi(self, psi_new, relax=1.0):
        if relax > 0.0 and relax < 1.0:
            psi_new = self._data['psi'].ravel() + relax * (psi_new - self._data['psi'].ravel())
        self._data['psi_error'] = np.nanmax(np.abs(psi_new - self._data['psi'].ravel())) / np.nanmax(np.abs(psi_new))
        self._data['psi'] = psi_new.reshape(self._data['nz'], self._data['nr'])


    def _update_q(self, q_new, q_old=None, relax=1.0):
        if relax > 0.0 and relax < 1.0:
            q_new = q_old + relax * (q_new - q_old)
        self._data['q_error'] = np.nanmax((np.abs(self._data['qpsi_target'] - q_new) / np.abs(self._data['qpsi_target']))[:-1])
        logger.debug(f'Error in q: {self._data["q_error"]}, {self._data["fpol"][0]}, {self._data["fpol"][-1]}')
        self._data['qpsi'] = copy.deepcopy(q_new)


    def compute_flux_surface_averaged_jtor_profile(self):
        if self._fs:
            jtor = np.zeros((len(self._fs), ), dtype=float)
            for i, (level, contour) in enumerate(self._fs.items()):
                psinorm = (level - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])
                ffp, pp = self.compute_ffprime_and_pprime_grid(np.array([psinorm]), internal_cutoff=-1.0)
                jtor[i] = compute_jtor_contour_integral(contour, ffp, pp)
            self._data['jpsi'] = copy.deepcopy(jtor)


    def solve_psi(
        self,
        nxiter=100,   # Max iterations in the equilibrium loop: recommend 100
        erreq=1.0e-8, # Convergence criteria in eq loop max(psiNew-psiOld)/max(psiNew) <= erreq: recommend 1.e-8
        relax=1.0,    # Relaxation parameter in psi correction in eq loop: recommend 1.0
        relaxj=1.0,   # Relaxation parameter in j correction in eq loop: recommend 1.0
        pnaxis=None,  # Normalized psi below which to apply j modification: recommend None (auto)
        approxq=False,
        symmetrical=True,
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
        if isinstance(pnaxis, float):
            self._options['pnaxis'] = pnaxis
        elif self.scratch:
            self._options['pnaxis'] = 0.1
            #logger.debug('fprime: ', splev(self._options['pnaxis'], self._fit['fpol_fs']['tck'], der=1) / splev(self._options['pnaxis'], self._fit['fpol_fs']['tck']))
            #logger.debug('pprime: ', splev(self._options['pnaxis'], self._fit['pres_fs']['tck'], der=1) / splev(self._options['pnaxis'], self._fit['pres_fs']['tck']))
        else:
            self._options['pnaxis'] = 1.0 / float(self._data['nr_orig']) if 'nr_orig' in self._data else 1.0 / float(self._data['nr'])

        if self.solver is None:
            self.make_solver()
        if 'pres_fs' not in self._fit:
            self.recompute_pressure_profile(smooth=symmetrical, symmetrical=symmetrical)
        if 'fpol_fs' not in self._fit:
            self.recompute_f_profile(smooth=symmetrical, symmetrical=symmetrical)
        if 'qpsi' in self._data and 'qpsi_fs' not in self._fit:
            self.recompute_q_profile(smooth=symmetrical)

        # Initial current profile
        self.compute_normalized_psi_map()
        self.zero_psi_outside_boundary()
        if 'cur' not in self._data:
            self.define_plasma_current(self._data['cpasma'])
        # Loop to solve psi using Picard iteration
        self._data['psi_error'] = np.inf
        for n in range(self._options['nxiter']):
            ffp, pp = self.compute_ffprime_and_pprime_grid(self._data['xpsi'], internal_cutoff=self._options['pnaxis'])
            cur_new = np.where(self._data['inout'] == 0, 0.0, compute_jtor(self._data['rpsi'].ravel(), ffp.ravel(), pp.ravel()))
            self._update_current(cur_new, relax=self._options['relaxj'] if n > 0 else 1.0)
            psi_new = compute_psi(self.solver, self._data['s5'], self._data['cur'])
            self._update_psi(psi_new, relax=self._options['relax'])
            self.find_magnetic_axis_from_grid()
            self.zero_magnetic_boundary()
            self.compute_normalized_psi_map()
            if self._data['psi_error'] <= self._options['erreq']: break
        #self.rescale_kinetic_profiles()
        #self.recompute_f_profile()
        #self.recompute_pressure_profile()
        self.create_boundary_gradient_splines(smooth=True)
        self.extend_psi_beyond_boundary()
        self.normalize_psi_to_original()
        self.compute_normalized_psi_map()
        self.generate_psi_bivariate_spline()
        self.find_magnetic_axis()
        self.recompute_q_profile_from_scratch(approximate_lcfs=approxq)

        if n + 1 == self._options['nxiter']:
            logger.info(f'Failed to converge after {n + 1} iterations with maximum psi error of {self._data["psi_error"]:8.2e}')
            self.converged = False
        else:
            logger.info(f'Converged after {n + 1} iterations with maximum psi error of {self._data["psi_error"]:8.2e}')
            self.converged = True

        if self.solver is not None:
            self._data['errsol'] = self.check_psi_solution()

        self._data['gcase'] = 'FiBE'
        self._data['gid'] = 0


    def solve_psi_using_q_profile(
        self,
        nxqiter=50,
        errq=1.0e-3,
        relaxq=1.0,
        nxiter=100,
        erreq=1.0e-8,
        relax=1.0,
        relaxj=1.0,
        pnaxis=None,
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
            self.define_plasma_current(self._data['cpasma'])
        if 'qpsi' not in self._data:
            self.recompute_q_profile_from_scratch()
        for n in range(self._options['nxqiter']):
            q_old = copy.deepcopy(self._data['qpsi'])
            if n > 0 or 'fpol' not in self._data:
                self.recompute_f_profile_from_scratch()
            self.solve_psi(nxiter=nxiter, erreq=erreq, relax=relax, relaxj=relaxj, pnaxis=pnaxis)
            self._update_q(self._data['qpsi'], q_old=q_old, relax=self._options['relaxq'])
            if self._data['q_error'] <= self._options['errq']: break

        if n + 1 == self._options['nxqiter']:
            logger.info(f'Failed to converge after {n + 1} iterations with maximum safety factor error of {self._data["q_error"]:8.2e}')
            self.converged = False
        else:
            logger.info(f'Converged after {n + 1} iterations with maximum safety factor error of {self._data["q_error"]:8.2e}')
            self.converged = True


    def check_psi_solution(self):
        '''Check accuracy of solution Delta*psi = mu0RJ'''
        # Compute Delta*psi and current density (force balance)
        ds = self._data['matrix'].dot(self._data['psi'].ravel())
        cur = self._data['s5'] * self._data['cur']
        curmax = np.abs(cur).max()
        errds  = np.abs(cur - ds).max() / curmax
        logger.debug(f'max(-Delta*psi-mu0RJ) / max(mu0RJ) = {errds}')
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


    def enforce_wall_duplicate_at_end(self):
        if 'rlim' in self._data and 'zlim' in self._data:
            df = pd.DataFrame(data={'rlim': self._data['rlim'], 'zlim': self._data['zlim']}, index=pd.RangeIndex(self._data['nlim']))
            df = df.drop_duplicates(subset=['rlim', 'zlim'], keep='first').reset_index(drop=True)
            rwall = df['rlim'].to_numpy()
            zwall = df['zlim'].to_numpy()
            self._data['rlim'] = np.concatenate([rwall, [rwall[0]]])
            self._data['zlim'] = np.concatenate([zwall, [zwall[0]]])
            self._data['nlim'] = len(self._data['rlim'])


    def compute_mxh_parameters(self, n_coeff=6):
        if 'rvec' not in self._data or 'zvec' not in self._data:
            self.create_grid_basis_meshes()
        if self._fs is None:
            self._fs = self.trace_flux_surfaces()
        r0 = []
        z0 = []
        rm = []
        kappa = []
        cosc = []
        sinc = []
        bpol = []
        btor = []
        for level, contour in self._fs.items():
            if level != self._data['simagx']:
                mxh = compute_mxh_coefficients_from_contours(contour, n_coeff=n_coeff)
                r0.append(mxh['r0'])
                z0.append(mxh['z0'])
                rm.append(mxh['r'])
                kappa.append(mxh['kappa'])
                cosc.append(np.atleast_2d(mxh['cos_coeffs']))
                sinc.append(np.atleast_2d(mxh['sin_coeffs']))
            else:
                r0.append(np.atleast_1d(np.mean(contour['r'])))
                z0.append(np.atleast_1d(np.mean(contour['z'])))
                rm.append(np.atleast_1d([0.0]))
                kappa.append(np.atleast_1d([1.0]))
                cosc.append(np.atleast_2d(np.zeros((n_coeff + 1, ))))
                sinc.append(np.atleast_2d(np.zeros((n_coeff + 1, ))))
        self._data['mxh_r0'] = np.concatenate(r0, axis=0)
        self._data['mxh_z0'] = np.concatenate(z0, axis=0)
        self._data['mxh_r'] = np.concatenate(rm, axis=0)
        self._data['mxh_kappa'] = np.concatenate(kappa, axis=0)
        self._data['mxh_cos'] = np.concatenate(cosc, axis=0)
        self._data['mxh_sin'] = np.concatenate(sinc, axis=0)
        self._data['mxh_kappa'][0] = 2.0 * self._data['mxh_kappa'][1] - self._data['mxh_kappa'][2]


    def set_bounding_box_as_wall(self):
        self.save_original_data(['nlim', 'rlim', 'zlim'])
        rmin = self._data['rleft']
        rmax = self._data['rleft'] + self._data['rdim']
        zmin = self._data['zmid'] - 0.5 * self._data['zdim']
        zmax = self._data['zmid'] + 0.5 * self._data['zdim']
        self._data['nlim'] = 5
        self._data['rlim'] = np.array([rmin, rmax, rmax, rmin, rmin])
        self._data['zlim'] = np.array([zmin, zmin, zmax, zmax, zmin])


    def load_geqdsk(self, path, clean=True, legacy_ip=False, bt_sign=None, ip_sign=None):
        if isinstance(path, (str, Path)):
            geqdsk_dict = read_geqdsk_file(path)
            self.insert_geqdsk_dict(geqdsk_dict, clean=clean, legacy_ip=legacy_ip, bt_sign=bt_sign, ip_sign=ip_sign)


    def insert_geqdsk_dict(self, geqdsk_dict, clean=True, legacy_ip=False, bt_sign=None, ip_sign=None):
        if isinstance(geqdsk_dict, dict) and 'nr' in geqdsk_dict and 'nz' in geqdsk_dict and 'rbdry' in geqdsk_dict and 'zbdry' in geqdsk_dict:
            if clean:
                self._data = {}
                self._fit = {}
                self.solver = None
                self.error = None
                self.converged = None
                self.fs = None
            self._data.update(geqdsk_dict)
            if 'cpasma' in self._data and legacy_ip:
                self._data['cpasma'] *= -1.0
            if 'cpasma' in self._data and isinstance(ip_sign, (int, float)):
                self.reset_plasma_current_sign(True if ip_sign < 0.0 else False)
            if 'bcentr' in self._data and isinstance(bt_sign, (int, float)):
                self.reset_toroidal_field_sign(True if bt_sign < 0.0 else False)
            if 'simagx' in self._data and 'sibdry' in self._data and 'psi' in self._data:
                if self._data['simagx'] > self._data['sibdry']:
                    self._data['psi'] *= -1.0
                    self._data['simagx'] *= -1.0
                    self._data['sibdry'] *= -1.0
                    if 'pprime' in self._data:
                        self._data['pprime'] *= -1.0
                    if 'ffprime' in self._data:
                        self._data['ffprime'] *= -1.0
                    if 'q' in self._data:
                        self._data['q'] *= -1.0
                dpsi_dpsinorm = (self._data['sibdry'] - self._data['simagx'])
                if 'pprime' in self._data:
                    self._data['pprime'] *= dpsi_dpsinorm
                if 'ffprime' in self._data:
                    self._data['ffprime'] *= dpsi_dpsinorm
            self.enforce_boundary_duplicate_at_end()
            self.enforce_wall_duplicate_at_end()
            self.scratch = False


    def extract_geqdsk_dict(self, cocos=None, legacy_ip=False):
        if not ('rlim' in self._data and len(self._data['rlim']) > 0) and not ('zlim' in self._data and len(self._data['zlim']) > 0):
            self.set_bounding_box_as_wall()
        geqdsk_dict = {k: v for k, v in self._data.items() if k in self.geqdsk_fields}
        dpsinorm_dpsi = 1.0 / (geqdsk_dict['sibdry'] - geqdsk_dict['simagx'])
        if 'pprime' in geqdsk_dict:
            geqdsk_dict['pprime'] *= dpsinorm_dpsi
        if 'ffprime' in geqdsk_dict:
            geqdsk_dict['ffprime'] *= dpsinorm_dpsi
        geqdsk_dict['gcase'] = 'FiBE'
        geqdsk_dict['gid'] = 2
        if isinstance(cocos, int):
            # FiBE should internally always be in COCOS=2
            current_cocos = detect_cocos(geqdsk_dict)
            logger.info(f'Converting GEQDSK from COCOS={current_cocos} to COCOS={cocos}')
            geqdsk_dict = convert_cocos(geqdsk_dict, current_cocos, cocos)
        if legacy_ip:
            geqdsk_dict['cpasma'] *= -1.0
        return geqdsk_dict


    @classmethod
    def from_geqdsk(cls, path, legacy_ip=False, bt_sign=None, ip_sign=None):
        return cls(geqdsk=path, legacy_ip=legacy_ip, bt_sign=bt_sign, ip_sign=ip_sign)


    def to_geqdsk(self, path, cocos=None, legacy_ip=False):
        geqdsk = self.extract_geqdsk_dict(cocos=cocos, legacy_ip=legacy_ip)
        write_geqdsk_file(path, geqdsk)


    @classmethod
    def from_imas(cls, path, legacy_ip=False):
        return cls()


    def to_imas(self, path, legacy_ip=False):
        # IMAS-4.0.0+ is defined with COCOS=17
        pass


    def plot_contour(self, save=None, show=True):
        plot_equilibrium_contours(self, save=save, show=show)


    def plot_heatmap(self, save=None, show=True):
        plot_equilibrium_heatmap(self, save=save, show=show)


    def plot_comparison_to_original(self, save=None, show=True):
        plot_equilibrium_comparison(self, save=save, show=show)


    def plot_grid_splitting(self, save=None, show=True):
        plot_equilibrium_grid(self, save=save, show=show)


    def plot_flux_surfaces(self, save=None, show=True):
        plot_equilibrium_flux_surfaces(self, save=save, show=show)


    def plot_profiles(self, save=None, show=True):
        plot_equilibrium_profiles(self, save=save, show=show)


    def plot_shaping_parameters(self, save=None, show=True):
        plot_equilibrium_shaping_coefficients(self, save=save, show=show)


    def plot_boundary_gradients(self, save=None, show=True):
        plot_equilibrium_boundary_gradients(self, save=save, show=show)

import argparse
import copy
from pathlib import Path
from typing import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d, splrep, splev, bisplrep, bisplev, RectBivariateSpline, RegularGridInterpolator, make_interp_spline
from scipy.sparse import spdiags
from scipy.sparse.linalg import factorized
from scipy.optimize import fsolve, brentq, root
import contourpy
from shapely import Point, Polygon

from .eqdsk import read_eqdsk_file, write_eqdsk_file, detect_cocos, convert_cocos


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
        self.A = None
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


    def define_boundary(self, rb, zb):
        '''Initialize last-closed-flux-surface. Use if no geqdsk is read.'''
        if 'nbdry' not in self._data and 'rbdry' not in self._data and 'zbdry' not in self._data and len(rb) == len(zb):
            self._data['nbdry'] = len(rb)
            self._data['rbdry'] = copy.deepcopy(rbdry)
            self._data['zbdry'] = copy.deepcopy(zbdry)


    def initialize_psi(self):
        '''Initialize psi. Use if no geqdsk is read.'''
        flat_psi = np.zeros((self._data['nrz'], ), dtype=float)
        r0 = 0.5 * (self._data['rbdry'].max() + self._data['rbdry'].min())
        z0 = 0.5 * (self._data['zbdry'].max() + self._data['zbdry'].min())
        rb = self._data['rbdry'] - x0
        zb = self._data['zbdry'] - y0
        rp = self._data['rbdry'] - x0
        zp = self._data['zbdry'] - y0
        drb = rb[1:] - rb[:-1]
        dzb = zb[1:] - zb[:-1]
        drzb = rb[:-1] * dzb - zb[:-1] * drb
        tcur = 0.0
        for k in self._data['ijin']:
            j = k // self._data['nr']
            i = k - j * self._data['nr']
            det = rp[i] * dzb - zp[j] * drb
            det = np.where(det==0.0, 1.e-10, det)
            rr = rp[i] * drzb / det
            zz = zp[j] * drzb / det
            xin = np.logical_or(
                np.logical_and(rr - rb[1:] <=  1.e-11, rr - rb[:-1] >= -1.e-11),
                np.logical_and(rr - rb[1:] >= -1.e-11, rr - rb[:-1] <=  1.e-11),
            )
            yin = np.logical_or(
                np.logical_and(zz - zb[1:] <=  1.e-11, zz - zb[:-1] >= -1.e-11),
                np.logical_and(zz - zb[1:] >= -1.e-11, zz - zb[:-1] <=  1.e-11),
            )
            rzin = np.logical_and(yin, xin)
            rc = rr.compress(rzin)
            zc = zz.compress(rzin)
            if rc.size == 0 or zc.size == 0:
                rho = 0.0
            else:
                db = rc[0] ** 2 + zc[0] ** 2 if (rc[0] * rp[i] + zc[0] * zp[j]) > 0 else rc[-1] ** 2 + zc[-1] ** 2
                rho = np.nanmin(np.sqrt((rp[i] ** 2 + zp[j] ** 2) / db), 1.0)
            flat_psi[k] = (1.0 - (rho ** 2)) ** 1.2   # Why 1.2?
        self._data['psi'] = flat_psi.reshape(self._data['nr'], self._data['nz'])
        self.find_magnetic_axis()
        self.newj(1.0)


    def define_pressure_profile(self, pressure, psinorm=None, smooth=True):
        if isinstance(pressure, np.ndarray) and len(pressure) > 0:
            if 'pres' in self._data and 'pres_orig' not in self._data:
                self._data['pres_orig'] = self._data['pres'].copy()
            if 'pprime' in self._data and 'pprime_orig' not in self._data:
                self._data['pprime_orig'] = self._data['pprime'].copy()
            w = 100.0 / pressure if smooth else None
            psin = np.linspace(0.0, 1.0, len(pressure))
            if isinstance(psinorm, np.ndarray) and len(psinorm) == len(pressure):
                psin = psinorm.copy()
            psin_mirror = -psin[::-1]
            pressure_mirror = pressure[::-1]
            w_mirror = w[::-1] if w is not None else None
            if np.isclose(psin[0], psin_mirror[-1]):
                psin_mirror = psin_mirror[:-1]
                pressure_mirror = pressure_mirror[:-1]
                w_mirror = w_mirror[:-1] if w_mirror is not None else None
            psin_fit = np.concatenate([psin_mirror, psin])
            pressure_fit = np.concatenate([pressure_mirror, pressure])
            w_fit = np.concatenate([w_mirror, w]) if w is not None else None
            self._fit['pres_fs'] = {'tck': splrep(psin_fit, pressure_fit, w_fit, xb=-1.0, xe=1.0, k=3, quiet=1), 'bounds': (-1.0, 1.0)}
            self._data['pres'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'])
            self._data['pprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['pres_fs']['tck'], der=1)


    def define_f_profile(self, f, psinorm=None, smooth=True):
        if isinstance(f, np.ndarray) and len(f) > 0:
            if 'fpol' in self._data and 'pres_orig' not in self._data:
                self._data['fpol_orig'] = self._data['fpol'].copy()
            if 'ffprime' in self._data and 'ffprime_orig' not in self._data:
                self._data['ffprime_orig'] = self._data['ffprime'].copy()
            w = 100.0 / f if smooth else None
            psin = np.linspace(0.0, 1.0, len(f))
            if isinstance(psinorm, np.ndarray) and len(psinorm) == len(f):
                psin = psinorm.copy()
            psin_mirror = -psin[::-1]
            f_mirror = f[::-1]
            w_mirror = w[::-1] if w is not None else None
            if np.isclose(psin[0], psin_mirror[-1]):
                psin_mirror = psin_mirror[:-1]
                f_mirror = f_mirror[:-1]
                w_mirror = w_mirror[:-1] if w_mirror is not None else None
            psin_fit = np.concatenate([psin_mirror, psin])
            f_fit = np.concatenate([f_mirror, f])
            w_fit = np.concatenate([w_mirror, w]) if w is not None else None
            self._fit['fpol_fs'] = {'tck': splrep(psin_fit, f_fit, w_fit, xb=-1.0, xe=1.0, k=3, quiet=1), 'bounds': (-1.0, 1.0)}
            self._data['fpol'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'])
            self._data['ffprime'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['fpol_fs']['tck'], der=1) * self._data['fpol']


    def compute_normalized_psi_map(self):
        if 'xpsi' in self._data and 'xpsi_orig' not in self._data:
            self._data['xpsi_orig'] = self._data['xpsi'].copy()
        self._data['xpsi'] = np.abs((self._data['simagx'] - self._data['psi']) / (self._data['simagx'] - self._data['sibdry']))


    def generate_psi_bivariate_spline(self):
        if 'psi_rz' in self._fit and 'psi_rz_orig' not in self._fit:
            self._fit['psi_rz_orig'] = self._fit['psi_rz']
        rmin = self._data['rleft']
        rmax = self._data['rleft'] + self._data['rdim']
        zmin = self._data['zmid'] - 0.5 * self._data['zdim']
        zmax = self._data['zmid'] + 0.5 * self._data['zdim']
        rvec = rmin + np.linspace(0.0, 1.0, self._data['nr']) * (rmax - rmin)
        zvec = zmin + np.linspace(0.0, 1.0, self._data['nz']) * (zmax - zmin)
        psi = RectBivariateSpline(rvec, zvec, self._data['psi'].T)
        #dpsidr = psi.partial_derivative(1, 0)
        #dpsidz = psi.partial_derivative(0, 1)
        #tr = psi.get_knots()[0]
        #tz = psi.get_knots()[1]
        #c = psi.get_coeffs()
        #kr = psi.kx
        #kz = psi.ky
        tr, tz, c = psi.tck
        kr, kz = psi.degrees
        self._fit['psi_rz'] = {'tck': (tr, tz, c, kr, kz), 'bounds': (rmin, zmin, rmax, zmax)}


    def find_magnetic_axis(self):

        if 'simagx' in self._data and 'simagx_orig' not in self._data:
            self._data['simagx_orig'] = self._data['simagx']
        if 'rmagx' in self._data and 'rmagx_orig' not in self._data:
            self._data['rmagx_orig'] = self._data['rmagx']
        if 'zmagx' in self._data and 'zmagx_orig' not in self._data:
            self._data['zmagx_orig'] = self._data['zmagx']

        def _grad_psi_vector(loc, tck):
            dpsidr = bisplev(loc[0], loc[1], tck, dx=1)
            dpsidz = bisplev(loc[0], loc[1], tck, dx=1)
            return np.array([dpsidr, dpsidz]).flatten()

        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()

        rmagx = self._data['rmagx'] if 'rmagx' in self._data else self._data['rleft'] + 0.5 * self._data['rdim']
        zmagx = self._data['zmagx'] if 'zmagx' in self._data else self._data['zmid']
        sol = root(lambda x: _grad_psi_vector(x, self._fit['psi_rz']['tck']), np.array([rmagx, zmagx]).flatten())
        if sol.success:
            r, z = sol.x
            self._data['rmagx'] = r
            self._data['zmagx'] = z
            self._data['simagx'] = bisplev(r, z, self._fit['psi_rz']['tck'])


    def find_x_points(self):

        if 'psi_rz' not in self._fit:
            self.generate_psi_bivariate_spline()
        if 'rmagx' not in self._data or 'zmagx' not in self._data:
            self.find_magnetic_axis()

        def _grad_psi_vector(loc, tck):
            dpsidr = bisplev(loc[0], loc[1], tck, dx=1)
            dpsidz = bisplev(loc[0], loc[1], tck, dx=1)
            return np.array([dpsidr, dpsidz]).flatten()

        vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']
        vbdry = self._data['rbdry'] + 1.0j * self._data['zbdry']
        abdry = np.angle(vbdry - vmagx)
        if np.any(abdry < 0.0):
            abdry[abdry < 0.0] = abdry[abdry < 0.0] + 2.0 * np.pi
        b = xr.Dataset(coords={'angle': abdry}, data_vars={'r': (['angle'], self._data['rbdry']), 'z': (['angle'], self._data['zbdry'])})
        b = b.drop_duplicates('angle').sortby('angle')
        orbdry = b['r'].to_numpy()
        ozbdry = b['z'].to_numpy()
        oabdry = b['angle'].to_numpy()
        orbdry = np.concatenate([orbdry, [orbdry[0]]])
        ozbdry = np.concatenate([ozbdry, [ozbdry[0]]])
        oabdry = np.concatenate([oabdry, [oabdry[0] + 2.0 * np.pi]])
        ovbdry = orbdry + 1.0j * ozbdry
        olbdry = np.abs(ovbdry - vmagx)
        dpsidr_obdry = np.array([bisplev(r, z, self._fit['psi_rz']['tck'], dx=1) for r, z in zip(orbdry, ozbdry)]).flatten()
        dpsidz_obdry = np.array([bisplev(r, z, self._fit['psi_rz']['tck'], dy=1) for r, z in zip(orbdry, ozbdry)]).flatten()

        xpoint_candidates = []
        xpoint_indices = []
        dpsidr_zero = np.where(dpsidr_obdry == 0.0)[0]
        dpsidz_zero = np.where(dpsidz_obdry == 0.0)[0]
        for idr in dpsidr_zero:
            if idr in dpsidz_zero:
                xpoint_indices.append(idr)
                xpoint_candidates.append(np.array([orbdry[idr], ozbdry[idr]]))

        split_indices = []
        dpsidr_change = np.where(dpsidr_obdry[:-1] * dpsidr_obdry[1:] < 0.0)[0]
        for idr in dpsidr_change:
            if idr not in xpoint_indices and idr + 1 not in xpoint_indices:
                split_indices.append(int(idr))
        dpsidz_change = np.where(dpsidz_obdry[:-1] * dpsidz_obdry[1:] < 0.0)[0]
        for idz in dpsidz_change:
            if idz not in xpoint_indices and idz + 1 not in xpoint_indices:
                split_indices.append(int(idz))
        split_indices = sorted(split_indices)

        split_lines = []
        for i, istart in enumerate(split_indices):
            ibdry = np.arange(self._data['nbdry'], dtype=int)
            vbdry_seg = np.array([])
            mask = (ibdry > istart)
            if i + 1 < len(split_indices):
                mask &= (ibdry <= split_indices[i + 1])
                vbdry_seg = ovbdry.compress(mask)
            else:
                mask[-1] = False
                vbdry_seg = ovbdry.compress(mask)
                mask2 = (ibdry <= split_indices[0])
                vbdry_seg = np.concatenate([vbdry_seg, ovbdry.compress(mask2)])
                mask |= mask2
            if (len(vbdry_seg) + 1) < len(ibdry):
                vdbdry = np.diff(vbdry_seg)
                adbdry = np.angle(vdbdry)
                if np.any(np.diff(adbdry) > 2.0 * np.pi):
                    adbdry[adbdry < 0.0] = adbdry[adbdry < 0.0] + 2.0 * np.pi
                amin = np.nanmin(adbdry)
                amax = np.nanmax(adbdry)
                if amax - amin > np.pi:
                    print('Spline angle error')
                rotation = np.exp(1.0j * (amax + amin - np.pi) / 2.0)
                vspline = (vbdry_seg - vmagx) * rotation
                xspline = vspline.real
                yspline = vspline.imag
                if xspline[0] > xspline[-1]:
                    xspline = xspline[::-1]
                    yspline = yspline[::-1]
                spl = make_interp_spline(xspline, yspline, bc_type='natural')
                l0 = np.array([vbdry_seg[0], vbdry_seg[0] - (1.0 + 1.0j * spl(vspline[0].real, 1)) / rotation])
                l1 = np.array([vbdry_seg[-1], vbdry_seg[-1] + (1.0 + 1.0j * spl(vspline[-1].real, 1)) / rotation])
                split_lines.append((l0, l1))
            else:
                nchop = vbdry_seg // 3
                vdbdry = np.diff(vbdry_seg)
                adbdry = np.angle(vdbdry)
                if np.any(np.diff(adbdry) > 2.0 * np.pi):
                    adbdry[adbdry < 0.0] = adbdry[adbdry < 0.0] + 2.0 * np.pi
                amin0 = np.nanmin(adbdry[:nchop-1])
                amax0 = np.nanmax(adbdry[:nchop-1])
                if amax0 - amin0 > np.pi:
                    print('Spline 0 angle error')
                rotation0 = np.exp(1.0j * (amax0 + amin0 - np.pi) / 2.0)
                vspline0 = (vbdry_seg[:nchop] - vmagx) * rotation0
                xspline0 = vspline0.real
                yspline0 = vspline0.imag
                if xspline0[0] > xspline0[-1]:
                    xspline0 = xspline0[::-1]
                    yspline0 = yspline0[::-1]
                spl0 = make_interp_spline(xspline0, yspline0, bc_type='natural')
                amin1 = np.nanmin(adbdry[-nchop+1:])
                amax1 = np.nanmax(adbdry[-nchop+1:])
                if amax1 - amin1 > np.pi:
                    print('Spline 1 angle error')
                rotation1 = np.exp(1.0j * (amax1 + amin1 - np.pi) / 2.0)
                vspline1 = (vbdry_seg[-nchop:] - vmagx) * rotation1
                xspline1 = vspline1.real
                yspline1 = vspline1.imag
                if xspline1[0] > xspline1[-1]:
                    xspline1 = xspline1[::-1]
                    yspline1 = yspline1[::-1]
                spl1 = make_interp_spline(xspline1, yspline1, bc_type='natural')
                l0 = np.array([vbdry_seg[0], vbdry_seg[0] - (1.0 + 1.0j * spl0(vspline[0].real, 1)) / rotation0])
                l1 = np.array([vbdry_seg[-1], vbdry_seg[-1] + (1.0 + 1.0j * spl1(vspline[-1].real, 1)) / rotation1])
                split_lines.append((l0, l1))

        intersections = []
        for i, lines in enumerate(split_lines):
            # u = (x1 - x3 + t * (x2 - x1)) / (x4 - x3)
            # t * (y2 - y1 - (x2 - x1) * (y4 - y3) / (x4 - x3)) = y3 - y1 - (x3 - x1) * (y4 - y3) / (x4 - x3)
            p1, p2 = lines[-1]
            p3, p4 = split_lines[i + 1][0] if i + 1 < len(split_lines) else split_lines[0][0]
            da = p2 - p1
            db = p4 - p3
            dx = p3 - p1
            ta = (dx.imag - dx.real * db.imag / db.real) / (da.imag - da.real * db.imag / db.real)
            tb = (ta * da.real - dx.real) / db.real
            px = p1 + ta * da
            if not np.isclose(np.abs(px - (p3 + tb * db)), 0.0):
                print('Intersection error')
            intersections.append(np.array([px.real, px.imag]))

        hr = self._data['hr'] if 'hr' in self._data else self._data['rdim'] / float(self._data['nr'] - 1)
        hz = self._data['hz'] if 'hz' in self._data else self._data['zdim'] / float(self._data['nz'] - 1)
        for i, inter in enumerate(intersections):
            drl = bisplev(inter[0] - 0.1 * float(self._data['nr']) * hr, inter[1], self._fit['psi_rz']['tck'], dx=1)
            drr = bisplev(inter[0] + 0.1 * float(self._data['nr']) * hr, inter[1], self._fit['psi_rz']['tck'], dx=1)
            dzb = bisplev(inter[0], inter[1] - 0.1 * float(self._data['nz']) * hz, self._fit['psi_rz']['tck'], dy=1)
            dza = bisplev(inter[0], inter[1] + 0.1 * float(self._data['nz']) * hz, self._fit['psi_rz']['tck'], dy=1)
            #print(inter, drl, drr, dzb, dza)
            if drl * drr <= 0.0 and dzb * dza <= 0.0:
                xpoint_candidates.append(inter)

        xpoints = []
        for xpc in xpoint_candidates:
            sol = root(lambda x: _grad_psi_vector(x, self._fit['psi_rz']['tck']), np.array([xpc[0], xpc[1]]).flatten())
            if sol.success:
                r, z = sol.x
                xpoints.append(np.array([r, z]))

        self._data['xpoints'] = xpoints


    def generate_boundary_splines(self):

        if 'lseg_abdry' not in self._fit:

            if 'xpoints' not in self._data:
                self.find_x_points()

            vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']
            vbdry = self._data['rbdry'] + 1.0j * self._data['zbdry']
            abdry = np.angle(vbdry - vmagx)
            if np.any(abdry < 0.0):
                abdry[abdry < 0.0] = abdry[abdry < 0.0] + 2.0 * np.pi
            b = xr.Dataset(coords={'angle': abdry}, data_vars={'r': (['angle'], self._data['rbdry']), 'z': (['angle'], self._data['zbdry'])})
            b = b.drop_duplicates('angle').sortby('angle')
            orbdry = b['r'].to_numpy()
            ozbdry = b['z'].to_numpy()
            oabdry = b['angle'].to_numpy()
            orbdry = np.concatenate([orbdry, [orbdry[0]]])
            ozbdry = np.concatenate([ozbdry, [ozbdry[0]]])
            oabdry = np.concatenate([oabdry, [oabdry[0] + 2.0 * np.pi]])
            ovbdry = orbdry + 1.0j * ozbdry
            olbdry = np.abs(ovbdry - vmagx)

            splines = []
            for i, xp in enumerate(self._data['xpoints']):
                vxp0 = xp[0] + 1.0j * xp[-1]
                lxp0 = np.abs(vxp0 - vmagx)
                axp0 = np.angle(vxp0 - vmagx)
                if axp0 < 0.0:
                    axp0 += 2.0 * np.pi
                mask = (oabdry >= axp0)
                olbdry_seg = None
                oabdry_seg = None
                if i + 1 < len(self._data['xpoints']):
                    vxp1 = self._data['xpoints'][i + 1][0] + 1.0j * self._data['xpoints'][i + 1][-1]
                    lxp1 = np.abs(vxp1 - vmagx)
                    axp1 = np.angle(vxp1 - vmagx)
                    if axp1 < 0.0:
                        axp1 += 2.0 * np.pi
                    if axp1 <= 2.0 * np.pi:
                        mask &= (oabdry <= axp1)
                    else:
                        mask |= (oabdry <= (axp1 - 2.0 * np.pi))
                    olbdry_seg = np.concatenate([[lxp0], olbdry.compress(mask), [lxp1]])
                    oabdry_seg = np.concatenate([[axp0], oabdry.compress(mask), [axp1]])
                else:
                    mask[-1] = False
                    olbdry_seg = np.concatenate([[lxp0], olbdry.compress(mask)])
                    oabdry_seg = np.concatenate([[axp0], oabdry.compress(mask)])
                    oabdry_seg = oabdry_seg - 2.0 * np.pi
                    vxp1 = self._data['xpoints'][0][0] + 1.0j * self._data['xpoints'][0][-1]
                    lxp1 = np.abs(vxp1 - vmagx)
                    axp1 = np.angle(vxp1 - vmagx)
                    if axp1 < 0.0:
                        axp1 += 2.0 * np.pi
                    mask2 = (oabdry <= axp1)
                    olbdry_seg = np.concatenate([olbdry_seg, olbdry.compress(mask2), [lxp1]])
                    oabdry_seg = np.concatenate([oabdry_seg, oabdry.compress(mask2), [axp1]])
                    mask |= mask2
                spl = make_interp_spline(oabdry_seg, olbdry_seg, bc_type=None)
                splines.append({'tck': spl.tck, 'bounds': (float(np.nanmin(oabdry_seg)), float(np.nanmax(oabdry_seg)))})
            if len(splines) == 0:
                spl = make_interp_spline(oabdry, olbdry, bc_type='periodic')
                splines.append({'tck': spl.tck, 'bounds': (float(np.nanmin(oabdry)), float(np.nanmax(oabdry)))})

            if len(splines) > 0:
                print(splines)
                self._fit['lseg_abdry'] = splines


    def refine_boundary_with_splines(self, nbdry=501):

        if 'nbdry_orig' not in self._data:
            self._data['nbdry_orig'] = self._data['nbdry']
        if 'rbdry_orig' not in self._data:
            self._data['rbdry_orig'] = self._data['rbdry'].copy()
        if 'zbdry_orig' not in self._data:
            self._data['zbdry_orig'] = self._data['zbdry'].copy()

        if 'lseg_abdry' not in self._fit:
            self.generate_boundary_splines()

        boundary = []
        for i, spline in enumerate(self._fit['lseg_abdry']):
            vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']
            npoints = int(np.rint(nbdry * (spline['bounds'][-1] - spline['bounds'][0]) / (2.0 * np.pi)))
            angle = np.linspace(spline['bounds'][0], spline['bounds'][-1], npoints)
            length = splev(angle, spline['tck'])
            vector = length * np.exp(1.0j * angle) + vmagx
            boundary.extend([v for v in vector])
        if len(boundary) > 0:  # May not be exact
            self._data['nbdry'] = len(boundary)
            self._data['rbdry'] = np.array(boundary).flatten().real
            self._data['zbdry'] = np.array(boundary).flatten().imag


    def generate_finite_difference_grid(self):
        '''Setup the grid and compute the differences matrix.'''

        # SETUP GRID
        rmin = self._data['rleft']
        rmax = self._data['rleft'] + self._data['rdim']
        zmin = self._data['zmid'] - 0.5 * self._data['zdim']
        zmax = self._data['zmid'] + 0.5 * self._data['zdim']

        # GRID POINTS
        self._data['rvec'] = rmin + np.linspace(0.0, 1.0, self._data['nr']) * (rmax - rmin)
        self._data['zvec'] = zmin + np.linspace(0.0, 1.0, self._data['nz']) * (zmax - zmin)
        self._data['rpsi'] = np.repeat(np.atleast_2d(self._data['rvec']), self._data['nz'], axis=0)
        self._data['zpsi'] = np.repeat(np.atleast_2d(self._data['zvec']).T, self._data['nr'], axis=1)

        self._data['nrz'] = self._data['nr'] * self._data['nz']
        self._data['hr'] = (rmax - rmin) / float(self._data['nr'] - 1)
        self._data['hrm1'] = 1.0 / self._data['hr']
        self._data['hrm2'] = self._data['hrm1'] ** 2
        self._data['hz'] = (zmax - zmin) / float(self._data['nz'] - 1)
        self._data['hzm1'] = 1.0 / self._data['hz']
        self._data['hzm2'] = self._data['hzm1'] ** 2
        self._data['hrz'] = self._data['hr'] * self._data['hz']

        inout = np.zeros((self._data['nrz'], ), dtype=int)

        # MIN,MAX OF GRID POINTS AROUND BOUNDARY
        # i = r, j = z
        ibmin = (1.0 + 0.5 * np.sign(self._data['rbdry'].min() - self._data['rvec'])).astype(int).sum() - 1
        ibmax = (1.0 + 0.5 * np.sign(self._data['rbdry'].max() - self._data['rvec'])).astype(int).sum()

        jmin = (1.0 + 0.5 * np.sign(self._data['zbdry'].min() - self._data['zvec'])).astype(int).sum()
        jmax = (1.0 + 0.5 * np.sign(self._data['zbdry'].max() - self._data['zvec'])).astype(int).sum() - 1

        # FIND GRID POINTS INSIDE BOUNDARY
        self._data['jmin'] = jmin
        self._data['jmax'] = jmax
        self._data['ibmin'] = ibmin
        self._data['ibmax'] = ibmax
        self._data['imin'] = np.zeros((self._data['nz'], ), dtype=int)
        self._data['imax'] = np.zeros((self._data['nz'], ), dtype=int)

        r0 = self._data['rbdry'][:-1]
        r1 = self._data['rbdry'][1:]
        z0 = self._data['zbdry'][:-1]
        z1 = self._data['zbdry'][1:]
        for j in range(jmin, jmax+1):
            k0 = np.arange(z0.size).astype(int).compress(
                np.logical_and(
                    self._data['zvec'][j] >= z0,
                    self._data['zvec'][j] < z1
                )
            )
            rc0 = None
            if len(k0) > 0:
                k = k0[0]
                rc0 = (r0[k] - r1[k]) * (self._data['zvec'][j] - z1[k]) / (z0[k] - z1[k]) + r1[k]
            k1 = np.arange(z0.size).astype(int).compress(
                np.logical_and(
                    self._data['zvec'][j] < z0,
                    self._data['zvec'][j] >= z1
                )
            )
            rc1 = None
            if len(k1) > 0:
                k = k1[0]
                rc1 = (r0[k] - r1[k]) * (self._data['zvec'][j] - z1[k]) / (z0[k] - z1[k]) + r1[k]
            rcmin = None
            rcmax = None
            if rc0 is not None and rc1 is not None:
                rcmin = rc0 if rc0 < rc1 else rc1
                rcmax = rc1 if rc0 < rc1 else rc0
            ir = np.arange(self._data['rvec'].size).astype(int).compress(
                np.logical_and(
                    self._data['rvec'] > rcmin,
                    self._data['rvec'] < rcmax
                )
            ) if rcmin is not None and rcmax is not None else np.array([])
            if ir.size > 0:
                self._data['imin'][j] = ir[0]
                self._data['imax'][j] = ir[-1]
                inout[ir[0] + self._data['nr'] * j:ir[-1] + self._data['nr'] * j + 1] = 1
            else:
                self._data['imin'][j] = 0
                self._data['imax'][j] = -1

        if self._data['imax'][self._data['jmin']] < 0:
            self._data['jmin'] += 1
        if self._data['imax'][self._data['jmax']] < 0:
            self._data['jmax'] -= 1
        ijin = np.arange(self._data['nrz']).astype(int).compress(inout)

        # BOUNDARY INSIDE GRID POINTS ARE THOSE WITH A NEIGHBOR THAT IS OUTSIDE
        # BOUNDARY INSIDE GRID POINTS HAVE SPECIAL DIFFERENCE EQUATIONS THAT
        # DEPEND ON LOCATION OF NEXT OUTSIDE POINT

        ijinl = ijin.compress(inout.take(ijin - 1) == 0) # Left
        ijinr = ijin.compress(inout.take(ijin + 1) == 0) # Right
        ijinb = ijin.compress(inout.take(ijin - self._data['nr']) == 0) # Below
        ijina = ijin.compress(inout.take(ijin + self._data['nr']) == 0) # Above
        inout.put(ijinl, inout.take(ijinl) | 0b10)
        inout.put(ijinr, inout.take(ijinr) | 0b100)
        inout.put(ijinb, inout.take(ijinb) | 0b1000)
        inout.put(ijina, inout.take(ijina) | 0b10000)
        del ijinl, ijinr, ijinb, ijina
        ijedge = np.arange(self._data['nrz']).astype(int).compress(inout > 1)

        ss1 = np.ones((self._data['nrz'], ))
        ss1.put(ijin, 2.0 * (self._data['hrm2'] + self._data['hzm2']))
        ss2 = np.zeros((self._data['nrz'], ))
        ss2.put(ijin, self._data['hrm2'])
        ss3 = ss2.copy()
        rxx = np.where(inout, (0.5 * self._data['hrm1']) / self._data['rpsi'].ravel(), 0.0)
        ss2 += rxx
        ss3 -= rxx
        ss4 = np.zeros((self._data['nrz'], ))
        ss4.put(ijin, self._data['hzm2'])
        ss5 = ss4.copy()
        ss6 = np.where(inout, self.mu0 * self._data['rpsi'].ravel(), 0.0)
        
        self._data['a1'] = np.ones((self._data['nrz'], ))
        self._data['a2'] = np.ones((self._data['nrz'], ))
        self._data['b1'] = np.ones((self._data['nrz'], ))
        self._data['b2'] = np.ones((self._data['nrz'], ))

        # COMPUTE DIFFERENCE EQUATION QUANTITIES
        c0 = self._data['rbdry'][:-1] + 1.0j * self._data['zbdry'][:-1]
        c1 = self._data['rbdry'][1:] + 1.0j * self._data['zbdry'][1:]
        dl = c1 - c0
        rmask = (dl.imag != 0.0)
        dlr  = dl.imag.compress(rmask)
        zmask = (dl.real != 0.0)
        dlz  = dl.real.compress(zmask)

        s = np.diff(self._data['rbdry']) / np.diff(self._data['zbdry'])
        # LOOP OVER EDGE POINTS
        for ij in ijedge:

            j = ij // self._data['nr']
            i = ij - self._data['nr'] * j
            rr = self._data['rvec'][i]
            zz = self._data['zvec'][j]

            # BOUNDARY GRID POINT: DIFFERENCES BASED ON LOCATION OF ADJACENT OUTSIDE POINT
            a1 = 1.0
            a2 = 1.0
            b1 = 1.0
            b2 = 1.0
            a = c0 - (rr + 1.0j * zz)
            cr = -a.imag.compress(rmask) / dlr
            cz = -a.real.compress(zmask) / dlz
            adl = (a.conj() * dl).imag
            dr = adl.compress(rmask) / dlr
            dz = -adl.compress(zmask) / dlz
            drc = dr.compress((cr <= 1.0) & (cr >= 0)) * self._data['hrm1']   # (a.r - dl.r * a.z / dl.z) / hr
            dzc = dz.compress((cz <= 1.0) & (cz >= 0)) * self._data['hzm1']   # (a.z - dl.z * a.r / dl.r) / hz

            # XM,YM = LOCATION OF ADJACENT OUTSIDE POINTS AS A FRACTION OF GRID SPACING
            if inout[ij] & 0b10:
                # POINT TO THE LEFT IS OUTSIDE
                if np.any(drc <= 0):
                    a1 = -(drc.compress(drc <= 0).max())
                self._data['a1'][ij] = a1
            if inout[ij] & 0b100:
                # POINT TO THE RIGHT IS OUTSIDE
                if np.any(drc >= 0):
                    a2 = drc.compress(drc >= 0).min()
                self._data['a2'][ij] = a2
            if inout[ij] & 0b1000:
                # POINT BELOW IS OUTSIDE
                if np.any(dzc <= 0):
                    b1 = -(dzc.compress(dzc <= 0).max())
                self._data['b1'][ij] = b1
            if inout[ij] & 0b10000:
                # POINT ABOVE IS OUTSIDE
                if np.any(dzc >= 0):
                    b2 = dzc.compress(dzc >= 0).min()
                self._data['b2'][ij] = b2

            # MODIFIED DIFFERENCE EQUATION QUANTITIES
            ss1[ij] = self._data['hrm1'] * (2.0 * self._data['hrm1'] + (a2 - a1) / rr) / (a1 * a2) + 2.0 * self._data['hzm2'] / (b1 * b2)
            ss2[ij] = self._data['hrm1'] * (2.0 * self._data['hrm1'] + a2 / rr) / (a1 * (a1 + a2))
            ss3[ij] = self._data['hrm1'] * (2.0 * self._data['hrm1'] - a1 / rr) / (a2 * (a1 + a2))
            ss4[ij] = 2.0 * self._data['hzm2'] / b1 / (b1 + b2)
            ss5[ij] = 2.0 * self._data['hzm2'] / b2 / (b1 + b2)

        s1 = ss2 / ss1
        s2 = ss3 / ss1
        s3 = ss4 / ss1
        s4 = ss5 / ss1
        s5 = ss6 / ss1

        self._data['inout'] = inout
        self._data['ijin'] = ijin
        self._data['ijedge'] = ijedge
        self._data['ijout'] = np.arange(self._data['nrz']).astype(int).compress(inout == 0)
        self._data['s5'] = s5

        # FULL DIFFERENCE MATRIX FOR SPARSE MATRIX INVERSION OR SOLUTION
        data = np.array([np.ones((self._data['nrz'], )), -s1, -s2, -s3, -s4])
        diags = np.array([0, 1, -1, self._data['nr'], -self._data['nr']])
        self.A = spdiags(data, diags, self._data['nrz'], self._data['nrz']).T
        del s1, s2, s3, s4

        #self._data['a1'] = self._data['a1'].take(ijedge)
        #self._data['a2'] = self._data['a2'].take(ijedge)
        #self._data['b1'] = self._data['b1'].take(ijedge)
        #self._data['b2'] = self._data['b2'].take(ijedge)


    def make_solver(self):
        self.solver = factorized(self.A.tocsc())


    def find_magnetic_axis_from_grid(self):
        '''Compute magnetic axis location and psi value using second order differences'''

        if 'simagx_orig' not in self._data:
            self._data['simagx_orig'] = self._data['simagx']
        if 'rmagx_orig' not in self._data:
            self._data['rmagx_orig'] = self._data['rmagx']
        if 'zmagx_orig' not in self._data:
            self._data['zmagx_orig'] = self._data['zmagx']

        # FINITE DIFFERENCE EQUATIONS FOR DERIVATIVES
        apsi = np.abs(self._data['psi'].ravel())
        nr = self._data['nr']
        k = apsi.argmax()
        ax = 0.5 * self._data['hrm1'] * (apsi[k + 1] - apsi[k - 1])
        ay = 0.5 * self._data['hzm1'] * (apsi[k + nr] - apsi[k - nr])
        axx = self._data['hrm2'] * (apsi[k + 1] + apsi[k - 1] - 2.0 * apsi[k])
        ayy = self._data['hzm2'] * (apsi[k + nr] + apsi[k - nr] - 2.0 * apsi[k])
        axy = 0.25 * self._data['hrm1'] * self._data['hzm1'] * (
            apsi[k + nr + 1] + apsi[k - nr - 1] - apsi[k - nr + 1] - apsi[k + nr - 1]
        )

        # SOLVE FOR LOCATION WHERE X AND Y DERIVATIVES ARE 0
        delta = axy * axy - axx * ayy
        xmax = (ayy * ax - axy * ay) / delta
        ymax = (-axy * ax + axx * ay) / delta

        # PEAK PSI VALUE
        self._data['simagx'] = np.sign(self._data['cpasma']) * (
            apsi[k] + 
            xmax * ax + ymax * ay + 
            0.5 * axx * xmax**2 + 0.5 * ayy * ymax**2 + 
            axy * xmax * ymax
        )

        # MAGNETIC AXIS LOCATION
        j = k // nr
        i = k - j * nr
        self._data['rmagx'] = xmax + self._data['rvec'][i]
        self._data['zmagx'] = ymax + self._data['zvec'][j]


    def zero_psi_outside_boundary(self):
        if 'psi_orig' not in self._data:
            self._data['psi_orig'] = self._data['psi'].copy()
        psi = self._data['psi'].copy().ravel()
        psi.put(self._data['ijout'], np.zeros((len(self._data['ijout']), ), dtype=float))
        self._data['psi'] = psi.reshape(self._data['nz'], self._data['nr'])


    def zero_magnetic_boundary(self):
        if 'sibdry_orig' not in self._data:
            self._data['sibdry_orig'] = self._data['sibdry']
        self._data['sibdry'] = 0.0


    def find_x_points_from_grid(self):
        # X-POINT LOCATION
        if 'gradr_bdry' not in self._fit or 'gradz_bdry' not in self._fit:
            self.compute_boundary_gradients()
        abdry = np.linspace(0.0, 2.0 * np.pi, 5000)
        mag_grad_psi = splev(abdry, self._fit['gradr_bdry']['tck']) ** 2 + splev(abdry, self._fit['gradz_bdry']['tck']) ** 2
        axs = []
        for i, magnitude in enumerate(mag_grad_psi):
            if magnitude < 1.0e-2:
                axs.append(abdry[i])
        self._data['theta_xpoint'] = np.array(axs)


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

        # SAVE OLD GRID
        if 'nr_orig' not in self._data:
            self._data['nr_orig'] = self._data['nr']
        if 'nz_orig' not in self._data:
            self._data['nz_orig'] = self._data['nz']
        if 'rleft_orig' not in self._data:
            self._data['rleft_orig'] = self._data['rleft']
        if 'rdim_orig' not in self._data:
            self._data['rdim_orig'] = self._data['rdim']
        if 'zmid_orig' not in self._data:
            self._data['zmid_orig'] = self._data['zmid']
        if 'zdim_orig' not in self._data:
            self._data['zdim_orig'] = self._data['zdim']
        if 'psi_orig' not in self._data:
            self._data['psi_orig'] = self._data['psi'].copy()

        # FUNCTION TO FIT GRID TO BOUNDARY
        def _optimize_grid(nr, nz, rbdry, zbdry):
            e = 3.5
            m = float(nr - 1)
            g = 1.0 / ((m - e)**2 - e**2)
            x0 = np.nanmin(rbdry)
            x1 = np.nanmax(rbdry)
            rmax = m * g * (x1 * (m - e) - x0 * e)
            rmin = m * g * (x0 * (m - e) - x1 * e)
            m = float(nz - 1)
            g = 1.0 / ((m - e)**2 - e**2)
            y0 = np.nanmin(zbdry)
            y1 = np.nanmax(zbdry)
            zmax = m * g * (y1 * (m - e) - y0 * e)
            zmin = m * g * (y0 * (m - e) - y1 * e)
            if zmax > -zmin:
                zmin = -zmax
            elif zmax < -zmin:
                zmax = -zmin
            return rmin, rmax, zmin, zmax

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

        # SETUP NEW GRID
        if optimal:
            rmin, rmax, zmin, zmax = _optimize_grid(nr, nz, self._data['rbdry'], self._data['zbdry'])
            self._data['rleft'] = rmin
            self._data['rdim'] = rmax - rmin
            self._data['zmid'] = (zmax + zmin) / 2.0
            self._data['zdim'] = zmax - zmin

        self._data['nr'] = nr
        self._data['nz'] = nz
        self.generate_finite_difference_grid()
        self.make_solver()

        # INTERPOLATE PSI ONTO NEW GRID
        self._data['psi'] = bisplev(self._data['rvec'], self._data['zvec'], self._fit['psi_rz']['tck']).T
        self.recompute_pressure_profile()
        self.recompute_f_profile()
        self.recompute_q_profile()


    def jtor(self, rmaj, psinorm):
        '''Function to compute current density from R and psiN'''
        ff = self._data['ffprime'].copy()
        pp = self._data['pprime'].copy()
        if 'ffprime' in self._fit:
            ff = splev(psinorm, self._fit['ffprime']['tck'])
        else:
            ff = np.interp(psinorm, np.linspace(0.0, 1.0, ff.size), ff)
        if 'pprime' in self._fit:
            pp = splev(psinorm, self._fit['pprime']['tck'])
        else:
            pp = np.interp(psinorm, np.linspace(0.0, 1.0, pp.size), pp)
        return -np.sign(self._data['cpasma']) * (ff / (self.mu0 * rmaj) + rmaj * pp)


    def newj(self, relax=1.0):
        '''Compute current density over grid. Scale to Ip'''
        if relax != 1.0:
            curold = self._data['cpasma'].copy()
            curnew = np.where(self._data['inout'] == 0, 0.0, self.jtor(self._data['rpsi'], self._data['xpsi']).ravel())
            self._data['cur'] = curold + relax * (curnew - curold)
        else:
            self._data['cur'] = np.where(self._data['inout'] == 0, 0.0, self.jtor(self._data['rpsi'], self._data['xpsi']).ravel())
        self._data['curscale'] = self._data['cpasma'] / (self._data['cur'].sum() * self._data['hrz'])
        self._data['cur'] *= self._data['curscale']


    def rescale_kinetic_profiles(self):
        if 'curscale' in self._data:
            if 'ffprime' in self._data:
                if 'ffprime_orig' not in self._data:
                    self._data['ffprime_orig'] = self._data['ffprime'].copy()
                self._data['ffprime'] *= self._data['curscale']
            if 'pprime' in self._data:
                if 'pprime_orig' not in self._data:
                    self._data['pprime_orig'] = self._data['pprime'].copy()
                self._data['pprime'] *= self._data['curscale']
            if 'fpol' in self._data:
                if 'fpol_orig' not in self._data:
                    self._data['fpol_orig'] = self._data['fpol'].copy()
                self._data['fpol'] *= np.sqrt(self._data['curscale'])
            if 'pres' in self._data:
                if 'pres_orig' not in self._data:
                    self._data['pres_orig'] = self._data['pres'].copy()
                self._data['pres'] *= self._data['curscale']


    def compute_boundary_gradients_from_grid(self, tol=1.0e-6):

        vgradr = []
        gradr = []
        vgradz = []
        gradz = []

        psi = self._data['psi'].copy().ravel()

        # DETERMINE GRADIENT OF PSI AT BOUNDARY POINTS
        jedge = self._data['ijedge'] // self._data['nr']
        iedge = self._data['ijedge'] - self._data['nr'] * jedge
        vedge = self._data['rvec'].take(iedge) + 1.0j * self._data['zvec'].take(jedge)
        k = -1
        for ij in self._data['ijedge']:

            k += 1
            j = ij // self._data['nr']
            i = ij - self._data['nr'] * j
            a1 = self._data['a1'][ij] #[k]
            a2 = self._data['a2'][ij] #[k]
            b1 = self._data['b1'][ij] #[k]
            b2 = self._data['b2'][ij] #[k]
            #print(i, j, cedge[k], self._data['xpsi'].ravel()[ij], '{:b}'.format(self._data['inout'][ij]))

            # dpsi/dx AT BOUNDARY POINTS
            if self._data['inout'][ij] & 0b10 or self._data['inout'][ij] & 0b100:
                vl = vedge[k] - a1 * self._data['hr']
                vr = vedge[k] + a2 * self._data['hr']
                grad = (a1 + a2) * psi[ij] / (a1 * a2)
                # LEFT OR RIGHT OUT
                if self._data['inout'][ij] & 0b10 and self._data['inout'][ij] & 0b100:
                    if (a1 < tol or a2 < tol): continue
                    # LEFT AND RIGHT OUT
                    vgradr.extend([vl, vr])
                    gradr.extend([grad, -grad])
                elif self._data['inout'][ij] & 0b10:
                    if (a1 < tol): continue
                    # ONLY LEFT OUT
                    vgradr.append(vl)
                    gradr.append((grad - a1 * psi[ij + 1] / (1.0 + a1)) / self._data['hr'])
                    #gradr.append(grad - a1 * psi[ij + 1] / (1.0 + a1))
                else:
                    if (a2 < tol): continue
                    # ONLY RIGHT OUT
                    vgradr.append(vr)
                    gradr.append((-grad + a2 * psi[ij - 1] / (1.0 + a2)) / self._data['hr'])
                    #gradr.append(-grad + a2 * psi[ij - 1] / (1.0 + a2))

            # dpsi/dy AT BOUNDARY POINTS
            if self._data['inout'][ij] & 0b1000 or self._data['inout'][ij] & 0b10000:
                ni = self._data['nr']
                vb = vedge[k] - 1.0j * b1 * self._data['hz']
                va = vedge[k] + 1.0j * b2 * self._data['hz']
                grad = (b1 + b2) * psi[ij] / (b1 * b2)
                # ABOVE OR BELOW OUT
                if self._data['inout'][ij] & 0b1000 and self._data['inout'][ij] & 0b10000:
                    if (b1 < tol or b2 < tol): continue
                    # ABOVE AND BELOW OUT
                    vgradz.extend([vb, va])
                    gradz.extend([grad, -grad])
                elif self._data['inout'][ij] & 0b1000:
                    if (b1 < tol): continue
                    # ONLY BELOW OUT
                    vgradz.append(vb)
                    gradz.append((grad - b1 * psi[ij + ni] / (1.0 + b1)) / self._data['hz'])
                else:
                    if (b2 < tol): continue
                    # ONLY ABOVE OUT
                    vgradz.append(va)
                    gradz.append((-grad + b2 * psi[ij - ni] / (1.0 + b2)) / self._data['hz'])
            #print(vgradr[-1], gradr[-1], vgradz[-1], gradz[-1])

        # SPLINE GRAD PSI AS A FUNCTION OF ANGLE ALONG BOUNDARY
        # USING MAGNETIC AXIS AS CENTER
        vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']
        vgradr = np.array(vgradr) - vmagx
        vgradz = np.array(vgradz) - vmagx
        gr = xr.Dataset(coords={'angle': np.angle(vgradr)}, data_vars={'length': (['angle'], np.abs(vgradr)), 'gradient': (['angle'], np.array(gradr))}).sortby('angle')
        gz = xr.Dataset(coords={'angle': np.angle(vgradz)}, data_vars={'length': (['angle'], np.abs(vgradz)), 'gradient': (['angle'], np.array(gradz))}).sortby('angle')
        agradr = gr['angle'].to_numpy()
        lgradr = gr['length'].to_numpy()
        gradr = gr['gradient'].to_numpy()
        agradz = gz['angle'].to_numpy()
        lgradz = gz['length'].to_numpy()
        gradz = gz['gradient'].to_numpy()
        self._data['lmin'] = np.nanmin(np.concatenate([np.abs(vgradr), np.abs(vgradz)])) - np.abs(self._data['hr'] + 1.0j * self._data['hz'])

        self._fit['gradr_bdry'] = {'tck': splrep(agradr, gradr, k=3, quiet=1), 'bounds': (float(np.nanmin(agradr)), float(np.nanmax(agradr)))}
        if np.abs(agradr[0] + np.pi) < tol:
            agradr[0] = -np.pi
        if np.abs(agradr[-1] - np.pi) < tol:
            agradr[-1] =  np.pi
        yy = None
        if agradr[0] == -np.pi:
            yy = gradr[0]
        elif agradr[-1] == np.pi:
            yy = gradr[-1]
        if agradr[0] > -np.pi:
            if yy is None:
                xx = agradr[0]
                yy = splev(xx, self._fit['gradr_bdry']['tck']) - (np.pi + xx) * splev(xx, self._fit['gradr_bdry']['tck'], der=1)  # Why the derivative?
            agradr = np.concatenate(([-np.pi], agradr))
            gradr = np.concatenate(([yy], gradr))
        if agradr[-1] < np.pi:
            agradr = np.concatenate((agradr, [np.pi]))
            gradr = np.concatenate((gradr, [yy]))
        self._fit['gradr_bdry'] = {'tck': splrep(agradr, gradr, k=3, quiet=1), 'bounds': (float(np.nanmin(agradr)), float(np.nanmax(agradr)))}

        self._fit['gradz_bdry'] = {'tck': splrep(agradz, gradz, k=3, quiet=1), 'bounds': (float(np.nanmin(agradz)), float(np.nanmax(agradz)))}
        if np.abs(agradz[0] + np.pi) < tol:
            agradz[0] = -np.pi
        if np.abs(agradz[-1] - np.pi) < tol:
            agradz[-1] =  np.pi
        yy = None
        if agradz[0] == -np.pi:
            yy = gradz[0]
        elif agradz[-1] == np.pi:
            yy = gradz[-1]
        if agradz[0] > -np.pi:
            if yy is None:
                xx = agradz[0]
                yy = splev(xx, self._fit['gradz_bdry']['tck']) - (np.pi + xx) * splev(xx, self._fit['gradz_bdry']['tck'], der=1)  # Why the derivative?
            agradz = np.concatenate(([-np.pi], agradz))
            gradz = np.concatenate(([yy], gradz))
        if agradz[-1] < np.pi:
            agradz = np.concatenate((agradz, [np.pi]))
            gradz = np.concatenate((gradz, [yy]))
        self._fit['gradz_bdry'] = {'tck': splrep(agradz, gradz, k=3, quiet=1), 'bounds': (float(np.nanmin(agradz)), float(np.nanmax(agradz)))}


    def extend_psi_beyond_boundary(self):

        if 'gradr_bdry' not in self._fit or 'gradz_bdry' not in self._fit:
            self.compute_boundary_gradients_from_grid()

        vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']

        # VECTORS TO AND BETWEEN BOUNDARY POINTS
        vbdry = self._data['rbdry'] + 1.0j * self._data['zbdry']
        vb0 = vbdry[:-1]
        vb1 = vbdry[1:]
        dvb = vb1 - vb0

        # VECTORS TO EXTERIOR GRID POINTS
        jout = self._data['ijout'] // self._data['nr']
        iout = self._data['ijout'] - self._data['nr'] * jout
        vvec = self._data['rvec'].take(iout) + 1.0j * self._data['zvec'].take(jout)

        psiout = np.zeros(vvec.shape, dtype=float)
        ivvec = np.arange(vvec.size)

        ## VECTORS BETWEEN EXTERIOR POINTS AND BOUNDARY POINTS
        for k in range(dvb.size):
            c0 = vb0[k] - vmagx
            angmin = np.angle(c0)
            angmax = np.angle(vb1[k] - vmagx)
            angvec = np.angle(vvec - vmagx)
            angmask = np.isfinite(angvec)
            if (angmin - angmax) > (2.0 * np.pi):
                angmask = np.logical_and(
                    angmask,
                    angvec > 0.0
                )
                angvec.put(np.logical_not(angmask), angvec.compress(np.logical_not(angmask)) + 2.0 * np.pi)
                angmax += 2.0 * np.pi
            numer = c0 * np.conj(dvb[k])
            mask = np.logical_and(
                angvec >= angmin,
                angvec < angmax
            )
            if not np.all(angmask):
                angvec.put(np.logical_not(angmask), angvec.compress(np.logical_not(angmask)) - 2.0 * np.pi)
            if not np.any(mask): continue
            ang = angvec.compress(mask)
            vvecc = vvec.compress(mask)
            ivvecc = ivvec.compress(mask)
            if ang.size > 0:
                dvvecc = vvecc - vmagx
                dv = dvvecc * (1.0 - numer.imag / (dvvecc * np.conj(dvb[k])).imag)
                vgradc = splev(ang, self._fit['gradr_bdry']['tck']) + 1.0j * splev(ang, self._fit['gradz_bdry']['tck'])
                psie = (vgradc * np.conj(dv)).real
                psiout.put(ivvecc, psie)
                vvec = vvec.compress(np.logical_not(mask))
                ivvec = ivvec.compress(np.logical_not(mask))
            if vvec.size == 0: break

        self._data['psi'].ravel().put(self._data['ijout'], psiout)


    def trace_rough_flux_surfaces(self):
        axis = Point([float(self._data['rmagx']), float(self._data['zmagx'])])
        psin = np.linspace(0.0, 1.0, self._data['nr'])
        psin[-1] = 0.9999
        psin = np.delete(psin, 0, axis=0)
        levels = psin * (self._data['sibdry'] - self._data['simagx']) + self._data['simagx']
        rmesh, zmesh = np.meshgrid(self._data['rvec'], self._data['zvec'])
        cg = contourpy.contour_generator(rmesh, zmesh, self._data['psi'])
        contours = {}
        for level in levels:
            vertices = cg.create_contour(level)
            for i in range(len(vertices)):
                if vertices[i] is not None:
                    polygon = Polygon(np.array(vertices[i]))
                    if polygon.contains(axis):
                        contours[float(level)] = vertices[i].copy()
                        break
        return contours


    def trace_fine_flux_surfaces(self, maxpoints=101):
        contours = self.trace_rough_flux_surfaces()
        psisign = np.sign(self._data['sibdry'] - self._data['simagx'])
        levels = np.sort(psisign * np.array(list(contours.keys())))
        vmagx = self._data['rmagx'] + 1.0j * self._data['zmagx']
        vbdry = self._data['rbdry'][:-1] + 1.0j * self._data['zbdry'][:-1]
        abdry = np.angle(vbdry - vmagx)
        if np.any(abdry < 0.0):
            abdry[abdry < 0.0] = abdry[abdry < 0.0] + 2.0 * np.pi
        psi = RectBivariateSpline(self._data['rvec'], self._data['zvec'], self._data['psi'].T)
        dpsidr = psi.partial_derivative(1, 0)
        dpsidz = psi.partial_derivative(0, 1)
        fine_contours = {}
        fine_contours[float(self._data['simagx'])] = {
            'r': np.array([self._data['rmagx']]).flatten(),
            'z': np.array([self._data['zmagx']]).flatten(),
            'fpol': np.array([self._data['fpol'][0]]).flatten(),
            'bpol': np.array([0.0]).flatten(),
            'btor': np.array([self._data['fpol'][0] / self._data['rmagx']]).flatten(),
            'dpsidr': np.array([0.0]).flatten(),
            'dpsidz': np.array([0.0]).flatten(),
        }
        for level in levels:
            ll = psisign * level
            ln = np.abs((ll - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx']))
            fp = splev(ln, self._fit['fpol_fs']['tck'])
            vcont = contours[ll][:, 0] + 1.0j * contours[ll][:, 1]
            vb = np.argmax(np.abs(vbdry - vmagx))
            vc = np.argmax(np.abs(vcont - vmagx))
            npoints = max(21, int(np.rint(float(maxpoints) * vc / vb)))
            angles = np.linspace(0.0, 2.0 * np.pi, npoints)
            rc = []
            zc = []
            for ang in angles[:-1]:
                fang = ang
                for i, segfit in enumerate(self._fit['lseg_abdry']):
                    anglb = segfit['bounds'][0]
                    angub = segfit['bounds'][1]
                    if anglb < 0.0:
                        fang -= 2.0 * np.pi
                    if fang >= anglb and fang <= angub:
                        break
                lbdry = splev(fang, self._fit['lseg_abdry'][i]['tck'])
                vvec = np.linspace(0.0, lbdry, 501) * np.exp(1.0j * ang) + vmagx
                vl = []
                psil = []
                for v in vvec[250:0:-1]:
                    psival = psisign * bisplev(v.real, v.imag, self._fit['psi_rz']['tck'])
                    if len(psil) == 0:
                        vl.append(v)
                        psil.append(psival)
                    elif psival < psil[-1] and psival >= psisign * self._data['simagx']:
                        vl.append(v)
                        psil.append(psival)
                vu = []
                psiu = []
                for v in vvec[251:-1]:
                    psival = psisign * bisplev(v.real, v.imag, self._fit['psi_rz']['tck'])
                    if len(psiu) == 0:
                        vu.append(v)
                        psiu.append(psival)
                    elif psival > psiu[-1] and psival <= psisign * self._data['sibdry']:
                        vu.append(v)
                        psiu.append(psival)
                vscan = np.concatenate([[vvec[0]], vl[::-1], vu, [vvec[-1]]])
                psiscan = np.concatenate([[psisign * self._data['simagx']], psil[::-1], psiu, [psisign * self._data['sibdry']]])
                lmin = np.abs(vscan[0] - vmagx)
                lmax = np.abs(vscan[-1] - vmagx)
                psifunc = interp1d(np.abs(vscan - vmagx), psiscan, bounds_error=False, fill_value='extrapolate')
                lc = brentq(lambda l, t: psifunc(l) - t, lmin, lmax, args=(level), xtol=1.0e-4)
                vroot = lc * np.exp(1.0j * ang) + vmagx
                rc.append(vroot.real)
                zc.append(vroot.imag)
            if len(rc) > 2:
                rc = np.array(rc + [rc[0]]).flatten()
                zc = np.array(zc + [zc[0]]).flatten()
                grpc = dpsidr(rc, zc, grid=False)
                gzpc = dpsidz(rc, zc, grid=False)
                bpc = np.sqrt(np.power(grpc, 2.0) + np.power(gzpc, 2.0)) / rc
                btc = fp / rc
                fine_contours[float(level)] = {
                    'r': rc.copy(),
                    'z': zc.copy(),
                    'fpol': np.array([fp]).flatten(),
                    'bpol': bpc.copy(),
                    'btor': btc.copy(),
                    'dpsidr': grpc.copy(),
                    'dpsidz': gzpc.copy(),
                }
        return fine_contours


    def recompute_pressure_profile(self):
        self.define_pressure_profile(self._data['pres'], smooth=False)


    def recompute_f_profile(self):
        self.define_f_profile(self._data['fpol'], smooth=False)


    def recompute_q_profile(self, smooth=False):
        if 'qpsi' in self._data and 'qpsi_orig' not in self._data:
            self._data['qpsi_orig'] = self._data['qpsi'].copy()
        w = 100.0 / self._data['qpsi'] if smooth else None
        psin = np.linspace(0.0, 1.0, len(self._data['qpsi']))
        psin_mirror = -psin[::-1]
        q_mirror = self._data['qpsi'][::-1]
        w_mirror = w[::-1] if w is not None else None
        if np.isclose(psin[0], psin_mirror[-1]):
            psin_mirror = psin_mirror[:-1]
            q_mirror = q_mirror[:-1]
            w_mirror = w_mirror[:-1] if w_mirror is not None else None
        psin_fit = np.concatenate([psin_mirror, psin])
        q_fit = np.concatenate([q_mirror, self._data['qpsi']])
        w_fit = np.concatenate([w_mirror, w]) if w is not None else None
        self._fit['qpsi_fs'] = {'tck': splrep(psin_fit, q_fit, w_fit, xb=-1.0, xe=1.0, k=3, quiet=1), 'bounds': (-1.0, 1.0)}
        self._data['qpsi'] = splev(np.linspace(0.0, 1.0, self._data['nr']), self._fit['qpsi_fs']['tck'])


    def recompute_q_profile_from_scratch(self):
        if self._data['psi'][0, 0] == self._data['psi'][-1, -1] and self._data['psi'][0, -1] == self._data['psi'][-1, 0]:
            self.extend_psi_beyond_boundary()
        if 'qpsi' in self._data and 'qpsi_orig' not in self._data:
            self._data['qpsi_orig'] = self._data['qpsi'].copy()
        self._fs = self.trace_fine_flux_surfaces()
        qpsi = np.zeros((len(self._fs), ), dtype=float)
        for i, (level, contour) in enumerate(self._fs.items()):
            if contour['r'].size > 1:
                dl = np.sqrt(np.square(np.diff(contour['r'])) + np.square(np.diff(contour['z']))).flatten()
                rcm = 0.5 * (contour['r'][1:] + contour['r'][:-1]).flatten()
                zcm = 0.5 * (contour['z'][1:] + contour['z'][:-1]).flatten()
                bpm = 0.5 * (contour['bpol'][1:] + contour['bpol'][:-1]).flatten()
                btm = 0.5 * (contour['btor'][1:] + contour['btor'][:-1]).flatten()
                dl_over_bp = dl / bpm
                vp = np.sum(dl_over_bp)
                rm2 = np.sum(dl_over_bp / np.square(rcm)) / vp
                qpsi[i] = contour['fpol'].item() * vp * rm2 / (2.0 * np.pi)
        qpsi[0] = 2.0 * qpsi[1] - qpsi[2]  # Linear interpolation to axis
        self._data['qpsi'] = qpsi


    def renormalize_psi(self, simagx=None, sibdry=None):
        if simagx is None and 'simagx_orig' in self._data:
            simagx = self._data['simagx_orig']
        if sibdry is None and 'sibdry_orig' in self._data:
            sibdry = self._data['sibdry_orig']
        if 'psi' in self._data and 'simagx' in self._data and 'sibdry' in self._data and simagx is not None and sibdry is not None:
            self._data['psi'] = ((sibdry - simagx) * (self._data['psi'] - self._data['simagx']) / (self._data['sibdry'] - self._data['simagx'])) + simagx
            self._data['simagx'] = simagx
            self._data['sibdry'] = sibdry


    def run(
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
        self.compute_normalized_psi_map()
        self.zero_psi_outside_boundary()
        self.newj(1.0)
        for n in range(self._options['nxiter']):
            psinew = self.solver(self._data['s5'] * self._data['cur']).reshape(self._data['nz'], self._data['nr'])
            if self._options['relax'] != 1.0:
                psinew = self._data['psi'] + self._options['relax'] * (psinew - self._data['psi'])
            psierror = np.abs(psinew - self._data['psi']).max() / np.abs(psinew).max()
            self._data['psi'] = psinew
            self.find_magnetic_axis_from_grid()
            self.zero_magnetic_boundary()
            self.compute_normalized_psi_map()
            #if self.nxiter < 0:
            #    print('max(psiNew-psiOld)/max(psiNew) = %8.2e'%(error))
            if psierror <= self._options['erreq']: break
            self.newj(self._options['relaxj'])
        self.rescale_kinetic_profiles()
        self.extend_psi_beyond_boundary()
        self.renormalize_psi()
        self.generate_psi_bivariate_spline()
        self.recompute_q_profile_from_scratch()

        self.error = psierror
        if n + 1 == self._options['nxiter']:
            #print ('Failed to converge after %i iterations with error = %8.2e. Time = %6.1f S'%(abs_nxiter,error,t0))
            self.converged = False
        else:
            #print ('Converged after %i iterations with error = %8.2e. Time = %6.1f S'%(n+1,error,t0))
            self.converged = True

        if self.A is not None:
            self._data['errsol'] = self.check_solution()

        self._data['gcase'] = 'FBE'
        self._data['gid'] = 42


    def check_solution(self):
        '''Check accuracy of solution Delta*psi = mu0RJ'''
        # Compute Delta*psi and current density (force balance)
        ds = self.A.dot(self._data['psi'].ravel())
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


    def plot_contour(self):
        if 'psi' in self._data:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 6))
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
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            fig.tight_layout()
            plt.show()
            plt.close(fig)


    def plot_flux_surfaces(self):
        if self._fs is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            for level, contour in self._fs.items():
                ax.plot(contour['r'], contour['z'], c='k', label=f'{level:.3f}')
            if 'rbdry' in self._data and 'zbdry' in self._data:
                ax.plot(self._data['rbdry'], self._data['zbdry'], c='r', label='Boundary')
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

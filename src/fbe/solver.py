import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import splrep, splev, RectBivariateSpline, RegularGridInterpolator
from scipy.sparse import spdiags
from scipy.sparse.linalg import factorized

from .eqdsk import read_eqdsk_file, write_eqdsk_file


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
        if isinstance(eqdsk, (str, Path)):
            self._data.update(read_eqdsk_file(eqdsk))


    def define_grid(self, nr, nz, rmin, rmax, zmin, zmax):
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
        if 'nbdry' not in self._data and 'rbdry' not in self._data and 'zbdry' not in self._data and len(rb) == len(zb):
            self._data['nbdry'] = len(rb)
            self._data['rbdry'] = copy.deepcopy(rbdry)
            self._data['zbdry'] = copy.deepcopy(zbdry)


    def refine_boundary(self, n_boundary=300, boundary_smoothing=1.0e-4):

        if 'rbdry_orig' not in self._data:
            self._data['rbdry_orig'] = self._data['rbdry'].copy()
        if 'zbdry_orig' not in self._data:
            self._data['zbdry_orig'] = self._data['zbdry'].copy()

        if n_boundary < 0:
            n_boundary = self._data['rbdry'].size

        rb0 = 0.5 * (self._data['rbdry'].max() + self._data['rbdry'].min())
        zb0 = 0.5 * (self._data['zbdry'].max() + self._data['zbdry'].min())

        # ORIGINAL BOUNDARY in r,theta
        complex_b = (self._data['rbdry'] - rb0) + 1.0j * (self._data['zbdry'] - zb0)
        theta = np.angle(complex_b)
        theta = np.where(theta < 0.0, 2.0 * np.pi + theta, theta)
        radius = np.abs(complex_b)

        b = xr.Dataset(coords={'angle': theta}, data_vars={'radius': (['angle'], radius)})
        b = b.sortby('angle')
        _, mask = b.angle.to_numpy().unique(return_index=True)
        b = b.isel(angle=mask)
        if b.angle[0] == 0.0:
            b = xr.concat([
                xr.Dataset(coords={'angle': [2.0 * np.pi]}, data_vars={'radius': [b.radius[0].to_numpy()]}),
                b,
            ], dim='angle')
        elif b.angle[-1] == 2.0 * np.pi:
            b = xr.concat([
                b,
                xr.Dataset(coords={'angle': [0.0]}, data_vars={'radius': [b.radius[-1].to_numpy()]})
            ], dim='angle')
        else:
            rezero = b.isel(angle=[0, -1]).radius.mean().to_numpy()
            b = xr.concat([
                xr.Dataset(coords={'angle': [2.0 * np.pi]}, data_vars={'radius': [rezero]}),
                b,
                xr.Dataset(coords={'angle': [0.0]}, data_vars={'radius': [rezero]}),
            ], dim='angle')

        newtheta = np.linspace(0.0, 2.0 * np.pi, 5000)
        newradius = np.ones_like(newtheta)
        if boundary_smoothing > 0.0:
            self._fit['bb'] = splrep(b.angle.to_numpy(), b.radius.to_numpy(), k=3, s=boundary_smoothing, per=True, quiet=1)
            newradius = splev(newtheta, self._fit['bb'])
        else:
            newradius = interp(newtheta, b.angle.to_numpy(), b.radius.to_numpy())
        newb = xr.Dataset(coords={'angle': newtheta}, data_vars={'radius': (['angle'], newradius)})

        # FINELY SPACED BOUNDARY POINTS ON SMOOTHED BOUNDARY
        rbs = (newb.radius.to_numpy() * np.cos(newb.angle.to_numpy()) + rb0)[::-1]
        zbs = (newb.radius.to_numpy() * np.sin(newb.angle.to_numpy()) + zb0)[::-1]

        # GET CURVE LENGTH TO DISTRIBUTE NEW BOUNDARY POINTS IN EQUAL LENGTH INTERVALS
        dl = np.concatenate([
            [0.0],
            np.sqrt(np.square(rbs[1:] - rbs[:-1]) + np.square(zbs[1:] - zbs[:-1])),
        ])
        length = np.cumulative_sum(dl)

        # theta AS A FUNCTION OF CURVE LENGTH
        t = xr.Dataset(coords={'length': length}, data_vars={'angle': (['length'], newtheta)})

        # theta FOR NEW BOUNDARY POINTS EQUALLY SPACED IN LENGTH
        newlength = np.linspace(length[0], length[-1], n_boundary)
        self._fit['ll'] = splrep(t.length.to_numpy(), t.angle.to_numpy(), k=3, quiet=1)
        finaltheta = splev(newlength, self._fit['ll'])

        # NEW BOUNDARY POINTS
        finalradius = np.ones_like(finaltheta)
        if 'bb' in self._fit:
            finalradius = splev(finaltheta, self._fit['bb'])
        else:
            finalradius= interp(finaltheta, b.angle.to_numpy(), b.radius.to_numpy())
        self._data['rbdry'] = (finalradius * np.cos(finaltheta) + rb0)[::-1]
        self._data['zbdry'] = (finalradius * np.sin(finaltheta) + zb0)[::-1]


    def optimize_grid(self, nr, nz):
        # FIT GRID TO BOUNDARY
        e = 3.5
        m = float(nx - 1)
        g = 1.0 / ((m - e)**2 - e**2)
        x0 = self._data['rbdry'].min()
        x1 = self._data['zbdry'].max()
        rmax = m * g * (x1 * (m - e) - x0 * e)
        rmin = m * g * (x0 * (m - e) - x1 * e)
        m = float(ny - 1)
        g = 1.0 / ((m - e)**2 - e**2)
        y0 = self._data['rbdry'].min()
        y1 = self._data['zbdry'].max()
        zmax = m * g * (y1 * (m - e) - y0 * e)
        zmin = m * g * (y0 * (m - e) - y1 * e)
        if zmax > -zmin:
            zmin = -zmax
        elif zmax < -zmin:
            zmax = -zmin
        return rmin, rmax, zmin, zmax


    def setup(self, solver=True):
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
        bma = c1 - c0
        rmask = (bma.imag != 0.0)
        bmar  = bma.imag.compress(rmask)
        zmask = (bma.real != 0.0)
        bmaz  = bma.real.compress(zmask)

        #n = self._data['rbdry'].size - 1
        #s = (self._data['rbdry'][1:] - self._data['rbdry'][:n]) / (self._data['zbdry'][1:] - self._data['zbdry'][:n])
        #s = (r1 - r0) / (z1 - z0)
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
            cr = -a.imag.compress(rmask) / bmar
            cz = -a.real.compress(zmask) / bmaz
            abma = (a.conj() * bma).imag
            dr = abma.compress(rmask) / bmar
            dz = -abma.compress(zmask) / bmaz
            drc = dr.compress((cr <= 1.0) & (cr >= 0)) * self._data['hrm1']
            dzc = dz.compress((cz <= 1.0) & (cz >= 0)) * self._data['hzm1']

            # XM,YM = LOCATION OF ADJACENT OUTSIDE POINTS AS A FRACTION OF GRID SPACING
            if inout[ij] & 0b10:
                # POINT TO THE LEFT IS OUTSIDE
                a1 = -(drc.compress(drc <= 0).max())
                self._data['a1'][ij] = a1
            if inout[ij] & 0b100:
                # POINT TO THE RIGHT IS OUTSIDE
                a2 = drc.compress(drc >= 0).min()
                self._data['a2'][ij] = a2
            if inout[ij] & 0b1000:
                # POINT BELOW IS OUTSIDE
                b1 = -(dzc.compress(dzc <= 0).max())
                self._data['b1'][ij] = b1
            if inout[ij] & 0b10000:
                # POINT ABOVE IS OUTSIDE
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

        self._data['a1'] = self._data['a1'].take(ijedge)
        self._data['a2'] = self._data['a2'].take(ijedge)
        self._data['b1'] = self._data['b1'].take(ijedge)
        self._data['b2'] = self._data['b2'].take(ijedge)

        if solver:
            self.make_solver()


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
        if 'rvec_orig' not in self._data:
            self._data['rvec_orig'] = self._data['rvec'].copy()
        if 'zvec_orig' not in self._data:
            self._data['zvec_orig'] = self._data['zvec'].copy()
        if 'psi_orig' not in self._data:
            self._data['psi_orig'] = self._data['psi'].copy()

        # LINEAR INTERPOLATING FUNCTION FOR PSI
        #meshes = np.meshgrid(self._data['rvec_orig'], self._data['zvec_orig'])
        psi_func = RegularGridInterpolator((self._data['rvec_orig'], self._data['zvec_orig']), self._data['psi_orig'].T)

        # SETUP NEW GRID
        if optimal:
            rmin, rmax, zmin, zmax = self.optimize_grid(nr, nz)
            self._data['rleft'] = rmin
            self._data['rdim'] = rmax - rmin
            self._data['zmid'] = (zmax + zmin) / 2.0
            self._data['zdim'] = zmax - zmin

        self._data['nr'] = nr
        self._data['nz'] = nz
        self.setup(solver=True)

        # INTERPOLATE PSI ONTO NEW GRID
        newpsi = np.zeros((self._data['nrz'], ), dtype=float)
        for j in range(self._data['jmin'], self._data['jmax'] + 1):
            i0 = self._data['imin'][j]
            i1 = self._data['imax'][j] + 1
            ip0 = j * self._data['nr'] + i0
            ip1 = j * self._data['nr'] + i1
            segment = np.stack([self._data['rpsi'][i0:i1, j].flatten(), self._data['zpsi'][i0:i1, j].flatten()], axis=-1)
            newpsi[ip0:ip1] = psi_func(segment).ravel()
        self._data['psi'] = newpsi.reshape(self._data['nr'], self._data['nz'])
        self.find_magnetic_axis()


    def make_solver(self):
        self.solver = factorized(self.A.tocsc())


    def init_psi(self):
        '''Initialize psi. Used if no geqdsk is read.'''

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
            r = rp[i] * drzb / det
            z = zp[j] * drzb / det
            xin = np.logical_or(
                np.logical_and(r - rb[1:] <=  1.e-11, r - rb[:-1] >= -1.e-11),
                np.logical_and(r - rb[1:] >= -1.e-11, r - rb[:-1] <=  1.e-11),
            )
            yin = np.logical_or(
                np.logical_and(z - zb[1:] <=  1.e-11, z - zb[:-1] >= -1.e-11),
                np.logical_and(z - zb[1:] >= -1.e-11, z - zb[:-1] <=  1.e-11),
            )
            rzin = np.logical_and(yin, xin)
            rc = r.compress(rzin)
            zc = z.compress(rzin)
            if rc.size == 0 or zc.size == 0:
                rho = 0.0
            else:
                db = rc[0] ** 2 + zc[0] ** 2 if (rc[0] * rp[i] + zc[0] * zp[j]) > 0 else rc[-1] ** 2 + zc[-1] ** 2
                rho = np.nanmin(np.sqrt((rp[i] ** 2 + zp[j] ** 2) / db), 1.0)
            flat_psi[k] = (1.0 - (rho ** 2)) ** 1.2   # Why 1.2?

        self._data['psi'] = flat_psi.reshape(self._data['nr'], self._data['nz'])
        self.find_magnetic_axis()
        self.newj(1.0)


    def jtor(self, rmaj, psinorm):
        '''Function to compute current density from R and psiN'''
        ff = self._data['ffprime'].copy()
        pp = self._data['pprime'].copy()
        if 'ffprime' in self._fit:
            ff = splev(psinorm, self._fit['ffprime'])
        else:
            ff = np.interp(psinorm, np.linspace(0.0, 1.0, ff.size), ff)
        if 'pprime' in self._fit:
            pp = splev(psinorm, self._fit['pprime'])
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
        self._data['cur'] *= self._data['cpasma'] / (self._data['cur'].sum() * self._data['hrz'])


    def find_magnetic_axis(self):
        '''Compute magnetic axis location and psi value using second order differences'''

        if 'simagx_orig' not in self._data:
            self._data['simagx_orig'] = self._data['simagx']
        if 'sibdry_orig' not in self._data:
            self._data['sibdry_orig'] = self._data['sibdry']
        if 'rmagx_orig' not in self._data:
            self._data['rmagx_orig'] = self._data['rmagx']
        if 'zmagx_orig' not in self._data:
            self._data['zmagx_orig'] = self._data['zmagx']

        # FINITE DIFFERENCE EQUATIONS FOR DERIVATIVES
        apsi = np.abs(self._data['psi'].ravel())
        nr = self._data['nr']
        k = apsi.argmax()
        ax = 0.5 * self._data['hrm1'] * (apsi[k+1] - apsi[k-1])
        ay = 0.5 * self._data['hzm1'] * (apsi[k+nr] - apsi[k-nr])
        axx = self._data['hrm2'] * (apsi[k+1] + apsi[k-1] - 2.0 * apsi[k])
        ayy = self._data['hzm2'] * (apsi[k+nr] + apsi[k-nr] - 2.0 * apsi[k])
        axy = 0.25 * self._data['hrm1'] * self._data['hzm1'] * (
            apsi[k+nr+1] + apsi[k-nr-1] - apsi[k-nr+1] - apsi[k+nr-1]
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
        self._data['sibdry'] = 0.0
        self._data['xpsi'] = np.abs((self._data['simagx'] - self._data['psi']) / (self._data['simagx'] - self._data['sibdry']))

        # MAGNETIC AXIS LOCATION
        j = k // nr
        i = k - j * nr
        self._data['rmagx'] = xmax + self._data['rvec'][i]
        self._data['zmagx'] = ymax + self._data['zvec'][j]


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
        self.newj(1.0)
        for n in range(self._options['nxiter']):
            psinew = self.solver(self._data['s5'] * self._data['cur']).reshape(self._data['nr'], self._data['nz'])
            if self._options['relax'] != 1.0:
                psinew = self._data['psi'] + self._options['relax'] * (psinew - self._data['psi'])
            psierror = np.abs(psinew - self._data['psi']).max() / np.abs(psinew).max()
            self._data['psi'] = psinew
            self.find_magnetic_axis()
            #if self.nxiter < 0:
            #    print('Mhdeq.run: max(psiNew-psiOld)/max(psiNew) = %8.2e'%(error))
            if psierror <= self._options['erreq']:
                break
            self.newj(self._options['relaxj'])

        self.error = psierror
        if n + 1 == self._options['nxiter']:
            #print ('Mhdeq.run: Failed to converge after %i iterations with error = %8.2e. Time = %6.1f S'%(abs_nxiter,error,t0))
            self.converged = False
        else:
            #print ('Mhdeq.run: Converged after %i iterations with error = %8.2e. Time = %6.1f S'%(n+1,error,t0))
            self.converged = True

        if self.A is not None:
            self._data['errsol'] = self.check_solution()


    def check_solution(self):
        '''Check accuracy of solution Delta*psi = mu0RJ'''
        # Compute Delta*psi and current density (force balance)
        ds = self.A.dot(self._data['psi'].ravel())
        cur = self._data['s5'] * self._data['cur']
        curmax = np.abs(cur).max()
        errds  = np.abs(cur - ds).max() / curmax
        #print('Mhdeq.check: max(-Delta*psi-mu0RJ)/max(mu0RJ) = %8.2e'%(errds))
        return errds


    @classmethod
    def from_eqdsk(cls, path):
        return cls(eqdsk=path)


    #def to_eqdsk(self, path):
    #    write_eqdsk_file(path, **{k: v, for k, v in self._data.items() if k in self.eqdsk_fields})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nr', type=int, required=True, help='Number of grid points in R')
    parser.add_argument('nz', type=int, required=True, help='Number of grid points in Z')
    parser.add_argument('niter', nargs='?', type=int, default=50, help='Maximum number of iterations for equilibrium solver')
    parser.add_argument('err', nargs='?', type=float, default=1.0e-8, help='Convergence criteria on psi error for equilibrium solver')
    parser.add_argument('relax', nargs='?', type=float, defualt=1.0, help='Relaxation constant to smoothen psi stepping for stability')
    parser.add_argument('relaxj', nargs='?', type=float, defualt=1.0, help='Relaxation constant to smoothen current stepping for stability')
    parser.add_argument('--ifile', type=str, default=None, help='Path to input g-eqdsk file')
    paresr.add_arguemnt('--optimize', default=False, action='store_true', help='Toggle on optimal grid dimensions to fit boundary contour')
    args = parser.parse_args()

    if args.ifile is None:
        # From scratch
        mhdeq = FixedBoundaryEquilibrium()
        #mhdeq.define_grid(args.nr, args.nz, args.rmin, args.rmax, args.zmin, args.zmax)
        #mhdeq.define_boundary(args.rb, args.zb)
        #mhdeq.setup()
        #mhdeq.init_psi()
        #mhdeq.run(args.niter, args.err, args.relax, args.relaxj)
    else:
        ipath = Path(args.ifile)
        if ipath.is_file():
            mhdeq = FixedBoundaryEquilibrium.from_eqdsk(ipath)
            mhdeq.setup(solver=False)
            mhdeq.regrid(args.nr, args.nz, optimal=args.optimize)
            mhdeq.run(args.niter, args.err, args.relax, args.relaxj)


if __name__ == '__main__':
    main()

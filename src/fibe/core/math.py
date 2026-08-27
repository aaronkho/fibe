import copy
import logging
import numpy as np
import xarray as xr
from scipy.interpolate import (
    interp1d,
    splrep,
    splev,
    bisplev,
    RectBivariateSpline,
    make_interp_spline,
    CubicHermiteSpline,
)
from scipy.sparse import spdiags
from scipy.optimize import brentq, least_squares, root, isotonic_regression
from scipy.integrate import quad, trapezoid
from scipy.signal import windows, convolve
from scipy.stats import gamma, beta
from scipy.special import j0, j1
import contourpy
from shapely import Point, Polygon
from megpy import (
    contour as contour_tracer,
    find_null_points,
)

logger = logging.getLogger('fibe')
logger.setLevel(logging.INFO)


def generate_bounded_1d_spline(y, xnorm=None, symmetrical=True, smooth=False):
    yn = copy.deepcopy(y)
    w = 1000.0 / yn if smooth else None  # Should this be made user-configurable?
    xn = np.linspace(0.0, 1.0, len(yn))
    if isinstance(xnorm, np.ndarray) and len(xnorm) == len(yn):
        xn = copy.deepcopy(xnorm)
    b = (xn[0], xn[-1])
    xn_mirror = []
    yn_mirror = []
    w_mirror = [] if w is not None else None
    if symmetrical:
        b = (-xn[-1], xn[-1])
        xn_mirror = -xn[::-1]
        yn_mirror = yn[::-1]
        w_mirror = w[::-1] if w is not None else None
        if np.isclose(xn[0], xn_mirror[-1]):
            xn_mirror = xn_mirror[:-1]
            yn_mirror = yn_mirror[:-1]
            w_mirror = w_mirror[:-1] if w_mirror is not None else None
    xn_fit = np.concatenate([xn_mirror, xn])
    yn_fit = np.concatenate([yn_mirror, yn])
    w_fit = np.concatenate([w_mirror, w]) if w is not None else None
    return {'tck': splrep(xn_fit, yn_fit, w_fit, xb=b[0], xe=b[-1], k=3, quiet=1), 'bounds': b}


def generate_2d_spline(x, y, z, s=0):
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    z_spline = RectBivariateSpline(x, y, z, s=s)
    tr, tz, c = z_spline.tck
    kr, kz = z_spline.degrees
    return {'tck': (tr, tz, c, kr, kz), 'bounds': (xmin, ymin, xmax, ymax)}


def generate_optimal_grid(nr, nz, rbdry, zbdry, rwall=None, zwall=None):
    # Fit grid to boundary
    e = 3.5
    m = float(nr - 1)
    g = 1.0 / ((m - e)**2 - e**2)
    x0 = np.nanmin(rbdry) if rwall is None else np.nanmin(rwall)
    x1 = np.nanmax(rbdry) if rwall is None else np.nanmax(rwall)
    rmax = m * g * (x1 * (m - e) - x0 * e)
    rmin = m * g * (x0 * (m - e) - x1 * e)
    m = float(nz - 1)
    g = 1.0 / ((m - e)**2 - e**2)
    y0 = np.nanmin(zbdry) if zwall is None else np.nanmin(zwall)
    y1 = np.nanmax(zbdry) if zwall is None else np.nanmax(zwall)
    zmax = m * g * (y1 * (m - e) - y0 * e)
    zmin = m * g * (y0 * (m - e) - y1 * e)
    if np.isclose(zmax, -zmin):
        # Make z grid symmetric if already close
        if zmax > -zmin:
            zmin = -zmax
        elif zmax < -zmin:
            zmax = -zmin
    return rmin, rmax, zmin, zmax


def generate_boundary_maps(rvec, zvec, rbdry, zbdry):

    nr = rvec.size
    nz = zvec.size

    inout = np.zeros((nr * nz, ), dtype=int)

    # i = r, j = z
    ibmin = (1.0 + 0.5 * np.sign(np.nanmin(rbdry) - rvec)).astype(int).sum() - 1
    ibmax = (1.0 + 0.5 * np.sign(np.nanmax(rbdry) - rvec)).astype(int).sum()
    jmin = (1.0 + 0.5 * np.sign(np.nanmin(zbdry) - zvec)).astype(int).sum()
    jmax = (1.0 + 0.5 * np.sign(np.nanmax(zbdry) - zvec)).astype(int).sum() - 1

    # Find grid points inside boundary
    imin = np.zeros((nz, ), dtype=int)
    imax = np.zeros((nz, ), dtype=int)

    r0 = rbdry[:-1]
    r1 = rbdry[1:]
    z0 = zbdry[:-1]
    z1 = zbdry[1:]
    for j in range(jmin, jmax+1):
        k0 = np.arange(z0.size).astype(int).compress(
            np.logical_and(zvec[j] >= z0, zvec[j] < z1)
        )
        rc0 = None
        if len(k0) > 0:
            k = k0[0]
            rc0 = (r0[k] - r1[k]) * (zvec[j] - z1[k]) / (z0[k] - z1[k]) + r1[k]
        k1 = np.arange(z0.size).astype(int).compress(
            np.logical_and(zvec[j] < z0, zvec[j] >= z1)
        )
        rc1 = None
        if len(k1) > 0:
            k = k1[0]
            rc1 = (r0[k] - r1[k]) * (zvec[j] - z1[k]) / (z0[k] - z1[k]) + r1[k]
        rcmin = None
        rcmax = None
        if rc0 is not None and rc1 is not None:
            rcmin = rc0 if rc0 < rc1 else rc1
            rcmax = rc1 if rc0 < rc1 else rc0
        ir = np.arange(rvec.size).astype(int).compress(
            np.logical_and(rvec > rcmin, rvec < rcmax)
        ) if rcmin is not None and rcmax is not None else np.array([])
        if ir.size > 0:
            imin[j] = ir[0]
            imax[j] = ir[-1]
            inout[ir[0] + nr * j:ir[-1] + nr * j + 1] = 1
        else:
            imin[j] = 0
            imax[j] = -1

    if imax[jmin] < 0:
        jmin += 1
    if imax[jmax] < 0:
        jmax -= 1
    ijin = np.arange(nr * nz).astype(int).compress(inout)

    # ijedge are points with a neighbour that is outside the boundary
    # These points have special difference equations

    ijinl = ijin.compress(inout.take(ijin - 1) == 0) # Left
    ijinr = ijin.compress(inout.take(ijin + 1) == 0) # Right
    ijinb = ijin.compress(inout.take(ijin - nr) == 0) # Below
    ijina = ijin.compress(inout.take(ijin + nr) == 0) # Above
    inout.put(ijinl, inout.take(ijinl) | 0b10)
    inout.put(ijinr, inout.take(ijinr) | 0b100)
    inout.put(ijinb, inout.take(ijinb) | 0b1000)
    inout.put(ijina, inout.take(ijina) | 0b10000)

    ijout = np.arange(nr * nz).astype(int).compress(inout == 0)
    ijedge = np.arange(nr * nz).astype(int).compress(inout > 1)

    return inout, ijin, ijout, ijedge


def compute_grid_spacing(rvec, zvec):
    nr = rvec.size
    nz = zvec.size
    rmin = np.nanmin(rvec)
    rmax = np.nanmax(rvec)
    zmin = np.nanmin(zvec)
    zmax = np.nanmax(zvec)
    hr = (rmax - rmin) / float(nr - 1)
    hz = (zmax - zmin) / float(nz - 1)
    hrm1 = 1.0 / hr
    hrm2 = (1.0 / hr) ** 2
    hzm1 = 1.0 / hz
    hzm2 = (1.0 / hz) ** 2
    return hr, hrm1, hrm2, hz, hzm1, hzm2


def generate_finite_difference_grid(rvec, zvec, rbdry, zbdry):

    mu0 = 4.0e-7 * np.pi
    nr = rvec.size
    nz = zvec.size
    rpsi = np.repeat(np.atleast_2d(rvec), nz, axis=0)
    zpsi = np.repeat(np.atleast_2d(zvec).T, nr, axis=1)

    hr, hrm1, hrm2, hz, hzm1, hzm2 = compute_grid_spacing(rvec, zvec)

    inout, ijin, ijout, ijedge = generate_boundary_maps(rvec, zvec, rbdry, zbdry)

    nrz = nr * nz
    hrz = hr * hz

    ss1 = np.ones((nrz, ))
    ss1.put(ijin, 2.0 * (hrm2 + hzm2))
    ss2 = np.zeros((nrz, ))
    ss2.put(ijin, hrm2)
    ss3 = copy.deepcopy(ss2)
    rxx = np.where(inout, (0.5 * hrm1) / rpsi.ravel(), 0.0)
    ss2 += rxx
    ss3 -= rxx
    ss4 = np.zeros((nrz, ))
    ss4.put(ijin, hzm2)
    ss5 = copy.deepcopy(ss4)
    ss6 = np.where(inout, mu0 * rpsi.ravel(), 0.0)

    a1 = np.ones((nrz, ))
    a2 = np.ones((nrz, ))
    b1 = np.ones((nrz, ))
    b2 = np.ones((nrz, ))

    # COMPUTE DIFFERENCE EQUATION QUANTITIES
    c0 = rbdry[:-1] + 1.0j * zbdry[:-1]
    c1 = rbdry[1:] + 1.0j * zbdry[1:]
    dl = c1 - c0
    rmask = (dl.imag != 0.0)
    dlr  = dl.imag.compress(rmask)
    zmask = (dl.real != 0.0)
    dlz  = dl.real.compress(zmask)

    # LOOP OVER EDGE POINTS
    for ij in ijedge:

        j = ij // nr
        i = ij - nr * j
        rr = rvec[i]
        zz = zvec[j]

        # BOUNDARY GRID POINT: DIFFERENCES BASED ON LOCATION OF ADJACENT OUTSIDE POINT
        a1ij = 1.0
        a2ij = 1.0
        b1ij = 1.0
        b2ij = 1.0
        a = c0 - (rr + 1.0j * zz)
        cr = -a.imag.compress(rmask) / dlr
        cz = -a.real.compress(zmask) / dlz
        adl = (a.conj() * dl).imag
        dr = adl.compress(rmask) / dlr
        dz = -adl.compress(zmask) / dlz
        drc = dr.compress((cr <= 1.0) & (cr >= 0)) * hrm1   # (a.r - dl.r * a.z / dl.z) / hr
        dzc = dz.compress((cz <= 1.0) & (cz >= 0)) * hzm1   # (a.z - dl.z * a.r / dl.r) / hz

        # XM,YM = LOCATION OF ADJACENT OUTSIDE POINTS AS A FRACTION OF GRID SPACING
        if inout[ij] & 0b10:
            # POINT TO THE LEFT IS OUTSIDE
            if np.any(drc <= 0):
                a1ij = -(drc.compress(drc <= 0).max())
            a1[ij] = a1ij
        if inout[ij] & 0b100:
            # POINT TO THE RIGHT IS OUTSIDE
            if np.any(drc >= 0):
                a2ij = drc.compress(drc >= 0).min()
            a2[ij] = a2ij
        if inout[ij] & 0b1000:
            # POINT BELOW IS OUTSIDE
            if np.any(dzc <= 0):
                b1ij = -(dzc.compress(dzc <= 0).max())
            b1[ij] = b1ij
        if inout[ij] & 0b10000:
            # POINT ABOVE IS OUTSIDE
            if np.any(dzc >= 0):
                b2ij = dzc.compress(dzc >= 0).min()
            b2[ij] = b2ij

        # MODIFIED DIFFERENCE EQUATION QUANTITIES
        ss1[ij] = hrm1 * (2.0 * hrm1 + (a2ij - a1ij) / rr) / (a1ij * a2ij) + 2.0 * hzm2 / (b1ij * b2ij)
        ss2[ij] = hrm1 * (2.0 * hrm1 + a2ij / rr) / (a1ij * (a1ij + a2ij))
        ss3[ij] = hrm1 * (2.0 * hrm1 - a1ij / rr) / (a2ij * (a1ij + a2ij))
        ss4[ij] = 2.0 * hzm2 / b1ij / (b1ij + b2ij)
        ss5[ij] = 2.0 * hzm2 / b2ij / (b1ij + b2ij)

    s1 = ss2 / ss1
    s2 = ss3 / ss1
    s3 = ss4 / ss1
    s4 = ss5 / ss1
    s5 = ss6 / ss1

    out = {
        'nr': nr,
        'nz': nz,
        'nrz': nrz,
        'rpsi': rpsi,
        'zpsi': zpsi,
        'hr': hr,
        'hrm1': hrm1,
        'hrm2': hrm2,
        'hz': hz,
        'hzm1': hzm1,
        'hzm2': hzm2,
        'hrz': hrz,
        'a1': a1,
        'a2': a2,
        'b1': b1,
        'b2': b2,
        's1': s1,
        's2': s2,
        's3': s3,
        's4': s4,
        's5': s5,
        'inout': inout,
        'ijin': ijin,
        'ijout': ijout,
        'ijedge': ijedge,
    }
    out['matrix'] = compute_finite_difference_matrix(nr, nz, s1, s2, s3, s4)

    return out


def compute_jtor(rpsi, ffprime, pprime):
    '''Compute current density over grid. Scale to Ip'''
    mu0 = 4.0e-7 * np.pi
    jtor = -1.0 * (ffprime / (mu0 * rpsi) + rpsi * pprime)  # This -1 is for COCOS convention
    return jtor


def compute_psi(solver, s5, current):
    return -1.0 * solver(s5 * current)  # This -1 is necessary from GS equation


def compute_jpar(btot, fpol, fprime, pprime):
    mu0 = 4.0e-7 * np.pi
    jpar = -1.0 * (fpol * pprime / btot + fprime * btot / mu0)  # This -1 is for COCOS convention
    return jpar


def compute_finite_difference_matrix(nr, nz, s1, s2, s3, s4):
    # Full difference matrix for sparse solution finding
    data = np.array([np.ones((nr * nz, )), -s1, -s2, -s3, -s4])
    diags = np.array([0, 1, -1, nr, -nr])
    return spdiags(data, diags, nr * nz, nr * nz).T


def generate_initial_psi(rvec, zvec, rbdry, zbdry, ijin):
    nr = rvec.size
    nz = zvec.size
    flat_psi = np.zeros((nr * nz, ), dtype=float)
    r0 = 0.5 * (np.nanmax(rbdry) + np.nanmin(rbdry))
    z0 = 0.5 * (np.nanmax(zbdry) + np.nanmin(zbdry))
    rb = rbdry - r0
    zb = zbdry - z0
    rp = rvec - r0
    zp = zvec - z0
    drb = rb[1:] - rb[:-1]
    dzb = zb[1:] - zb[:-1]
    drzb = rb[:-1] * dzb - zb[:-1] * drb
    for k in ijin:
        j = k // nr
        i = k - j * nr
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
            rho = np.nanmin([np.sqrt((rp[i] ** 2 + zp[j] ** 2) / db), 1.0])
        flat_psi[k] = (1.0 - (rho ** 2)) ** 1.2   # Why 1.2?
    flat_psi *= -1.0
    return flat_psi.reshape(nz, nr)


def compute_grad_psi_vector_from_2d_spline(point, spline_tck):
    ddim1 = bisplev(point[0], point[1], spline_tck, dx=1)
    ddim2 = bisplev(point[0], point[1], spline_tck, dy=1)
    return np.array([ddim1, ddim2]).flatten()


def find_saddle_point_near_contour(r_contour, z_contour, psi_tck):
    '''Locate the psi null point (X-point-like saddle) closest to a boundary
    contour, independent of the contour's own point ordering.
    Returns ((r, z), success).
    '''
    dpsidr = np.array([bisplev(r, z, psi_tck, dx=1) for r, z in zip(r_contour, z_contour)]).flatten()
    dpsidz = np.array([bisplev(r, z, psi_tck, dy=1) for r, z in zip(r_contour, z_contour)]).flatten()
    iseed = int(np.argmin(dpsidr ** 2 + dpsidz ** 2))
    sol = root(compute_grad_psi_vector_from_2d_spline, np.array([r_contour[iseed], z_contour[iseed]]), args=(psi_tck, ))
    return (float(sol.x[0]), float(sol.x[1])), bool(sol.success)


def remove_contour_points_beyond_saddle(r_contour, z_contour, r_reference, z_reference, saddle, angle_tol=0.35, radius_margin=1.02):
    r_contour = np.asarray(r_contour)
    z_contour = np.asarray(z_contour)
    theta = np.arctan2(z_contour - z_reference, r_contour - r_reference)
    radius = np.hypot(r_contour - r_reference, z_contour - z_reference)
    theta_s = np.arctan2(saddle[1] - z_reference, saddle[0] - r_reference)
    radius_s = np.hypot(saddle[0] - r_reference, saddle[1] - z_reference)
    dtheta = np.abs(np.mod(theta - theta_s + np.pi, 2.0 * np.pi) - np.pi)
    is_leg = (dtheta < angle_tol) & (radius > radius_s * radius_margin)
    return ~is_leg


def order_contour_points_by_angle(r_contour, z_contour, r_reference=None, z_reference=None, close_contour=True, **extra_data):
    if r_reference is None:
        r_reference = 0.5 * (np.nanmax(r_contour) + np.nanmin(r_contour))
    if z_reference is None:
        z_reference = 0.5 * (np.nanmax(z_contour) + np.nanmin(z_contour))
    contour = r_contour + 1.0j * z_contour
    reference = r_reference + 1.0j * z_reference
    angle = np.mod(np.angle(contour - reference), 2.0 * np.pi)
    coords = {'angle': angle}
    data_vars = {'r': (['angle'], r_contour), 'z': (['angle'], z_contour)}
    for k, v in extra_data:
        if len(v) == len(angle):
            data_vars[f'{k}'] = (['angle'], v)
    ds = xr.Dataset(coords=coords, data_vars=data_vars)
    ds = ds.drop_duplicates('angle').sortby('angle')
    r_ordered = ds['r'].to_numpy()
    z_ordered = ds['z'].to_numpy()
    angle_ordered = ds['angle'].to_numpy()
    extra_ordered = {k: ds[k].to_numpy() for k in data_vars if k not in ['r', 'z']}
    # Ensure returned contour closes
    if close_contour:
        r_ordered = np.concatenate([r_ordered, [r_ordered[0]]])
        z_ordered = np.concatenate([z_ordered, [z_ordered[0]]])
        angle_ordered = np.concatenate([angle_ordered, [angle_ordered[0] + 2.0 * np.pi]])
        for k in extra_ordered:
            extra_ordered[k] = np.concatenate([extra_ordered[k], [extra_ordered[k][0]]])
    return r_ordered, z_ordered, angle_ordered, extra_ordered


def generate_segments(r_contour, z_contour, indices, cuts=None, r_reference=None, z_reference=None):
    if r_reference is None:
        r_reference = 0.5 * (np.nanmax(r_contour) + np.nanmin(r_contour))
    if z_reference is None:
        z_reference = 0.5 * (np.nanmax(z_contour) + np.nanmin(z_contour))
    v_reference = r_reference + 1.0j * z_reference
    v_contour = r_contour + 1.0j * z_contour
    index_contour = np.arange(len(v_contour), dtype=int)
    cut_indices = cuts if isinstance(cuts, list) else []
    lines = []
    for i, index in enumerate(indices):
        v_segment = np.array([])
        mask = (index_contour > index)
        i_next = i + 1 if index not in cut_indices else i + 2
        if i_next < len(indices):
            mask &= (index_contour >= indices[i_next - 1]) & (index_contour <= indices[i_next])
            v_segment = v_contour.compress(mask)
        else:
            mask[-1] = False
            v_segment = v_contour.compress(mask)
            mask2 = (index_contour <= indices[0])
            v_segment = np.concatenate([v_segment, v_contour.compress(mask2)])
            mask |= mask2
        if len(v_segment) < 2:
            widen = 1
            lo_index = index
            hi_index = indices[i_next] if i_next < len(indices) else indices[0]
            while len(v_segment) < 2 and widen <= len(index_contour):
                lo = max(lo_index - widen, 0)
                hi = min(hi_index + widen, len(index_contour) - 1)
                mask_pad = (index_contour >= lo) & (index_contour <= hi)
                v_segment = v_contour.compress(mask_pad)
                widen += 1
        if (len(v_segment) + 1) < len(index_contour):
            dv_segment = np.diff(v_segment)
            angle_dv_segment = np.angle(dv_segment)
            if angle_dv_segment[-1] < angle_dv_segment[0]:
                angle_dv_segment = np.where(angle_dv_segment < 0.0, angle_dv_segment + 2.0 * np.pi, angle_dv_segment)
            amin = np.nanmin(angle_dv_segment)
            amax = np.nanmax(angle_dv_segment)
            if amax - amin > np.pi:
                logger.error(f'Spline angle error: {amax - amin} > pi')
            rotation = np.exp(1.0j * (amax + amin - np.pi) / 2.0)
            vspline = (v_segment - v_reference) * rotation
            xspline = vspline.real
            yspline = vspline.imag
            if xspline[0] > xspline[-1]:
                xspline = xspline[::-1]
                yspline = yspline[::-1]
            spl = make_interp_spline(xspline, yspline, bc_type='natural')
            l0 = np.array([v_segment[0], v_segment[0] - (1.0 + 1.0j * spl(vspline[0].real, 1)) / rotation])
            l1 = np.array([v_segment[-1], v_segment[-1] + (1.0 + 1.0j * spl(vspline[-1].real, 1)) / rotation])
            lines.append((l0, l1))
        else:
            nchop = v_segment // 3
            dv_segment = np.diff(v_segment)
            angle_dv_segment = np.angle(dv_segment)
            if angle_dv_segment[-1] < angle_dv_segment[0]:
                angle_dv_segment = np.where(angle_dv_segment < 0.0, angle_dv_segment + 2.0 * np.pi, angle_dv_segment)
            amin0 = np.nanmin(angle_dv_segment[:nchop-1])
            amax0 = np.nanmax(angle_dv_segment[:nchop-1])
            if amax0 - amin0 > np.pi:
                logger.error(f'Spline 0 angle error: {amax0 - amin0} > pi')
            rotation0 = np.exp(1.0j * (amax0 + amin0 - np.pi) / 2.0)
            vspline0 = (v_segment[:nchop] - v_reference) * rotation0
            xspline0 = vspline0.real
            yspline0 = vspline0.imag
            if xspline0[0] > xspline0[-1]:
                xspline0 = xspline0[::-1]
                yspline0 = yspline0[::-1]
            spl0 = make_interp_spline(xspline0, yspline0, bc_type='natural')
            amin1 = np.nanmin(angle_dv_segment[-nchop+1:])
            amax1 = np.nanmax(angle_dv_segment[-nchop+1:])
            if amax1 - amin1 > np.pi:
                logger.error(f'Spline 1 angle error: {amax1 - amin1} > pi')
            rotation1 = np.exp(1.0j * (amax1 + amin1 - np.pi) / 2.0)
            vspline1 = (v_segment[-nchop:] - v_reference) * rotation1
            xspline1 = vspline1.real
            yspline1 = vspline1.imag
            if xspline1[0] > xspline1[-1]:
                xspline1 = xspline1[::-1]
                yspline1 = yspline1[::-1]
            spl1 = make_interp_spline(xspline1, yspline1, bc_type='natural')
            l0 = np.array([v_segment[0], v_segment[0] - (1.0 + 1.0j * spl0(vspline[0].real, 1)) / rotation0])
            l1 = np.array([v_segment[-1], v_segment[-1] + (1.0 + 1.0j * spl1(vspline[-1].real, 1)) / rotation1])
            lines.append((l0, l1))
    return lines


def generate_x_point_candidates(rbdry, zbdry, rmagx, zmagx, psi_tck, dr, dz):

    vmagx = rmagx + 1.0j * zmagx
    orbdry, ozbdry, oabdry, _ = order_contour_points_by_angle(rbdry, zbdry, rmagx, zmagx)
    ovbdry = orbdry + 1.0j * ozbdry
    olbdry = np.abs(ovbdry - vmagx)
    dpsidr_obdry = np.array([bisplev(r, z, psi_tck, dx=1) for r, z in zip(orbdry, ozbdry)]).flatten()
    dpsidz_obdry = np.array([bisplev(r, z, psi_tck, dy=1) for r, z in zip(orbdry, ozbdry)]).flatten()

    xpoint_candidates = []
    xpoint_indices = []
    dpsidr_zero = np.where(np.isclose(dpsidr_obdry, 0.0))[0]
    dpsidz_zero = np.where(np.isclose(dpsidz_obdry, 0.0))[0]
    for idr in dpsidr_zero:
        if idr in dpsidz_zero:
            xpoint_indices.append(idr)
            xpoint_candidates.append(np.array([orbdry[idr], ozbdry[idr]]))

    split_indices = []
    split_cut_indices = []
    dpsidr_change = np.where(dpsidr_obdry[:-1] * dpsidr_obdry[1:] < 0.0)[0]
    for idr in dpsidr_change:
        if idr not in xpoint_indices and idr + 1 not in xpoint_indices:
            split_indices.append(int(idr))
    dpsidz_change = np.where(dpsidz_obdry[:-1] * dpsidz_obdry[1:] < 0.0)[0]
    for idz in dpsidz_change:
        if idz not in xpoint_indices and idz + 1 not in xpoint_indices:
            split_indices.append(int(idz))
    split_indices = sorted(set(split_indices))
    for index in split_indices:
        if index + 1 in split_indices:
            split_cut_indices.append(index)
        if index == len(dpsidr_obdry) - 2 and 0 in split_indices:
            split_cut_indices.append(index)

    split_lines = generate_segments(orbdry, ozbdry, split_indices, split_cut_indices, rmagx, zmagx)

    intersections = []
    for i, lines in enumerate(split_lines):
        p1, p2 = lines[-1]
        p3, p4 = split_lines[i + 1][0] if i + 1 < len(split_lines) else split_lines[0][0]
        ta, tb = compute_intersection_from_line_segment_complex(p1, p2, p3, p4)
        px = p1 + ta * (p2 - p1)
        if not np.isclose(np.abs(px - (p3 + tb * (p4 - p3))), 0.0):
            logger.error('Intersection error')
        intersections.append(np.array([px.real, px.imag]))

    for i, inter in enumerate(intersections):
        drl = bisplev(inter[0] - dr, inter[1], psi_tck, dx=1)
        drr = bisplev(inter[0] + dr, inter[1], psi_tck, dx=1)
        dzb = bisplev(inter[0], inter[1] - dz, psi_tck, dy=1)
        dza = bisplev(inter[0], inter[1] + dz, psi_tck, dy=1)
        if drl * drr <= 0.0 and dzb * dza <= 0.0:
            xpoint_candidates.append(inter)

    return xpoint_candidates


def compute_intersection_from_ray_complex(p1s, p1d, p2s, p2d):
    dx = p2s - p1s
    l1 = np.nan
    l2 = np.nan
    if not np.isclose(p1d.imag * p2d.real - p1d.real * p2d.imag, 0.0):
        l1 = (dx.imag * p2d.real - dx.real * p2d.imag) / (p1d.imag * p2d.real - p1d.real * p2d.imag)
        l2 = (dx.imag * p1d.real - dx.real * p1d.imag) / (p1d.imag * p2d.real - p1d.real * p2d.imag)
    return l1, l2


def compute_intersection_from_line_segment_complex(p1s, p1e, p2s, p2e):
    # Implemented equations:
    # u = (x1 - x3 + t * (x2 - x1)) / (x4 - x3)
    # t * (y2 - y1 - (x2 - x1) * (y4 - y3) / (x4 - x3)) = y3 - y1 - (x3 - x1) * (y4 - y3) / (x4 - x3)
    p1d = p1e - p1s
    p2d = p2e - p2s
    return compute_intersection_from_ray_complex(p1s, p1d, p2s, p2d)


def compute_intersection_from_line_segment_coordinates(r1s, z1s, r1e, z1e, r2s, z2s, r2e, z2e):
    p1s = r1s + 1.0j * z1s
    p1e = r1e + 1.0j * z1e
    p2s = r2s + 1.0j * z2s
    p2e = r2e + 1.0j * z2e
    return compute_intersection_from_line_segment_complex(p1s, p1e, p2s, p2e)


def avoid_convex_curvature(r_contour, z_contour, r_point, z_point, r_reference=None, z_reference=None, r_vertex=None, z_vertex=None):
    r_new = r_point
    z_new = z_point
    if r_reference is None:
        r_reference = 0.5 * (np.nanmax(r_contour) + np.nanmin(r_contour))
    if z_reference is None:
        z_reference = 0.5 * (np.nanmax(z_contour) + np.nanmin(z_contour))
    if r_vertex is None:
        r_vertex = 0.5 * (np.nanmax(r_contour) + np.nanmin(r_contour))
    if z_vertex is None:
        z_vertex = 0.5 * (np.nanmax(z_contour) + np.nanmin(z_contour))
    v_reference = r_reference + 1.0j * z_reference
    v_vertex = r_vertex + 1.0j * z_vertex
    v_point = r_point + 1.0j * z_point
    r_ordered, z_ordered, angle_ordered, _ = order_contour_points_by_angle(r_contour, z_contour, r_reference, z_reference)
    angle_point = np.mod(np.angle(v_point - v_reference), 2.0 * np.pi)
    dangle_ordered = angle_ordered - angle_point
    iangle = np.argmin(np.abs(dangle_ordered))
    jangle = iangle
    sign_ddangle = np.where(dangle_ordered[:-1] * dangle_ordered[1:] <= 0.0)[0]
    if len(sign_ddangle) == 0:
        if dangle_ordered[0] >= 0.0:
            iangle = -1
        else:
            jangle = 0
    else:
        if sign_ddangle[0] >= iangle:
            jangle = sign_ddangle[0] + 1 if (sign_ddangle[0] + 2) < len(dangle_ordered) else -1
        else:
            iangle = sign_ddangle[0]
    i0 = r_ordered[iangle - 1] + 1.0j * z_ordered[iangle - 1]
    i1 = r_ordered[iangle] + 1.0j * z_ordered[iangle]
    tai, tbi = compute_intersection_from_line_segment_complex(v_vertex, v_point, i0, i1)
    j0 = r_ordered[jangle + 1] + 1.0j * z_ordered[jangle + 1]
    j1 = r_ordered[jangle] + 1.0j * z_ordered[jangle]
    taj, tbj = compute_intersection_from_line_segment_complex(v_vertex, v_point, j0, j1)
    ta = np.nanmin([tai, taj])
    if ta < 1.0:
        v_new = v_vertex + 0.99 * ta * (v_point - v_vertex)
        r_new = v_new.real
        z_new = v_new.imag
    return r_new, z_new


def generate_boundary_splines(rbdry, zbdry, rmagx, zmagx, xpoints, enforce_concave=True):
    vmagx = rmagx + 1.0j * zmagx
    r_ordered, z_ordered, angle_ordered, _ = order_contour_points_by_angle(rbdry, zbdry, rmagx, zmagx, close_contour=True)
    v_ordered = r_ordered + 1.0j * z_ordered
    length_ordered = np.abs(v_ordered - vmagx)
    rxps = np.array([xp[0] for xp in xpoints])
    zxps = np.array([xp[-1] for xp in xpoints])
    rxp_ordered, zxp_ordered, axp_ordered, _ = order_contour_points_by_angle(rxps, zxps, rmagx, zmagx, close_contour=False)
    xps = [np.array([r, z]) for r, z in zip(rxp_ordered, zxp_ordered)]
    mask = np.isfinite(angle_ordered)
    for i, xp in enumerate(xps):
        rxp = xp[0]
        zxp = xp[-1]
        vxp = rxp + 1.0j * zxp
        length_xp = np.abs(vxp - vmagx)
        angle_xp = np.angle(vxp - vmagx)
        if angle_xp < 0.0:
            angle_xp += 2.0 * np.pi
        dangle_xp = np.abs(angle_ordered - angle_xp)
        mask &= ~((dangle_xp < (np.pi / 6.0)) & (length_ordered > (0.99 * length_xp)))
        if dangle_xp[0] < (np.pi / 6.0):
            mask &= ~((np.abs(angle_ordered - 2.0 * np.pi - angle_xp) < (np.pi / 6.0)) & (length_ordered > (0.99 * length_xp)))
        if dangle_xp[-1] < (np.pi / 6.0):
            mask &= ~((np.abs(angle_ordered + 2.0 * np.pi - angle_xp) < (np.pi / 6.0)) & (length_ordered > (0.99 * length_xp)))
    r_ordered = r_ordered[mask]
    z_ordered = z_ordered[mask]
    angle_ordered = angle_ordered[mask]
    length_ordered = length_ordered[mask]
    splines = []
    for i, xp in enumerate(xps):
        r_xpa = xp[0]
        z_xpa = xp[-1]
        v_xpa = r_xpa + 1.0j * z_xpa
        length_xpa = np.abs(v_xpa - vmagx)
        angle_xpa = np.angle(v_xpa - vmagx)
        if angle_xpa < 0.0:
            angle_xpa += 2.0 * np.pi
        smask = (angle_ordered >= angle_xpa)
        r_segment = None
        z_segment = None
        angle_segment = None
        length_segment = None
        if i + 1 < len(xps):
            r_xpb = xps[i + 1][0]
            z_xpb = xps[i + 1][-1]
            v_xpb = r_xpb + 1.0j * z_xpb
            length_xpb = np.abs(v_xpb - vmagx)
            angle_xpb = np.angle(v_xpb - vmagx)
            if angle_xpb < 0.0:
                angle_xpb += 2.0 * np.pi
            if angle_xpb <= 2.0 * np.pi:
                smask &= (angle_ordered <= angle_xpb)
            else:
                smask |= (angle_ordered <= (angle_xpb - 2.0 * np.pi))
            r_segment = np.concatenate([[r_xpa], r_ordered.compress(smask), [r_xpb]])
            z_segment = np.concatenate([[z_xpa], z_ordered.compress(smask), [z_xpb]])
            angle_segment = np.concatenate([[angle_xpa], angle_ordered.compress(smask), [angle_xpb]])
            length_segment = np.concatenate([[length_xpa], length_ordered.compress(smask), [length_xpb]])
        else:
            smask[-1] = False
            r_segment = np.concatenate([[r_xpa], r_ordered.compress(smask)])
            z_segment = np.concatenate([[z_xpa], z_ordered.compress(smask)])
            length_segment = np.concatenate([[length_xpa], length_ordered.compress(smask)])
            angle_segment = np.concatenate([[angle_xpa], angle_ordered.compress(smask)])
            angle_segment = angle_segment - 2.0 * np.pi
            r_xpb = xps[0][0]
            z_xpb = xps[0][-1]
            v_xpb = r_xpb + 1.0j * z_xpb
            length_xpb = np.abs(v_xpb - vmagx)
            angle_xpb = np.angle(v_xpb - vmagx)
            if angle_xpb < 0.0:
                angle_xpb += 2.0 * np.pi
            smask2 = (angle_ordered <= angle_xpb)
            r_segment = np.concatenate([r_segment, r_ordered.compress(smask2), [r_xpb]])
            z_segment = np.concatenate([z_segment, z_ordered.compress(smask2), [z_xpb]])
            length_segment = np.concatenate([length_segment, length_ordered.compress(smask2), [length_xpb]])
            angle_segment = np.concatenate([angle_segment, angle_ordered.compress(smask2), [angle_xpb]])
            smask |= smask2
        if enforce_concave:
            for j in range(2, len(r_segment)):
                ta, tb = compute_intersection_from_line_segment_coordinates(
                    r_segment[j - 2],
                    z_segment[j - 2],
                    r_segment[j],
                    z_segment[j],
                    rmagx,
                    zmagx,
                    r_segment[j - 1],
                    z_segment[j - 1]
                )
                if tb > 1.0:
                    newp = vmagx + tb * (r_segment[j - 1] + 1.0j * z_segment[j - 1] - vmagx) / 0.99
                    r_segment[j - 1] = newp.real
                    z_segment[j - 1] = newp.imag
                    length_segment[j - 1] = np.abs(newp - vmagx)
                if tb < 0.96:
                    newp = vmagx + tb * (r_segment[j - 1] + 1.0j * z_segment[j - 1] - vmagx) / 0.97
                    r_segment[j - 1] = newp.real
                    z_segment[j - 1] = newp.imag
                    length_segment[j - 1] = np.abs(newp - vmagx)
        #if angle_segment[0] > angle_segment[-1]:
        #    angle_segment[angle_segment > np.pi] = angle_segment[angle_segment > np.pi] - 2.0 * np.pi
        if len(angle_segment) > 2:
            spl = make_interp_spline(angle_segment, length_segment, bc_type=None)
            splines.append({'tck': spl.tck, 'bounds': (float(np.nanmin(angle_segment)), float(np.nanmax(angle_segment)))})
    if len(splines) == 0:
        spl = make_interp_spline(angle_ordered, length_ordered, bc_type='periodic')
        splines.append({'tck': spl.tck, 'bounds': (float(np.nanmin(angle_ordered)), float(np.nanmax(angle_ordered)))})
    return splines


def find_extrema_with_taylor_expansion(rvec, zvec, psi):

    nr = rvec.size
    nz = zvec.size
    hr, hrm1, hrm2, hz, hzm1, hzm2 = compute_grid_spacing(rvec, zvec)
    apsi = np.abs(psi.ravel())
    k = apsi.argmax()
    ar = 0.5 * hrm1 * (apsi[k + 1] - apsi[k - 1])
    az = 0.5 * hzm1 * (apsi[k + nr] - apsi[k - nr])
    arr = hrm2 * (apsi[k + 1] + apsi[k - 1] - 2.0 * apsi[k])
    azz = hzm2 * (apsi[k + nr] + apsi[k - nr] - 2.0 * apsi[k])
    arz = 0.25 * hrm1 * hzm1 * (
        apsi[k + nr + 1] + apsi[k - nr - 1] - apsi[k - nr + 1] - apsi[k + nr - 1]
    )
    delta = arz * arz - arr * azz
    rmax = (azz * ar - arz * az) / delta
    zmax = (arr * az - arz * ar) / delta

    j = k // nr
    i = k - j * nr
    r_extrema = rmax + rvec[i]
    z_extrema = zmax + zvec[j]
    psi_extrema = (
        apsi[k] + 
        rmax * ar + zmax * az + 
        0.5 * arr * rmax**2 + 0.5 * azz * zmax**2 + 
        arz * rmax * zmax
    ) * np.sign(psi.ravel()[k])
    return r_extrema, z_extrema, psi_extrema


def compute_gradients_at_boundary(rvec, zvec, flat_psi, inout, ijedge, a1, a2, b1, b2, tol=1.0e-6):

    rgradr = []
    zgradr = []
    gradr = []
    rgradz = []
    zgradz = []
    gradz = []

    # Determine gradient of psi at boundary points
    nr = rvec.size
    nz = zvec.size
    hr, hrm1, hrm2, hz, hzm1, hzm2 = compute_grid_spacing(rvec, zvec)
    jedge = ijedge // nr
    iedge = ijedge - nr * jedge
    vedge = rvec.take(iedge) + 1.0j * zvec.take(jedge)
    for k, ij in enumerate(ijedge):

        # dpsi/dx at boundary points
        if inout[ij] & 0b10 or inout[ij] & 0b100:
            vl = vedge[k] - a1[ij] * hr
            vr = vedge[k] + a2[ij] * hr
            grad = (a1[ij] + a2[ij]) * flat_psi[ij] / (a1[ij] * a2[ij])
            # LEFT OR RIGHT OUT
            if inout[ij] & 0b10 and inout[ij] & 0b100:
                if (a1[ij] < tol or a2[ij] < tol): continue
                # LEFT AND RIGHT OUT
                rgradr.extend([vl.real, vr.real])
                zgradr.extend([vl.imag, vr.imag])
                gradr.extend([grad, -grad])
            elif inout[ij] & 0b10:
                if (a1[ij] < tol): continue
                # ONLY LEFT OUT
                rgradr.append(vl.real)
                zgradr.append(vl.imag)
                gradr.append((grad - a1[ij] * flat_psi[ij + 1] / (1.0 + a1[ij])) / hr)
            else:
                if (a2[ij] < tol): continue
                # ONLY RIGHT OUT
                rgradr.append(vr.real)
                zgradr.append(vr.imag)
                gradr.append((-grad + a2[ij] * flat_psi[ij - 1] / (1.0 + a2[ij])) / hr)

        # dpsi/dy at boundary points
        if inout[ij] & 0b1000 or inout[ij] & 0b10000:
            vb = vedge[k] - 1.0j * b1[ij] * hz
            va = vedge[k] + 1.0j * b2[ij] * hz
            grad = (b1[ij] + b2[ij]) * flat_psi[ij] / (b1[ij] * b2[ij])
            # ABOVE OR BELOW OUT
            if inout[ij] & 0b1000 and inout[ij] & 0b10000:
                if (b1[ij] < tol or b2[ij] < tol): continue
                # ABOVE AND BELOW OUT
                rgradz.extend([vb.real, va.real])
                zgradz.extend([vb.imag, va.imag])
                gradz.extend([grad, -grad])
            elif inout[ij] & 0b1000:
                if (b1[ij] < tol): continue
                # ONLY BELOW OUT
                rgradz.append(vb.real)
                zgradz.append(vb.imag)
                gradz.append((grad - b1[ij] * flat_psi[ij + nr] / (1.0 + b1[ij])) / hz)
            else:
                if (b2[ij] < tol): continue
                # ONLY ABOVE OUT
                rgradz.append(va.real)
                zgradz.append(va.imag)
                gradz.append((-grad + b2[ij] * flat_psi[ij - nr] / (1.0 + b2[ij])) / hz)

    rgradr = np.array(rgradr)
    zgradr = np.array(zgradr)
    gradr = np.array(gradr)
    rgradz = np.array(rgradz)
    zgradz = np.array(zgradz)
    gradz = np.array(gradz)

    return rgradr, zgradr, gradr, rgradz, zgradz, gradz


def generate_boundary_gradient_spline(r_points, z_points, grad_points, r_reference, z_reference, s=0):

    # Spline psi gradient along boundary angle, using magnetic axis as center
    v_reference = r_reference + 1.0j * z_reference
    v_points = r_points + 1.0j * z_points - v_reference
    ds = xr.Dataset(coords={'angle': np.angle(v_points)}, data_vars={'gradient': (['angle'], grad_points)})
    ds = ds.drop_duplicates('angle').sortby('angle')
    angle_ordered = ds['angle'].to_numpy()
    grad_ordered = ds['gradient'].to_numpy()
    roll_mask = (angle_ordered < 0.0)
    ang1 = angle_ordered[~roll_mask][-1]
    ang2 = angle_ordered[roll_mask][0] + 2.0 * np.pi
    grad1 = grad_ordered[~roll_mask][-1]
    grad2 = grad_ordered[roll_mask][0]

    mask = (angle_ordered > -np.pi) & (angle_ordered < np.pi)
    if np.any(mask):
        angle_ordered = angle_ordered[mask]
        grad_ordered = grad_ordered[mask]
    for angle_mult in np.linspace(0.9, 1.0, 21)[:-1]:
        if angle_ordered[0] > (angle_mult * -np.pi):
            yy = grad1 + (grad2 - grad1) * ((2.0 - angle_mult) * np.pi - ang1) / (ang2 - ang1)
            angle_ordered = np.concatenate(([angle_mult * -np.pi], angle_ordered))
            grad_ordered = np.concatenate(([yy], grad_ordered))
        if angle_ordered[-1] < (angle_mult * np.pi):
            yy = grad1 + (grad2 - grad1) * (angle_mult * np.pi - ang1) / (ang2 - ang1)
            angle_ordered = np.concatenate((angle_ordered, [angle_mult * np.pi]))
            grad_ordered = np.concatenate((grad_ordered, [yy]))
    yy = grad1 + (grad2 - grad1) * (np.pi - ang1) / (ang2 - ang1)
    angle_ordered = np.concatenate(([-np.pi], angle_ordered, [np.pi]))
    grad_ordered = np.concatenate(([yy], grad_ordered, [yy]))
    b = (-np.pi, np.pi)
    return {'tck': splrep(angle_ordered, grad_ordered, xb=b[0], xe=b[-1], k=3, s=s, per=True, quiet=1), 'bounds': b}


def generate_boundary_gradient_spline_with_windows(r_points, z_points, grad_points, r_reference, z_reference, s=0):

    v_reference = r_reference + 1.0j * z_reference
    v_points = r_points + 1.0j * z_points - v_reference
    ds = xr.Dataset(coords={'angle': np.angle(v_points)}, data_vars={'gradient': (['angle'], grad_points)})
    ds = ds.drop_duplicates('angle').sortby('angle')
    angle_ordered = ds['angle'].to_numpy()
    grad_ordered = ds['gradient'].to_numpy()
    angle_ordered_extended = np.concatenate([angle_ordered, [angle_ordered[0] + 2.0 * np.pi]])
    grad_ordered_extended = np.concatenate([grad_ordered, [grad_ordered[0]]])

    tck = splrep(angle_ordered_extended, grad_ordered_extended, k=3, per=True, quiet=1)
    n_output = 201
    angle_even = np.linspace(-np.pi, np.pi, n_output)
    grad_even = splev(angle_even, tck)
    if angle_ordered_extended[-1] > np.pi:
        grad_even[0] = splev(np.pi, tck)
    else:
        grad_even[-1] = splev(-np.pi, tck)
    window_length = n_output // 10
    if window_length % 2 == 0:
        window_length += 1
    window = windows.hann(window_length)
    window /= np.sum(window)
    padded_grad_even = np.pad(grad_even, (window_length // 2, window_length // 2), mode='wrap')
    smooth_grad_even = convolve(padded_grad_even, window, mode='valid')
    b = (-np.pi, np.pi)
    return {'tck': splrep(angle_even, smooth_grad_even, xb=b[0], xe=b[-1], k=3, s=s, per=True, quiet=1), 'bounds': b}


def old_compute_psi_extension(rvec, zvec, rbdry, zbdry, rmagx, zmagx, psi, ijout, gradr_tck, gradz_tck, grad_norm=1.0):

    # VECTORS TO AND BETWEEN BOUNDARY POINTS
    vmagx = rmagx + 1.0j * zmagx
    vbdry = rbdry + 1.0j * zbdry
    vb0 = vbdry[:-1]
    vb1 = vbdry[1:]
    dvb = vb1 - vb0
    cb = vb0 - vmagx
    dcb = 2.0 * np.pi / float(len(rbdry) - 1)

    # VECTORS TO EXTERIOR GRID POINTS
    nr = rvec.size
    jout = ijout // nr
    iout = ijout - nr * jout
    vvec = rvec.take(iout) + 1.0j * zvec.take(jout)

    psiout = np.zeros(vvec.shape, dtype=float)
    ivvec = np.arange(vvec.size)

    ## VECTORS BETWEEN EXTERIOR POINTS AND BOUNDARY POINTS
    for k in range(dvb.size):
        angmin = np.angle(cb[k])
        angmax = np.angle(vb1[k] - vmagx)
        angvec = np.angle(vvec - vmagx)
        angmask = np.isfinite(angvec)
        numer = cb[k] * np.conj(dvb[k])
        mask = np.logical_and(
            angvec >= angmin,
            angvec < angmax
        )
        if np.abs(angmax - angmin) > (2.0 * np.pi - 2.0 * dcb):
            mask = np.logical_or(
                angvec >= angmin,
                angvec < angmax
            )
        if not np.any(mask): continue
        ang = angvec.compress(mask)
        vvecc = vvec.compress(mask)
        ivvecc = ivvec.compress(mask)
        if ang.size > 0:
            dvvecc = vvecc - vmagx
            dv = dvvecc * (1.0 - numer.imag / (dvvecc * np.conj(dvb[k])).imag)
            vgradc = (splev(ang, gradr_tck) + 1.0j * splev(ang, gradz_tck)) * grad_norm
            psie = (vgradc * np.conj(dv)).real
            psiout.put(ivvecc, psie)
            vvec = vvec.compress(np.logical_not(mask))
            ivvec = ivvec.compress(np.logical_not(mask))
        if vvec.size == 0: break
    psi.ravel().put(ijout, psiout)

    return psi


def compute_psi_extension(rvec, zvec, rbdry, zbdry, rmagx, zmagx, psi, ijout, gradr_tck, gradz_tck, grad_norm=1.0):

    # Vectors to and between boundary points
    vmagx = rmagx + 1.0j * zmagx
    vbdry = rbdry + 1.0j * zbdry
    abdry = np.angle(vbdry - vmagx)
    dvb = vbdry[1:] - vbdry[:-1]
    nvb = -1.0j * dvb / np.abs(dvb)
    nbdry = np.concatenate([[nvb[0] + nvb[-1]], nvb[1:] + nvb[:-1], [nvb[-1] + nvb[0]]])
    for k in range(nbdry.size):
        if k == 0:
            nbdry[k] /= np.abs(nbdry[k])
        else:
            _, nfac = compute_intersection_from_ray_complex(vbdry[k - 1] + nbdry[k - 1], dvb[k - 1], vbdry[k], nbdry[k])
            nbdry[k] *= nfac

    # Vectors to exterior grid points
    nr = rvec.size
    jout = ijout // nr
    iout = ijout - nr * jout
    vvec = rvec.take(iout) + 1.0j * zvec.take(jout)
    gbdry = (splev(abdry, gradr_tck) + 1.0j * splev(abdry, gradz_tck)) * grad_norm

    psiout = np.zeros(vvec.shape, dtype=float)
    ivvec = np.arange(vvec.size)

    # Use vectors between exterior points and boundary surface to extend gradient
    for k in range(dvb.size):
        angmin = np.mod(np.angle(vvec - vbdry[k]) - np.angle(nbdry[k]), 2.0 * np.pi)
        angmin[angmin > np.pi] = angmin[angmin > np.pi] - 2.0 * np.pi
        angmax = np.mod(np.angle(vvec - vbdry[k + 1]) - np.angle(nbdry[k + 1]), 2.0 * np.pi)
        angmax[angmax > np.pi] = angmax[angmax > np.pi] - 2.0 * np.pi
        mask = np.logical_and(
            np.logical_and(angmin >= 0.0, angmin <= (0.5 * np.pi)),
            np.logical_and(angmax < 0.0, angmax >= (-0.5 * np.pi))
        )
        if not np.any(mask): continue
        vvecc = vvec.compress(mask)
        ivvecc = ivvec.compress(mask)
        if vvecc.size > 0:
            _, t = compute_intersection_from_ray_complex(vvecc, dvb[k], vbdry[k], nbdry[k])
            vlevel = vbdry[k] + t * nbdry[k]
            s, _ = compute_intersection_from_ray_complex(vlevel, dvb[k], vvecc, nvb[k])
            vgradc = gbdry[k] + s * (gbdry[k + 1] - gbdry[k])
            dv = vvecc - (vbdry[k] + s * dvb[k])
            psie = (vgradc * np.conj(dv)).real
            psiout.put(ivvecc, psie)
            vvec = vvec.compress(np.logical_not(mask))
            ivvec = ivvec.compress(np.logical_not(mask))
        if vvec.size == 0: break
    psi.ravel().put(ijout, psiout)

    return psi


def compute_flux_surface_quantities(psinorm, r_contour, z_contour, psi_tck=None, fpol_tck=None):
    fpol = splev(psinorm, fpol_tck) if fpol_tck is not None else 0.0
    gradr_contour = np.array([0.0])
    gradz_contour = np.array([0.0])
    if psi_tck is not None and len(r_contour) > 1 and len(z_contour) > 1:
        gradr_contour = np.array([bisplev(r, z, psi_tck, dx=1) for r, z in zip(r_contour, z_contour)])
        gradz_contour = np.array([bisplev(r, z, psi_tck, dy=1) for r, z in zip(r_contour, z_contour)])
    out = {
        'r': r_contour,
        'z': z_contour,
        'fpol': np.array([fpol]).flatten(),
        'bpol': np.sqrt(np.power(gradr_contour, 2.0) + np.power(gradz_contour, 2.0)) / r_contour,
        'btor': fpol / r_contour,
        'dpsidr': gradr_contour,
        'dpsidz': gradz_contour,
    }
    return out


def compute_flux_surface_quantities_boundary(psinorm, r_contour, z_contour, r_reference, z_reference, gradr_tck=None, gradz_tck=None, fpol_tck=None):
    fpol = splev(psinorm, fpol_tck) if fpol_tck is not None else 0.0
    gradr_contour = np.array([0.0])
    gradz_contour = np.array([0.0])
    if len(r_contour) > 1 and len(z_contour) > 1:
        a_contour = np.angle(r_contour + 1.0j * z_contour - r_reference - 1.0j * z_reference)
        if gradr_tck is not None:
            gradr_contour = splev(a_contour, gradr_tck)
        if gradz_tck is not None:
            gradz_contour = splev(a_contour, gradz_tck)
    out = {
        'r': r_contour,
        'z': z_contour,
        'fpol': np.array([fpol]).flatten(),
        'bpol': np.sqrt(np.power(gradr_contour, 2.0) + np.power(gradz_contour, 2.0)) / r_contour,
        'btor': fpol / r_contour,
        'dpsidr': gradr_contour,
        'dpsidz': gradz_contour,
    }
    return out


def compute_flux_surface_line_integral_factors(contour):
    dl = np.array([0.0])
    bpm = np.array([0.0])
    btm = np.array([contour['btor'][0]]) if contour.get('btor', np.array([])).size > 0 else np.array([0.0])
    rcm = np.array([contour['r'][0]]) if contour.get('r', np.array([])).size > 0 else np.array([0.0])
    zcm = np.array([contour['z'][0]]) if contour.get('z', np.array([])).size > 0 else np.array([0.0])
    if contour.get('r', np.array([])).size > 1:
        dl = np.sqrt(np.square(np.diff(contour['r'])) + np.square(np.diff(contour['z']))).flatten()
        bpm = 0.5 * (contour['bpol'][1:] + contour['bpol'][:-1]).flatten()
        btm = 0.5 * (contour['btor'][1:] + contour['btor'][:-1]).flatten()
        rcm = 0.5 * (contour['r'][1:] + contour['r'][:-1]).flatten()
        zcm = 0.5 * (contour['z'][1:] + contour['z'][:-1]).flatten()
    return dl, bpm, btm, rcm, zcm


def compute_flux_surface_cross_sectional_area(contour, r_reference=None, z_reference=None):
    dl, bpm, btm, rcm, zcm = compute_flux_surface_line_integral_factors(contour)
    if r_reference is None:
        r_reference = 0.5 * (np.nanmax(rcm) + np.nanmin(rcm))
    if z_reference is None:
        z_reference = 0.5 * (np.nanmax(zcm) + np.nanmin(zcm))
    area = 0.0
    if rcm.size > 2 and zcm.size > 2:
        # theta = np.angle(rcm + 1.0j * zcm - r_reference - 1.0j * z_reference).flatten()
        # theta -= theta[0]
        # mask = (theta < 0.0)
        # theta[mask] = theta[mask] + 2.0 * np.pi
        theta = np.unwrap(np.angle(rcm + 1.0j * zcm - r_reference - 1.0j * z_reference).flatten())
        area = float(trapezoid(0.5 * (rcm * np.gradient(zcm, theta) - zcm * np.gradient(rcm, theta)), x=theta))
    return area


def compute_flux_surface_average_factors(contour):
    dl, bpm, btm, rcm, zcm = compute_flux_surface_line_integral_factors(contour)
    vprime = 0.0
    r = float(contour['r'][0]) if contour.get('r', np.array([])).size > 0 else 0.0
    r2 = float(contour['r'][0]) ** 2 if contour.get('r', np.array([])).size > 0 else 0.0
    z = float(contour['z'][0]) if contour.get('z', np.array([])).size > 0 else 0.0
    ir = 1.0 / float(contour['r'][0]) if contour.get('r', np.array([])).size > 0 else 0.0
    ir2 = 1.0 / float(contour['r'][0]) ** 2 if contour.get('r', np.array([])).size > 0 else 0.0
    bp2 = float(contour['bpol'][0]) ** 2 if contour.get('bpol', np.array([])).size > 0 else 0.0
    bt2 = float(contour['btor'][0]) ** 2 if contour.get('bpol', np.array([])).size > 0 else 0.0
    if dl.size > 1:
        dl_over_bp = dl / bpm
        vprime = np.sum(dl_over_bp)
        r = np.sum(rcm * dl_over_bp)
        r2 = np.sum(np.square(rcm) * dl_over_bp)
        z = np.sum(zcm * dl_over_bp)
        ir = np.sum(dl_over_bp / rcm)
        ir2 = np.sum(dl_over_bp / np.square(rcm))
        bp2 = np.sum(dl_over_bp * np.square(bpm))
        bt2 = np.sum(dl_over_bp * np.square(btm))
    out = {
        'fs_vprime': vprime,
        'fs_r': r,
        'fs_r2': r2,
        'fs_z': z,
        'fs_ir': ir,
        'fs_ir2': ir2,
        'fs_bp2': bp2,
        'fs_bt2': bt2,
    }
    return out


def compute_volume_derivative_contour_integral(contour):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        fsa = compute_flux_surface_average_factors(contour)
        val = fsa['fs_vprime']
    return val


def compute_safety_factor_contour_integral(contour, current_inside=None):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        fsa = compute_flux_surface_average_factors(contour)
        val = 0.5 * contour['fpol'].item() * fsa['fs_ir2'] / np.pi
        # Current constraint needs more testing, advised NOT to use
        if isinstance(current_inside, float):
            val = (0.5 * contour['fpol'].item() * fsa['fs_ir2'] / np.pi) * fsa['fs_bp2'] / (4.0e-7 * np.pi * current_inside)
    return val


def compute_ffprime_from_jstar_pprime_and_contour(jstar, pp, contour):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        mu0 = 4.0e-7 * np.pi
        fsa = compute_flux_surface_average_factors(contour)
        val = (-mu0 / fsa['fs_ir2']) * (jstar * fsa['fs_ir'] + fsa['fs_vprime'] * pp)
    return val


def compute_f_from_safety_factor_and_contour(q, contour):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        fsa = compute_flux_surface_average_factors(contour)
        val = 2.0 * np.pi * q / fsa['fs_ir2']
    return val


def compute_jtor_from_f_contour_integral(contour, ffp):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        dl, bpm, _, rcm, _ = compute_flux_surface_line_integral_factors(contour)
        jtor_f = compute_jtor(contour['r'], np.zeros_like(contour['r']) + ffp, np.zeros_like(contour['r']))
        jtfm = 0.5 * (jtor_f[1:] + jtor_f[:-1]).flatten()
        val = np.sum(jtfm * dl / (bpm * rcm)) / np.sum(dl / bpm)
    return val


def compute_jtor_from_p_contour_integral(contour, pp):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        dl, bpm, _, rcm, _ = compute_flux_surface_line_integral_factors(contour)
        jtor_p = compute_jtor(contour['r'], np.zeros_like(contour['r']), np.zeros_like(contour['r']) + pp)
        jtpm = 0.5 * (jtor_p[1:] + jtor_p[:-1]).flatten()
        val = np.sum(jtpm * dl / (bpm * rcm)) / np.sum(dl / bpm)
    return val


def compute_jtor_contour_integral(contour, ffp, pp):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        dl, bpm, _, rcm, _ = compute_flux_surface_line_integral_factors(contour)
        jtor = compute_jtor(contour['r'], np.zeros_like(contour['r']) + ffp, np.zeros_like(contour['r']) + pp)
        jtm = 0.5 * (jtor[1:] + jtor[:-1]).flatten()
        val = np.sum(jtm * dl / (bpm * rcm)) / np.sum(dl / bpm)
    return val


def compute_jpar_contour_integral(contour, f, fp, pp):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        dl, bpm, _, _, _ = compute_flux_surface_line_integral_factors(contour)
        btot = np.sqrt(np.square(contour['bpol']) + np.square(contour['btor'])).flatten()
        jpar = compute_jpar(btot, np.zeros_like(contour['r']) + f, np.zeros_like(contour['r']) + fp, np.zeros_like(contour['r']) + pp)
        jpm = 0.5 * (jpar[1:] + jpar[:-1]).flatten()
        val = np.sum(jpm * dl / bpm) / np.sum(dl / bpm)
    return val


def compute_jstar_contour_integral(contour, ffp, pp):
    val = 0.0
    if contour.get('r', np.array([])).size > 1:
        dl, bpm, _, rcm, _ = compute_flux_surface_line_integral_factors(contour)
        jtor = compute_jtor(contour['r'], np.zeros_like(contour['r']) + ffp, np.zeros_like(contour['r']) + pp)
        jtm = 0.5 * (jtor[1:] + jtor[:-1]).flatten()
        val = np.sum(jtm * dl / (bpm * rcm)) / np.sum(dl / (bpm * rcm))
    return val


def build_core_smoothed_jstar_target(psinorm, jstar, trust_from=0.5):
    '''Replaces jstar(psinorm) for psinorm < trust_from with a smooth
    extrapolation from the trusted (psinorm >= trust_from) region: the
    physical slope d(jstar)/d(psinorm) is taken to ramp *linearly* from
    zero at the true axis (psinorm=0 -- the physically-required symmetry
    condition for a well-behaved flux-surface-averaged profile) up to the
    trusted region's own local slope at the join point (psinorm=trust_from,
    estimated from the two trusted grid points nearest the join, a local
    finite difference, not a global property of the trusted data).
    Integrating that linear slope ramp gives a profile that is quadratic in
    psinorm across the core and, by construction, continuous in both value
    and slope (C1) at the join -- not an independent least-squares fit to
    the trusted region that merely approximates the join point (which, tried
    first, produced a visible kink right at trust_from: confirmed on a real
    device G-EQDSK, an unconstrained quadratic-in-psinorm**2 fit was off by
    ~9% of the local jstar value at a trust_from=0.2 join, on the still-
    steeply-falling tail of the near-axis spike being smoothed away, and by
    ~3% at trust_from=0.5 further out where the profile is flatter -- small
    enough there to be easy to miss, but not actually zero).

    This exists because jstar read directly off a real, as-loaded
    equilibrium (compute_flux_surface_averaged_jstar_profile) can carry a
    sharp, unphysical spike in the first few grid points near the axis --
    a near-axis flux-surface-tracing artifact (tiny, poorly-resolved
    contours right around the magnetic axis; confirmed in practice against
    a real device G-EQDSK by megpy's own "Polyfit may be poorly
    conditioned" warning firing in exactly that region), not real physics.
    Feeding that raw spike into
    FixedBoundaryEquilibrium.derive_f_profile_from_jstar_target as a
    current-density target can produce a genuinely unphysical (non-
    monotonic/folded) resolved psi map in the core; smoothing it first
    with this function fixes that.

    `trust_from` is left to the caller to judge (no single default suits
    every device/equilibrium) -- if the near-axis instability visibly
    extends past the default trust_from=0.5, lower it.
    '''
    psinorm = np.asarray(psinorm, dtype=float)
    jstar = np.asarray(jstar, dtype=float)
    trusted = psinorm >= trust_from
    order = np.argsort(psinorm[trusted])
    idx = np.flatnonzero(trusted)[order]
    p0, p1 = psinorm[idx[0]], psinorm[idx[1]]
    y0, y1 = jstar[idx[0]], jstar[idx[1]]
    value_join = y0
    slope_join = (y1 - y0) / (p1 - p0)  # d(jstar)/d(psinorm) at the join, in physical psinorm space

    # d(jstar)/d(psinorm)(s) = slope_join * s / p0 for s in [0, p0] (zero at
    # the axis, slope_join at the join); integrating from p0 down to psinorm
    # gives jstar(psinorm) = value_join - integral_{psinorm}^{p0} of that,
    # i.e. a quadratic in psinorm with coefficient b below.
    b = slope_join / (2.0 * p0)
    jstar_target = jstar.copy()
    core = ~trusted
    jstar_target[core] = value_join + b * (psinorm[core] ** 2 - p0 ** 2)
    return jstar_target


def check_radial_flux_surface_monotonicity(
    rmagx, zmagx, simagx, sibdry, psi_tck, rbounds, zbounds,
    n_angles=144, n_samples=200, tol=1.0e-3, xpsi_margin=1.0,
):
    '''Shape-agnostic check for a "current hole" / non-nested flux surfaces:
    traces radial rays outward from the magnetic axis (rmagx, zmagx) at
    n_angles evenly spaced angles (a bivariate-spline evaluation of psi
    along each ray, not a grid-index walk) and confirms xpsi = (psi -
    simagx) / (sibdry - simagx) is non-decreasing along each ray, from the
    axis out to the LCFS.

    A naive row/column-based check (splitting a horizontal or vertical
    grid cut at the magnetic axis's own R/Z index and calling the two
    halves "inward"/"outward") is *not* shape-agnostic: for an up-down- or
    left-right-asymmetric plasma, a cut at a fixed Z (or R) far from the
    axis's own height does not pass through the local center of the flux
    surfaces there, so splitting it at the axis's own index is not a
    meaningful monotonicity test away from the axis's own row/column --
    confirmed to raise false alarms (a "49/61 bad rows" false positive on a
    genuinely good equilibrium) before this radial-ray approach replaced
    it. This function only walks each ray up to its first crossing of
    xpsi_margin: points further out are in the extrapolated exterior
    (beyond the LCFS, not a physical flux map -- see
    extend_psi_beyond_boundary) and are excluded rather than checked, and
    `tol` (default 1e-3) absorbs small spline-evaluation noise right at
    the LCFS itself (observed up to a few times 1e-4 in xpsi on real,
    good equilibria; an order of magnitude or more larger than that on a
    deliberately injected core defect, so this default cleanly separates
    the two).

    Returns (n_bad, bad_angles): the count of angles (out of n_angles)
    whose ray showed non-monotonic xpsi, and the array of their angles (in
    radians, counter-clockwise from the R-axis) for inspection. n_bad == 0
    means no current hole / properly nested flux surfaces were found.
    '''
    rmin, rmax = rbounds
    zmin, zmax = zbounds
    dpsi_dpsinorm = sibdry - simagx
    bad_angles = []
    for i in range(n_angles):
        theta = 2.0 * np.pi * i / n_angles
        dR, dZ = np.cos(theta), np.sin(theta)
        s_candidates = []
        if dR > 0.0:
            s_candidates.append((rmax - rmagx) / dR)
        elif dR < 0.0:
            s_candidates.append((rmin - rmagx) / dR)
        if dZ > 0.0:
            s_candidates.append((zmax - zmagx) / dZ)
        elif dZ < 0.0:
            s_candidates.append((zmin - zmagx) / dZ)
        s_max = min(s_candidates) if s_candidates else 1.0

        s = np.linspace(0.0, s_max, n_samples)[1:]  # skip s=0, the axis itself
        r = rmagx + s * dR
        z = zmagx + s * dZ
        psi_ray = np.array([bisplev(rr, zz, psi_tck) for rr, zz in zip(r, z)])
        xpsi_ray = (psi_ray - simagx) / dpsi_dpsinorm

        inside = xpsi_ray <= xpsi_margin
        n_inside = int(np.argmax(~inside)) if not np.all(inside) else len(inside)
        xpsi_ray = xpsi_ray[:n_inside]
        if len(xpsi_ray) < 3:
            continue
        if np.any(np.diff(xpsi_ray) < -tol):
            bad_angles.append(theta)
    return len(bad_angles), np.array(bad_angles)


def enforce_monotonic_diamagnetic_fpol(fpol, psinorm=None):
    '''Replaces fpol(psinorm) (psinorm assumed axis-first, i.e. index 0 is
    the magnetic axis, index -1 the boundary -- fibe's own internal
    convention throughout) with a single smooth cubic Hermite/Bezier
    profile whose *magnitude* |F| is monotonically non-increasing from the
    axis to the boundary -- the physically-expected diamagnetic behavior
    (the diamagnetic plasma current most strongly modifies the vacuum
    field at the magnetic axis, tapering monotonically outward to the
    vacuum value at the edge; a non-monotonic |F| -- a mid-radius hump that
    reverses before the edge -- is not physical).

    Built on `F^2 = fpol**2` (not `fpol` itself -- |F| monotonicity, not a
    same-signed F's own monotonicity, is the actual physical requirement,
    and squaring sidesteps needing F's own sign already resolved before
    this runs). If `F^2` is already non-increasing throughout, it is
    returned essentially unchanged (only the tie-breaking nudge below
    moves it) -- no correction is needed, so none is applied. Otherwise, a
    single `scipy.interpolate.CubicHermiteSpline` segment is fit across
    the *entire* psinorm domain -- so the result has no internal plateau/
    steep-transition boundary at all, unlike a direct `scipy.optimize.
    isotonic_regression` (PAVA) projection: PAVA's own output is
    piecewise-constant wherever it corrects a violation, and exactly
    interpolating that (as `FixedBoundaryEquilibrium.define_f_profile`'s
    `smooth=False` call does) leaves a genuine derivative discontinuity at
    each pool boundary -- visible as an F(psi)/jstar(psi) profile with a
    flat plateau followed by an abrupt steep drop, confirmed against real
    scenarios and already acknowledged as "PAVA's own pooling-boundary
    jitter" in `solve_psi_with_f_iteration`'s own convergence-gate
    docstring.

    The Hermite fit's own endpoint value/derivative are *not* read
    directly off the raw (violating) `F^2` -- the violation is exactly a
    local gradient-sign flip, most often right near the edge (a
    mid-radius hump that reverses before the boundary), so a naive
    `np.gradient` there risks reading that same flip back out and feeding
    it straight into the fit, propagating the violation rather than
    fixing it. Instead, PAVA is still run first, purely as a source of
    trustworthy endpoint conditions: its own output is monotonic by
    construction, so its value and local slope at the axis and boundary
    are always correctly signed regardless of what the raw data does
    nearby. `edge_weight` anchors PAVA's own boundary point close to its
    true (bcentr-tied) value, same role as before. Endpoint tangents are
    additionally clamped to `[0, 3*secant]` (same sign as the axis-to-edge
    secant slope) before fitting -- the standard Fritsch-Carlson sufficient
    condition for a monotone cubic Hermite segment -- so the final result
    is guaranteed monotonic, not just smooth. This does discard the true
    profile's own interior shape between the two endpoints in exchange for
    a single smooth curve -- accepted deliberately, only in the violating
    case, since the interior region containing the violation is, by
    definition, not something worth reproducing exactly.

    Returns a new fpol array with the same sign as the input's own edge
    value (`fpol[-1]`, matching `bcentr`'s sign) applied uniformly
    throughout -- not a per-point re-sign, since a physically sensible
    F(psi) does not change sign across the domain.

    Exists because `derive_f_profile_from_jstar_target`'s own F(psi), held
    to a `jstar_target` fixed to the original file's own current profile
    (including its untouched edge region -- `build_core_smoothed_jstar_
    target` only smooths the *core*), can come out genuinely non-monotonic
    when the new pressure profile's edge gradient differs substantially
    from the original's own -- confirmed on a real device G-EQDSK, where F
    peaked mid-radius then reversed toward the edge. See
    `FixedBoundaryEquilibrium.derive_monotonic_f_profile_from_jstar_
    target`, which applies this as a post-processing step.
    '''
    fpol = np.asarray(fpol, dtype=float)
    f2 = fpol ** 2
    if psinorm is None:
        psinorm = np.linspace(0.0, 1.0, len(fpol))
    psinorm = np.asarray(psinorm, dtype=float)

    # Already non-increasing: no violation to correct, so skip the Hermite
    # replacement entirely and leave the true profile's own interior shape
    # untouched (only the tie-breaking nudge below moves it) -- the
    # replacement below discards the true interior shape between the two
    # endpoints (that's the whole point, for the violating case), so it
    # should not run at all when nothing needs correcting.
    if np.all(np.diff(f2) <= 0.0):
        f2_smooth = f2
    else:
        # The violation is exactly a local gradient-sign flip, so a naive
        # np.gradient(f2, ...) evaluated at the endpoints risks reading
        # that same flip back out (most likely right at the edge, since
        # the docstring's own "mid-radius hump that reverses before the
        # edge" case sits closest to index -1) and feeding it straight
        # back into the fit -- propagating the violation instead of fixing
        # it. Endpoint tangents are instead read off the isotonic
        # -regression (PAVA) projection of f2, which is monotonic by
        # construction and therefore always has a correctly-signed local
        # slope at both ends, whatever the raw data does nearby.
        # `edge_weight` anchors the isotonic fit's boundary point close to
        # its true (bcentr-tied) value, and the small linear nudge (same
        # one applied to the final result below) guarantees a genuine
        # nonzero slope to sample even where PAVA pooled a flat region
        # straight through the axis or edge.
        edge_weight = 1.0e8
        weights = np.ones_like(f2)
        weights[-1] = edge_weight
        f2_pava = isotonic_regression(f2, weights=weights, increasing=False).x
        f2_pava = f2_pava - np.linspace(0.0, 1.0e-6 * (np.nanmax(f2) - np.nanmin(f2) + 1.0), len(f2))
        grad_pava = np.gradient(f2_pava, psinorm)
        d0, d1 = float(grad_pava[0]), float(grad_pava[-1])

        secant = (f2_pava[-1] - f2_pava[0]) / (psinorm[-1] - psinorm[0])
        lo, hi = (3.0 * secant, 0.0) if secant <= 0.0 else (0.0, 3.0 * secant)
        d0 = float(np.clip(d0, lo, hi))
        d1 = float(np.clip(d1, lo, hi))

        hermite = CubicHermiteSpline([psinorm[0], psinorm[-1]], [f2_pava[0], f2_pava[-1]], [d0, d1])
        f2_smooth = hermite(psinorm)

    f2_strict = f2_smooth - np.linspace(0.0, 1.0e-6 * (np.nanmax(f2) - np.nanmin(f2) + 1.0), len(f2))
    f2_strict = np.clip(f2_strict, a_min=0.0, a_max=None)  # F^2 can't be negative
    sign_ref = np.sign(fpol[-1])
    return sign_ref * np.sqrt(f2_strict)


def rescale_fpol_uniformly_for_target_current(fpol, i_f_current, target_current):
    '''Uniformly rescales fpol (already re-signed and, typically,
    monotonicity-projected by enforce_monotonic_diamagnetic_fpol) by a
    single multiplicative factor so its own F-driven grid current matches
    `target_current` exactly, *given*
    `i_f_current` (the F-driven current fpol, as currently shaped, already
    implies -- the caller must compute this via the real solve_psi/
    define_f_profile spline machinery, e.g. FixedBoundaryEquilibrium.
    compute_ffprime_and_pprime_grid + compute_jtor, not this function's
    concern; a naive np.gradient-based estimate of ffprime does *not*
    match what that spline-based construction actually produces downstream
    and silently fails to correct anything -- confirmed empirically).

    Scaling F uniformly (not just its deviation from some fixed reference
    point) is a deliberate choice: F^2 scales quadratically under a
    uniform scale of F, and Jtor's F-driven part depends linearly on
    d(F^2)/dpsi, so `i_f_current` scales by exactly the *square* of
    whatever factor scales F -- an exact, single-evaluation relationship,
    no iteration needed. This lets bcentr (tied to fpol[-1]) shift as a
    result, matching this module/class's own established convention
    elsewhere (`redefine_bcentre=True`): F's diamagnetic current genuinely
    contributes to the toroidal field, so letting the edge value move is
    physically expected, not something to guard against.

    Exists because a monotonic-|F| projection changes fpol's own shape
    (and hence its own implied current), silently undoing whatever
    magnitude correction `_estimate_flux_surface_averaged_fpol`'s own
    `scale` had already applied *before* the projection ran. Confirmed
    empirically (a real device G-EQDSK): without this rescale, `curscalef`
    (`solve_psi`'s own inner Picard-loop current-matching factor, see
    `_update_current_fixed_pressure`) settled at a *stable but wrong* ~1.33
    rather than the ~1.0 a genuinely self-consistent F(psi)/p'(psi)
    combination implies -- `cpasma` itself still came out exactly right
    (`curscalef` always forces that, regardless of its own value), but only
    via a persistent ~33% solver-level correction that the written-out
    F(psi)/pressure profiles, if independently re-integrated without that
    correction, would not themselves reproduce -- i.e. the equilibrium was
    not actually self-consistent, just numerically patched to look right.

    Raises ValueError if the needed scale factor is not positive -- the
    same self-inconsistency `_estimate_flux_surface_averaged_fpol`'s own
    `scale` check (`scale <= 0.0`) guards against.
    '''
    fpol = np.asarray(fpol, dtype=float)
    if abs(i_f_current) < 1.0e-30:
        return fpol
    ratio = float(target_current) / float(i_f_current)
    if ratio <= 0.0:
        raise ValueError(
            'Rescaling fpol to match the target F-driven current would require a '
            'non-positive scale factor -- self-inconsistent request.'
        )
    return fpol * np.sqrt(ratio)


def trace_contours_with_contourpy(rvec, zvec, dmap, levels, rcheck, zcheck):
    point_inside = Point([float(rcheck), float(zcheck)])
    rmesh, zmesh = np.meshgrid(rvec, zvec)
    cg = contourpy.contour_generator(rmesh, zmesh, dmap)
    contours = {}
    for level in levels:
        vertices = cg.create_contour(level)
        for i in range(len(vertices)):
            if vertices[i] is not None:
                polygon = Polygon(np.array(vertices[i]))
                if polygon.contains(point_inside):
                    contours[float(level)] = vertices[i].copy()
                    break
    return contours


def trace_contour_with_splines(dmap, level, npoints, rmagx, zmagx, psimagx, psibdry, psi_tck, boundary_splines, resolution=251):
    rc = []
    zc = []
    nvecl = resolution // 2
    nvecu = nvecl + 1
    vmagx = rmagx + 1.0j * zmagx
    psisign = np.sign(psibdry - psimagx)
    angles = np.linspace(0.0, 2.0 * np.pi, npoints)
    for ang in angles[:-1]:
        fang = ang
        for i, segfit in enumerate(boundary_splines):
            anglb = segfit['bounds'][0]
            angub = segfit['bounds'][-1]
            if anglb < 0.0 and fang > angub:
                fang -= 2.0 * np.pi
            if fang >= anglb and fang <= angub:
                break
        lbdry = splev(fang, boundary_splines[i]['tck'])
        vvec = np.linspace(0.0, lbdry, resolution) * np.exp(1.0j * ang) + vmagx
        vl = []
        psil = []
        for v in vvec[nvecl:0:-1]:
            psival = psisign * bisplev(v.real, v.imag, psi_tck)
            if len(psil) == 0:
                vl.append(v)
                psil.append(psival)
            elif psival < psil[-1] and psival >= psisign * psimagx:
                vl.append(v)
                psil.append(psival)
        vu = []
        psiu = []
        for v in vvec[nvecu:-1]:
            psival = psisign * bisplev(v.real, v.imag, psi_tck)
            if len(psiu) == 0:
                vu.append(v)
                psiu.append(psival)
            elif psival > psiu[-1] and psival <= psisign * psibdry:
                vu.append(v)
                psiu.append(psival)
        vscan = np.concatenate([[vvec[0]], vl[::-1], vu, [vvec[-1]]])
        psiscan = np.concatenate([[psisign * psimagx], psil[::-1], psiu, [psisign * psibdry]])
        lmin = np.abs(vscan[0] - vmagx)
        lmax = np.abs(vscan[-1] - vmagx)
        psifunc = interp1d(np.abs(vscan - vmagx), psisign * psiscan, bounds_error=False, fill_value='extrapolate')
        lc = brentq(lambda l, t: psifunc(l) - t, lmin, lmax, args=(level), xtol=1.0e-4)
        vroot = lc * np.exp(1.0j * ang) + vmagx
        rc.append(vroot.real)
        zc.append(vroot.imag)
    if len(rc) > 2:
        rc = np.array(rc + [rc[0]]).flatten()
        zc = np.array(zc + [zc[0]]).flatten()
    return rc, zc


def trace_contour_with_megpy(rvec, zvec, psi, level, rcheck, zcheck, boundary=False):
    contour_out = {}
    check = [float(rcheck), float(zcheck)]
    loops = contour_tracer(
        rvec,
        zvec,
        psi,
        level=level,
        kind='s',
        ref_point=np.array(check),
        x_point=boundary
    )
    if len(loops['contours']) > 0:
        loop = np.array(loops['contours'][0]).T
        contour_out['r'] = loop[:, 0].flatten()
        contour_out['z'] = loop[:, 1].flatten()
        if contour_out['r'][0] != contour_out['r'][-1]:
            contour_out['r'] = np.concatenate([contour_out['r'], np.array([contour_out['r'][0]])])
        if contour_out['z'][0] != contour_out['z'][-1]:
            contour_out['z'] = np.concatenate([contour_out['z'], np.array([contour_out['z'][0]])])
        signed_area = 0.5 * np.sum(
            contour_out['r'][:-1] * contour_out['z'][1:] - contour_out['r'][1:] * contour_out['z'][:-1]
        )
        if signed_area < 0.0:
            contour_out['r'] = contour_out['r'][::-1]
            contour_out['z'] = contour_out['z'][::-1]
    return contour_out


def compute_adjusted_contour_resolution(r_axis, z_axis, r_boundary, z_boundary, r_contour, z_contour, maxpoints=51, minpoints=21):
    v_axis = r_axis + 1.0j * z_axis
    v_boundary = r_boundary + 1.0j * z_boundary
    if not np.all(np.isfinite(v_boundary)):
        v_boundary = v_boundary[np.isfinite(v_boundary)]
    v_contour = r_contour + 1.0j * z_contour
    if not np.all(np.isfinite(v_contour)):
        v_contour = v_contour[np.isfinite(v_contour)]
    lmax_boundary = np.nanmax(np.abs(v_boundary - v_axis))
    lmax_contour = np.nanmax(np.abs(v_contour - v_axis))
    npoints = min(maxpoints, max(minpoints, int(np.rint(1.5 * float(maxpoints) * np.sqrt(lmax_contour / lmax_boundary)))))
    return npoints


def compute_mxh_coefficients_from_contours(fs, n_coeff=6):
    r0 = (np.nanmax(fs['r']) + np.nanmin(fs['r'])) / 2.0
    z0 = (np.nanmax(fs['z']) + np.nanmin(fs['z'])) / 2.0
    r_ordered, z_ordered, angle_ordered, _ = order_contour_points_by_angle(fs['r'], fs['z'], float(r0), float(z0), close_contour=True)
    v_ordered = r_ordered + 1.0j * z_ordered
    v0 = r0 + 1.0j * z0
    length_ordered = np.abs(v_ordered - v0)
    r = (np.nanmax(r_ordered) - np.nanmin(r_ordered)) / 2.0
    kappa = ((np.nanmax(z_ordered) - np.nanmin(z_ordered)) / 2.0) / r
    rc = np.clip((r_ordered - r0) / r, -1.0, 1.0)
    angle_ordered_r = np.where(z_ordered[:-1] < z0, 2.0 * np.pi - np.arccos(rc[:-1]), np.arccos(rc[:-1]))
    angle_ordered_r = np.concatenate([angle_ordered_r, [angle_ordered_r[0] + 2.0 * np.pi]])
    zc = np.clip((z_ordered - z0) / (kappa * r), -1.0, 1.0)
    angle_ordered_z = np.where(r_ordered[:-1] < r0, np.pi - np.arcsin(zc[:-1]), np.arcsin(zc[:-1]))
    angle_ordered_z = np.where(angle_ordered_z < 0.0, 2.0 * np.pi + angle_ordered_z, angle_ordered_z)
    angle_ordered_z = np.concatenate([angle_ordered_z, [angle_ordered_z[0] + 2.0 * np.pi]])
    sinc = np.zeros((n_coeff + 1, ))
    cosc = np.zeros((n_coeff + 1, ))
    for i in range(n_coeff + 1):
        sinc[i] = quad(np.interp, 0.0, 2.0 * np.pi, weight='sin', wvar=i, args=(angle_ordered_z, angle_ordered_r - angle_ordered_z))[0] / np.pi
        cosc[i] = quad(np.interp, 0.0, 2.0 * np.pi, weight='cos', wvar=i, args=(angle_ordered_z, angle_ordered_r - angle_ordered_z))[0] / np.pi
    cosc[0] /= 2.0
    return {'r0': np.array([r0]).flatten(), 'z0': np.array([z0]).flatten(), 'r': np.array([r]).flatten(), 'kappa': np.array([kappa]), 'cos_coeffs': cosc.flatten(), 'sin_coeffs': sinc.flatten()}


def compute_contours_from_mxh_coefficients(mxh, theta):
    r0 = mxh['r0']
    z0 = mxh['z0']
    r = mxh['r']
    kappa = mxh['kappa']
    cosc = mxh['cos_coeffs']
    sinc = mxh['sin_coeffs']
    if sinc.shape[-1] == (cosc.shape[-1] - 1):
        sinc = np.concatenate([np.atleast_2d(np.zeros(r0.shape)).T, sinc], axis=-1)
    theta_ex = np.repeat(np.atleast_2d(theta), r0.shape[0], axis=0)
    theta_R = copy.deepcopy(theta_ex)
    for n in range(cosc.shape[-1]):
        cos_R = cosc[:, n] * np.cos(float(n) * theta_ex)
        sin_R = sinc[:, n] * np.sin(float(n) * theta_ex)
        theta_R += cos_R + sin_R
    r_contour = r0 + r * np.cos(theta_R)
    z_contour = z0 + kappa * r * np.sin(theta_ex)
    return {'r': r_contour, 'z': z_contour}


def check_fully_contained_contours(r_inner, z_inner, r_outer, z_outer):
    polygon = Polygon(np.array([[ro, zo] for ro, zo in zip(r_outer, z_outer)]))
    for ri, zi in zip(r_inner, z_inner):
        point_inside = Point([float(ri), float(zi)])
        if not polygon.contains(point_inside):
            return False
    return True


def f_profile_shape_guesser(psinorm, linearity=0.0):
    f_norm = 0.75 * (np.exp(-5.0 * psinorm) - psinorm * np.exp(-5.0)) + 0.25 - 0.25 * psinorm
    f_base = 0.75 * (np.exp(-5.0 * psinorm) - psinorm * np.exp(-5.0)) + 0.25 - 0.25 * psinorm
    if linearity > 0.0 and linearity <= 1.0:
        # f_extreme = (1.0 - 0.2 * psinorm) - 0.8 * (psinorm ** 20)
        f_extreme = 0.9 * (np.exp(-2.0 * psinorm) - psinorm * np.exp(-2.0)) + 0.1 - 0.1 * (psinorm ** 15)
        f_norm = (1.0 - linearity) * f_base + linearity * f_extreme
    if linearity < 0.0 and linearity >= -1.0:
        # f_extreme = 0.9 * (np.exp(-6.0 * psinorm) - psinorm * np.exp(-6.0)) + 0.1 - 0.1 * psinorm
        f_extreme = 0.9 * (1.0 - psinorm ** 10) - 0.1 * psinorm + 0.1
        f_norm = (1.0 + linearity) * f_base - linearity * f_extreme
    return f_norm


def jtor_profile_shape_guesser(psinorm, linearity=0.0):
    j_norm = 0.5 * (1.0 - (psinorm ** 2)) * np.exp(-(psinorm ** 2) / 0.2) - 0.5 * (psinorm - 1.0)
    j_base = 0.5 * (1.0 - (psinorm ** 2)) * np.exp(-(psinorm ** 2) / 0.2) - 0.5 * (psinorm - 1.0)
    if linearity > 0.0 and linearity <= 1.0:
        j_extreme = (1.0 - 0.1 * psinorm) * (1.0 - psinorm ** 9)
        j_norm = (1.0 - linearity) * j_base + linearity * j_extreme
    if linearity < 0.0 and linearity >= -1.0:
        j_extreme = 0.9 * np.exp(-10.0 * psinorm) - 0.1 * psinorm + 0.1
        j_norm = (1.0 + linearity) * j_base - linearity * j_extreme
    return j_norm


def generalized_zero_integral_cubic_polynomial_shape(psinorm, skew=0.0):
    form = 1.0 - skew * psinorm - (6.0 - 2.0 * skew) * (psinorm ** 2)
    return (1.0 - psinorm) * form


def generalized_zero_integral_exponential_shape(psinorm, exponent=1.0):
    factor = 2.0 * (1.0 / exponent - (1.0 - np.exp(-exponent)) / exponent ** 2)
    form = (np.exp(-exponent * psinorm) - factor) / (1.0 - factor)
    return (1.0 - psinorm) * form


def generalized_zero_integral_shape(psinorm, degree=1):
    form = 1.0 - 3.0 * psinorm
    if degree >= 1:
        d = int(degree)
        form = 1.0 - 0.5 * float((d + 1) * (d + 2)) * psinorm ** d
    return (1.0 - psinorm) * form


def weighted_beta_shape(psinorm, weight=None, skew=0.0):
    w = weight if isinstance(weight, (float, int, np.ndarray)) else np.ones_like(psinorm)
    left = 1.0 - 4.0 * skew if skew < 0.0 and skew >= -1.0 else 5.0
    right = 1.0 + 4.0 * skew if skew > 0.0 and skew <= 1.0 else 5.0
    func = beta(left, right).pdf(psinorm) / w
    return func


def weighted_exponential_shape(psinorm, weight=None, exponent=1.0):
    w = weight if isinstance(weight, (float, int, np.ndarray)) else np.ones_like(psinorm)
    # norm = 1.0 - 1.0 / np.exp(exponent)
    norm = 1.0
    func = np.exp(-exponent * psinorm) / (norm * w)
    return func


def optimize_ffprime(psinorm, functions, ffp_axis_target, current_target, psinorm_grid, r_grid, mask, area_grid, ffp_max=1.0, exponent_target=3.0, required=None):
    axis_weight = 1.0
    current_weight = 1.0
    regpar = 0.1
    exp = exponent_target
    ffprime_required = required if isinstance(required, np.ndarray) else np.zeros_like(psinorm)
    def objective(x):
        ffprime = np.sum(np.atleast_2d(x).T * functions, axis=0)
        ffp_grid = np.interp(psinorm_grid, psinorm, ffprime)
        j_f_grid = np.where(mask == 0, 0.0, compute_jtor(r_grid.ravel(), ffp_grid.ravel(), 0.0))
        i_f_grid = np.sum(j_f_grid) * area_grid
        form_target = (ffprime[0] - ffprime[-1]) * (np.exp(-exp * psinorm) - np.exp(-exp)) / (1.0 - np.exp(-exp)) + ffprime[-1]
        # print(ffprime[0] / ffp_axis_target, i_f_grid / current_target, np.sum((ffprime - form_target) ** 2))
        ffprime += ffprime_required
        residual = (
            axis_weight * (1.0 - ffprime[0] / ffp_axis_target) ** 2 +
            current_weight * (1.0 - i_f_grid / current_target) ** 2 +
            regpar * np.sum((1.0 - ffprime / form_target) ** 2)
        )
        return residual
    guess = np.zeros((functions.shape[0], )) - 0.1
    bounds = (-5.0 * ffp_max * np.ones_like(guess), 5.0 * ffp_max * np.ones_like(guess))
    result = least_squares(objective, x0=guess, bounds=bounds, method='trf', ftol=1.0e-4, xtol=1.0e-4)
    return np.atleast_2d(result.x).T

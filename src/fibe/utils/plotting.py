import copy
from pathlib import Path
import numpy as np
from scipy.interpolate import splev, bisplev


def plot_equilibrium_contours(eq_obj, save=None, show=True, debug=False):
    if 'rleft' in eq_obj._data and 'rdim' in eq_obj._data and 'zmid' in eq_obj._data and 'zdim' in eq_obj._data:
        lvec = np.array([0.01, 0.04, 0.09, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.02, 1.05, 1.1, 1.2])
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)
        rmin = eq_obj._data['rleft']
        rmax = eq_obj._data['rleft'] + eq_obj._data['rdim']
        zmin = eq_obj._data['zmid'] - 0.5 * eq_obj._data['zdim']
        zmax = eq_obj._data['zmid'] + 0.5 * eq_obj._data['zdim']
        rvec = rmin + np.linspace(0.0, 1.0, eq_obj._data['nr']) * (rmax - rmin)
        zvec = zmin + np.linspace(0.0, 1.0, eq_obj._data['nz']) * (zmax - zmin)
        if 'psi' in eq_obj._data:
            rmesh, zmesh = np.meshgrid(rvec, zvec)
            dpsi = eq_obj._data['sibdry'] - eq_obj._data['simagx']
            levels = lvec * dpsi + eq_obj._data['simagx']
            if levels[0] > levels[-1]:
                levels = levels[::-1]
            ax.contour(rmesh, zmesh, eq_obj._data['psi'], levels=levels)
        if 'rbdry' in eq_obj._data and 'zbdry' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c='r', label='Boundary')
        if 'rlim' in eq_obj._data and 'zlim' in eq_obj._data:
            ax.plot(eq_obj._data['rlim'], eq_obj._data['zlim'], c='k', label='Limiter')
        if 'rmagx' in eq_obj._data and 'zmagx' in eq_obj._data:
            ax.scatter(eq_obj._data['rmagx'], eq_obj._data['zmagx'], marker='o', facecolors='none', edgecolors='r', label='O-points')
        if 'xpoints' in eq_obj._data and len(eq_obj._data['xpoints']) > 0:
            xparr = np.atleast_2d(eq_obj._data['xpoints'])
            ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points')
        if debug:
            if 'inout' in eq_obj._data:
                mask = eq_obj._data['inout'] == 0
                ax.scatter(rmesh.ravel()[~mask], zmesh.ravel()[~mask], c='g', marker='.', s=0.1)
                ax.scatter(rmesh.ravel()[mask], zmesh.ravel()[mask], c='k', marker='x')
            if 'gradr_bdry' in eq_obj._fit and 'gradz_bdry' in eq_obj._fit:
                abdry = np.angle(eq_obj._data['rbdry'] + 1.0j * eq_obj._data['zbdry'] - eq_obj._data['rmagx'] - 1.0j * eq_obj._data['zmagx'])
                mag_grad_psi = splev(abdry, eq_obj._fit['gradr_bdry']['tck']) ** 2 + splev(abdry, eq_obj._fit['gradz_bdry']['tck']) ** 2
                mag_grad_psi_norm = mag_grad_psi / (np.nanmax(mag_grad_psi) - np.nanmin(mag_grad_psi))
                ax.scatter(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c=mag_grad_psi_norm, cmap='cividis')
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.legend(loc='best')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_heatmap(eq_obj, save=None, show=True, debug=False):
    fig = None
    if 'rleft' in eq_obj._data and 'rdim' in eq_obj._data and 'zmid' in eq_obj._data and 'zdim' in eq_obj._data:
        lvec = np.array([0.01, 0.04, 0.09, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.02, 1.05, 1.1, 1.2])
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)
        rmin = eq_obj._data['rleft']
        rmax = eq_obj._data['rleft'] + eq_obj._data['rdim']
        zmin = eq_obj._data['zmid'] - 0.5 * eq_obj._data['zdim']
        zmax = eq_obj._data['zmid'] + 0.5 * eq_obj._data['zdim']
        rvec = rmin + np.linspace(0.0, 1.0, eq_obj._data['nr']) * (rmax - rmin)
        zvec = zmin + np.linspace(0.0, 1.0, eq_obj._data['nz']) * (zmax - zmin)
        dr = rvec[1] - rvec[0]
        dz = zvec[1] - zvec[0]
        if 'xpsi' in eq_obj._data:
            vmin = 0.9
            vmax = 1.1
            ax.imshow(eq_obj._data['xpsi'], origin='lower', extent=(rmin - dr, rmax + dr, zmin - dz, zmax + dz), vmin=vmin, vmax=vmax)
        if 'rbdry' in eq_obj._data and 'zbdry' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c='r', label='Boundary')
        if 'rlim' in eq_obj._data and 'zlim' in eq_obj._data:
            ax.plot(eq_obj._data['rlim'], eq_obj._data['zlim'], c='k', label='Limiter')
        if 'rmagx' in eq_obj._data and 'zmagx' in eq_obj._data:
            ax.scatter(eq_obj._data['rmagx'], eq_obj._data['zmagx'], marker='o', facecolors='none', edgecolors='r', label='O-points')
        if 'xpoints' in eq_obj._data and len(eq_obj._data['xpoints']) > 0:
            xparr = np.atleast_2d(eq_obj._data['xpoints'])
            ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points')
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.legend(loc='best')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_comparison(eq_obj, save=None, show=True):
    if 'psi' in eq_obj._data and 'psi_orig' in eq_obj._data:
        lvec = np.array([0.01, 0.04, 0.09, 0.15, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.02, 1.05])
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)
        nr_new = eq_obj._data['nr']
        nz_new = eq_obj._data['nz']
        rleft_new = eq_obj._data['rleft']
        rdim_new = eq_obj._data['rdim']
        zmid_new = eq_obj._data['zmid']
        zdim_new = eq_obj._data['zdim']
        simagx_new = eq_obj._data['simagx']
        sibdry_new = eq_obj._data['sibdry']
        nr_old = eq_obj._data['nr_orig'] if 'nr_orig' in eq_obj._data else copy.deepcopy(nr_new)
        nz_old = eq_obj._data['nz_orig'] if 'nz_orig' in eq_obj._data else copy.deepcopy(nz_new)
        rleft_old = eq_obj._data['rleft_orig'] if 'rleft_orig' in eq_obj._data else copy.deepcopy(rleft_new)
        rdim_old = eq_obj._data['rdim_orig'] if 'rdim_orig' in eq_obj._data else copy.deepcopy(rdim_new)
        zmid_old = eq_obj._data['zmid_orig'] if 'zmid_orig' in eq_obj._data else copy.deepcopy(zmid_new)
        zdim_old = eq_obj._data['zdim_orig'] if 'zdim_orig' in eq_obj._data else copy.deepcopy(zdim_new)
        simagx_old = eq_obj._data['simagx_orig'] if 'simagx_orig' in eq_obj._data else copy.deepcopy(simagx_new)
        sibdry_old = eq_obj._data['sibdry_orig'] if 'sibdry_orig' in eq_obj._data else copy.deepcopy(sibdry_new)
        rmin_old = rleft_old
        rmax_old = rleft_old + rdim_old
        zmin_old = zmid_old - 0.5 * zdim_old
        zmax_old = zmid_old + 0.5 * zdim_old
        rvec_old = rmin_old + np.linspace(0.0, 1.0, nr_old) * (rmax_old - rmin_old)
        zvec_old = zmin_old + np.linspace(0.0, 1.0, nz_old) * (zmax_old - zmin_old)
        rmesh_old, zmesh_old = np.meshgrid(rvec_old, zvec_old)
        dpsi_old = sibdry_old - simagx_old
        levels_old = lvec * dpsi_old + simagx_old
        if levels_old[0] > levels_old[-1]:
            levels_old = levels_old[::-1]
        ax.contour(rmesh_old, zmesh_old, eq_obj._data['psi_orig'], levels=levels_old, colors='r', alpha=0.6)
        if 'rbdry_orig' in eq_obj._data and 'zbdry_orig' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry_orig'], eq_obj._data['zbdry_orig'], c='r', label='Boundary (old)')
        elif 'rbdry' in eq_obj._data and 'zbdry' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c='r', label='Boundary (old)')
        if 'rmagx_orig' in eq_obj._data and 'zmagx_orig' in eq_obj._data:
            ax.scatter(eq_obj._data['rmagx_orig'], eq_obj._data['zmagx_orig'], marker='o', facecolors='none', edgecolors='r', label='O-points (old)')
        elif 'rmagx' in eq_obj._data and 'zmagx' in eq_obj._data:
            ax.scatter(eq_obj._data['rmagx'], eq_obj._data['zmagx'], marker='o', facecolors='none', edgecolors='r', label='O-points (old)')
        if 'xpoints_orig' in eq_obj._data and len(eq_obj._data['xpoints_orig']) > 0:
            xparr = np.atleast_2d(eq_obj._data['xpoints_orig'])
            ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points (old)')
        #elif 'xpoints' in eq_obj._data and len(eq_obj._data['xpoints']) > 0:
        #    xparr = np.atleast_2d(eq_obj._data['xpoints'])
        #    ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='r', label='X-points (old)')
        rmin_new = rleft_new
        rmax_new = rleft_new + rdim_new
        zmin_new = zmid_new - 0.5 * zdim_new
        zmax_new = zmid_new + 0.5 * zdim_new
        rvec_new = rmin_new + np.linspace(0.0, 1.0, nr_new) * (rmax_new - rmin_new)
        zvec_new = zmin_new + np.linspace(0.0, 1.0, nz_new) * (zmax_new - zmin_new)
        rmesh_new, zmesh_new = np.meshgrid(rvec_new, zvec_new)
        dpsi_new = sibdry_new - simagx_new
        levels_new = lvec * dpsi_new + simagx_new
        if levels_new[0] > levels_new[-1]:
            levels_new = levels_new[::-1]
        ax.contour(rmesh_new, zmesh_new, eq_obj._data['psi'], levels=levels_new, colors='b', alpha=0.6)
        if 'rbdry' in eq_obj._data and 'zbdry' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c='b', label='Boundary (new)')
        if 'rmagx' in eq_obj._data and 'zmagx' in eq_obj._data:
            ax.scatter(eq_obj._data['rmagx'], eq_obj._data['zmagx'], marker='o', facecolors='none', edgecolors='b', label='O-points (new)')
        if 'xpoints' in eq_obj._data and len(eq_obj._data['xpoints']) > 0:
            xparr = np.atleast_2d(eq_obj._data['xpoints'])
            ax.scatter(xparr[:, 0], xparr[:, 1], marker='x', facecolors='b', label='X-points (new)')
        if rmin_new > rmin_old:
            ax.plot([rmin_new, rmin_new], [zmin_new, zmax_new], ls='-', c='b')
        if rmax_new < rmax_old:
            ax.plot([rmax_new, rmax_new], [zmin_new, zmax_new], ls='-', c='b')
        if zmin_new > zmin_old:
            ax.plot([rmin_new, rmax_new], [zmin_new, zmin_new], ls='-', c='b')
        if zmax_new < zmax_old:
            ax.plot([rmin_new, rmax_new], [zmax_new, zmax_new], ls='-', c='b')
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
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_grid(eq_obj, save=None, show=True):
    if 'inout' in eq_obj._data:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)
        rmin = np.nanmin(eq_obj._data['rvec'])
        rmax = np.nanmax(eq_obj._data['rvec'])
        zmin = np.nanmin(eq_obj._data['zvec'])
        zmax = np.nanmax(eq_obj._data['zvec'])
        rmesh = copy.deepcopy(eq_obj._data['rpsi']).ravel()
        zmesh = copy.deepcopy(eq_obj._data['zpsi']).ravel()
        mask = eq_obj._data['inout'] == 0
        ax.scatter(rmesh[~mask], zmesh[~mask], c='g', marker='.', s=0.1)
        ax.scatter(rmesh[mask], zmesh[mask], c='k', marker='x')
        if 'rbdry' in eq_obj._data and 'zbdry' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c='r', label='Boundary')
        if 'rlim' in eq_obj._data and 'zlim' in eq_obj._data:
            ax.plot(eq_obj._data['rlim'], eq_obj._data['zlim'], c='k', label='Limiter')
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_flux_surfaces(eq_obj, save=None, show=True):
    if eq_obj._fs is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)
        rmin = np.nanmin(eq_obj._data['rvec'])
        rmax = np.nanmax(eq_obj._data['rvec'])
        zmin = np.nanmin(eq_obj._data['zvec'])
        zmax = np.nanmax(eq_obj._data['zvec'])
        for level, contour in eq_obj._fs.items():
            ax.plot(contour['r'], contour['z'], c='b', label=f'{level:.3f}', alpha=0.4)
        if 'rbdry' in eq_obj._data and 'zbdry' in eq_obj._data:
            ax.plot(eq_obj._data['rbdry'], eq_obj._data['zbdry'], c='r', label='Boundary')
        if 'rlim' in eq_obj._data and 'zlim' in eq_obj._data:
            ax.plot(eq_obj._data['rlim'], eq_obj._data['zlim'], c='k', label='Limiter')
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_profiles(eq_obj, save=None, show=True):
    if 'fpol' in eq_obj._data and 'pres' in eq_obj._data:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        psinorm = np.linspace(0.0, 1.0, eq_obj._data['nr'])
        dpsinorm_dpsi = 1.0 / (eq_obj._data['sibdry'] - eq_obj._data['simagx'])
        f_factor = 1.0e-1 * np.sign(eq_obj._data['bcentr'])
        p_factor = 1.0e-5
        q_factor = np.sign(eq_obj._data['bcentr'] * eq_obj._data['cpasma'])
        phi_factor = np.sign(eq_obj._data['bcentr'])
        j_factor = 1.0e-6 * np.sign(eq_obj._data['cpasma'])
        d_factor = np.sign(eq_obj._data['cpasma'])
        ax1.plot(psinorm, f_factor * eq_obj._data['fpol'], c='b', label='F [10**-1 Tm]')
        if 'ffprime' in eq_obj._data:
            ax2.plot(psinorm, f_factor * d_factor * eq_obj._data['ffprime'] * dpsinorm_dpsi / eq_obj._data['fpol'], c='b', label='Fp')
        if 'fpol_fs' in eq_obj._fit:
            ax1.plot(psinorm, f_factor * splev(psinorm, eq_obj._fit['fpol_fs']['tck']), c='b', ls='--', label='F Fit [10**-1 Tm]')
            ax2.plot(psinorm, f_factor * d_factor * splev(psinorm, eq_obj._fit['fpol_fs']['tck'], der=1) * dpsinorm_dpsi, c='b', ls='--', label='Fp Fit')
        ax1.plot(psinorm, p_factor * eq_obj._data['pres'], c='r', label='p [10**-5 Pa]')
        if 'pprime' in eq_obj._data:
            ax2.plot(psinorm, p_factor * d_factor * eq_obj._data['pprime'] * dpsinorm_dpsi, c='r', label='pp')
        if 'pres_fs' in eq_obj._fit:
            ax1.plot(psinorm, p_factor * splev(psinorm, eq_obj._fit['pres_fs']['tck']), c='r', ls='--', label='p Fit [10**-5 Pa]')
            ax2.plot(psinorm, p_factor * d_factor * splev(psinorm, eq_obj._fit['pres_fs']['tck'], der=1) * dpsinorm_dpsi, c='r', ls='--', label='pp Fit')
        if 'qpsi' in eq_obj._data:
            ax1.plot(psinorm, q_factor * eq_obj._data['qpsi'], c='g', label='q [-]')
            if 'qpsi_fs' in eq_obj._fit:
                ax1.plot(psinorm, q_factor * splev(psinorm, eq_obj._fit['qpsi_fs']['tck']), c='g', ls='--', label='q Fit [-]')
                ax2.plot(psinorm, q_factor * d_factor * splev(psinorm, eq_obj._fit['qpsi_fs']['tck'], der=1) * dpsinorm_dpsi, c='g', ls='--', label='qp Fit')
        if 'phi' in eq_obj._data:
            ax1.plot(psinorm, phi_factor * eq_obj._data['phi'], c='m', label='phi [Wb/rad]')
        if 'jpsi' in eq_obj._data:
            ax1.plot(psinorm, j_factor * eq_obj._data['jpsi'], c='#800080', label='jtor [MA m**-2]')
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel('psi_norm [-]')
        ax1.set_ylabel('Profiles')
        ax1.legend(loc='best')
        ax2.set_xlim(0.0, 1.0)
        ax2.set_xlabel('psi_norm [-]')
        ax2.set_ylabel('Gradients')
        ax2.legend(loc='best')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_shaping_coefficients(eq_obj, save=None, show=True):
    if eq_obj._fs is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        psinorm = np.linspace(0.0, 1.0, eq_obj._data['nr'])
        if 'mxh_r0' in eq_obj._data:
            ax1.plot(psinorm, eq_obj._data['mxh_r0'], label='R0')
        if 'mxh_z0' in eq_obj._data:
            ax1.plot(psinorm, eq_obj._data['mxh_z0'], label='Z0')
        if 'mxh_r' in eq_obj._data:
            ax1.plot(psinorm, eq_obj._data['mxh_r'], label='r')
        if 'mxh_kappa' in eq_obj._data:
            ax1.plot(psinorm, eq_obj._data['mxh_kappa'], label='kappa')
        if 'mxh_cos' in eq_obj._data:
            for i in range(eq_obj._data['mxh_cos'].shape[1]):
                if i > 0:
                    ax2.plot(psinorm, eq_obj._data['mxh_cos'][:, i], label=f'c{i:d}')
                else:
                    ax1.plot(psinorm, eq_obj._data['mxh_cos'][:, i], label='c0')
        if 'mxh_sin' in eq_obj._data:
            for i in range(eq_obj._data['mxh_sin'].shape[1]):
                if i > 0:
                    ax3.plot(psinorm, eq_obj._data['mxh_sin'][:, i], label=f's{i:d}')
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel('psi_norm [-]')
        ax1.set_ylabel('Coefficients')
        ax1.legend(loc='best')
        ax2.set_xlim(0.0, 1.0)
        ax2.set_xlabel('psi_norm [-]')
        ax2.set_ylabel('Coefficients')
        ax2.legend(loc='best')
        ax3.set_xlim(0.0, 1.0)
        ax3.set_xlabel('psi_norm [-]')
        ax3.set_ylabel('Coefficients')
        ax3.legend(loc='best')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)


def plot_equilibrium_boundary_gradients(eq_obj, save=None, show=True):
    if 'gradr_bdry' in eq_obj._fit and 'gradz_bdry' in eq_obj._fit:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        abdry = np.angle(eq_obj._data['rbdry'] + 1.0j * eq_obj._data['zbdry'] - eq_obj._data['rmagx'] - 1.0j * eq_obj._data['zmagx'])
        gradr_fit = splev(abdry, eq_obj._fit['gradr_bdry']['tck'])
        gradz_fit = splev(abdry, eq_obj._fit['gradz_bdry']['tck'])
        ax.scatter(eq_obj._data['agradr'], eq_obj._data['gradr'], label='dpsi/dr')
        ax.scatter(eq_obj._data['agradz'], eq_obj._data['gradz'], label='dpsi/dz')
        ax.plot(abdry, gradr_fit, label='dpsi/dr Fit')
        ax.plot(abdry, gradz_fit, label='dpsi/dz Fit')
        #mag_grad_psi = splev(abdry, eq_obj._fit['gradr_bdry']['tck']) ** 2 + splev(abdry, eq_obj._fit['gradz_bdry']['tck']) ** 2
        #ax.plot(abdry, mag_grad_psi, label='|grad(psi)|^2')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xlabel('Boundary Angle [rad]')
        #ax.set_ylabel('|grad(psi)|^2 [Wb^2/m^2]')
        ax.set_ylabel('Gradient of Psi [Wb/m]')
        ax.legend(loc='best')
        fig.tight_layout()
        if isinstance(save, (str, Path)):
            fig.savefig(save, dpi=100)
        if show:
            plt.show()
        plt.close(fig)
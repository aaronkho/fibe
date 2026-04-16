import copy
import numpy as np
from scipy.integrate import cumulative_simpson, trapezoid
from fibe import FixedBoundaryEquilibrium
from fibe.core.math import (
    compute_jtor_from_f_contour_integral,
    compute_jtor_from_p_contour_integral,
    weighted_exponential_shape,
    weighted_beta_shape,
)

optuna = None
try:
    import optuna
except ImportError:
    print("Optuna is not installed. Please install it to run the Bayesian optimization.")


def solve_with_scaled_f_contribution(eq):
    mu0 = 4.0e-7 * np.pi
    dpsi_dpsinorm = (eq._data['sibdry'] - eq._data['simagx'])
    psinorm = np.zeros_like(eq._data['qpsi'])
    vprime_fsa = np.zeros_like(eq._data['qpsi'])
    ir2_fsa = np.zeros_like(eq._data['qpsi'])
    j_f_fsa = np.zeros_like(eq._data['qpsi'])
    j_p_fsa = np.zeros_like(eq._data['qpsi'])
    for k, (level, contour) in enumerate(eq._fs.items()):
        psinorm[k] = (level - eq._data['simagx']) / (eq._data['sibdry'] - eq._data['simagx']) #renormalization
        vprime_fsa[k] = contour['fsa_vprime']
        ir2_fsa[k] = contour['fsa_ir2']
        j_f_fsa[k] = compute_jtor_from_f_contour_integral(contour, eq._data['ffprime'][k] / dpsi_dpsinorm)
        j_p_fsa[k] = compute_jtor_from_p_contour_integral(contour, eq._data['pprime'][k] / dpsi_dpsinorm)
    vprime_fsa[0] = 2.0 * vprime_fsa[1] - vprime_fsa[2]
    ir2_fsa[0] = 1.0 / eq._data['rmagx'] ** 2
    j_f_fsa[0] = 2.0 * j_f_fsa[1] - j_f_fsa[2]
    i_f_fsa = trapezoid(j_f_fsa * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    i_p_fsa = trapezoid(j_p_fsa * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    i_f_target_fsa = (eq._data['cpasma'] - i_p_fsa)
    print('Current contributions:', i_f_fsa, i_p_fsa, i_f_target_fsa / i_f_fsa)
    j_f_add = j_f_fsa * np.exp((psinorm - 0.5) ** 2 / -0.05) * (1.0 - psinorm)
    i_f_add = trapezoid(j_f_add * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    j_f_scaled = j_f_fsa + (j_f_add / i_f_add) * (i_f_target_fsa / i_f_fsa - 1.0)
    f2prime = -2.0 * mu0 * j_f_scaled / ir2_fsa
    fpol_edge = 2 * np.pi * eq._data['qpsi'][-1] / ir2_fsa[-1]
    if 'bvacuum' in eq._data and 'rvacuum' in eq._data:
        fpol_edge = eq._data['bvacuum'] * eq._data['rvacuum']
    f2_flipped = -1.0 * cumulative_simpson(f2prime[::-1], x=np.abs(psinorm[::-1] - psinorm[-1]), initial=0.0) * dpsi_dpsinorm
    f2 = f2_flipped[::-1] + fpol_edge ** 2
    fpol = np.sqrt(f2)
    print(fpol)
    eq.define_f_profile(fpol, smooth=True, symmetrical=False, redefine_bcentre=True)
    eq.solve_psi(
        nxiter=100,
        erreq=1.0e-4,
        relax=1.0,
        relaxj=1.0,
        pnaxis=None,
        approxq=False,
        symmetrical=False,
    )
    eq.compute_flux_surface_averaged_jstar_profile()
    return eq


def solve_with_parameterized_j_profile(eq, q_axis, q_edge, lpar, rpar, lscale, rscale):
    mu0 = 4.0e-7 * np.pi
    dpsi_dpsinorm = (eq._data['sibdry'] - eq._data['simagx'])
    psinorm = np.zeros_like(eq._data['qpsi'])
    vprime_fsa = np.zeros_like(eq._data['qpsi'])
    ir2_fsa = np.zeros_like(eq._data['qpsi'])
    j_f_fsa = np.zeros_like(eq._data['qpsi'])
    j_p_fsa = np.zeros_like(eq._data['qpsi'])
    for k, (level, contour) in enumerate(eq._fs.items()):
        psinorm[k] = (level - eq._data['simagx']) / (eq._data['sibdry'] - eq._data['simagx']) #renormalization
        vprime_fsa[k] = contour['fsa_vprime']
        ir2_fsa[k] = contour['fsa_ir2']
        j_f_fsa[k] = compute_jtor_from_f_contour_integral(contour, eq._data['ffprime'][k] / dpsi_dpsinorm)
        j_p_fsa[k] = compute_jtor_from_p_contour_integral(contour, eq._data['pprime'][k] / dpsi_dpsinorm)
    vprime_fsa[0] = 2.0 * vprime_fsa[1] - vprime_fsa[2]
    j_f_fsa[0] = 2.0 * j_f_fsa[1] - j_f_fsa[2]
    i_f_fsa = trapezoid(j_f_fsa * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    i_p_fsa = trapezoid(j_p_fsa * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    i_f_target_fsa = (eq._data['cpasma'] - i_p_fsa)
    fpol_edge = 2 * np.pi * eq._data['qpsi'][-1] / ir2_fsa[-1]
    if 'bvacuum' in eq._data and 'rvacuum' in eq._data:
        fpol_edge = eq._data['bvacuum'] * eq._data['rvacuum']
    j_axis = (j_f_fsa[0] + j_p_fsa[0]) * (eq._data['qpsi'][0] / q_axis)
    j_edge = 0.0
    # print('F_edge constraint: F =', fpol_edge, eq._data['fpol'][-1])
    j_mod_left = weighted_beta_shape(psinorm, weight=None, skew=lpar)
    j_mod_right = weighted_beta_shape(psinorm, weight=None, skew=rpar)
    j_f_new = j_f_fsa * (1.0 + j_mod_left - j_mod_right)
    j_mod_left_bound = (j_axis - j_f_new[0]) * weighted_exponential_shape(psinorm, weight=None, exponent=10.0 + lscale)
    j_mod_right_bound = (j_edge - j_f_new[-1]) * weighted_exponential_shape(psinorm[::-1], weight=None, exponent=10.0 + rscale)
    j_f_bound = j_mod_left_bound + j_mod_right_bound[::-1]
    i_f_bound = trapezoid(j_f_bound * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    i_f_new = trapezoid(j_f_new * vprime_fsa, x=psinorm) * dpsi_dpsinorm
    j_f_scaled = (j_f_new + j_f_bound) * (1.0 - i_f_target_fsa / (i_f_new + i_f_bound))
    f2prime = -2.0 * mu0 * j_f_scaled / ir2_fsa
    f2_flipped = -1.0 * cumulative_simpson(f2prime[::-1], x=np.abs(psinorm[::-1] - psinorm[-1]), initial=0.0) * dpsi_dpsinorm
    f2 = f2_flipped[::-1] + fpol_edge ** 2
    fpol = np.sqrt(f2)
    eq.define_f_profile(fpol, smooth=False, symmetrical=False, redefine_bcentre=True)
    eq.solve_psi(
        nxiter=100,
        erreq=1.0e-4,
        relax=1.0,
        relaxj=1.0,
        pnaxis=None,
        approxq=False,
        symmetrical=True,
    )
    eq.compute_flux_surface_averaged_jstar_profile()
    return eq


def q_constraints_with_bayesian_optimization(
    eq,
    q_axis=1.0,
    q_edge=15.0,
    seed=42,
):
    if eq.scratch:
        eq.solve_psi(
            nxiter=100,
            erreq=1.0e-4,
            relax=1.0,
            relaxj=1.0,
            pnaxis=None,
            approxq=False,
            symmetrical=True,
        )
        eq.scratch = False

    # ffp_grid, pp_grid = eq.compute_ffprime_and_pprime_grid(eq._data['xpsi'], internal_cutoff=eq._options['pnaxis'])
    # cur_new = np.where(eq._data['inout'] == 0, 0.0, compute_jtor(eq._data['rpsi'].ravel(), ffp_grid.ravel(), pp_grid.ravel()))
    # j_f_grid = np.where(eq._data['inout'] == 0, 0.0, compute_jtor(eq._data['rpsi'].ravel(), ffp_grid.ravel(), 0.0))
    # j_p_grid = np.where(eq._data['inout'] == 0, 0.0, compute_jtor(eq._data['rpsi'].ravel(), 0.0, pp_grid.ravel()))

    # Define the function to optimize
    def loss_function(lpar, rpar, lscale, rscale):
        eq_test = copy.deepcopy(eq)
        eq_test = solve_with_parameterized_j_profile(eq_test, q_axis, q_edge, lpar, rpar, lscale, rscale)
        return (eq_test._data['qpsi'][0] - q_axis) ** 2 + (eq_test._data['qpsi'][-1] - q_edge) ** 2 + (eq_test._data['curscale'] - 1.0) ** 2

    # Define the objective function for Optuna
    def objective(trial):
        # Define parameters to optimize
        left_beta_parameter = trial.suggest_float('lpar', -1.0, 0.0)
        right_beta_parameter = trial.suggest_float('rpar', 0.0, 1.0)
        left_scale_parameter = trial.suggest_float('lscale', 0.0, 10.0)
        right_scale_parameter = trial.suggest_float('rscale', 0.0, 10.0)
        return loss_function(left_beta_parameter, right_beta_parameter, left_scale_parameter, right_scale_parameter)

    # Create study object and optimize
    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=2)

    # Print results
    print(f'Best parameters: {study.best_params}')
    print(f'Best value: {study.best_value}')
    print(f'Number of trials: {len(study.trials)}')

    eq = solve_with_parameterized_j_profile(eq, q_axis, q_edge, **study.best_params)
    return eq


def main():

    ip = 10.0e6
    btvac = 12.0
    r = 4.62
    a = 1.18
    kappa = 1.7
    delta = 0.6
    zeta = -0.1
    pressure = [5.0e5, 4.0e5, 2.5e5, 1.5e5, 7.0e4, 2.0e4]
    psinorm = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rwall = None
    zwall = None
    q_axis = 1.0
    q_edge = 15.0

    if optuna is not None:
        eq = FixedBoundaryEquilibrium()
        eq.define_grid_and_boundary_with_mxh(
            nr=129,
            nz=129,
            rgeo=r,
            zgeo=1.0,
            rminor=a,
            kappa=kappa,
            cos_coeffs=[0.0, 0.0, 0.0],
            sin_coeffs=[0.0, np.arcsin(delta), -zeta],
            rwall=rwall,
            zwall=zwall,
        )
        eq.define_plasma_current(ip)
        eq.define_vacuum_toroidal_field(btvac)
        # eq.define_axis_safety_factor(q_axis)
        # eq.define_edge_safety_factor(q_edge)
        eq.define_pressure_profile(pressure, psinorm=psinorm, smooth=True)
        eq.initialize_psi()
        eq.solve_psi(
            nxiter=100,
            erreq=1.0e-4,
            relax=1.0,
            relaxj=1.0,
            pnaxis=None,
            approxq=False,
            symmetrical=False,
        )
        eq.scratch = False
        eq.compute_flux_surface_averaged_jstar_profile()
        eq.plot_contour(save=f'contour_initial.png', show=False)
        eq.plot_profiles(save=f'profiles_initial.png', show=False)
        print('q initial:', eq._data['qpsi'][0], eq._data['qpsi'][-1])
        eq = solve_with_scaled_f_contribution(eq)
        eq.plot_contour(save=f'contour_scaled.png', show=False)
        eq.plot_profiles(save=f'profiles_scaled.png', show=False)
        print('q scaled:', eq._data['qpsi'][0], eq._data['qpsi'][-1])
        eq = q_constraints_with_bayesian_optimization(eq, q_axis=q_axis, q_edge=q_edge)
        eq.set_bounding_box_as_wall()
        eq.plot_contour(save=f'contour_optimized.png', show=False)
        eq.plot_profiles(save=f'profiles_optimized.png', show=False)
        eq.plot_boundary_gradients(save=f'boundary_optimized.png', show=False)
        print('q optimized:', eq._data['qpsi'][0], eq._data['qpsi'][-1])


if __name__ == '__main__':
    main()
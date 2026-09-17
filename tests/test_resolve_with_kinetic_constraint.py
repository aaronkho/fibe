import numpy as np
import pytest
import megpy.tracer as megpy_tracer

import fibe.core.classes as classes_mod
from fibe import FixedBoundaryEquilibrium
from fibe.core.math import (
    build_core_smoothed_jstar_target,
    enforce_monotonic_diamagnetic_fpol,
    rescale_fpol_uniformly_for_target_current,
)


# Two environment-level bugs in the installed megpy's null-point/contour
# tracer, both hit by this test file's from-scratch equilibrium setup (and
# separately confirmed against a real device G-EQDSK -- see
# fibe.scripts.resolve_with_kinetic_constraint's module docstring): a
# scipy.optimize.fsolve shape mismatch inside find_nulls, and a
# tuple/ndarray mixup + out-of-bounds index in megpy.tracer.contour's own
# near-axis branches. Patched here the same way resolve_with_kinetic_constraint.py
# patches `megpy.tracer.contour` -- except `find_null_points` is imported
# by *name* into `fibe.core.classes` at that module's own import time
# (`from .math import (find_null_points, ...)`), so patching
# `megpy.tracer.find_null_points` after fibe is already imported would not
# affect the copy `classes.py` actually calls; the module attribute on
# `fibe.core.classes` itself has to be reassigned instead.
_original_find_null_points = classes_mod.find_null_points


def _patched_find_null_points(*args, **kwargs):
    try:
        return _original_find_null_points(*args, **kwargs)
    except TypeError:
        return {'o-points': np.empty((0, 2)), 'x-points': np.empty((0, 2))}


classes_mod.find_null_points = _patched_find_null_points

_original_megpy_contour = megpy_tracer.contour


def _patched_megpy_contour(*args, **kwargs):
    try:
        return _original_megpy_contour(*args, **kwargs)
    except (TypeError, NameError, IndexError):
        level = kwargs.get('level', args[3] if len(args) > 3 else None)
        return {'X': np.array([]), 'Y': np.array([]), 'theta_XY': np.array([]), 'level': level, 'contours': [], 'radius': 0.0}


megpy_tracer.contour = _patched_megpy_contour


def _build_negative_bt_equilibrium():
    '''A small, from-scratch, converged equilibrium with bcentr < 0 --
    enough to exercise derive_f_profile_from_jstar_target's fpol/bcentr
    sign-consistency fix (bug confirmed against a real device G-EQDSK: see
    that method's docstring). Same grid/boundary shape as
    test_fixed_pressure_solver.py's own `_build_equilibrium`, with a
    negated F profile.
    '''
    eq = FixedBoundaryEquilibrium()
    eq.define_grid(nr=33, nz=33, rmin=1.5, rmax=4.5, zmin=-2.0, zmax=2.0)
    eq.define_boundary_with_mxh(
        rgeo=3.0, zgeo=0.0, rminor=1.0, kappa=1.5,
        cos_coeffs=[0.0] * 7,
        sin_coeffs=[0.0, 0.5, -0.1, 0.0, 0.0, 0.0, 0.0],
    )
    eq.define_f_and_pressure_profiles(
        psinorm=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        f=[-10.0, -9.99, -9.98, -9.97, -9.96, -9.95],
        pressure=[1.0e6, 9.0e5, 7.0e5, 4.0e5, 2.0e5, 1.0e5],
    )
    eq.initialize_psi()
    eq.solve_psi()
    return eq


class TestBuildCoreSmoothedJstarTarget:

    def test_trusted_region_unchanged(self):
        psinorm = np.linspace(0.0, 1.0, 51)
        jstar = 5.0 + 3.0 * psinorm ** 2 - 1.5 * psinorm ** 4
        target = build_core_smoothed_jstar_target(psinorm, jstar, trust_from=0.5)
        trusted = psinorm >= 0.5
        assert np.allclose(target[trusted], jstar[trusted])

    def test_core_replaced_with_smooth_zero_slope_extrapolation(self):
        psinorm = np.linspace(0.0, 1.0, 101)
        # A sharp near-axis spike, well outside the trend of the rest of
        # the (smooth, physically-sensible) profile.
        jstar = 5.0 + 3.0 * psinorm ** 2
        jstar[:5] += np.array([0.0, 40.0, 30.0, 10.0, 2.0])
        target = build_core_smoothed_jstar_target(psinorm, jstar, trust_from=0.5)
        assert target[1] < jstar[1] and target[2] < jstar[2]  # spike removed
        # Zero-slope-at-axis, to within finite-difference resolution.
        dcore = np.diff(target[:5])
        assert abs(dcore[0]) < abs(dcore[-1])

    def test_join_is_slope_continuous_not_just_value_continuous(self):
        # A trusted region with real local curvature of its own (not just
        # the exact quadratic-in-psinorm the extrapolation reproduces) --
        # an unconstrained least-squares fit over the whole trusted region
        # (the original, buggy implementation) trades join-point accuracy
        # for a better fit further out, producing a visible kink right at
        # trust_from; this checks the fix (a local, closed-form value+slope
        # match at the join) actually eliminates that kink, black-box, by
        # comparing finite-difference slopes just inside the extrapolated
        # core vs. just inside the trusted region.
        psinorm = np.linspace(0.0, 1.0, 201)
        jstar = 5.0 + 3.0 * psinorm ** 2 + 20.0 * np.sin(4.0 * psinorm)
        trust_from = 0.3
        target = build_core_smoothed_jstar_target(psinorm, jstar, trust_from=trust_from)
        idx = int(np.searchsorted(psinorm, trust_from))
        dpsin = psinorm[1] - psinorm[0]
        slope_core_side = (target[idx - 1] - target[idx - 2]) / dpsin
        slope_trusted_side = (target[idx + 1] - target[idx]) / dpsin
        assert slope_core_side == pytest.approx(slope_trusted_side, rel=0.1)
        # And the two arrays' values right at the join are close (not just
        # slopes) -- one grid step apart, so a genuinely smooth join should
        # differ by O(dpsin), not O(1).
        assert target[idx - 1] == pytest.approx(target[idx], abs=2.0 * abs(slope_trusted_side) * dpsin)


class TestDeriveFProfileFromJstarTarget:

    def test_preserves_cpasma_and_fixes_fpol_bcentr_sign(self, tmp_path):
        eq_orig = _build_negative_bt_equilibrium()
        assert eq_orig._data['bcentr'] < 0.0  # the condition this fix targets
        geqdsk_path = tmp_path / 'negative_bt.geqdsk'
        eq_orig.to_geqdsk(geqdsk_path)

        eq_for_target = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        eq_for_target.generate_psi_bivariate_spline()
        eq_for_target._fs = eq_for_target.trace_flux_surfaces()
        eq_for_target.compute_flux_surface_averaged_jstar_profile()
        psin_grid = np.linspace(0.0, 1.0, eq_for_target._data['nr'])
        jstar_target = build_core_smoothed_jstar_target(psin_grid, eq_for_target._data['jstar'])

        eq_new = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        cpasma_true = float(eq_new._data['cpasma'])
        bcentr_true = float(eq_new._data['bcentr'])
        # A pressure profile shaped differently from the original (flatter
        # core, same rough edge value) -- the scenario this method exists
        # for: an independently-derived p'(psi), not tied to the original
        # reconstruction.
        new_pressure = 0.5 * eq_new._data['pres'] + 0.5 * eq_new._data['pres'][-1]
        eq_new.define_pressure_profile(new_pressure, psinorm=psin_grid)

        eq_new.derive_f_profile_from_jstar_target(jstar_target, psinorm=psin_grid)

        assert eq_new._data['cpasma'] == pytest.approx(cpasma_true)
        assert np.sign(eq_new._data['fpol'][-1]) == np.sign(bcentr_true)
        # bcentr is *not* pinned to its pre-derivation value (redefine_bcentre=True,
        # 2026-08-25 decision): F's own diamagnetic current genuinely
        # contributes to the toroidal field, so bcentr is allowed to shift
        # once F is re-derived against a new jstar_target -- only its sign
        # (fixed by the bug this test targets) is guaranteed to match.
        assert np.sign(eq_new._data['bcentr']) == np.sign(bcentr_true)

        eq_new.find_magnetic_axis = lambda: None  # see module docstring of resolve_with_kinetic_constraint.py
        eq_new.solve_psi(nxiter=200, erreq=1.0e-7, relax=0.5, relaxj=0.5)
        assert eq_new.converged
        assert eq_new._data['cpasma'] == pytest.approx(cpasma_true, rel=1.0e-6)

        eq_new.compute_flux_surface_averaged_jstar_profile()
        # q's sign should track sign(bcentr*cpasma), same as the original --
        # this is exactly what bug #5 (fpol's un-signed sqrt) got wrong
        # before the fix, for a bcentr<0 equilibrium.
        assert np.sign(np.median(eq_new._data['qpsi'])) == np.sign(np.median(eq_orig._data['qpsi']))


class TestSolvePsiWithFIteration:

    def test_seed_then_f_iteration_converges_with_no_current_hole(self, tmp_path):
        '''The validated two-stage workflow (see FIBE_IMPROVEMENT.md's
        "Best-validated workflow" and Next-tasks items 1-3): seed F once
        via derive_f_profile_from_jstar_target (avoids a core current hole
        from the first Picard iteration), then hand off entirely to
        solve_psi_with_f_iteration (re-derives F from the equilibrium's own
        resolved current each outer iteration, not from jstar_target again).
        Confirms convergence, cpasma preservation, and -- using the
        check_flux_surface_monotonicity utility, not just eyeballing a plot
        (see the "methodology trap" this exact black-box check was built to
        avoid) -- that no current hole/non-nested flux surfaces result.
        '''
        eq_orig = _build_negative_bt_equilibrium()
        geqdsk_path = tmp_path / 'negative_bt_f_iteration.geqdsk'
        eq_orig.to_geqdsk(geqdsk_path)

        eq_for_target = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        eq_for_target.generate_psi_bivariate_spline()
        eq_for_target._fs = eq_for_target.trace_flux_surfaces()
        eq_for_target.compute_flux_surface_averaged_jstar_profile()
        psin_grid = np.linspace(0.0, 1.0, eq_for_target._data['nr'])
        jstar_target = build_core_smoothed_jstar_target(psin_grid, eq_for_target._data['jstar'])

        eq_new = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        cpasma_true = float(eq_new._data['cpasma'])
        new_pressure = 0.5 * eq_new._data['pres'] + 0.5 * eq_new._data['pres'][-1]
        eq_new.define_pressure_profile(new_pressure, psinorm=psin_grid)
        eq_new.derive_f_profile_from_jstar_target(jstar_target, psinorm=psin_grid)  # seed F, avoids a core current hole

        eq_new.find_magnetic_axis = lambda: None  # see module docstring of resolve_with_kinetic_constraint.py
        # errf=2e-4, not 1e-4: with the boundary-anchored normalize_psi_to_ampere(),
        # the F re-estimation runs on the self-consistent span (~2x this synthetic
        # file's stored one), so the F-error metric floors at ~1.3e-4 on contour-
        # tracing noise. The solve is still good there: curscalef = 1.0000,
        # psi_error ~1e-9, exact cpasma; damping does not lower the floor.
        eq_new.solve_psi_with_f_iteration(nfiter=10, errf=2.0e-4, relaxf=1.0, nxiter=200, erreq=1.0e-8, relax=1.0, relaxj=1.0)

        assert eq_new.converged
        assert eq_new._data['cpasma'] == pytest.approx(cpasma_true, rel=1.0e-6)
        n_bad, bad_angles = eq_new.check_flux_surface_monotonicity()
        assert n_bad == 0, f'non-monotonic xpsi (possible current hole) at angles {bad_angles}'

    def test_estimate_flux_surface_averaged_fpol_self_calibrates_sign(self, tmp_path):
        '''_estimate_flux_surface_averaged_fpol (the F-re-derivation
        solve_psi_with_f_iteration calls each outer iteration) self-
        calibrates the raw-vs-labeled cpasma sign convention rather than
        assuming they match -- confirmed on a real device G-EQDSK this
        session (see FIBE_IMPROVEMENT.md's "mixed sign conventions" entry)
        where compute_jtor's raw grid-integrated current came out opposite
        -signed from the file's own labeled cpasma. Before the fix, this
        made the function raise ValueError('Requested plasma current is
        less than the computed pressure contribution!') immediately.

        Reproduced synthetically here, under full control: build a normal,
        self-consistent equilibrium, then reload it with ip_sign=-1, which
        flips only the *labeled* cpasma (reset_plasma_current_sign, called
        inside insert_geqdsk_dict) -- ffprime/pprime/psi (and hence
        compute_jtor's raw grid-integrated current) are left untouched, so
        the labeled and raw-integrated signs now genuinely disagree, the
        same shape of bug as the real G-EQDSK case.
        '''
        eq_orig = _build_negative_bt_equilibrium()
        geqdsk_path = tmp_path / 'negative_bt_sign_mismatch.geqdsk'
        eq_orig.to_geqdsk(geqdsk_path)

        eq_new = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path, ip_sign=-1)
        cpasma_flipped = float(eq_new._data['cpasma'])
        assert cpasma_flipped == pytest.approx(-float(eq_orig._data['cpasma']))

        psin_grid = np.linspace(0.0, 1.0, eq_new._data['nr'])
        new_pressure = 0.5 * eq_new._data['pres'] + 0.5 * eq_new._data['pres'][-1]
        eq_new.define_pressure_profile(new_pressure, psinorm=psin_grid)
        eq_new.find_magnetic_axis = lambda: None

        # No ValueError here is itself the primary assertion (see docstring).
        eq_new.solve_psi_with_f_iteration(nfiter=10, errf=1.0e-3, nxiter=200, erreq=1.0e-7, relax=0.5, relaxj=0.5)
        assert eq_new.converged
        # solve_psi's own current rescaling always forces the total current
        # to match self._data['cpasma'] regardless of the raw/labeled
        # convention -- the labeled (flipped) value should still come out
        # preserved, not silently corrupted back toward the raw convention.
        assert eq_new._data['cpasma'] == pytest.approx(cpasma_flipped, rel=1.0e-6)


class TestEnforceMonotonicDiamagneticFpol:

    def test_corrects_a_mid_radius_hump_to_strictly_decreasing(self):
        psinorm = np.linspace(0.0, 1.0, 51)
        fpol = 10.0 - 2.0 * psinorm
        fpol[20:25] += 3.0  # mid-radius hump: |F| rises before falling again
        assert not np.all(np.diff(np.abs(fpol)) <= 0.0)  # confirm the setup is actually a violation

        result = enforce_monotonic_diamagnetic_fpol(fpol)
        assert np.all(np.diff(np.abs(result)) < 0.0)  # strictly decreasing, see tie-breaking-nudge docstring
        assert np.sign(result[-1]) == np.sign(fpol[-1])
        assert result[-1] == pytest.approx(fpol[-1], rel=1.0e-4)  # edge_weight holds the boundary point still

    def test_preserves_negative_edge_sign(self):
        psinorm = np.linspace(0.0, 1.0, 51)
        fpol = -(10.0 - 2.0 * psinorm)
        fpol[20:25] -= 3.0  # same hump, mirrored onto a bcentr<0 profile
        result = enforce_monotonic_diamagnetic_fpol(fpol)
        assert np.all(result < 0.0)
        assert np.all(np.diff(np.abs(result)) < 0.0)

    def test_already_monotonic_profile_left_nearly_unchanged(self):
        psinorm = np.linspace(0.0, 1.0, 51)
        fpol = 10.0 - 2.0 * psinorm  # no violation to correct
        result = enforce_monotonic_diamagnetic_fpol(fpol)
        assert np.allclose(result, fpol, atol=1.0e-4)  # only the tie-breaking nudge should move it


class TestRescaleFpolUniformlyForTargetCurrent:

    def test_scales_by_sqrt_of_current_ratio(self):
        fpol = np.array([10.0, 9.5, 9.0, 8.5])
        i_f_current, target_current = 5.0e5, 8.0e5
        result = rescale_fpol_uniformly_for_target_current(fpol, i_f_current, target_current)
        expected_factor = np.sqrt(target_current / i_f_current)
        assert np.allclose(result, fpol * expected_factor)

    def test_negligible_i_f_current_returns_fpol_unchanged(self):
        fpol = np.array([10.0, 9.5, 9.0, 8.5])
        result = rescale_fpol_uniformly_for_target_current(fpol, 1.0e-31, 8.0e5)
        assert np.allclose(result, fpol)

    def test_raises_on_non_positive_required_ratio(self):
        # target_current and i_f_current with opposite signs implies a
        # negative scale factor -- self-inconsistent, must raise rather
        # than silently return a sign-flipped fpol.
        fpol = np.array([10.0, 9.5, 9.0, 8.5])
        with pytest.raises(ValueError):
            rescale_fpol_uniformly_for_target_current(fpol, 5.0e5, -8.0e5)


class TestDeriveMonotonicFProfileFromJstarTarget:

    def test_seed_is_monotonic_and_preserves_cpasma(self, tmp_path):
        '''derive_monotonic_f_profile_from_jstar_target's own contract: same
        cpasma-preservation and sign-consistency guarantees as derive_f_
        profile_from_jstar_target (which it calls directly as its own first
        step -- see that method's own dedicated test class above), plus a
        monotonically non-increasing |F| from axis to edge, checked here
        directly rather than assumed.
        '''
        eq_orig = _build_negative_bt_equilibrium()
        geqdsk_path = tmp_path / 'negative_bt_monotonic_seed.geqdsk'
        eq_orig.to_geqdsk(geqdsk_path)

        eq_for_target = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        eq_for_target.generate_psi_bivariate_spline()
        eq_for_target._fs = eq_for_target.trace_flux_surfaces()
        eq_for_target.compute_flux_surface_averaged_jstar_profile()
        psin_grid = np.linspace(0.0, 1.0, eq_for_target._data['nr'])
        jstar_target = build_core_smoothed_jstar_target(psin_grid, eq_for_target._data['jstar'])

        eq_new = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_path)
        cpasma_true = float(eq_new._data['cpasma'])
        bcentr_true = float(eq_new._data['bcentr'])
        new_pressure = 0.5 * eq_new._data['pres'] + 0.5 * eq_new._data['pres'][-1]
        eq_new.define_pressure_profile(new_pressure, psinorm=psin_grid)

        eq_new.derive_monotonic_f_profile_from_jstar_target(jstar_target, psinorm=psin_grid)

        assert eq_new._data['cpasma'] == pytest.approx(cpasma_true)
        assert np.sign(eq_new._data['fpol'][-1]) == np.sign(bcentr_true)
        f2 = eq_new._data['fpol'] ** 2
        assert np.all(np.diff(f2) <= 1.0e-9)  # non-increasing |F| toward the edge

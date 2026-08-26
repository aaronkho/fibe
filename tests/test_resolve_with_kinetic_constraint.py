import numpy as np
import pytest
import megpy.tracer as megpy_tracer

import fibe.core.classes as classes_mod
from fibe import FixedBoundaryEquilibrium
from fibe.core.math import build_core_smoothed_jstar_target


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
        assert eq_new._data['bcentr'] == pytest.approx(bcentr_true)  # untouched by the fix

        eq_new.find_magnetic_axis = lambda: None  # see module docstring of resolve_with_kinetic_constraint.py
        eq_new.solve_psi(nxiter=200, erreq=1.0e-7, relax=0.5, relaxj=0.5)
        assert eq_new.converged
        assert eq_new._data['cpasma'] == pytest.approx(cpasma_true, rel=1.0e-6)

        eq_new.compute_flux_surface_averaged_jstar_profile()
        # q's sign should track sign(bcentr*cpasma), same as the original --
        # this is exactly what bug #5 (fpol's un-signed sqrt) got wrong
        # before the fix, for a bcentr<0 equilibrium.
        assert np.sign(np.median(eq_new._data['qpsi'])) == np.sign(np.median(eq_orig._data['qpsi']))

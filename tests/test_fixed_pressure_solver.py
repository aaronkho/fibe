import sys

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

import fibe.core.classes as classes_mod
from fibe import FixedBoundaryEquilibrium


def _capture_locals(bound_method, *args, **kwargs):
    '''Capture the local variables of bound_method's frame at return, without
    modifying it, so internals not otherwise exposed on the instance can be
    asserted on directly.'''
    target_code = bound_method.__func__.__code__
    captured = {}

    def tracer(frame, event, arg):
        if frame.f_code is target_code:
            captured.update(frame.f_locals)
            return tracer
        return None

    old_trace = sys.settrace(tracer)
    try:
        result = bound_method(*args, **kwargs)
    finally:
        sys.settrace(old_trace)
    return result, captured


def _build_equilibrium(nr=33, f=None, pressure=None):
    eq = FixedBoundaryEquilibrium()
    eq.define_grid(nr=nr, nz=nr, rmin=1.5, rmax=4.5, zmin=-2.0, zmax=2.0)
    eq.define_boundary_with_mxh(
        rgeo=3.0, zgeo=0.0, rminor=1.0, kappa=1.5,
        cos_coeffs=[0.0] * 7,
        sin_coeffs=[0.0, 0.5, -0.1, 0.0, 0.0, 0.0, 0.0],
    )
    eq.define_f_and_pressure_profiles(
        psinorm=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        f=f if f is not None else [10.0, 9.99, 9.98, 9.97, 9.96, 9.95],
        pressure=pressure if pressure is not None else [1.0e6, 9.0e5, 7.0e5, 4.0e5, 2.0e5, 1.0e5],
    )
    eq.initialize_psi()
    return eq


def _patch_area_to_force_trust_split(monkeypatch):
    '''self._fs's level==simagx contour is always a single point, so index 0 of
    _estimate_flux_surface_averaged_fpol's trust_fs is always False; but
    compute_flux_surface_cross_sectional_area happens to return negative areas
    for every surface on the equilibria this test file builds, so trust_fs ends
    up all-False and the near-axis extrapolation block never actually runs.
    Patching this (unrelated to the code under test) dependency to return
    small/large areas by call order forces the real trust/untrusted split so
    that block executes on real data.'''
    call_count = [0]

    def fake_area(contour, r_reference=None, z_reference=None):
        idx = call_count[0]
        call_count[0] += 1
        return 0.0 if idx == 0 else (1.0e-6 if idx <= 3 else 10.0)

    monkeypatch.setattr(classes_mod, 'compute_flux_surface_cross_sectional_area', fake_area)


class TestNearAxisFluxSurfaceExtrapolation:
    '''Regression coverage for the missing-anchor-term bug in
    FixedBoundaryEquilibrium._estimate_flux_surface_averaged_fpol's near-axis
    vprime_fs/ir2_fs/area_fs extrapolation (classes.py).'''

    def test_axis_values_recovered_when_patch_runs(self, monkeypatch):
        eq = _build_equilibrium()
        eq.solve_psi(nxiter=15, erreq=1.0e-3)
        eq.define_axis_safety_factor(1.3)
        _patch_area_to_force_trust_split(monkeypatch)

        _, loc = _capture_locals(eq._estimate_flux_surface_averaged_fpol)

        # Confirm the near-axis patch block actually ran (not skipped).
        assert loc['idx_trust'] > 0
        assert np.any(~loc['trust_fs'][1:loc['idx_trust']])

        # The axis point must recover its explicit override, not a value
        # corrupted by extrapolating without adding back the anchor value.
        rmagx = eq._data['rmagx']
        assert loc['ir2_fs'][0] == pytest.approx(1.0 / rmagx ** 2)
        assert loc['area_fs'][0] == pytest.approx(0.0, abs=1.0e-12)

    def test_differs_from_pre_fix_formula_on_real_data(self, monkeypatch):
        eq = _build_equilibrium()
        eq.solve_psi(nxiter=15, erreq=1.0e-3)
        eq.define_axis_safety_factor(1.3)
        _patch_area_to_force_trust_split(monkeypatch)

        _, loc = _capture_locals(eq._estimate_flux_surface_averaged_fpol)
        idx_trust = loc['idx_trust']
        psinorm = loc['psinorm']
        trust_fs = loc['trust_fs']
        ir2_fs = loc['ir2_fs']
        area_fs = loc['area_fs']
        vprime_fs = loc['vprime_fs']

        # Reconstruct what the pre-fix (missing anchor term) formula would have
        # produced, from the same real, captured trusted-region data -- if the
        # anchor term regresses out of classes.py, these will start matching
        # the real (fixed) values again and this test will fail.
        buggy_ir2 = (psinorm[~trust_fs] - psinorm[idx_trust]) * (ir2_fs[0] - ir2_fs[idx_trust]) / (psinorm[0] - psinorm[idx_trust])
        buggy_area = (psinorm[~trust_fs] - psinorm[idx_trust]) * (area_fs[0] - area_fs[idx_trust]) / (psinorm[0] - psinorm[idx_trust])
        buggy_vprime = (psinorm[~trust_fs] - psinorm[idx_trust]) * (vprime_fs[idx_trust + 1] - vprime_fs[idx_trust]) / (psinorm[idx_trust + 1] - psinorm[idx_trust])

        assert ir2_fs[~trust_fs][0] != pytest.approx(buggy_ir2[0])
        assert area_fs[~trust_fs][0] != pytest.approx(buggy_area[0])
        assert vprime_fs[~trust_fs][0] != pytest.approx(buggy_vprime[0])


class TestFixedPressureCurrentScaling:
    '''Regression coverage for solve_psi(fixed_pressure=...): opt-in F-only
    current rescale that holds the pressure-driven current fixed to p(psi),
    with a fallback guard for when the F-driven current is too weak to safely
    divide by.'''

    def test_default_leaves_curscalef_untouched(self):
        eq = _build_equilibrium()
        eq.solve_psi(nxiter=30, erreq=1.0e-6)
        assert eq._options['fixed_pressure'] is False
        assert eq._data['curscalef'] == pytest.approx(1.0)

    def test_fixed_pressure_matches_cpasma_and_uses_f_only_split(self):
        eq = _build_equilibrium()
        eq.solve_psi(nxiter=30, erreq=1.0e-6, fixed_pressure=True)
        total_current = float(np.sum(eq._data['cur']) * eq._data['hrz'])
        assert total_current == pytest.approx(eq._data['cpasma'], rel=1.0e-6)
        assert eq._data['curscalef'] != pytest.approx(1.0)

    def test_fixed_pressure_curscalef_stays_bounded_for_realistic_profile(self):
        eq = _build_equilibrium()
        trace = []
        orig = eq._update_current_fixed_pressure

        def wrapped(current_new, pp_grid, relax=1.0, min_f_fraction=0.05):
            orig(current_new, pp_grid, relax=relax, min_f_fraction=min_f_fraction)
            trace.append(eq._data['curscalef'])
        eq._update_current_fixed_pressure = wrapped

        eq.solve_psi(nxiter=30, erreq=1.0e-6, fixed_pressure=True)

        assert trace
        assert max(abs(c) for c in trace) < 100
        assert not any(np.isnan(c) or np.isinf(c) for c in trace)

    def test_fixed_pressure_falls_back_for_weak_f_current(self):
        # Near-flat F -> F-driven current starts ~0. Without the fallback guard
        # this diverges into chaotic, sign-flipping curscalef values (observed
        # up to ~1e15 in manual testing) instead of converging.
        eq = _build_equilibrium(f=[10.0] * 6)

        curscalef_trace = []
        fallback_calls = []
        orig_fixed = eq._update_current_fixed_pressure
        orig_uniform = eq._update_current

        def wrapped_fixed(current_new, pp_grid, relax=1.0, min_f_fraction=0.05):
            orig_fixed(current_new, pp_grid, relax=relax, min_f_fraction=min_f_fraction)
            curscalef_trace.append(eq._data['curscalef'])
        eq._update_current_fixed_pressure = wrapped_fixed

        def wrapped_uniform(current_new, relax=1.0):
            fallback_calls.append(True)
            orig_uniform(current_new, relax=relax)
        eq._update_current = wrapped_uniform

        eq.solve_psi(nxiter=30, erreq=1.0e-6, fixed_pressure=True)

        assert fallback_calls, 'expected the weak-F-current guard to trigger the uniform-rescale fallback at least once'
        assert not any(np.isnan(c) or np.isinf(c) for c in curscalef_trace)
        assert max(abs(c) for c in curscalef_trace) < 1000

    def test_solve_psi_with_f_iteration_opts_into_fixed_pressure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        eq = _build_equilibrium()
        eq.solve_psi_with_f_iteration(nfiter=2, errf=1.0e-2, nxiter=20, erreq=1.0e-4)
        assert eq._options['fixed_pressure'] is True
        assert eq._data['curscalef'] != pytest.approx(1.0)

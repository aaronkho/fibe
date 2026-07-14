# custom_root wiring: differentiate through the converged GS solution by
# treating it as a root of gs_residual (residual.py) and applying the
# implicit function theorem (jax.lax.custom_root), instead of unrolling the
# numpy Picard loop through autodiff.
#
# The *forward* solve reuses the existing, battle-tested numpy solver
# (classes.FixedBoundaryEquilibrium.solve_psi) unchanged, bridged into JAX
# via jax.pure_callback -- gs_residual only needs to be JAX-traceable enough
# to supply the *backward* pass. The backward pass solves the linearized
# system J @ dpsi = y (J = d(gs_residual)/d(psi) at the root).
#
# tangent_solve must be built entirely from JAX-native, transposable ops --
# NOT a pure_callback. custom_root's reverse-mode rule is derived by
# symbolically transposing tangent_solve itself (not just calling it with a
# transposed operator), so a callback anywhere inside it fails with "Pure
# callbacks do not support transpose."
#
# jax.scipy.sparse.linalg.gmres looks like the natural fit (J isn't
# symmetric -- the curscale renormalization and the axis-value coupling in
# gs_residual break symmetry -- so CG isn't valid here), but nesting it as
# tangent_solve inside custom_root reliably raises
# `NotImplementedError: open an issue at https://github.com/google/jax !!`
# from deep inside lax.custom_linear_solve, even for a trivial linear f with
# no sparse ops at all -- reproduced in isolation, so this looks like a
# genuine incompatibility between custom_root's and custom_linear_solve's
# AD rules in this JAX version (0.10.2), not something specific to
# gs_residual. Worked around here by materializing the (dense) Jacobian of
# g via jax.jacfwd -- g is linear (it's f linearized at the root), so its
# Jacobian is just its own matrix, independent of where it's evaluated --
# and solving with jnp.linalg.solve, both of which are plain, well-supported
# primitives with no custom transpose rule to conflict with custom_root's.
# This is O(n^2) memory for an n-point grid, so it will not scale past the
# grid sizes this scaffold has been tested on (tens of thousands of points);
# revisit once the custom_root/custom_linear_solve nesting issue is
# understood, or replace with a matrix-free Krylov solve implemented by
# hand instead of going through jax.scipy.sparse.linalg.
#
# Known limitation: the numpy bridge function below (used only in the
# forward `solve`, which *is* allowed to be a callback) mutates `eq` in
# place, which violates pure_callback's "no side effects" contract. This is
# a deliberate simplification for the single-equilibrium, non-vmapped case
# this scaffold targets; it will misbehave under jax.vmap over multiple
# parameter sets (they'd all fight over the same mutable `eq`) and should
# not be relied on there without reworking to build a fresh solver state per
# call.
#
# Validation status (33x33 test grid, against independent from-scratch
# finite differences of the numpy solver, not just internal consistency):
#   - Forward solve matches a direct eq.solve_psi() call to ~1e-8.
#   - d(psi)/d(pressure spline coefficient) matches FD to ~1e-7 relative
#     error -- the primary use case this was built for (gradient-based
#     profile shaping) looks solid.
#   - d(psi)/d(cpasma) initially looked systematically off by ~17%, stable
#     across FD step sizes and unchanged by Newton-correcting gs_residual to
#     machine-precision zero first, and reproduced exactly by a manual
#     implicit-function-theorem calculation bypassing custom_root entirely
#     -- so not a wiring bug and not truncation/residual-nonzero-ness. Root
#     cause turned out to be in the *test*, not this module: the from-
#     scratch FD reference called classes.define_vacuum_toroidal_field()
#     for each perturbed cpasma to seed fpol, and that seed generator's
#     f_span is itself a function of cpasma (f_span = 0.005*(1e-6*cpasma)),
#     so the "reference" was silently varying the F-profile shape along
#     with cpasma while gs_residual (correctly) holds it fixed -- two
#     different partial derivatives being compared. Holding fpol truly
#     fixed in the FD reference (defining it once via define_f_profile and
#     reusing the same array for every cpasma value) brought the relative
#     error down to ~1e-10. d(psi)/d(cpasma) is validated.
from typing import Any

import numpy as np
from scipy.interpolate import splev
import jax
import jax.numpy as jnp

from .residual import GSGridConstants, GSProfileParams, gs_residual


def _numpy_forward_solve(
    eq: Any,
    pres_c: np.ndarray,
    fpol_c: np.ndarray,
    cpasma: float,
    internal_cutoff: float,
    solver_kwargs: dict,
) -> np.ndarray:
    '''Mutate eq's profile fit + cpasma, re-converge with the existing numpy
    solver, and return flat psi with exterior points pinned to 0.

    Warm-started from eq's current psi (solve_psi's Picard loop starts from
    whatever self._data['psi'] already holds), so repeated calls during an
    optimization loop are cheap after the first.
    '''
    t_pres, _, k_pres = eq._fit['pres_fs']['tck']
    eq._fit['pres_fs']['tck'] = (t_pres, np.asarray(pres_c, dtype=np.float64), k_pres)
    t_fpol, _, k_fpol = eq._fit['fpol_fs']['tck']
    eq._fit['fpol_fs']['tck'] = (t_fpol, np.asarray(fpol_c, dtype=np.float64), k_fpol)

    psinorm = np.linspace(0.0, 1.0, eq._data['nr'])
    eq._data['pres'] = splev(psinorm, eq._fit['pres_fs']['tck'])
    eq._data['pprime'] = splev(psinorm, eq._fit['pres_fs']['tck'], der=1)
    eq._data['fpol'] = splev(psinorm, eq._fit['fpol_fs']['tck'])
    eq._data['ffprime'] = splev(psinorm, eq._fit['fpol_fs']['tck'], der=1) * eq._data['fpol']
    eq._data['cpasma'] = float(np.asarray(cpasma))

    # solve_psi()'s final normalize_psi_to_original() step affinely rescales
    # psi to match simagx_orig/sibdry_orig on any *non-scratch* call -- i.e.
    # the axis flux from whatever solve happened to run first -- which would
    # silently erase the true params sensitivity on every call after the
    # first (found by comparing against an independent from-scratch finite
    # difference, which disagreed sharply with a warm-started one before this
    # fix). Forcing scratch=True makes that step just update the _orig
    # bookkeeping instead of rescaling. pnaxis is already passed explicitly
    # below, so scratch's other use (a pnaxis default) is unaffected.
    eq.scratch = True
    eq.solve_psi(pnaxis=float(internal_cutoff), **solver_kwargs)

    psi_flat = np.asarray(eq._data['psi'], dtype=np.float64).ravel().copy()
    # gs_residual encodes psi == 0 outside the plasma (the FD Dirichlet
    # condition the Picard loop actually solves under); solve_psi's final
    # extend_psi_beyond_boundary() overwrites those points with an
    # extrapolated field for plotting, which would otherwise make
    # f(solve(...)) != 0 there and break custom_root's root assumption.
    psi_flat[eq._data['ijout']] = 0.0
    return psi_flat


def solve_gs_implicit(
    params: GSProfileParams,
    grid: GSGridConstants,
    eq: Any,
    solver_kwargs: dict | None = None,
) -> jnp.ndarray:
    '''Solve gs_residual(psi, params, grid) = 0 via the numpy Picard solver,
    with gradients w.r.t. params (and any grid leaves) defined by implicit
    differentiation through jax.lax.custom_root.
    '''
    solver_kwargs = dict(solver_kwargs) if solver_kwargs else {}
    psi_shape_dtype = jax.ShapeDtypeStruct((grid.nr * grid.nz,), jnp.float64)

    def f(psi):
        return gs_residual(psi, params, grid)

    def solve(f, initial_guess):
        def _cb(pres_c, fpol_c, cpasma):
            return _numpy_forward_solve(eq, pres_c, fpol_c, cpasma, params.internal_cutoff, solver_kwargs)
        return jax.pure_callback(_cb, psi_shape_dtype, params.pres.c, params.fpol.c, params.cpasma)

    def tangent_solve(g, y):
        J = jax.jacfwd(g)(y)
        return jnp.linalg.solve(J, y)

    psi_init = jnp.asarray(np.asarray(eq._data['psi'], dtype=np.float64).ravel())
    return jax.lax.custom_root(f, psi_init, solve, tangent_solve)

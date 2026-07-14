# JAX-traceable discretized Grad-Shafranov residual.
#
# The existing Picard step (core.math.compute_psi) is
#     psi_new = -solver(s5 * cur(psi_old))     where solver = factorized(A)
# so the converged solution is a root of
#     R(psi, theta) = A @ psi + s5 * cur(psi, theta) = 0
# with A the fixed pentadiagonal FD operator (core.math.compute_finite_
# difference_matrix) and cur(psi, theta) = curscale * jtor(psi, theta) the
# scaled toroidal current density. This module implements R() in JAX so that
# jax.lax.custom_root (or an equivalent implicit-diff wrapper, added in
# solve.py) can differentiate through the *root*, not through the Picard
# iteration.
#
# Scope of this first cut: grid/boundary geometry (A, s5, inout mask) and the
# magnetic-axis grid *index* are frozen constants, taken from an already-
# converged FixedBoundaryEquilibrium. Only the profile shape (fpol/pres
# B-spline coefficients) and cpasma are differentiable parameters. Freezing
# the axis index is justified by the envelope theorem: the axis is a
# stationary point of psi, so its location is insensitive to psi to first
# order, and only the *value* there (a smooth function of the frozen index's
# neighborhood) needs to carry gradients. Differentiating w.r.t. boundary
# shape (MXH coefficients) would additionally require A, s5 and the axis
# index itself to become functions of theta -- deferred to a later pass.
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from .profiles import BSplineTCK, from_scipy_fit, ffprime_and_pprime_grid

MU0 = 4.0e-7 * np.pi


class GSGridConstants:
    '''Frozen FD-grid/geometry constants for one equilibrium.

    Registered as a pytree with `nr`, `nz`, `axis_index` as static aux data,
    since they're used for concrete-integer offsets (`k+1`, `k+nr`, ...) in
    magnetic_axis_flux -- a plain NamedTuple would trace them under jax.jit
    and break that indexing (see profiles.BSplineTCK for the same issue).
    '''

    def __init__(self, nr, nz, rpsi, hrz, hrm1, hrm2, hzm1, hzm2, inout_mask, s5, A, axis_index, sibdry, simagx_orig, sibdry_orig):
        self.nr = nr                    # static
        self.nz = nz                    # static
        self.rpsi = rpsi                # flat (nr*nz,), grid R value at each point
        self.hrz = hrz                  # cell area hr * hz
        self.hrm1 = hrm1
        self.hrm2 = hrm2
        self.hzm1 = hzm1
        self.hzm2 = hzm2
        self.inout_mask = inout_mask    # flat bool (nr*nz,), True inside the plasma
        self.s5 = s5                    # flat (nr*nz,), current-term grid metric factor
        self.A = A                      # (nr*nz, nr*nz), frozen FD operator
        self.axis_index = axis_index    # static, flat grid index nearest the magnetic axis
        # Always 0.0 (see grid_from_equilibrium) -- NOT read from the source
        # equilibrium's own eq._data['sibdry'], which can be a nonzero
        # "cosmetic" physical value (e.g. ~8.8 for a loaded, F-solver-
        # converged G-EQDSK) that solve.py's _numpy_forward_solve never
        # actually returns psi in, since it permanently forces scratch=True.
        # This is the scale gs_residual/custom_root require for correctness
        # -- see psi_to_physical_scale below for recovering the source
        # equilibrium's own psi convention for display/comparison purposes
        # without disturbing this.
        self.sibdry = sibdry
        # simagx_orig/sibdry_orig: the source equilibrium's true physical
        # axis/boundary flux (e.g. simagx~0, sibdry~8.8 for a loaded,
        # F-solver-converged G-EQDSK), captured once here before any
        # differentiable perturbation runs. Frozen constants, only consumed
        # by psi_to_physical_scale -- gs_residual itself must keep using
        # `sibdry` (always 0) above, not these.
        self.simagx_orig = simagx_orig
        self.sibdry_orig = sibdry_orig

    def tree_flatten(self):
        children = (
            self.rpsi, self.hrz, self.hrm1, self.hrm2, self.hzm1, self.hzm2,
            self.inout_mask, self.s5, self.A, self.sibdry, self.simagx_orig, self.sibdry_orig,
        )
        aux_data = (self.nr, self.nz, self.axis_index)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nr, nz, axis_index = aux_data
        rpsi, hrz, hrm1, hrm2, hzm1, hzm2, inout_mask, s5, A, sibdry, simagx_orig, sibdry_orig = children
        return cls(nr, nz, rpsi, hrz, hrm1, hrm2, hzm1, hzm2, inout_mask, s5, A, axis_index, sibdry, simagx_orig, sibdry_orig)


jax.tree_util.register_pytree_node(GSGridConstants, GSGridConstants.tree_flatten, GSGridConstants.tree_unflatten)


class GSProfileParams:
    def __init__(self, fpol: BSplineTCK, pres: BSplineTCK, cpasma: float, internal_cutoff: float = 0.1):
        self.fpol = fpol
        self.pres = pres
        self.cpasma = cpasma
        self.internal_cutoff = internal_cutoff

    def tree_flatten(self):
        return (self.fpol, self.pres, self.cpasma, self.internal_cutoff), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fpol, pres, cpasma, internal_cutoff = children
        return cls(fpol, pres, cpasma, internal_cutoff)


jax.tree_util.register_pytree_node(GSProfileParams, GSProfileParams.tree_flatten, GSProfileParams.tree_unflatten)


def grid_from_equilibrium(eq) -> GSGridConstants:
    '''Extract frozen geometry/FD constants from a converged FixedBoundaryEquilibrium.'''
    data = eq._data
    psi_flat = np.asarray(data['psi']).ravel()
    # eq._data['psi'] has generally been through extend_psi_beyond_boundary(),
    # which extrapolates *exterior* (ijout) grid points to nonzero values for
    # plotting/analysis -- restrict the argmax to interior (ijin) points, or
    # a stray large extrapolated value outside the LCFS gets mistaken for the
    # magnetic axis.
    #
    # argmax(|psi - sibdry|), not argmax(|psi|): the magnetic axis is the
    # interior extremum of psi -- i.e. the point *farthest* from the
    # boundary flux value -- and only coincides with argmax(|psi|) when
    # sibdry happens to be pinned at 0 (true mid-Picard-iteration, and hence
    # also true for a from-scratch equilibrium whose solve_psi() call never
    # got past scratch=True to rescale it away from that). For an
    # already-converged equilibrium loaded from a G-EQDSK (sibdry~8.8, not
    # 0, once solve_psi()'s final normalize_psi_to_original() rescales psi
    # to match the source file's physical scale), argmax(|psi|) instead
    # picks a point *near the boundary* (where |psi| is largest) and
    # magnetic_axis_flux ends up evaluating a Taylor expansion centered
    # there -- silently wrong for every derived quantity, not just a
    # labeling error, since gs_residual's current term is evaluated relative
    # to this same (bogus) "axis" value. Found via a ~1800x-too-small,
    # roughly sign-scrambled JAX gradient (confirmed not a nonlinearity
    # effect: the mismatch factor stayed ~constant across 4 orders of
    # magnitude of perturbation size) that traced back to
    # magnetic_axis_flux(psi, grid) returning a value near 0 instead of the
    # true simagx.
    ijin = np.asarray(data['ijin'])
    axis_index = int(ijin[np.abs(psi_flat[ijin] - data['sibdry']).argmax()])
    A = jsparse.BCOO.from_scipy_sparse(data['matrix'].tocoo())
    return GSGridConstants(
        nr=int(data['nr']),
        nz=int(data['nz']),
        rpsi=jnp.asarray(np.asarray(data['rpsi']).ravel()),
        hrz=float(data['hrz']),
        hrm1=float(data['hrm1']),
        hrm2=float(data['hrm2']),
        hzm1=float(data['hzm1']),
        hzm2=float(data['hzm2']),
        inout_mask=jnp.asarray(data['inout'] != 0),
        s5=jnp.asarray(data['s5']),
        A=A,
        axis_index=axis_index,
        # 0.0, not float(data['sibdry']): data['sibdry'] here is eq's current
        # *cosmetic* physical boundary flux (e.g. ~8.8 for a loaded,
        # F-solver-converged G-EQDSK), used just above to correctly locate
        # axis_index in that reference frame. But every psi this frozen grid
        # will ever be paired with (via gs_residual, from
        # solve.py's _numpy_forward_solve) comes back on the raw internal
        # Picard-loop convention instead (boundary pinned to 0 by
        # zero_magnetic_boundary(), independent of eq's cosmetic scale),
        # because that bridge function permanently forces eq.scratch=True to
        # dodge a *different* bug (normalize_psi_to_original() otherwise
        # rescaling every subsequent solve back toward whichever equilibrium
        # happened to converge first) and never rescales back afterward.
        # Freezing sibdry=8.8 here while gs_residual is actually evaluated
        # against sibdry=0 psi arrays put every downstream quantity on the
        # wrong psinorm scale -- confirmed by gs_residual's own norm at the
        # true numpy solution: ~2.5e-05 with sibdry=0 (a genuine root) vs.
        # ~0.27 with sibdry=8.8 (not a root at all). If solve.py's forced
        # scratch=True/raw-convention behavior ever changes, this must too.
        sibdry=0.0,
        simagx_orig=float(data['simagx']),
        sibdry_orig=float(data['sibdry']),
    )


def params_from_equilibrium(eq, internal_cutoff: float | None = None) -> GSProfileParams:
    '''Extract differentiable profile parameters from the same equilibrium.'''
    cutoff = internal_cutoff if internal_cutoff is not None else eq._options.get('pnaxis', 0.1)
    return GSProfileParams(
        fpol=from_scipy_fit(eq._fit['fpol_fs']),
        pres=from_scipy_fit(eq._fit['pres_fs']),
        cpasma=float(eq._data['cpasma']),
        internal_cutoff=float(cutoff),
    )


def compute_jtor(rpsi: jnp.ndarray, ffprime: jnp.ndarray, pprime: jnp.ndarray) -> jnp.ndarray:
    '''JAX port of core.math.compute_jtor -- keep the -1.0 COCOS sign in sync with it.'''
    return -1.0 * (ffprime / (MU0 * rpsi) + rpsi * pprime)


def normalized_psi(psi_flat: jnp.ndarray, simagx: jnp.ndarray, sibdry: float) -> jnp.ndarray:
    xpsi = (psi_flat - simagx) / (sibdry - simagx)
    return jnp.where(xpsi < 0.0, 0.0, xpsi)


def magnetic_axis_flux(psi_flat: jnp.ndarray, grid: GSGridConstants) -> jnp.ndarray:
    '''JAX port of core.math.find_extrema_with_taylor_expansion, frozen at grid.axis_index.'''
    k = grid.axis_index
    nr = grid.nr
    apsi = jnp.abs(psi_flat)
    ar = 0.5 * grid.hrm1 * (apsi[k + 1] - apsi[k - 1])
    az = 0.5 * grid.hzm1 * (apsi[k + nr] - apsi[k - nr])
    arr = grid.hrm2 * (apsi[k + 1] + apsi[k - 1] - 2.0 * apsi[k])
    azz = grid.hzm2 * (apsi[k + nr] + apsi[k - nr] - 2.0 * apsi[k])
    arz = 0.25 * grid.hrm1 * grid.hzm1 * (
        apsi[k + nr + 1] + apsi[k - nr - 1] - apsi[k - nr + 1] - apsi[k + nr - 1]
    )
    delta = arz * arz - arr * azz
    rmax = (azz * ar - arz * az) / delta
    zmax = (arr * az - arz * ar) / delta
    psi_extrema = (
        apsi[k] + rmax * ar + zmax * az
        + 0.5 * arr * rmax ** 2 + 0.5 * azz * zmax ** 2
        + arz * rmax * zmax
    )
    return psi_extrema * jnp.sign(psi_flat[k])


def psi_to_physical_scale(psi_flat: jnp.ndarray, grid: GSGridConstants) -> jnp.ndarray:
    '''Affinely rescale a raw psi (grid.sibdry=0 convention, as returned by
    solve.py's solve_gs_implicit) onto the source equilibrium's true physical
    scale (grid.simagx_orig, grid.sibdry_orig) -- e.g. to match psi as
    originally stored in a loaded G-EQDSK, or to compare directly against
    eq._data['psi'] saved *before* any of this differentiable machinery ran.

    This is the same affine remapping classes.renormalize_psi performs
    (psi_new = scale*(psi_old - simagx_old) + simagx_new with
    scale = (sibdry_new-simagx_new)/(sibdry_old-simagx_old)), reimplemented
    here in JAX so it can be composed with solve_gs_implicit and
    differentiated through directly -- unlike calling eq.renormalize_psi()
    itself, which would require a second numpy round-trip and isn't
    traceable. `simagx_raw = magnetic_axis_flux(psi_flat, grid)` is
    recomputed dynamically (not cached), so gradients through this
    rescaling stay correct even though the raw axis value shifts slightly
    with the perturbed profile. Purely a presentation/comparison transform:
    gs_residual/custom_root must keep operating on the raw (un-rescaled)
    psi -- see grid.sibdry's docstring.
    '''
    simagx_raw = magnetic_axis_flux(psi_flat, grid)
    scale = (grid.sibdry_orig - grid.simagx_orig) / (grid.sibdry - simagx_raw)
    return scale * (psi_flat - simagx_raw) + grid.simagx_orig


def gs_residual(psi_flat: jnp.ndarray, params: GSProfileParams, grid: GSGridConstants) -> jnp.ndarray:
    '''R(psi, theta); a converged fibe solution is (approximately) a root of this.'''
    simagx = magnetic_axis_flux(psi_flat, grid)
    psinorm = normalized_psi(psi_flat, simagx, grid.sibdry)
    dpsinorm_dpsi = 1.0 / (grid.sibdry - simagx)

    ffp, pp = ffprime_and_pprime_grid(psinorm, params.fpol, params.pres, dpsinorm_dpsi, params.internal_cutoff)
    cur_raw = jnp.where(grid.inout_mask, compute_jtor(grid.rpsi, ffp, pp), 0.0)
    curscale = params.cpasma / (jnp.sum(cur_raw) * grid.hrz)
    cur = curscale * cur_raw

    return grid.A @ psi_flat + grid.s5 * cur

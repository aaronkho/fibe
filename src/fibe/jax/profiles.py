# JAX-traceable evaluation of the cubic B-splines fibe uses for p(psi)/F(psi).
#
# fibe fits profiles with `scipy.interpolate.splrep` (see
# `core.math.generate_bounded_1d_spline`), which is not JAX-traceable and
# whose knot placement is a nonlinear function of the input samples when
# smoothing is enabled. Rather than reimplementing splrep, we treat an
# already-fitted `(t, c, k)` tuple as the interface: knots `t` and degree `k`
# are frozen/static, and only the coefficient vector `c` is a differentiable
# leaf. This is the natural free-parameter choice for gradient-based profile
# shaping anyway.
#
# The evaluator below is a direct Cox-de Boor recursion, validated against
# `scipy.interpolate.splev` (value and first derivative, including mirrored
# fits and boundary points) to ~1e-10 before being ported here.
import jax
# fibe's numpy solver runs in float64 and converges to erreq~1e-8; JAX
# defaults to float32, which silently limits both this evaluator and any
# gradients through it to ~1e-7 relative precision. Must be set before any
# JAX arrays are created, so it happens at import time here.
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


class BSplineTCK:
    '''(t, c, k) B-spline, registered as a pytree with `k` as static aux data.

    A plain NamedTuple would make `k` a traced leaf under jax.jit/vmap, which
    breaks since `k` is used for concrete-integer slicing (`c[1:n]`, python
    range() loops in the de Boor recursion below).
    '''

    def __init__(self, t: jnp.ndarray, c: jnp.ndarray, k: int):
        self.t = t
        self.c = c
        self.k = k

    def tree_flatten(self):
        return (self.t, self.c), self.k

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        t, c = children
        return cls(t, c, aux_data)


jax.tree_util.register_pytree_node(BSplineTCK, BSplineTCK.tree_flatten, BSplineTCK.tree_unflatten)


def from_scipy_fit(fit: dict) -> BSplineTCK:
    '''Build a BSplineTCK from a fibe `_fit[...]` entry, e.g. eq._fit["pres_fs"].'''
    t, c, k = fit['tck']
    return BSplineTCK(jnp.asarray(t), jnp.asarray(c), int(k))


def _eval_bspline_scalar(x: jnp.ndarray, t: jnp.ndarray, c: jnp.ndarray, k: int) -> jnp.ndarray:
    n = t.shape[0] - k - 1
    x = jnp.clip(x, t[k], t[n])
    i = jnp.clip(jnp.searchsorted(t, x, side='right') - 1, k, n - 1)
    d = [c[i - k + j] for j in range(k + 1)]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = i - k + j
            alpha = (x - t[left]) / (t[left + k - r + 1] - t[left])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


def eval_bspline(x: jnp.ndarray, t: jnp.ndarray, c: jnp.ndarray, k: int) -> jnp.ndarray:
    x = jnp.atleast_1d(x)
    return jax.vmap(_eval_bspline_scalar, in_axes=(0, None, None, None))(x, t, c, k)


def bspline_derivative(t: jnp.ndarray, c: jnp.ndarray, k: int) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    '''Coefficients of the (k-1)-degree spline that is the derivative of (t, c, k).'''
    n = t.shape[0] - k - 1
    c_new = k * (c[1:n] - c[0:n - 1]) / (t[k + 1:k + n] - t[1:n])
    t_new = t[1:-1]
    return t_new, c_new, k - 1


def eval_profile_and_derivative(spline: BSplineTCK, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    val = eval_bspline(x, spline.t, spline.c, spline.k)
    t1, c1, k1 = bspline_derivative(spline.t, spline.c, spline.k)
    der = eval_bspline(x, t1, c1, k1)
    return val, der


def ffprime_and_pprime_grid(
    psinorm: jnp.ndarray,
    fpol: BSplineTCK,
    pres: BSplineTCK,
    dpsinorm_dpsi: jnp.ndarray,
    internal_cutoff: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''JAX port of classes.compute_ffprime_and_pprime_grid (spline branch only).'''
    cutoff = jnp.array([internal_cutoff])

    fpol_val, fpol_der = eval_profile_and_derivative(fpol, psinorm)
    fpol_val_c, fpol_der_c = eval_profile_and_derivative(fpol, cutoff)
    ffp_internal = (fpol_der_c * fpol_val_c * dpsinorm_dpsi)[0]
    ffp = fpol_der * fpol_val * dpsinorm_dpsi
    ffp = jnp.where(psinorm < internal_cutoff, ffp_internal, ffp)

    _, pres_der = eval_profile_and_derivative(pres, psinorm)
    _, pres_der_c = eval_profile_and_derivative(pres, cutoff)
    pp_internal = (pres_der_c * dpsinorm_dpsi)[0]
    pp = pres_der * dpsinorm_dpsi
    pp = jnp.where(psinorm < internal_cutoff, pp_internal, pp)

    return ffp, pp

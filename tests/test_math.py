import numpy as np
import pytest
from scipy.interpolate import splev, bisplev
from scipy.sparse import issparse

from fibe.core.math import (
    generate_bounded_1d_spline,
    generate_2d_spline,
    generate_optimal_grid,
    generate_boundary_maps,
    compute_grid_spacing,
    generate_finite_difference_grid,
    compute_jtor,
    compute_psi,
    compute_finite_difference_matrix,
    generate_initial_psi,
    compute_grad_psi_vector_from_2d_spline,
    order_contour_points_by_angle,
    generate_segments,
    generate_x_point_candidates,
    compute_intersection_from_line_segment_complex,
    compute_intersection_from_line_segment_coordinates,
    avoid_convex_curvature,
    generate_boundary_splines,
    find_extrema_with_taylor_expansion,
    compute_gradients_at_boundary,
    generate_boundary_gradient_spline,
    compute_psi_extension,
    compute_flux_surface_quantities,
    compute_safety_factor_contour_integral,
    trace_contours_with_contourpy,
    trace_contour_with_splines,
    compute_adjusted_contour_resolution,
)


# Shared analytic setup: a circular boundary and a paraboloid psi field, both
# centered at (R0, Z0). psi = (R-R0)^2 + (Z-Z0)^2 has an exact minimum (the
# magnetic axis) at the center and its level sets are circles of radius
# sqrt(level), which gives most geometry helpers an exact, hand-derivable
# expected answer instead of a purely qualitative smoke test.
R0 = 2.0
Z0 = 0.3
A_BDRY = 0.8


def circle_points(radius, n=64, r0=R0, z0=Z0):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return r0 + radius * np.cos(theta), z0 + radius * np.sin(theta)


@pytest.fixture(scope='module')
def grid():
    rvec = np.linspace(0.8, 3.2, 49)
    zvec = np.linspace(-1.5, 1.5, 49)
    return rvec, zvec


@pytest.fixture(scope='module')
def boundary():
    # Explicitly closed (first point repeated at the end): generate_boundary_maps /
    # generate_finite_difference_grid / generate_initial_psi iterate rbdry[:-1],
    # rbdry[1:] as edges and need the closing edge included exactly this way.
    r, z = circle_points(A_BDRY)
    return np.concatenate([r, [r[0]]]), np.concatenate([z, [z[0]]])


@pytest.fixture(scope='module')
def psi_grid(grid):
    rvec, zvec = grid
    rmesh, zmesh = np.meshgrid(rvec, zvec)
    return (rmesh - R0) ** 2 + (zmesh - Z0) ** 2


@pytest.fixture(scope='module')
def psi_spline(grid, psi_grid):
    rvec, zvec = grid
    return generate_2d_spline(rvec, zvec, psi_grid.T, s=0)


@pytest.fixture(scope='module')
def psi_grid_fd(grid):
    # The finite-difference solver's own convention (matched by
    # find_extrema_with_taylor_expansion, compute_gradients_at_boundary,
    # compute_psi_extension): psi is zero outside the plasma boundary and
    # most-negative (largest magnitude) at the magnetic axis, not the smooth,
    # unclipped quadratic used elsewhere in this file.
    rvec, zvec = grid
    rmesh, zmesh = np.meshgrid(rvec, zvec)
    rho2 = (rmesh - R0) ** 2 + (zmesh - Z0) ** 2
    return np.where(rho2 <= A_BDRY ** 2, rho2 - A_BDRY ** 2, 0.0)


@pytest.fixture(scope='module')
def fd_grid(grid, boundary):
    rvec, zvec = grid
    rbdry, zbdry = boundary
    return generate_finite_difference_grid(rvec, zvec, rbdry, zbdry)


class TestSplineUtilities:

    def test_generate_bounded_1d_spline_roundtrip_asymmetrical(self):
        xnorm = np.linspace(0.0, 1.0, 11)
        y = 1.0 + 2.0 * xnorm + 3.0 * xnorm ** 2
        fit = generate_bounded_1d_spline(y, xnorm=xnorm, symmetrical=False, smooth=False)
        assert fit['bounds'] == (0.0, 1.0)
        y_eval = splev(xnorm, fit['tck'])
        assert y_eval == pytest.approx(y, abs=1.0e-8)

    def test_generate_bounded_1d_spline_symmetrical_mirrors_derivative_to_zero(self):
        xnorm = np.linspace(0.0, 1.0, 11)
        y = 1.0 + xnorm ** 2
        fit = generate_bounded_1d_spline(y, xnorm=xnorm, symmetrical=True, smooth=False)
        assert fit['bounds'] == (-1.0, 1.0)
        # Symmetrical fitting mirrors y about x=0, so the fitted curve is even
        # and its derivative at the origin must vanish.
        assert splev(0.0, fit['tck'], der=1) == pytest.approx(0.0, abs=1.0e-8)
        assert splev(-0.5, fit['tck']) == pytest.approx(splev(0.5, fit['tck']), abs=1.0e-8)

    def test_generate_2d_spline_roundtrip(self, grid, psi_grid, psi_spline):
        rvec, zvec = grid
        assert psi_spline['bounds'] == (rvec.min(), zvec.min(), rvec.max(), zvec.max())
        for i in (0, len(rvec) // 2, -1):
            for j in (0, len(zvec) // 2, -1):
                value = bisplev(rvec[i], zvec[j], psi_spline['tck'])
                assert value == pytest.approx(psi_grid[j, i], abs=1.0e-6)


class TestGridGeneration:

    def test_generate_optimal_grid_places_boundary_extrema_at_offset_index(self, boundary):
        rbdry, zbdry = boundary
        nr, nz = 33, 33
        rmin, rmax, zmin, zmax = generate_optimal_grid(nr, nz, rbdry, zbdry)
        assert rmin < rmax
        assert zmin < zmax
        # e=3.5 is the fixed offset baked into generate_optimal_grid: the boundary's
        # extrema should land e grid cells in from the edges of the resulting grid.
        rvec = np.linspace(rmin, rmax, nr)
        zvec = np.linspace(zmin, zmax, nz)
        e = 3.5
        m = nr - 1
        assert np.interp(e, np.arange(nr), rvec) == pytest.approx(rbdry.min(), abs=1.0e-8)
        assert np.interp(m - e, np.arange(nr), rvec) == pytest.approx(rbdry.max(), abs=1.0e-8)
        m = nz - 1
        assert np.interp(e, np.arange(nz), zvec) == pytest.approx(zbdry.min(), abs=1.0e-8)
        assert np.interp(m - e, np.arange(nz), zvec) == pytest.approx(zbdry.max(), abs=1.0e-8)

    def test_compute_grid_spacing(self, grid):
        rvec, zvec = grid
        hr, hrm1, hrm2, hz, hzm1, hzm2 = compute_grid_spacing(rvec, zvec)
        assert hr == pytest.approx((rvec[-1] - rvec[0]) / (len(rvec) - 1))
        assert hz == pytest.approx((zvec[-1] - zvec[0]) / (len(zvec) - 1))
        assert hrm1 == pytest.approx(1.0 / hr)
        assert hrm2 == pytest.approx((1.0 / hr) ** 2)
        assert hzm1 == pytest.approx(1.0 / hz)
        assert hzm2 == pytest.approx((1.0 / hz) ** 2)

    def test_generate_boundary_maps_partitions_grid(self, grid, boundary):
        rvec, zvec = grid
        rbdry, zbdry = boundary
        inout, ijin, ijout, ijedge = generate_boundary_maps(rvec, zvec, rbdry, zbdry)
        nrz = rvec.size * zvec.size
        assert inout.size == nrz
        # ijin and ijout must exactly partition every grid point.
        assert set(ijin.tolist()) | set(ijout.tolist()) == set(range(nrz))
        assert set(ijin.tolist()) & set(ijout.tolist()) == set()
        assert np.all(inout[ijin] > 0)
        assert np.all(inout[ijout] == 0)
        # Every edge point must have at least one of the four "neighbor outside" bits set.
        assert np.all((inout[ijedge] & 0b11110) > 0)
        # Rough area sanity check: interior point count times cell area should be in the
        # right ballpark of the true circle area (grid is coarse, so keep this loose).
        hr = (rvec[-1] - rvec[0]) / (len(rvec) - 1)
        hz = (zvec[-1] - zvec[0]) / (len(zvec) - 1)
        approx_area = ijin.size * hr * hz
        true_area = np.pi * A_BDRY ** 2
        assert approx_area == pytest.approx(true_area, rel=0.15)

    def test_generate_boundary_maps_left_flag_neighbor_is_actually_outside(self, grid, boundary):
        rvec, zvec = grid
        rbdry, zbdry = boundary
        inout, ijin, ijout, ijedge = generate_boundary_maps(rvec, zvec, rbdry, zbdry)
        left_flagged = ijedge[(inout[ijedge] & 0b10) > 0]
        assert left_flagged.size > 0
        assert np.all(inout[left_flagged - 1] == 0)

    def test_generate_finite_difference_grid_shapes_and_matrix(self, grid, boundary, fd_grid):
        rvec, zvec = grid
        nrz = rvec.size * zvec.size
        for key in ('s1', 's2', 's3', 's4', 's5', 'a1', 'a2', 'b1', 'b2'):
            assert fd_grid[key].shape == (nrz,)
        assert issparse(fd_grid['matrix'])
        assert fd_grid['matrix'].shape == (nrz, nrz)
        # Interior (non-edge) points use the textbook symmetric 5-point stencil,
        # so s1 (left) + s2 (right) should equal 2*hrm2/ss1 there (a1=a2=1 off the
        # edge means ss2=ss3=hrm2, so s1 and s2 individually equal hrm2/ss1).
        interior = np.setdiff1d(fd_grid['ijin'], fd_grid['ijedge'])
        assert interior.size > 0
        assert fd_grid['a1'][interior] == pytest.approx(1.0)
        assert fd_grid['a2'][interior] == pytest.approx(1.0)


class TestCoreGSFunctions:

    def test_compute_jtor_matches_formula(self):
        mu0 = 4.0e-7 * np.pi
        rpsi = np.array([1.5, 2.0, 2.5])
        ffprime = np.array([0.1, -0.2, 0.3])
        pprime = np.array([1.0e5, -2.0e5, 0.0])
        jtor = compute_jtor(rpsi, ffprime, pprime)
        expected = -1.0 * (ffprime / (mu0 * rpsi) + rpsi * pprime)
        assert jtor == pytest.approx(expected)

    def test_compute_jtor_zero_profiles_gives_zero_current(self):
        rpsi = np.array([1.0, 2.0, 3.0])
        assert compute_jtor(rpsi, 0.0, 0.0) == pytest.approx(np.zeros(3))

    def test_compute_psi_applies_solver_and_sign(self):
        s5 = np.array([2.0, 3.0, 4.0])
        current = np.array([1.0, -1.0, 0.5])
        solver = lambda x: 2.0 * x  # noqa: E731
        result = compute_psi(solver, s5, current)
        assert result == pytest.approx(-2.0 * (s5 * current))

    def test_compute_finite_difference_matrix_small_case(self):
        nr, nz = 2, 2
        nrz = nr * nz
        s1 = np.array([0.1, 0.2, 0.3, 0.4])
        s2 = np.array([0.5, 0.6, 0.7, 0.8])
        s3 = np.array([0.9, 1.0, 1.1, 1.2])
        s4 = np.array([1.3, 1.4, 1.5, 1.6])
        matrix = compute_finite_difference_matrix(nr, nz, s1, s2, s3, s4).toarray()
        expected = np.eye(nrz)
        for i in range(nrz):
            if i + 1 < nrz:
                expected[i, i + 1] -= s2[i]
            if i - 1 >= 0:
                expected[i, i - 1] -= s1[i]
            if i + nr < nrz:
                expected[i, i + nr] -= s4[i]
            if i - nr >= 0:
                expected[i, i - nr] -= s3[i]
        assert matrix == pytest.approx(expected)


class TestInitialPsi:

    def test_generate_initial_psi_is_nonpositive_and_deepest_at_center(self, grid, boundary):
        rvec, zvec = grid
        rbdry, zbdry = boundary
        _, ijin, _, _ = generate_boundary_maps(rvec, zvec, rbdry, zbdry)
        psi = generate_initial_psi(rvec, zvec, rbdry, zbdry, ijin)
        assert psi.shape == (zvec.size, rvec.size)
        flat = psi.ravel()
        assert np.all(flat.take(ijin) <= 1.0e-12)
        # The interior grid point nearest the boundary's centroid should be at
        # (or very near) the deepest point of the (1-rho^2)^1.2 bowl, i.e. -1.
        i0 = int(np.argmin(np.abs(rvec - R0)))
        j0 = int(np.argmin(np.abs(zvec - Z0)))
        assert psi[j0, i0] == pytest.approx(-1.0, abs=0.05)


class TestGradientAndGeometryHelpers:

    def test_compute_grad_psi_vector_from_2d_spline_matches_analytic_gradient(self, psi_spline):
        r, z = R0 + 0.3, Z0 - 0.2
        grad = compute_grad_psi_vector_from_2d_spline((r, z), psi_spline['tck'])
        expected = np.array([2.0 * (r - R0), 2.0 * (z - Z0)])
        assert grad == pytest.approx(expected, abs=1.0e-4)

    def test_find_extrema_with_taylor_expansion_locates_axis(self, grid, psi_grid_fd):
        rvec, zvec = grid
        r_ext, z_ext, psi_ext = find_extrema_with_taylor_expansion(rvec, zvec, psi_grid_fd)
        assert r_ext == pytest.approx(R0, abs=1.0e-6)
        assert z_ext == pytest.approx(Z0, abs=1.0e-6)
        assert psi_ext == pytest.approx(-(A_BDRY ** 2), abs=1.0e-6)

    def test_order_contour_points_by_angle_sorts_and_closes(self):
        rng = np.random.default_rng(0)
        theta = rng.permutation(np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False))
        r = R0 + A_BDRY * np.cos(theta)
        z = Z0 + A_BDRY * np.sin(theta)
        r_ord, z_ord, angle_ord, _ = order_contour_points_by_angle(r, z, R0, Z0)
        assert np.all(np.diff(angle_ord) > 0.0)
        assert r_ord[0] == pytest.approx(r_ord[-1])
        assert z_ord[0] == pytest.approx(z_ord[-1])
        assert angle_ord[-1] == pytest.approx(angle_ord[0] + 2.0 * np.pi)

    def test_compute_intersection_from_line_segment_complex_crossing_lines(self):
        # Two segments crossing at (0, 0): one along the real axis, one along the imaginary axis.
        p1s, p1e = complex(-1.0, 0.0), complex(1.0, 0.0)
        p2s, p2e = complex(0.0, -1.0), complex(0.0, 1.0)
        ta, tb = compute_intersection_from_line_segment_complex(p1s, p1e, p2s, p2e)
        assert ta == pytest.approx(0.5)
        assert tb == pytest.approx(0.5)
        crossing = p1s + ta * (p1e - p1s)
        assert crossing == pytest.approx(0.0 + 0.0j)

    def test_compute_intersection_from_line_segment_coordinates_matches_complex_version(self):
        ta, tb = compute_intersection_from_line_segment_coordinates(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0)
        assert ta == pytest.approx(0.5)
        assert tb == pytest.approx(0.5)

    def test_compute_intersection_from_line_segment_parallel_lines_returns_nan(self):
        ta, tb = compute_intersection_from_line_segment_complex(
            complex(0.0, 0.0), complex(1.0, 0.0), complex(0.0, 1.0), complex(1.0, 1.0)
        )
        assert np.isnan(ta)
        assert np.isnan(tb)

    def test_avoid_convex_curvature_pulls_point_toward_vertex_when_outside_contour(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 65)
        r_contour = R0 + A_BDRY * np.cos(theta)
        z_contour = Z0 + A_BDRY * np.sin(theta)
        # A point well outside the circle, straight out along +R from center.
        r_point = R0 + 2.0 * A_BDRY
        z_point = Z0
        r_new, z_new = avoid_convex_curvature(
            r_contour, z_contour, r_point, z_point, r_reference=R0, z_reference=Z0, r_vertex=R0, z_vertex=Z0
        )
        # The corrected point must be pulled back closer to the vertex (axis) than
        # the original, out-of-bounds point was.
        assert np.hypot(r_new - R0, z_new - Z0) < np.hypot(r_point - R0, z_point - Z0)

    def test_avoid_convex_curvature_leaves_interior_point_untouched(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 65)
        r_contour = R0 + A_BDRY * np.cos(theta)
        z_contour = Z0 + A_BDRY * np.sin(theta)
        r_point = R0 + 0.1
        z_point = Z0
        r_new, z_new = avoid_convex_curvature(
            r_contour, z_contour, r_point, z_point, r_reference=R0, z_reference=Z0, r_vertex=R0, z_vertex=Z0
        )
        assert r_new == pytest.approx(r_point)
        assert z_new == pytest.approx(z_point)


class TestBoundaryAndXPoints:

    def test_generate_boundary_splines_no_xpoints_gives_constant_radius(self, boundary):
        rbdry, zbdry = boundary
        splines = generate_boundary_splines(rbdry, zbdry, R0, Z0, [], enforce_concave=True)
        assert len(splines) == 1
        angles = np.linspace(splines[0]['bounds'][0] + 1.0e-3, splines[0]['bounds'][1] - 1.0e-3, 25)
        lengths = splev(angles, splines[0]['tck'])
        assert lengths == pytest.approx(A_BDRY, abs=1.0e-2)

    def test_generate_x_point_candidates_finds_none_for_convex_boundary(self, boundary, psi_spline):
        rbdry, zbdry = boundary
        candidates = generate_x_point_candidates(rbdry, zbdry, R0, Z0, psi_spline['tck'], dr=1.0e-3, dz=1.0e-3)
        assert candidates == []

    def test_generate_segments_returns_tangent_lines_spanning_indices(self, boundary):
        rbdry, zbdry = boundary
        r_ord, z_ord, _, _ = order_contour_points_by_angle(rbdry, zbdry, R0, Z0, close_contour=True)
        n = len(r_ord)
        # Quarter-circle-ish splits, mirroring the ~4 cardinal-point splits
        # generate_x_point_candidates naturally produces for a convex boundary;
        # a near-half-circle span makes the segment's local tangent-line fit
        # numerically degenerate.
        indices = [0, n // 4, n // 2, 3 * n // 4]
        lines = generate_segments(r_ord, z_ord, indices, cuts=None, r_reference=R0, z_reference=Z0)
        assert len(lines) == len(indices)
        for l0, l1 in lines:
            assert len(l0) == 2
            assert len(l1) == 2
            assert np.all(np.isfinite(np.array(l0, dtype=complex).view(float)))
            assert np.all(np.isfinite(np.array(l1, dtype=complex).view(float)))


class TestFluxSurfaceHelpers:

    def test_compute_flux_surface_quantities_matches_analytic_bpol(self, psi_spline):
        rho = 0.4
        theta = np.linspace(0.0, 2.0 * np.pi, 33)
        r_contour = R0 + rho * np.cos(theta)
        z_contour = Z0 + rho * np.sin(theta)
        fpol_tck = generate_bounded_1d_spline(np.array([5.0, 5.0, 5.0, 5.0]), symmetrical=False, smooth=False)['tck']
        out = compute_flux_surface_quantities(0.5, r_contour, z_contour, psi_tck=psi_spline['tck'], fpol_tck=fpol_tck)
        # |grad psi| = 2*rho exactly on this circle (radial gradient of (R-R0)^2+(Z-Z0)^2).
        expected_bpol = 2.0 * rho / r_contour
        assert out['bpol'] == pytest.approx(expected_bpol, rel=1.0e-3)
        assert out['fpol'] == pytest.approx(5.0)
        assert out['btor'] == pytest.approx(5.0 / r_contour)

    def test_compute_safety_factor_contour_integral_scales_with_fpol(self, psi_spline):
        rho = 0.4
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        r_contour = R0 + rho * np.cos(theta)
        z_contour = Z0 + rho * np.sin(theta)
        contour_1 = compute_flux_surface_quantities(0.5, r_contour, z_contour, psi_tck=psi_spline['tck'])
        contour_1['fpol'] = np.array([3.0])
        contour_2 = compute_flux_surface_quantities(0.5, r_contour, z_contour, psi_tck=psi_spline['tck'])
        contour_2['fpol'] = np.array([6.0])
        q1 = compute_safety_factor_contour_integral(contour_1)
        q2 = compute_safety_factor_contour_integral(contour_2)
        assert q1 > 0.0
        assert q2 == pytest.approx(2.0 * q1)


class TestContourTracing:

    def test_trace_contours_with_contourpy_recovers_circle(self, grid, psi_grid):
        rvec, zvec = grid
        level = 0.4 ** 2
        contours = trace_contours_with_contourpy(rvec, zvec, psi_grid, [level], R0, Z0)
        assert len(contours) == 1
        vertices = contours[float(level)]
        radii = np.hypot(vertices[:, 0] - R0, vertices[:, 1] - Z0)
        assert radii == pytest.approx(0.4, rel=0.05)

    def test_trace_contour_with_splines_recovers_circle(self, psi_spline, boundary):
        rbdry, zbdry = boundary
        boundary_splines = generate_boundary_splines(rbdry, zbdry, R0, Z0, [], enforce_concave=True)
        level = 0.4 ** 2
        rc, zc = trace_contour_with_splines(
            None, level, npoints=33, rmagx=R0, zmagx=Z0, psimagx=0.0, psibdry=A_BDRY ** 2,
            psi_tck=psi_spline['tck'], boundary_splines=boundary_splines, resolution=101,
        )
        radii = np.hypot(np.array(rc) - R0, np.array(zc) - Z0)
        assert radii == pytest.approx(0.4, rel=1.0e-2)

    def test_compute_adjusted_contour_resolution_scales_with_relative_size(self):
        r_boundary, z_boundary = circle_points(A_BDRY)
        r_small, z_small = circle_points(0.2)
        r_big, z_big = circle_points(0.7)
        n_small = compute_adjusted_contour_resolution(R0, Z0, r_boundary, z_boundary, r_small, z_small)
        n_big = compute_adjusted_contour_resolution(R0, Z0, r_boundary, z_boundary, r_big, z_big)
        assert n_small < n_big
        assert n_big <= 51
        assert n_small >= 21


class TestBoundaryGradients:

    def test_compute_gradients_at_boundary_matches_analytic_gradient(self, grid, boundary, psi_grid_fd, fd_grid):
        rvec, zvec = grid
        flat_psi = psi_grid_fd.ravel()
        rgradr, zgradr, gradr, rgradz, zgradz, gradz = compute_gradients_at_boundary(
            rvec, zvec, flat_psi, fd_grid['inout'], fd_grid['ijedge'], fd_grid['a1'], fd_grid['a2'], fd_grid['b1'], fd_grid['b2']
        )
        assert rgradr.size > 0
        assert rgradz.size > 0
        expected_gradr = 2.0 * (rgradr - R0)
        expected_gradz = 2.0 * (zgradz - Z0)
        # A handful of "sliver" cells (both left and right, or both above and below,
        # neighbors outside the boundary) use a cruder one-sided estimate with no
        # neighbor-based correction term, so allow more slack there than for the
        # corrected single-sided-out cells.
        assert gradr == pytest.approx(expected_gradr, rel=0.3, abs=0.15)
        assert gradz == pytest.approx(expected_gradz, rel=0.3, abs=0.15)

    def test_generate_boundary_gradient_spline_reproduces_points(self, grid, boundary, psi_grid_fd, fd_grid):
        rvec, zvec = grid
        flat_psi = psi_grid_fd.ravel()
        rgradr, zgradr, gradr, _, _, _ = compute_gradients_at_boundary(
            rvec, zvec, flat_psi, fd_grid['inout'], fd_grid['ijedge'], fd_grid['a1'], fd_grid['a2'], fd_grid['b1'], fd_grid['b2']
        )
        fit = generate_boundary_gradient_spline(rgradr, zgradr, gradr, R0, Z0, s=0)
        angles = np.mod(np.angle((rgradr - R0) + 1.0j * (zgradr - Z0)), 2.0 * np.pi)
        angles = np.where(angles > np.pi, angles - 2.0 * np.pi, angles)
        fitted = splev(angles, fit['tck'])
        assert fitted == pytest.approx(gradr, abs=0.2)


class TestPsiExtension:

    def test_compute_psi_extension_extrapolates_outward_gradient(self, grid, boundary, psi_grid_fd, fd_grid):
        rvec, zvec = grid
        rbdry, zbdry = boundary
        flat_psi = psi_grid_fd.ravel()
        rgradr, zgradr, gradr, rgradz, zgradz, gradz = compute_gradients_at_boundary(
            rvec, zvec, flat_psi, fd_grid['inout'], fd_grid['ijedge'], fd_grid['a1'], fd_grid['a2'], fd_grid['b1'], fd_grid['b2']
        )
        gradr_fit = generate_boundary_gradient_spline(rgradr, zgradr, gradr, R0, Z0, s=0)
        gradz_fit = generate_boundary_gradient_spline(rgradz, zgradz, gradz, R0, Z0, s=0)
        psi_ext = compute_psi_extension(
            rvec, zvec, rbdry, zbdry, R0, Z0, flat_psi.reshape(zvec.size, rvec.size).copy(),
            fd_grid['ijout'], gradr_fit['tck'], gradz_fit['tck'],
        )
        flat_ext = psi_ext.ravel()
        assert np.all(np.isfinite(flat_ext.take(fd_grid['ijout'])))
        j_out = fd_grid['ijout'] // rvec.size
        i_out = fd_grid['ijout'] - rvec.size * j_out
        r_out = rvec.take(i_out)
        z_out = zvec.take(j_out)
        rho_out = np.hypot(r_out - R0, z_out - Z0)
        # psi is 0 exactly at the boundary and grows radially outward with gradient
        # magnitude 2*A_BDRY there; compute_psi_extension is a linear (gradient-based)
        # extrapolation, so close to the boundary it should track 2*A_BDRY*(rho-A_BDRY).
        close = (rho_out > A_BDRY) & (rho_out < A_BDRY * 1.1)
        assert close.sum() > 0
        expected_close = 2.0 * A_BDRY * (rho_out[close] - A_BDRY)
        assert flat_ext.take(fd_grid['ijout'])[close] == pytest.approx(expected_close, rel=0.3, abs=0.05)
        # And it should keep growing (not fold back toward zero) further out.
        far = (rho_out >= A_BDRY * 1.1) & (rho_out < A_BDRY * 1.4)
        if far.sum() > 0:
            assert np.all(flat_ext.take(fd_grid['ijout'])[far] > 0.0)

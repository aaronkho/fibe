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


#class TestMathematicalFunctions():

#    def test_optimized_grid():
#        generate_optimal_grid()

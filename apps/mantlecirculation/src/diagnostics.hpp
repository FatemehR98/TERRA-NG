#pragma once

#include "communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/solvers/diagonal_solver.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector_q1.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "parameters.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace terra::mantlecirculation {

/// @brief Compute ∫_Γ ∇T · n̂ dΓ on the surface or CMB boundary.
inline ScalarType compute_boundary_heat_flux_integral(
    const grid::shell::DistributedDomain&                           domain,
    const grid::Grid4DDataScalar< ScalarType >&                     T_grid,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
    const bool                                                      at_surface )
{
    using namespace fe::wedge;

    const int num_subdomains = domain.subdomains().size();
    const int nx             = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int nr             = domain.domain_info().subdomain_num_nodes_radially();

    const int  r_cell          = at_surface ? ( nr - 2 ) : 0;
    const int  r_boundary_node = at_surface ? ( nr - 1 ) : 0;
    const auto expected_flag =
        at_surface ? grid::shell::ShellBoundaryFlag::SURFACE : grid::shell::ShellBoundaryFlag::CMB;
    const ScalarType zeta_boundary = at_surface ? ScalarType( 1 ) : ScalarType( -1 );
    const ScalarType normal_sign   = at_surface ? ScalarType( 1 ) : ScalarType( -1 );

    ScalarType local_integral = 0;

    Kokkos::parallel_reduce(
        "nusselt_surface_integral",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_subdomains, nx - 1, nx - 1 } ),
        KOKKOS_LAMBDA( const int sd, const int x_cell, const int y_cell, ScalarType& sum ) {
            // Skip subdomains that are not at the actual boundary (radial subdomain decomposition).
            if ( boundary_mask( sd, x_cell, y_cell, r_boundary_node ) != expected_flag )
                return;
            // Skip cells whose anchor node is not owned to avoid double-counting at lateral subdomain boundaries.
            if ( ownership_mask( sd, x_cell, y_cell, r_cell ) != grid::NodeOwnershipFlag::OWNED )
                return;
            constexpr int               nqp = quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[nqp];
            ScalarType                  quad_weights[nqp];
            quadrature::quad_felippa_3x2_quad_points( quad_points );
            quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            dense::Vec< ScalarType, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, coords_shell, sd, x_cell, y_cell );

            const ScalarType r_1 = coords_radii( sd, r_cell );
            const ScalarType r_2 = coords_radii( sd, r_cell + 1 );

            dense::Vec< ScalarType, 6 > local_T[num_wedges_per_hex_cell] = {};
            extract_local_wedge_scalar_coefficients( local_T, sd, x_cell, y_cell, r_cell, T_grid );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < nqp; ++q )
                {
                    dense::Vec< ScalarType, 3 > qp = quad_points[q];
                    qp( 2 )                        = zeta_boundary;

                    const auto J       = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                    const auto det     = J.det();
                    const auto abs_det = Kokkos::abs( det );

                    if ( abs_det < ScalarType( 1e-20 ) )
                        continue;

                    const auto J_inv_T = J.inv().transposed();

                    dense::Vec< ScalarType, 3 > grad_T_phys{ 0, 0, 0 };
                    for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    {
                        const auto grad_phi_ref = grad_shape< ScalarType >( i, qp );
                        grad_T_phys             = grad_T_phys + ( J_inv_T * grad_phi_ref ) * local_T[wedge]( i );
                    }

                    const auto x_phys = forward_map(
                        wedge_phy_surf[wedge][0],
                        wedge_phy_surf[wedge][1],
                        wedge_phy_surf[wedge][2],
                        r_1,
                        r_2,
                        qp( 0 ),
                        qp( 1 ),
                        qp( 2 ) );
                    const auto r_hat = x_phys.normalized();

                    const ScalarType grad_T_dot_n = normal_sign * grad_T_phys.dot( r_hat );

                    const ScalarType radial_scale = ( r_2 - r_1 ) / ScalarType( 2 );
                    // The 3×2 tensor-product quadrature has 2 radial points that collapse
                    // to the same (ξ,η) when ζ is fixed to the boundary, so divide by 2.
                    const ScalarType dA = abs_det * quad_weights[q] / radial_scale / ScalarType( 2 );

                    sum += grad_T_dot_n * dA;
                }
            }
        },
        Kokkos::Sum< ScalarType >( local_integral ) );

    Kokkos::fence();

    ScalarType global_integral = 0;
    MPI_Allreduce( &local_integral, &global_integral, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return global_integral;
}

/// @brief Compute the Nusselt number at the top (surface) or bottom (CMB) boundary.
///
/// Nu = ∫_Γ ∇T · n̂ dΓ  /  ∫_Γ ∇T_ref · n̂ dΓ
///
/// The denominator is computed numerically from the reference conductive profile T_ref,
/// matching the approach used in HyTeG (Ilangovan et al.).
///
/// @param at_surface  If true, compute Nu at the outer surface; if false, at the CMB.
inline ScalarType compute_nusselt(
    const grid::shell::DistributedDomain&                           domain,
    const linalg::VectorQ1Scalar< ScalarType >&                     T,
    const linalg::VectorQ1Scalar< ScalarType >&                     T_ref,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
    const bool                                                      at_surface )
{
    const ScalarType numerator = compute_boundary_heat_flux_integral(
        domain, T.grid_data(), coords_shell, coords_radii, boundary_mask, ownership_mask, at_surface );
    const ScalarType denominator = compute_boundary_heat_flux_integral(
        domain, T_ref.grid_data(), coords_shell, coords_radii, boundary_mask, ownership_mask, at_surface );

    return Kokkos::abs( numerator ) / Kokkos::abs( denominator );
}

/// @brief Volume-averaged root-mean-square velocity over the shell.
///
/// V_rms = sqrt( ∫_Ω |u|² dV / ∫_Ω dV )
///
/// Integrated with the Felippa 3×2 wedge quadrature on each hex cell.  Cells
/// are not shared between ranks (only nodes are), so each rank's contribution
/// is a partition of the global volume integral; the two partial sums are
/// MPI_Allreduce'd before the final ratio.
inline ScalarType compute_v_rms(
    const grid::shell::DistributedDomain&       domain,
    const linalg::VectorQ1Vec< ScalarType, 3 >& velocity,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii )
{
    using namespace fe::wedge;

    const auto u_data   = velocity.grid_data();
    const auto grid_lat = coords_shell;
    const auto radii_v  = coords_radii;

    constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

    ScalarType sum_u2_dV = 0;
    ScalarType sum_dV    = 0;

    Kokkos::parallel_reduce(
        "v_rms_volume_int",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int id, int xc, int yc, int rc, ScalarType& acc_u2, ScalarType& acc_V ) {
            dense::Vec< ScalarType, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_lat, id, xc, yc );

            const ScalarType r_1 = radii_v( id, rc );
            const ScalarType r_2 = radii_v( id, rc + 1 );

            dense::Vec< ScalarType, 3 > qp[num_q];
            ScalarType                  qw[num_q];
            quadrature::quad_felippa_3x2_quad_points( qp );
            quadrature::quad_felippa_3x2_quad_weights( qw );

            dense::Vec< ScalarType, num_nodes_per_wedge > u_w[num_wedges_per_hex_cell][3];
            for ( int d = 0; d < 3; ++d )
            {
                dense::Vec< ScalarType, num_nodes_per_wedge > comp[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients( comp, id, xc, yc, rc, d, u_data );
                u_w[0][d] = comp[0];
                u_w[1][d] = comp[1];
            }

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_q; ++q )
                {
                    const dense::Mat< ScalarType, 3, 3 > J       = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const ScalarType                     abs_det = Kokkos::abs( J.det() );
                    if ( abs_det < ScalarType( 1e-30 ) )
                    {
                        continue;
                    }

                    ScalarType uq[3] = { 0, 0, 0 };
                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                    {
                        const ScalarType N_j = shape( j, qp[q] );
                        uq[0] += N_j * u_w[wedge][0]( j );
                        uq[1] += N_j * u_w[wedge][1]( j );
                        uq[2] += N_j * u_w[wedge][2]( j );
                    }
                    const ScalarType u_mag2 = uq[0] * uq[0] + uq[1] * uq[1] + uq[2] * uq[2];

                    acc_u2 += qw[q] * abs_det * u_mag2;
                    acc_V += qw[q] * abs_det;
                }
            }
        },
        sum_u2_dV,
        sum_dV );
    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum_u2_dV, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &sum_dV, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return ( sum_dV > ScalarType( 0 ) ) ? Kokkos::sqrt( sum_u2_dV / sum_dV ) : ScalarType( 0 );
}

/// @brief Returns true if node (sd, x, y, r) lies on the surface or the CMB.
///
/// Only the outermost radial layer of a subdomain can be on the surface, only the innermost one on the CMB, so
/// nodes on interfaces between radial subdomains are not counted as boundary nodes.
KOKKOS_INLINE_FUNCTION bool is_cbf_boundary_node(
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const int                                                       sd,
    const int                                                       x,
    const int                                                       y,
    const int                                                       r,
    const int                                                       nr )
{
    return ( r == nr - 1 && boundary_mask( sd, x, y, r ) == grid::shell::ShellBoundaryFlag::SURFACE ) ||
           ( r == 0 && boundary_mask( sd, x, y, r ) == grid::shell::ShellBoundaryFlag::CMB );
}

/// @brief Adds the consistent boundary mass matrix  M_ij = ∫_Γ φ_i φ_j dΓ  of the surface or CMB applied to `src`
///        (or, with `diagonal_only`, the diagonal of M) to the boundary layer of `dst`.
///
/// The boundary faces of the wedge mesh are flat triangles at a fixed radius, so the P1 element matrix is exact:
/// M_tri = |tri| / 12 * [[2, 1, 1], [1, 2, 1], [1, 1, 2]]. Only the local contributions are added; nodes on
/// subdomain interfaces must afterwards be summed with an additive communication.
inline void add_consistent_boundary_mass(
    grid::Grid4DDataScalar< ScalarType >&                           dst,
    const grid::Grid4DDataScalar< ScalarType >&                     src,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const bool                                                      at_surface,
    const bool                                                      diagonal_only )
{
    const int num_subdomains = static_cast< int >( dst.extent( 0 ) );
    const int nx             = static_cast< int >( dst.extent( 1 ) );
    const int nr             = static_cast< int >( dst.extent( 3 ) );

    const int  r_b = at_surface ? ( nr - 1 ) : 0;
    const auto expected_flag =
        at_surface ? grid::shell::ShellBoundaryFlag::SURFACE : grid::shell::ShellBoundaryFlag::CMB;

    Kokkos::parallel_for(
        "cbf_consistent_boundary_mass",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_subdomains, nx - 1, nx - 1 } ),
        KOKKOS_LAMBDA( const int sd, const int xc, const int yc ) {
            if ( boundary_mask( sd, xc, yc, r_b ) != expected_flag )
                return;

            // Boundary triangles of the two wedges of this hex cell (same node order as
            // fe::wedge::extract_local_wedge_scalar_coefficients()).
            const int tri_x[2][3] = { { xc, xc + 1, xc }, { xc + 1, xc, xc + 1 } };
            const int tri_y[2][3] = { { yc, yc, yc + 1 }, { yc + 1, yc + 1, yc } };

            for ( int w = 0; w < 2; ++w )
            {
                const auto p0 = grid::shell::coords( sd, tri_x[w][0], tri_y[w][0], r_b, coords_shell, coords_radii );
                const auto p1 = grid::shell::coords( sd, tri_x[w][1], tri_y[w][1], r_b, coords_shell, coords_radii );
                const auto p2 = grid::shell::coords( sd, tri_x[w][2], tri_y[w][2], r_b, coords_shell, coords_radii );

                // |tri| / 12
                const ScalarType area_12 = ( p1 - p0 ).cross( p2 - p0 ).norm() / ScalarType( 24 );

                if ( diagonal_only )
                {
                    for ( int i = 0; i < 3; ++i )
                        Kokkos::atomic_add( &dst( sd, tri_x[w][i], tri_y[w][i], r_b ), ScalarType( 2 ) * area_12 );
                    continue;
                }

                ScalarType s[3];
                for ( int j = 0; j < 3; ++j )
                    s[j] = src( sd, tri_x[w][j], tri_y[w][j], r_b );
                const ScalarType s_sum = s[0] + s[1] + s[2];

                // (M_tri s)_i = |tri| / 12 * ( s_i + Σ_j s_j )
                for ( int i = 0; i < 3; ++i )
                    Kokkos::atomic_add( &dst( sd, tri_x[w][i], tri_y[w][i], r_b ), area_12 * ( s[i] + s_sum ) );
            }
        } );
}

/// @brief dst_i = src_i (or 1 with `diagonal_only`) for all nodes that are not on the surface or the CMB
///        (identity rows of the CBF operator).
inline void set_cbf_identity_rows(
    grid::Grid4DDataScalar< ScalarType >&                           dst,
    const grid::Grid4DDataScalar< ScalarType >&                     src,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const bool                                                      diagonal_only )
{
    const int num_subdomains = static_cast< int >( dst.extent( 0 ) );
    const int nx             = static_cast< int >( dst.extent( 1 ) );
    const int nr             = static_cast< int >( dst.extent( 3 ) );

    Kokkos::parallel_for(
        "cbf_identity_rows",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_subdomains, nx, nx, nr } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            if ( !is_cbf_boundary_node( boundary_mask, sd, x, y, r, nr ) )
                dst( sd, x, y, r ) = diagonal_only ? ScalarType( 1 ) : src( sd, x, y, r );
        } );
}

/// @brief Right-hand side of the CBF system: b_i = R_i · n_i on surface and CMB nodes (outward normal n = ±r̂),
///        0 elsewhere.
inline void compute_cbf_rhs(
    grid::Grid4DDataScalar< ScalarType >&                           rhs,
    const grid::Grid4DDataVec< ScalarType, 3 >&                     residual,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask )
{
    const int num_subdomains = static_cast< int >( rhs.extent( 0 ) );
    const int nx             = static_cast< int >( rhs.extent( 1 ) );
    const int nr             = static_cast< int >( rhs.extent( 3 ) );

    Kokkos::parallel_for(
        "cbf_rhs",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_subdomains, nx, nx, nr } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            if ( !is_cbf_boundary_node( boundary_mask, sd, x, y, r, nr ) )
            {
                rhs( sd, x, y, r ) = ScalarType( 0 );
                return;
            }

            const ScalarType normal_sign = ( r == nr - 1 ) ? ScalarType( 1 ) : ScalarType( -1 );

            dense::Vec< ScalarType, 3 > r_hat;
            for ( int d = 0; d < 3; ++d )
                r_hat( d ) = coords_shell( sd, x, y, d );
            r_hat = r_hat.normalized();

            ScalarType R_dot_n = 0;
            for ( int d = 0; d < 3; ++d )
                R_dot_n += residual( sd, x, y, r, d ) * r_hat( d );

            rhs( sd, x, y, r ) = normal_sign * R_dot_n;
        } );
}

/// @brief Copies the surface or CMB layer of the CBF solution into a (subdomain, x, y) field. Entries of
///        subdomains not on that boundary are set to 0.
inline void extract_cbf_boundary_layer(
    grid::Grid3DDataScalar< ScalarType >&                           sigma_rr,
    const grid::Grid4DDataScalar< ScalarType >&                     solution,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const bool                                                      at_surface )
{
    const int num_subdomains = static_cast< int >( solution.extent( 0 ) );
    const int nx             = static_cast< int >( solution.extent( 1 ) );
    const int nr             = static_cast< int >( solution.extent( 3 ) );

    const int  r_b = at_surface ? ( nr - 1 ) : 0;
    const auto expected_flag =
        at_surface ? grid::shell::ShellBoundaryFlag::SURFACE : grid::shell::ShellBoundaryFlag::CMB;

    Kokkos::parallel_for(
        "cbf_extract_boundary_layer",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_subdomains, nx, nx } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y ) {
            sigma_rr( sd, x, y ) =
                ( boundary_mask( sd, x, y, r_b ) == expected_flag ) ? solution( sd, x, y, r_b ) : ScalarType( 0 );
        } );
}

/// @brief Consistent boundary mass operator of the CBF method, acting on Q1 scalar vectors.
///
///   (A s)_i = Σ_j M_ij s_j    for nodes i on the surface or the CMB  (M_ij = ∫_Γ φ_i φ_j dΓ, Γ = surface ∪ CMB)
///   (A s)_i = s_i             for all other nodes
///
/// Surface and CMB do not share nodes, so A is block diagonal and SPD, and one CG solve handles both boundaries.
/// The identity rows decouple the interior (which then simply stays zero), so standard Q1 vectors and the
/// library PCG can be used. Unlike fe::wedge::operators::shell::BoundaryMass this also works with radial
/// subdomain decomposition.
class CBFBoundaryMassOperator
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarType >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarType >;
    using ScalarType    = mantlecirculation::ScalarType;

    CBFBoundaryMassOperator(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask )
    : domain_( domain )
    , coords_shell_( coords_shell )
    , coords_radii_( coords_radii )
    , boundary_mask_( boundary_mask )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) { apply_or_diagonal( src, dst, false ); }

    /// @brief Writes the diagonal of the operator into `diag` (for Jacobi preconditioning).
    void diagonal( DstVectorType& diag ) { apply_or_diagonal( diag, diag, true ); }

  private:
    void apply_or_diagonal( const SrcVectorType& src, DstVectorType& dst, const bool diagonal_only )
    {
        auto dst_data = dst.grid_data();

        // With diagonal_only, src is not read, so src may alias dst.
        Kokkos::deep_copy( dst_data, ScalarType( 0 ) );
        for ( const bool at_surface : { true, false } )
            add_consistent_boundary_mass(
                dst_data, src.grid_data(), coords_shell_, coords_radii_, boundary_mask_, at_surface, diagonal_only );
        Kokkos::fence();

        communication::shell::send_recv(
            domain_, dst_data, send_buffers_, recv_buffers_, communication::CommunicationReduction::SUM );

        // After the additive communication, so that the identity rows are not summed over subdomain copies.
        set_cbf_identity_rows( dst_data, src.grid_data(), boundary_mask_, diagonal_only );
        Kokkos::fence();
    }

    grid::shell::DistributedDomain                           domain_;
    grid::Grid3DDataVec< ScalarType, 3 >                     coords_shell_;
    grid::Grid2DDataScalar< ScalarType >                     coords_radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > recv_buffers_;
};

static_assert( linalg::OperatorLike< CBFBoundaryMassOperator > );

/// @brief Nodal radial stress σ_rr = r̂ · σ · r̂ on the surface and the CMB via the consistent boundary flux
///        (CBF) method (Zhong, Gurnis & Hulbert 1993).
///
/// Instead of differentiating the FE velocity at the boundary (which is only O(h) accurate and ignores the
/// body force), the traction is recovered from the momentum residual of the discrete solution:
///
///   R_i = ( A u + B^T p - M b )_i = ∫_Γ (σ n) · φ_i dΓ     (boundary node i, outward normal n)
///
/// (see StokesContext::compute_momentum_residual()). Writing σ_rr in the P1 boundary space and testing with φ_i
/// gives the consistent boundary mass system
///
///   Σ_j M_ij σ_rr,j = R_i · n_i,     M_ij = ∫_Γ φ_i φ_j dΓ,
///
/// which is solved with Jacobi-preconditioned CG (CBFBoundaryMassOperator). For P1 mass matrices the Jacobi
/// preconditioned condition number is bounded independently of the mesh size, so CG needs only a few tens of
/// iterations at any resolution. Since n = ±r̂ enters twice in n·σ·n, the result is r̂·σ·r̂ on both boundaries
/// (positive = tension, σ includes -p).
///
/// The operator diagonal depends on the geometry only and is assembled once on construction. compute() is
/// meant to be called only on output steps. Everything runs in the default execution space.
///
/// Results are stored per subdomain as (subdomain, x, y). Entries of subdomains that do not touch the
/// respective boundary (radial subdomain decomposition) are zero.
///
/// The kernels live in free functions: CUDA does not allow extended (device) lambdas in constructors or in
/// private member functions.
///
/// Not copyable or movable: the preconditioner of the CG solver references the member diag_.
class BoundaryRadialStressCBF
{
  public:
    using MassOperator   = CBFBoundaryMassOperator;
    using Preconditioner = linalg::solvers::DiagonalSolver< MassOperator >;
    using Solver         = linalg::solvers::PCG< MassOperator, Preconditioner >;

    BoundaryRadialStressCBF(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
        const int                                                       max_iterations     = 200,
        const double                                                    relative_tolerance = 1e-10 )
    : coords_shell_( coords_shell )
    , boundary_mask_( boundary_mask )
    , mass_( domain, coords_shell, coords_radii, boundary_mask )
    , rhs_( "cbf_rhs", domain, ownership_mask )
    , solution_( "cbf_solution", domain, ownership_mask )
    , diag_( "cbf_inv_diag", domain, ownership_mask )
    , cg_statistics_( std::make_shared< util::Table >() )
    {
        const int num_subdomains = static_cast< int >( domain.subdomains().size() );
        const int nx             = domain.domain_info().subdomain_num_nodes_per_side_laterally();

        // Jacobi preconditioner. DiagonalSolver inverts diag_ in place on construction.
        mass_.diagonal( diag_ );

        std::vector< linalg::VectorQ1Scalar< ScalarType > > tmps;
        for ( int i = 0; i < 4; ++i )
            tmps.emplace_back( "cbf_cg_tmp_" + std::to_string( i ), domain, ownership_mask );

        solver_ = std::make_unique< Solver >(
            linalg::solvers::IterativeSolverParameters{ max_iterations, relative_tolerance, 1e-30 },
            cg_statistics_,
            tmps,
            Preconditioner( diag_ ) );
        solver_->set_tag( "cbf_boundary_mass_cg" );

        sigma_rr_surface_ = grid::Grid3DDataScalar< ScalarType >( "sigma_rr_surface", num_subdomains, nx, nx );
        sigma_rr_cmb_     = grid::Grid3DDataScalar< ScalarType >( "sigma_rr_cmb", num_subdomains, nx, nx );
    }

    BoundaryRadialStressCBF( const BoundaryRadialStressCBF& )            = delete;
    BoundaryRadialStressCBF& operator=( const BoundaryRadialStressCBF& ) = delete;
    BoundaryRadialStressCBF( BoundaryRadialStressCBF&& )                 = delete;
    BoundaryRadialStressCBF& operator=( BoundaryRadialStressCBF&& )      = delete;

    /// @brief Computes σ_rr on both boundaries from the momentum residual R (see class description).
    void compute( const linalg::VectorQ1Vec< ScalarType >& residual )
    {
        compute_cbf_rhs( rhs_.grid_data(), residual.grid_data(), coords_shell_, boundary_mask_ );
        Kokkos::fence();

        linalg::assign( solution_, ScalarType( 0 ) );
        cg_statistics_->clear();
        linalg::solvers::solve( *solver_, mass_, solution_, rhs_ );

        log_cg_convergence();

        extract_cbf_boundary_layer( sigma_rr_surface_, solution_.grid_data(), boundary_mask_, true );
        extract_cbf_boundary_layer( sigma_rr_cmb_, solution_.grid_data(), boundary_mask_, false );
        Kokkos::fence();
    }

    const grid::Grid3DDataScalar< ScalarType >& sigma_rr_surface() const { return sigma_rr_surface_; }
    const grid::Grid3DDataScalar< ScalarType >& sigma_rr_cmb() const { return sigma_rr_cmb_; }

  private:
    void log_cg_convergence() const
    {
        const auto& rows = cg_statistics_->rows();
        if ( rows.empty() )
            return;

        const auto& last = rows.back();
        const auto  it   = last.find( "iteration" );
        const auto  rr   = last.find( "relative_residual" );
        if ( it == last.end() || rr == last.end() )
            return;

        const int*    iterations = std::get_if< int >( &it->second );
        const double* rel_res    = std::get_if< double >( &rr->second );
        util::logroot << "CBF consistent boundary mass CG: " << ( iterations ? *iterations : -1 )
                      << " iterations, relative residual " << ( rel_res ? *rel_res : -1.0 ) << std::endl;
    }

    grid::Grid3DDataVec< ScalarType, 3 >                     coords_shell_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    MassOperator                         mass_;
    linalg::VectorQ1Scalar< ScalarType > rhs_;
    linalg::VectorQ1Scalar< ScalarType > solution_;
    linalg::VectorQ1Scalar< ScalarType > diag_;

    std::shared_ptr< util::Table > cg_statistics_;
    std::unique_ptr< Solver >      solver_;

    grid::Grid3DDataScalar< ScalarType > sigma_rr_surface_;
    grid::Grid3DDataScalar< ScalarType > sigma_rr_cmb_;
};

} // namespace terra::mantlecirculation

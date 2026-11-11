#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::geophysics::viscosity {

/// @brief Recovers the second invariant of the (deviatoric) strain rate of a W1 velocity field as a W1 nodal field.
///
/// The strain rate \f$ \dot\varepsilon(\mathbf{u}) = \tfrac{1}{2} ( \nabla \mathbf{u} + \nabla \mathbf{u}^T ) \f$ of a
/// piecewise (bi)linear velocity is discontinuous across elements, whereas the viscosity is stored at the nodes.
/// This class therefore computes a lumped-mass \f$ L^2 \f$ projection of the squared second invariant
///
/// \f[
///   \dot\varepsilon_{II}^2 = \tfrac{1}{2} \, \dot\varepsilon' : \dot\varepsilon' , \qquad
///   \dot\varepsilon' = \dot\varepsilon - \tfrac{1}{3} ( \nabla \cdot \mathbf{u} ) I ,
/// \f]
///
/// onto the nodes,
///
/// \f[
///   ( \dot\varepsilon_{II}^2 )_i = \frac{ \int_\Omega \dot\varepsilon_{II}^2 \, \phi_i \, \mathrm{d}x }
///                                       { \int_\Omega \phi_i \, \mathrm{d}x } ,
/// \f]
///
/// and returns its square root. Since the shape functions are non-negative at the quadrature points, the nodal
/// values are a positively weighted average of the surrounding quadrature-point values, i.e., the recovered
/// invariant is non-negative and bounded by the element-wise extrema.
///
/// The lumped mass \f$ \int_\Omega \phi_i \f$ only depends on the geometry and is assembled once in the constructor.
template < typename ScalarT >
class StrainRateInvariant
{
  public:
    using ScalarType = ScalarT;

    /// @param domain  distributed domain of the velocity field
    /// @param grid    unit sphere node coordinates, see grid::shell::subdomain_unit_sphere_single_shell_coords()
    /// @param radii   shell radii, see grid::shell::subdomain_shell_radii()
    StrainRateInvariant(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    , lumped_mass_(
          "strain_rate_lumped_mass",
          domain.subdomains().size(),
          domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          domain.domain_info().subdomain_num_nodes_radially() )
    {
        assemble_lumped_mass();
    }

    /// @brief Computes the nodal second invariant of the deviatoric strain rate of the passed velocity.
    ///
    /// @param velocity        [in]  W1 velocity field
    /// @param strain_rate_II  [out] nodal second invariant \f$ \dot\varepsilon_{II} \geq 0 \f$
    void compute( const linalg::VectorW1Vec< ScalarT, 3 >& velocity, linalg::VectorW1Scalar< ScalarT >& strain_rate_II )
    {
        project( velocity, strain_rate_II.grid_data() );
    }

    /// @brief Relaxed update of a nodal strain rate invariant (for fixed-point iterations on the viscosity).
    ///
    /// Computes the invariant of the passed velocity and blends it geometrically into the passed field,
    ///
    /// \f[ \dot\varepsilon_{II} \leftarrow \dot\varepsilon_{II}^{\,1-\omega} \;
    ///      \dot\varepsilon_{II}(\mathbf{u})^{\,\omega} , \f]
    ///
    /// i.e., a relaxation of \f$ \log \dot\varepsilon_{II} \f$ with factor \f$ \omega \f$
    /// (\f$ \omega = 1 \f$: plain update as in compute(), \f$ \omega < 1 \f$: damping,
    /// \f$ \omega > 1 \f$: over-relaxation). Nodes at which either value vanishes get the plain update.
    ///
    /// @param velocity        [in]     W1 velocity field
    /// @param strain_rate_II  [in/out] nodal second invariant to update
    /// @param relaxation      [in]     relaxation factor \f$ \omega \f$
    /// @param tmp             [out]    scratch field, holds the invariant of the passed velocity on return
    void compute_relaxed(
        const linalg::VectorW1Vec< ScalarT, 3 >& velocity,
        linalg::VectorW1Scalar< ScalarT >&       strain_rate_II,
        const ScalarT                            relaxation,
        linalg::VectorW1Scalar< ScalarT >&       tmp )
    {
        project( velocity, tmp.grid_data() );

        const auto computed = tmp.grid_data();
        const auto relaxed  = strain_rate_II.grid_data();
        Kokkos::parallel_for(
            "strain_rate_relax",
            grid::shell::local_domain_md_range_policy_nodes( domain_ ),
            KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
                const ScalarT e_new = computed( id, x, y, r );
                const ScalarT e_old = relaxed( id, x, y, r );
                if ( e_new > ScalarT( 0 ) && e_old > ScalarT( 0 ) )
                {
                    relaxed( id, x, y, r ) = e_old * Kokkos::pow( e_new / e_old, relaxation );
                }
                else
                {
                    relaxed( id, x, y, r ) = e_new;
                }
            } );
        Kokkos::fence();
    }

    /// @brief Projection kernel driver: writes the nodal invariant of the passed velocity into @p dst.
    ///        Public only because of device-lambda restrictions, use compute() / compute_relaxed().
    void project( const linalg::VectorW1Vec< ScalarT, 3 >& velocity, const grid::Grid4DDataScalar< ScalarT >& dst_grid )
    {
        vel_ = velocity.grid_data();
        dst_ = dst_grid;

        kernels::common::set_constant( dst_, ScalarT( 0 ) );
        run_kernel_and_communicate();

        // Note: device lambdas must live in public member functions (CUDA restriction).
        const auto dst         = dst_;
        const auto lumped_mass = lumped_mass_;
        Kokkos::parallel_for(
            "strain_rate_scale_and_sqrt",
            grid::shell::local_domain_md_range_policy_nodes( domain_ ),
            KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
                const ScalarT squared = dst( id, x, y, r ) / lumped_mass( id, x, y, r );
                dst( id, x, y, r )    = Kokkos::sqrt( Kokkos::max( squared, ScalarT( 0 ) ) );
            } );
        Kokkos::fence();
    }

    /// @brief Kokkos kernel: per-cell contribution to either the lumped mass or to the projection's right-hand side.
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        using fe::wedge::num_nodes_per_wedge;
        using fe::wedge::num_nodes_per_wedge_surface;
        using fe::wedge::num_wedges_per_hex_cell;

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        constexpr int num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
        fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        fe::wedge::wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< ScalarT, 6 > vel_coeffs[3][num_wedges_per_hex_cell];
        if ( !assemble_mass_ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                fe::wedge::extract_local_wedge_vector_coefficients(
                    vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_ );
            }
        }

        for ( int q = 0; q < num_quad_points; q++ )
        {
            const auto w  = quad_weights[q];
            const auto qp = quad_points[q];

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const auto J   = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                const auto det = Kokkos::abs( J.det() );

                // Lumped mass: integrand is 1. Projection: integrand is the squared second invariant.
                ScalarT integrand = ScalarT( 1 );

                if ( !assemble_mass_ )
                {
                    const auto J_inv_transposed = J.inv().transposed();

                    // du[a][b] = d u_a / d x_b
                    ScalarT du[3][3] = {};

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto grad_j = J_inv_transposed * fe::wedge::grad_shape( j, qp );

                        for ( int a = 0; a < 3; a++ )
                        {
                            for ( int b = 0; b < 3; b++ )
                            {
                                du[a][b] += grad_j( b ) * vel_coeffs[a][wedge]( j );
                            }
                        }
                    }

                    const ScalarT div_u = du[0][0] + du[1][1] + du[2][2];

                    // Frobenius norm squared of the deviatoric strain rate.
                    ScalarT eps_dev_sq = ScalarT( 0 );
                    for ( int a = 0; a < 3; a++ )
                    {
                        for ( int b = 0; b < 3; b++ )
                        {
                            ScalarT eps_ab = ScalarT( 0.5 ) * ( du[a][b] + du[b][a] );
                            if ( a == b )
                            {
                                eps_ab -= div_u / ScalarT( 3 );
                            }
                            eps_dev_sq += eps_ab * eps_ab;
                        }
                    }

                    integrand = ScalarT( 0.5 ) * eps_dev_sq;
                }

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    dst[wedge]( i ) += w * integrand * fe::wedge::shape( i, qp ) * det;
                }
            }
        }

        fe::wedge::atomically_add_local_wedge_scalar_coefficients(
            dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }

  private:
    /// Assembles the lumped mass and sums up the contributions of neighboring subdomains.
    void assemble_lumped_mass()
    {
        assemble_mass_ = true;
        dst_           = lumped_mass_;
        run_kernel_and_communicate();
        assemble_mass_ = false;
    }

    /// Runs the cell kernel (accumulating into dst_, which must be zero on entry) and adds up the contributions
    /// of all subdomains sharing a node.
    void run_kernel_and_communicate()
    {
        Kokkos::parallel_for(
            "strain_rate_invariant", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();

        communication::shell::pack_send_and_recv_local_subdomain_boundaries(
            domain_, dst_, send_buffers_, recv_buffers_ );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
    }

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarT > lumped_mass_;

    // Set before each kernel launch.
    bool                                 assemble_mass_ = false;
    grid::Grid4DDataScalar< ScalarT >    dst_;
    grid::Grid4DDataVec< ScalarT, 3 >    vel_;
};

} // namespace terra::geophysics::viscosity

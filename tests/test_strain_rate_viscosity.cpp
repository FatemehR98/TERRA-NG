/// \file test_strain_rate_viscosity.cpp
///
/// \brief Verifies the building blocks of the strain-rate dependent (non-linear) rheology:
///        geophysics::viscosity::StrainRateInvariant and geophysics::viscosity::strain_rate_dependent_viscosity().
///
/// What is tested
/// --------------
/// 1) StrainRateInvariant recovers the nodal second invariant of the deviatoric strain rate
///
///        eps_II = sqrt( 1/2 eps' : eps' ),    eps' = eps(u) - 1/3 (div u) I,
///
///    of a W1 velocity field. For a velocity that is linear in x the strain rate is constant, and the
///    (isoparametric) wedge elements reproduce linear fields exactly, so every node must carry the same
///    analytical value up to round-off:
///      a) u = G x with a full gradient G (shear + divergence)  =>  eps_II = sqrt( 7.25 ),
///      b) a rigid body rotation                                =>  eps_II = 0,
///      c) a constant translation                               =>  eps_II = 0.
/// 2) The relaxed update blends geometrically: eps <- eps_old^(1-w) * eps_new^w.
/// 3) The pointwise viscosity laws (Newtonian limit, power-law scaling, composite law, viscosity limits).

#include <cmath>

#include "geophysics/viscosity/strain_rate_dependent_viscosity.hpp"
#include "geophysics/viscosity/strain_rate_invariant.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/vector_w1.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;

using geophysics::viscosity::strain_rate_dependent_viscosity;
using geophysics::viscosity::StrainRateInvariant;
using geophysics::viscosity::StrainRateLaw;
using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;

enum class VelocityField
{
    LINEAR,      ///< u = G x, G = [ 1 1 1; 2 1 1; 1 3 1 ]
    ROTATION,    ///< u = omega x x, omega = ( 1, -2, 3 )
    TRANSLATION, ///< u = ( 1, 2, 3 )
};

struct VelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;
    VelocityField              field_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        double u[3] = { 1.0, 2.0, 3.0 };

        if ( field_ == VelocityField::LINEAR )
        {
            u[0] = c( 0 ) + c( 1 ) + c( 2 );
            u[1] = 2.0 * c( 0 ) + c( 1 ) + c( 2 );
            u[2] = c( 0 ) + 3.0 * c( 1 ) + c( 2 );
        }
        else if ( field_ == VelocityField::ROTATION )
        {
            const double w[3] = { 1.0, -2.0, 3.0 };
            u[0]              = w[1] * c( 2 ) - w[2] * c( 1 );
            u[1]              = w[2] * c( 0 ) - w[0] * c( 2 );
            u[2]              = w[0] * c( 1 ) - w[1] * c( 0 );
        }

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) = u[d];
        }
    }
};

static int num_failures = 0;

static void check( const bool ok, const std::string& what, const double value, const double bound )
{
    util::logroot << ( ok ? "[ ok ] " : "[FAIL] " ) << what << ": " << value << " (bound " << bound << ")" << std::endl;
    if ( !ok )
    {
        num_failures++;
    }
}

static void check_close( const std::string& what, const double value, const double expected, const double rel_tol )
{
    const double err = std::abs( value - expected ) / std::abs( expected );
    util::logroot << ( err <= rel_tol ? "[ ok ] " : "[FAIL] " ) << what << ": " << value << ", expected " << expected
                  << std::endl;
    if ( !( err <= rel_tol ) )
    {
        num_failures++;
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    using ScalarType = double;

    const int  level = 4;
    const auto r_min = 0.5;
    const auto r_max = 1.0;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );

    const auto mask_data    = grid::setup_node_ownership_mask_data( domain );
    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorW1Vec< ScalarType, 3 > velocity( "velocity", domain, mask_data );
    VectorW1Scalar< ScalarType > strain_rate( "strain_rate", domain, mask_data );
    VectorW1Scalar< ScalarType > relaxed( "relaxed", domain, mask_data );
    VectorW1Scalar< ScalarType > tmp( "tmp", domain, mask_data );

    StrainRateInvariant< ScalarType > strain_rate_invariant( domain, coords_shell, coords_radii );

    const auto interpolate_velocity = [&]( const VelocityField field ) {
        Kokkos::parallel_for(
            "velocity_interpolation",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            VelocityInterpolator{ coords_shell, coords_radii, velocity.grid_data(), field } );
        Kokkos::fence();
    };

    // ---------------------------------------------------------------------------------------------------------
    // 1) Strain rate invariant of analytical velocity fields.
    // ---------------------------------------------------------------------------------------------------------

    // a) Linear velocity: eps' : eps' = 14.5 (same field as in test_shear_heating)  =>  eps_II = sqrt( 7.25 ).
    const double eps_II_exact = std::sqrt( 7.25 );

    interpolate_velocity( VelocityField::LINEAR );
    strain_rate_invariant.compute( velocity, strain_rate );

    const double eps_min = linalg::min_entry( strain_rate );
    const double eps_max = linalg::max_entry( strain_rate );
    const double err_lin = std::max( std::abs( eps_min - eps_II_exact ), std::abs( eps_max - eps_II_exact ) ) /
                           eps_II_exact;
    util::logroot << "linear velocity: eps_II in [" << eps_min << ", " << eps_max << "], exact " << eps_II_exact
                  << std::endl;
    check( err_lin < 1e-10, "linear velocity, max. relative nodal error of eps_II", err_lin, 1e-10 );

    // b) Rigid body rotation: no deformation. Scale: |omega| = sqrt( 14 ).
    interpolate_velocity( VelocityField::ROTATION );
    strain_rate_invariant.compute( velocity, strain_rate );
    const double err_rot = linalg::norm_inf( strain_rate ) / std::sqrt( 14.0 );
    check( err_rot < 1e-10, "rigid rotation, max. eps_II / |omega|", err_rot, 1e-10 );

    // c) Translation: exactly representable, no deformation.
    interpolate_velocity( VelocityField::TRANSLATION );
    strain_rate_invariant.compute( velocity, strain_rate );
    const double err_trans = linalg::norm_inf( strain_rate );
    check( err_trans < 1e-10, "translation, max. eps_II", err_trans, 1e-10 );

    // ---------------------------------------------------------------------------------------------------------
    // 2) Relaxed update: starting from eps_old = 1, eps <- eps_old^(1-w) * eps_new^w = eps_new^w.
    // ---------------------------------------------------------------------------------------------------------

    interpolate_velocity( VelocityField::LINEAR );
    strain_rate_invariant.compute( velocity, strain_rate );

    for ( const double w : { 1.0, 0.5, 1.5 } )
    {
        linalg::assign( relaxed, 1.0 );
        strain_rate_invariant.compute_relaxed( velocity, relaxed, w, tmp );

        const double lo = linalg::min_entry( relaxed );
        const double hi = linalg::max_entry( relaxed );
        const double err_relaxed =
            std::max( std::abs( lo - std::pow( eps_min, w ) ), std::abs( hi - std::pow( eps_max, w ) ) );
        check( err_relaxed < 1e-12, "relaxed update, w = " + std::to_string( w ), err_relaxed, 1e-12 );
    }

    // ---------------------------------------------------------------------------------------------------------
    // 3) Pointwise viscosity laws.
    // ---------------------------------------------------------------------------------------------------------

    const double eta_N   = 3.0;
    const double eps_0   = 2.0;
    const double eta_min = 1e-3;
    const double eta_max = 1e3;

    const auto eta = [&]( const StrainRateLaw law, const double eps, const double n ) {
        return strain_rate_dependent_viscosity( law, eta_N, eps, eps_0, n, eta_min, eta_max );
    };

    // Newtonian: independent of the strain rate.
    check_close( "none", eta( StrainRateLaw::NONE, 123.0, 3.5 ), eta_N, 1e-14 );
    // Power law: eta_N at the reference strain rate, Newtonian for n = 1, ( eps / eps_0 )^( ( 1 - n ) / n ) scaling.
    check_close( "power law at reference strain rate", eta( StrainRateLaw::POWER_LAW, eps_0, 3.5 ), eta_N, 1e-14 );
    check_close( "power law, n = 1", eta( StrainRateLaw::POWER_LAW, 50.0 * eps_0, 1.0 ), eta_N, 1e-14 );
    check_close(
        "power law, n = 3, 8 x reference strain rate",
        eta( StrainRateLaw::POWER_LAW, 8.0 * eps_0, 3.0 ),
        eta_N / 4.0,
        1e-12 );
    check_close(
        "power law, n = 3, 1/8 x reference strain rate",
        eta( StrainRateLaw::POWER_LAW, eps_0 / 8.0, 3.0 ),
        eta_N * 4.0,
        1e-12 );
    // Composite: harmonic average of Newtonian and power-law viscosity.
    check_close( "composite at transition strain rate", eta( StrainRateLaw::COMPOSITE, eps_0, 3.5 ), eta_N / 2.0, 1e-14 );
    check_close(
        "composite, n = 3, 8 x transition strain rate",
        eta( StrainRateLaw::COMPOSITE, 8.0 * eps_0, 3.0 ),
        1.0 / ( 1.0 / eta_N + 4.0 / eta_N ),
        1e-12 );
    check_close( "composite, vanishing strain rate", eta( StrainRateLaw::COMPOSITE, 0.0, 3.5 ), eta_N, 1e-8 );
    // Viscosity limits.
    check_close( "power law, vanishing strain rate (upper limit)", eta( StrainRateLaw::POWER_LAW, 0.0, 3.5 ), eta_max, 1e-14 );
    check_close( "power law, huge strain rate (lower limit)", eta( StrainRateLaw::POWER_LAW, 1e30, 3.5 ), eta_min, 1e-14 );

    if ( num_failures > 0 )
    {
        Kokkos::abort( "test_strain_rate_viscosity failed." );
    }

    util::logroot << "All checks passed." << std::endl;

    return 0;
}

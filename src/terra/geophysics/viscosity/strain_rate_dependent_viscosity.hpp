#pragma once

#include "grid/grid_types.hpp"
#include "kokkos/kokkos_wrapper.hpp"

namespace terra::geophysics::viscosity {

/// @brief Strain-rate dependence of the viscosity (non-Newtonian / non-linear rheology).
///
/// All laws modify a given Newtonian viscosity \f$ \eta_N \f$ (which may itself depend on temperature and depth)
/// using the second invariant of the strain rate \f$ \dot\varepsilon_{II} \f$, a reference strain rate
/// \f$ \dot\varepsilon_0 \f$ and the stress exponent \f$ n \geq 1 \f$:
///
///   NONE      : \f$ \eta = \eta_N \f$ (Newtonian, linear rheology).
///   POWER_LAW : \f$ \eta = \eta_N \, ( \dot\varepsilon_{II} / \dot\varepsilon_0 )^{(1-n)/n} \f$.
///               Pure power-law (dislocation) creep. \f$ \eta_N \f$ is the viscosity at the reference strain rate.
///               Unbounded for \f$ \dot\varepsilon_{II} \to 0 \f$, i.e., relies on the upper viscosity limit.
///   COMPOSITE : \f$ 1 / \eta = 1 / \eta_N + 1 / \eta_{PL} \f$ with \f$ \eta_{PL} \f$ the POWER_LAW viscosity, i.e.,
///               \f$ \eta = \eta_N / ( 1 + ( \dot\varepsilon_{II} / \dot\varepsilon_0 )^{(n-1)/n} ) \f$.
///               Newtonian (diffusion) and power-law (dislocation) creep acting simultaneously.
///               \f$ \dot\varepsilon_0 \f$ is the transition strain rate at which both mechanisms contribute
///               equally. Recovers \f$ \eta_N \f$ for \f$ \dot\varepsilon_{II} \ll \dot\varepsilon_0 \f$.
///
/// \f$ n = 1 \f$ recovers a Newtonian viscosity for both laws (\f$ \eta_N \f$ and \f$ \eta_N / 2 \f$, respectively).
enum class StrainRateLaw
{
    NONE,
    POWER_LAW,
    COMPOSITE
};

/// @brief Pointwise evaluation of the strain-rate dependent (effective) viscosity, see StrainRateLaw.
///
/// The result is clamped to [eta_min, eta_max].
template < typename ScalarT >
KOKKOS_INLINE_FUNCTION ScalarT strain_rate_dependent_viscosity(
    const StrainRateLaw law,
    const ScalarT       eta_newtonian,
    const ScalarT       strain_rate_II,
    const ScalarT       reference_strain_rate,
    const ScalarT       stress_exponent,
    const ScalarT       eta_min,
    const ScalarT       eta_max )
{
    // Floor to keep the power well-defined for a vanishing strain rate. The viscosity limits take over long
    // before this is reached.
    const ScalarT ratio = Kokkos::max( strain_rate_II / reference_strain_rate, ScalarT( 1e-30 ) );

    // ( eps_II / eps_0 )^( ( n - 1 ) / n ) >= 0, increasing with the strain rate.
    const ScalarT weakening = Kokkos::pow( ratio, ( stress_exponent - ScalarT( 1 ) ) / stress_exponent );

    ScalarT eta = eta_newtonian;

    switch ( law )
    {
    case StrainRateLaw::POWER_LAW:
        eta = eta_newtonian / weakening;
        break;
    case StrainRateLaw::COMPOSITE:
        eta = eta_newtonian / ( ScalarT( 1 ) + weakening );
        break;
    case StrainRateLaw::NONE:
    default:
        break;
    }

    return Kokkos::clamp( eta, eta_min, eta_max );
}

/// @brief Kokkos kernel computing the effective viscosity from a Newtonian viscosity and the nodal second
///        invariant of the strain rate (see StrainRateInvariant), node by node.
///
/// Launch on a node range, e.g., grid::shell::local_domain_md_range_policy_nodes().
template < typename ScalarT >
struct StrainRateDependentViscosityKernel
{
    StrainRateLaw                           law_;
    grid::Grid4DDataScalar< ScalarT >       eta_;            ///< [out] effective viscosity
    const grid::Grid4DDataScalar< ScalarT > eta_newtonian_;  ///< [in]  Newtonian viscosity
    const grid::Grid4DDataScalar< ScalarT > strain_rate_II_; ///< [in]  second invariant of the strain rate
    const ScalarT                           reference_strain_rate_;
    const ScalarT                           stress_exponent_;
    const ScalarT                           eta_min_;
    const ScalarT                           eta_max_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        eta_( id, x, y, r ) = strain_rate_dependent_viscosity(
            law_,
            eta_newtonian_( id, x, y, r ),
            strain_rate_II_( id, x, y, r ),
            reference_strain_rate_,
            stress_exponent_,
            eta_min_,
            eta_max_ );
    }
};

} // namespace terra::geophysics::viscosity

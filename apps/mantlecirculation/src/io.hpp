
#pragma once

#include <fstream>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/xdmf.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "parameters.hpp"
#include "shell/radial_profiles.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/result.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using util::logroot;
using util::Ok;
using util::Result;

using terra::kernels::common::scale;

using ScalarType = double;

namespace terra::mantlecirculation {

inline Result<> create_directories( const IOParameters& io_parameters )
{
    const auto xdmf_dir            = io_parameters.outdir + "/" + io_parameters.xdmf_dir;
    const auto radial_profiles_dir = io_parameters.outdir + "/" + io_parameters.radial_profiles_out_dir;
    const auto timer_trees_dir     = io_parameters.outdir + "/" + io_parameters.timer_trees_dir;

    if ( !io_parameters.overwrite && std::filesystem::exists( io_parameters.outdir ) )
    {
        return { "Will not overwrite existing directory (to not accidentally delete old simulation data). "
                 "Use -h for help and look for an overwrite option or choose a different output dir name." };
    }

    util::prepare_empty_directory( io_parameters.outdir );
    util::prepare_empty_directory( xdmf_dir );
    util::prepare_empty_directory( radial_profiles_dir );
    util::prepare_empty_directory( timer_trees_dir );

    return { Ok{} };
}

// Xdmf output bundle
template < typename FieldType >
struct ScalableField
{
    FieldType* data;
    ScalarType scale_factor;
    bool       restore;

    ScalableField( FieldType& data_, ScalarType scale_factor_, bool restore_ = true )
    : data( &data_ )
    , scale_factor( scale_factor_ )
    , restore( restore_ )
    {}
};

using ScalarXdmfField = ScalableField< Grid4DDataScalar< ScalarType > >;
using VectorXdmfField = ScalableField< Grid4DDataVec< ScalarType, 3 > >;

struct XdmfFields
{
    std::vector< ScalarXdmfField >   scalar_fields;
    std::vector< VectorXdmfField >   vector_fields;
    std::optional< ScalarXdmfField > pressure_field;
};

inline Result<> write_xdmf(
    std::optional< io::XDMFOutput< ScalarType > >& xdmf_output,
    std::optional< io::XDMFOutput< ScalarType > >& xdmf_output_pressure,
    const int                                      timestep,
    const bool                                     output_dimensional,
    const std::vector< ScalarXdmfField >&          scalar_fields,
    const std::vector< VectorXdmfField >&          vector_fields,
    const std::optional< ScalarXdmfField >&        pressure_field )
{
    logroot << "Writing XDMF output ..." << std::endl;

    if ( output_dimensional )
    {
        // Redimensionalise ...
        for ( auto& f : scalar_fields )
            scale( *f.data, f.scale_factor );
        for ( auto& f : vector_fields )
            scale( *f.data, f.scale_factor );

        xdmf_output->write( timestep );

        // ... and nondimensionalise again, if required
        for ( auto& f : scalar_fields )
            if ( f.restore )
                scale( *f.data, ScalarType( 1 ) / f.scale_factor );
        for ( auto& f : vector_fields )
            if ( f.restore )
                scale( *f.data, ScalarType( 1 ) / f.scale_factor );

        // Redim, write and nondim pressure
        if ( pressure_field )
        {
            scale( *pressure_field->data, pressure_field->scale_factor );

            xdmf_output_pressure->write( timestep );

            if ( pressure_field->restore )
                scale( *pressure_field->data, ScalarType( 1 ) / pressure_field->scale_factor );
        }
    }
    else
    {
        xdmf_output->write( timestep );
        if ( pressure_field )
            xdmf_output_pressure->write( timestep );
    }
    return { Ok{} };
}

inline Result<> compute_and_write_radial_profiles(
    const VectorQ1Scalar< ScalarType >& scalar_function,
    const Grid2DDataScalar< int >&      subdomain_shell_idx,
    const std::vector< double >&        radial_coords,
    const Parameters&                   prm,
    const int                           timestep,
    const ScalarType                    field_scale  = ScalarType( 1 ),
    const std::string&                  radial_label = "radius" )
{
    const auto profiles = shell::radial_profiles_to_table< ScalarType >(
        shell::radial_profiles( scalar_function, subdomain_shell_idx, static_cast< int >( radial_coords.size() ) ),
        radial_coords,
        field_scale,
        radial_label );

    // Zero padding
    std::ostringstream sst;
    sst << std::setfill( '0' )
        << std::setw(
               std::to_string(
                   prm.time_stepping_parameters.timestep_initial + prm.time_stepping_parameters.max_timesteps - 1 )
                   .size() )
        << timestep;

    if ( mpi::rank() == 0 )
    {
        std::ofstream out(
            prm.io_parameters.outdir + "/" + prm.io_parameters.radial_profiles_out_dir + "/radial_profiles_" +
            scalar_function.grid_data().label() + "_" + sst.str() + ".csv" );
        profiles.print_csv( out );
    }

    return { Ok{} };
}

/// Decompose a Q1 velocity field into its radial (signed) and tangential (magnitude)
/// components and write radial profiles for both, reusing
/// `compute_and_write_radial_profiles`. Output files are named
/// `radial_profiles_u_r_<step>.csv` and `radial_profiles_u_t_<step>.csv`.
inline Result<> compute_and_write_velocity_radial_profiles(
    const linalg::VectorQ1Vec< ScalarType, 3 >&              velocity,
    const grid::Grid3DDataVec< ScalarType, 3 >&              coords_shell,
    const grid::Grid2DDataScalar< int >&                     subdomain_shell_idx,
    const DistributedDomain&                                 domain,
    const std::vector< double >&                             radial_coords,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask,
    const Parameters&                                        prm,
    const int                                                timestep,
    const ScalarType                                         field_scale  = ScalarType( 1 ),
    const std::string&                                       radial_label = "radius" )
{
    VectorQ1Scalar< ScalarType > u_r( "u_r", domain, mask );
    VectorQ1Scalar< ScalarType > u_t( "u_t", domain, mask );

    const auto u_grid   = velocity.grid_data();
    auto       u_r_grid = u_r.grid_data();
    auto       u_t_grid = u_t.grid_data();

    Kokkos::parallel_for(
        "decompose velocity into radial and tangential",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 }, { u_grid.extent( 0 ), u_grid.extent( 1 ), u_grid.extent( 2 ), u_grid.extent( 3 ) } ),
        KOKKOS_LAMBDA( int sd, int x, int y, int r ) {
            const ScalarType ux      = u_grid( sd, x, y, r, 0 );
            const ScalarType uy      = u_grid( sd, x, y, r, 1 );
            const ScalarType uz      = u_grid( sd, x, y, r, 2 );
            const ScalarType rx      = coords_shell( sd, x, y, 0 );
            const ScalarType ry      = coords_shell( sd, x, y, 1 );
            const ScalarType rz      = coords_shell( sd, x, y, 2 );
            const ScalarType ur_node = ux * rx + uy * ry + uz * rz;
            const ScalarType umag2   = ux * ux + uy * uy + uz * uz;
            const ScalarType ut2     = umag2 - ur_node * ur_node;
            u_r_grid( sd, x, y, r )  = ur_node;
            u_t_grid( sd, x, y, r )  = ( ut2 > ScalarType( 0 ) ) ? Kokkos::sqrt( ut2 ) : ScalarType( 0 );
        } );
    Kokkos::fence();

    auto res1 = compute_and_write_radial_profiles(
        u_r, subdomain_shell_idx, radial_coords, prm, timestep, field_scale, radial_label );
    if ( res1.is_err() )
    {
        return res1;
    }
    return compute_and_write_radial_profiles(
        u_t, subdomain_shell_idx, radial_coords, prm, timestep, field_scale, radial_label );
}

/// @brief XDMF output of nodal fields that live on a single boundary (surface or CMB) of the shell.
///
/// Writes a 2D triangle mesh at the boundary radius (two triangles per boundary hex face, i.e. the boundary
/// faces of the wedges) plus one binary file per field and write() call, and a step_<time>.xmf file to open in
/// Paraview. Only subdomains that touch the boundary contribute; like io::XDMFOutput, nodes on subdomain
/// interfaces are duplicated.
///
/// Fields are Grid3DDataScalar indexed (local_subdomain, x, y). Visualization only - not a checkpoint.
class BoundaryXDMFOutput
{
  public:
    BoundaryXDMFOutput(
        const std::string&                                              directory_path,
        const Grid3DDataVec< ScalarType, 3 >&                           coords_shell,
        const Grid2DDataScalar< ScalarType >&                           coords_radii,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&       boundary_mask,
        const bool                                                      at_surface,
        const ScalarType                                                coords_scale_factor = ScalarType( 1 ) )
    : directory_path_( directory_path )
    , coords_shell_( coords_shell )
    , coords_radii_( coords_radii )
    , nx_( static_cast< int >( coords_shell.extent( 1 ) ) )
    , r_b_( at_surface ? static_cast< int >( coords_radii.extent( 1 ) ) - 1 : 0 )
    , coords_scale_factor_( coords_scale_factor )
    {
        const auto expected_flag =
            at_surface ? grid::shell::ShellBoundaryFlag::SURFACE : grid::shell::ShellBoundaryFlag::CMB;

        const auto boundary_mask_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, boundary_mask );
        for ( int sd = 0; sd < static_cast< int >( boundary_mask_host.extent( 0 ) ); ++sd )
            if ( boundary_mask_host( sd, 0, 0, r_b_ ) == expected_flag )
                boundary_subdomains_.push_back( sd );

        const int64_t num_sd = static_cast< int64_t >( boundary_subdomains_.size() );
        const int64_t local[2] = { num_sd * nx_ * nx_, num_sd * ( nx_ - 1 ) * ( nx_ - 1 ) * 2 };
        int64_t       incl[2];
        MPI_Scan( local, incl, 2, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );
        nodes_offset_    = incl[0] - local[0];
        elements_offset_ = incl[1] - local[1];

        int64_t global[2];
        MPI_Allreduce( local, global, 2, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );
        nodes_global_    = global[0];
        elements_global_ = global[1];
    }

    void set_pad_width( const int pad_width ) { pad_width_ = pad_width; }

    /// @brief Registers a field. Its Kokkos label is used as the field name. Values are multiplied by
    ///        scale_factor on output (e.g. for redimensionalisation); the field itself is not modified.
    void add( const Grid3DDataScalar< ScalarType >& data, const ScalarType scale_factor = ScalarType( 1 ) )
    {
        // Persistent host mirror: allocated once here (aliases `data` if it is host-accessible already).
        fields_.push_back( { data, Kokkos::create_mirror_view( data ), scale_factor } );
    }

    void write( const int time )
    {
        using util::XML;

        if ( !first_write_happened_ )
        {
            util::prepare_empty_directory( directory_path_ );
            write_mesh();
            first_write_happened_ = true;
        }

        std::ostringstream oss;
        oss << std::setw( pad_width_ ) << std::setfill( '0' ) << time;
        const std::string time_str = oss.str();

        auto grid = XML( "Grid", { { "Name", "Grid" }, { "GridType", "Uniform" } } );
        grid.add_child( XML( "Time", { { "Value", std::to_string( time ) } } ) );
        grid.add_child( XML( "Geometry", { { "Type", "XYZ" } } )
                            .add_child( data_item( "Float", 4, std::to_string( nodes_global_ ) + " 3", "geometry.bin" ) ) );
        grid.add_child(
            XML( "Topology", { { "Type", "Triangle" }, { "NumberOfElements", std::to_string( elements_global_ ) } } )
                .add_child( data_item( "Int", 8, std::to_string( elements_global_ ) + " 3", "topology.bin" ) ) );

        for ( auto& [data, host, scale_factor] : fields_ )
        {
            // Only the (small) boundary field is copied; no-op if host and device memory coincide.
            Kokkos::deep_copy( host, data );

            buffer_.clear();
            for ( const int sd : boundary_subdomains_ )
                for ( int y = 0; y < nx_; ++y )
                    for ( int x = 0; x < nx_; ++x )
                        buffer_.push_back( static_cast< float >( host( sd, x, y ) * scale_factor ) );

            const std::string file_base = data.label() + "_" + time_str + ".bin";
            write_collectively( file_base, buffer_, nodes_offset_ );

            grid.add_child(
                XML( "Attribute", { { "Name", data.label() }, { "AttributeType", "Scalar" }, { "Center", "Node" } } )
                    .add_child( data_item( "Float", 4, std::to_string( nodes_global_ ), file_base ) ) );
        }

        auto xml = XML( "Xdmf", { { "Version", "2.0" } } );
        xml.add_child( XML( "Domain" ).add_child( grid ) );

        if ( mpi::rank() == 0 )
        {
            std::ofstream step_stream( directory_path_ + "/step_" + time_str + ".xmf" );
            step_stream << "<?xml version=\"1.0\" ?>\n";
            step_stream << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            step_stream << xml.to_string();
        }
    }

  private:
    static util::XML data_item(
        const std::string& data_type,
        const int          precision,
        const std::string& dimensions,
        const std::string& file )
    {
        return util::XML(
            "DataItem",
            { { "Format", "Binary" },
              { "DataType", data_type },
              { "Precision", std::to_string( precision ) },
              { "Endian", "Little" },
              { "Dimensions", dimensions } },
            file );
    }

    /// Each rank writes its contiguous chunk, starting at element offset `offset` (in units of T).
    template < typename T >
    void write_collectively( const std::string& file_base, const std::vector< T >& buffer, const int64_t offset )
    {
        const std::string path = directory_path_ + "/" + file_base;

        MPI_File fh;
        MPI_File_open( MPI_COMM_WORLD, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh );
        MPI_File_set_view(
            fh, static_cast< MPI_Offset >( offset * sizeof( T ) ), MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL );
        MPI_File_write_all(
            fh,
            buffer.data(),
            static_cast< int >( buffer.size() * sizeof( T ) ),
            MPI_CHAR,
            MPI_STATUS_IGNORE );
        MPI_File_close( &fh );
    }

    void write_mesh()
    {
        // Explicit host_mirror_type: on device builds, a mirror in HostSpace is a different type, which
        // grid::shell::coords() does not accept.
        const typename Grid3DDataVec< ScalarType, 3 >::host_mirror_type coords_shell_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell_ );
        const typename Grid2DDataScalar< ScalarType >::host_mirror_type coords_radii_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii_ );

        std::vector< float > points;
        points.reserve( boundary_subdomains_.size() * nx_ * nx_ * 3 );
        for ( const int sd : boundary_subdomains_ )
            for ( int y = 0; y < nx_; ++y )
                for ( int x = 0; x < nx_; ++x )
                {
                    const auto c = grid::shell::coords( sd, x, y, r_b_, coords_shell_host, coords_radii_host );
                    for ( int d = 0; d < 3; ++d )
                        points.push_back( static_cast< float >( c( d ) * coords_scale_factor_ ) );
                }
        write_collectively( "geometry.bin", points, nodes_offset_ * 3 );

        // Same triangle split as the wedges of io::XDMFOutput: (v0, v1, v2) and (v3, v2, v1).
        std::vector< int64_t > triangles;
        triangles.reserve( boundary_subdomains_.size() * ( nx_ - 1 ) * ( nx_ - 1 ) * 6 );
        int64_t sd_offset = nodes_offset_;
        for ( std::size_t i = 0; i < boundary_subdomains_.size(); ++i, sd_offset += int64_t( nx_ ) * nx_ )
            for ( int y = 0; y < nx_ - 1; ++y )
                for ( int x = 0; x < nx_ - 1; ++x )
                {
                    const int64_t v0 = sd_offset + int64_t( y ) * nx_ + x;
                    const int64_t v1 = v0 + 1;
                    const int64_t v2 = v0 + nx_;
                    const int64_t v3 = v2 + 1;
                    triangles.insert( triangles.end(), { v0, v1, v2, v3, v2, v1 } );
                }
        write_collectively( "topology.bin", triangles, elements_offset_ * 3 );
    }

    std::string                    directory_path_;
    Grid3DDataVec< ScalarType, 3 > coords_shell_;
    Grid2DDataScalar< ScalarType > coords_radii_;
    int                            nx_;
    int                            r_b_;
    ScalarType                     coords_scale_factor_;

    std::vector< int > boundary_subdomains_;
    int64_t            nodes_offset_    = 0;
    int64_t            elements_offset_ = 0;
    int64_t            nodes_global_    = 0;
    int64_t            elements_global_ = 0;

    struct Field
    {
        Grid3DDataScalar< ScalarType >                     data;
        Grid3DDataScalar< ScalarType >::host_mirror_type host;
        ScalarType                                         scale_factor;
    };
    std::vector< Field > fields_;
    std::vector< float > buffer_; // host-side output buffer, reused across writes

    int  pad_width_            = 1;
    bool first_write_happened_ = false;
};

inline Result<> write_timer_tree( const IOParameters& io_parameters, const int timestep )
{
    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        const auto timer_tree_file = io_parameters.outdir + "/" + io_parameters.timer_trees_dir + "/timer_tree_" +
                                     std::to_string( timestep ) + ".json";
        logroot << "Writing timer tree to " << timer_tree_file << std::endl;
        std::ofstream out( timer_tree_file );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }

    return { Ok{} };
}

} // namespace terra::mantlecirculation

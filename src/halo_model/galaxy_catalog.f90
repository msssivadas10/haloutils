module galaxy_catalog_mod
    !! An extension module to `halo_model_mod` module, containing methods
    !! for generating galaxy catalogs from a given halo catalog.
    
    use omp_lib
    use iso_c_binding
    use constants_mod
    use halo_model_mod
    use rfunctions_mod
    use interpolate_mod
    use random_mod
    implicit none

    private
    public :: setup_catalog_generation, generate_galaxies, generate_galaxy_catalog
    public :: cgenerate_galaxy_catalog

    integer, parameter :: PATH_LEN = 1024
    
    type, public, bind(c) :: halodata_t
        !! A struct containing data about a halo
        integer(c_int64_t) :: id     !! Unique halo ID
        real(c_double)     :: pos(3) !! Halo position coordinates in Mpc
        real(c_double)     :: mass   !! Halo mass in Msun
    end type 

    type, public, bind(c) :: galaxydata_t
        !! A struct containing data about a galaxy
        integer(c_int64_t) :: halo_id !! Parent halo ID
        real(c_double)     :: pos(3)  !! Halo position coordinates in Mpc
        real(c_double)     :: mass    !! Halo mass in Msun
        character(c_char)  :: typ     !! Galaxy type: `c`=central, `s`=satellite
    end type

    type, public, bind(c) :: cgargs_t
        !! Struct containing various parameters for galaxy catalog
        !! generation (execpt halo model parameters). 

        real(c_double)     :: lnm    !! Natural log of halo mass in Msun
        real(c_double)     :: pos(3) !! Coordinates of the halo in Mpc units
        real(c_double)     :: lnr    !! Natural log of halo radius in Mpc
        real(c_double)     :: s      !! Matter variance corresponding to this halo mass
        real(c_double)     :: c      !! Concentration parameter for the halo
        integer(c_int64_t) :: n_cen  !! Number of central galaxies (0 or 1)
        integer(c_int64_t) :: n_sat  !! Number of satellite galaxies
        integer(c_int64_t) :: rstate !! Random number generator state

        real(c_double) :: boxsize(3)
        !! Size of the bounding box containing all the halos in the 
        !! simulation. Used for periodic wrapping galaxy position.

        real(c_double) :: offset(3)
        !! Coordinates of the bottom-lower-left corner of the bounding 
        !! box. Used for periodic wrapping galaxy position.

    end type 

contains

    subroutine setup_catalog_generation(params, args) bind(c)
        !! Calculate various parameters for a galaxy catalog generation

        type(hmargs_t), intent(in) :: params 
        !! Halo model parameters

        type(cgargs_t), intent(inout) :: args
        !! Arguments for catalog generation

        real(c_double) :: rho_m, rho_h, p_cen, lam_sat

        rho_m = params%Om0 * ( critical_density_const * params%H0**2 ) ! Matter density at z=0 in Msun/Mpc^3 
        rho_h = rho_m * params%Delta_m ! Halo density (TODO: chek if the halo density is rho_m * self.Delta)

        ! Lagrangian radius (r) corresponding to halo mass
        args%lnr = ( args%lnm + log(3._c_double / (4*pi) / rho_h ) ) / 3._c_double ! r in Mpc

        ! Central galaxy count: this is drawn from a binomial distribution of n = 1, 
        ! so that the value is either 1 or 0. If the model uses a step function use 
        ! the average count as the actual count. 
        p_cen = central_count(params, args%lnm)
        if ( abs(params%sigma_m) < 1e-08_c_double ) then
            args%n_cen = int(p_cen, kind=c_int64_t)
        else
            args%n_cen = binomial_rv(args%rstate, 1_c_int64_t, p_cen)
        end if
        if ( args%n_cen < 1 ) return ! No central galaxies -> no satellites also 

        ! Satellite galaxy count: this drawn from a poisson distribution with the 
        ! calculated average.
        lam_sat    = satellite_count(params, args%lnm)
        args%n_sat = poisson_rv(args%rstate, lam_sat)
        
        if ( params%scale_shmf < exp(params%lnm_min - args%lnm) ) then 
            ! Halo mass correspond to an invalid satellite galaxy mass range: no satellites
            args%n_sat = 0
        end if

        ! Calculating halo concentration parameter
        args%c = halo_concentration(params, args%s)

    end subroutine setup_catalog_generation

    subroutine generate_galaxies(params, args, n, gdata) bind(c)
        !! Generate galaxy positions and mass.

        type(hmargs_t), intent(in) :: params 
        !! Halo model parameters

        type(cgargs_t), intent(inout) :: args
        !! Arguments for catalog generation

        integer(c_int64_t), intent(in), value :: n
        !! Size of the position and mass array: must be same total number 
        !! of galaxies i.e., `(args%n_cen+args%n_sat)`.

        type(galaxydata_t), intent(out) :: gdata(n)
        !! Galaxy positions and masses. 

        integer(c_int64_t) :: i
        real(c_double) :: m_halo, r_halo, c_halo, f, r, theta, phi, Ac, k1, k2, p

        if ( args%n_cen < 1 ) return ! No galaxies in this halo

        m_halo = exp(args%lnm) ! Halo mass in Msun
        r_halo = exp(args%lnr) ! Halo radius in Mpc
        c_halo = args%c        ! Halo concentration parameter

        ! Halo has a central galaxy: the position and mass of this galaxy is same
        ! as that of the parent halo.
        gdata(1)%typ  = 'c'
        gdata(1)%pos  = args%pos(1:3) ! in Mpc
        gdata(1)%mass = m_halo        ! in Msun

        if ( args%n_sat < 1 ) return ! No satellite galaxies in this halo
        do i = 1, args%n_sat

            gdata(i+1)%typ  = 's'
            
            ! Assigning random mass values to the satellite galaxies: These masses 
            ! are drown from a bounded pareto distribution, with bounds `m_min` and  
            ! `scale_shmf*m_halo`, and slope given by slope_shmf.
            !  
            ! NOTE: RVs are generated using inverse transform sampling 
            ! (<https://en.wikipedia.org/wiki/Pareto_distribution>)
            p  = -1._c_double / params%slope_shmf
            k1 = exp( (params%lnm_min - args%lnm)*params%slope_shmf )
            k2 = params%scale_shmf**params%slope_shmf
            f  = ( ( k2 - (k2 - k1) * uniform_rv(args%rstate) ) / (k1*k2) )**p ! m_sat / m_halo
            
            ! Generating random values corresponding to the distance of the galaxy 
            ! from the halo center. These RVs should follow a distribution matching 
            ! the NFW density profile of the halo. Sampling is done using the 
            ! inverse transformation method. 
            Ac    = uniform_rv(args%rstate)*( log(1 + c_halo) - c_halo / (1 + c_halo) )
            r     = (r_halo / c_halo) * nfw_c(Ac)
            theta = acos( 2*uniform_rv(args%rstate) - 1 ) ! -pi to pi
            phi   = 2*pi*uniform_rv(args%rstate)          !   0 to 2pi
            
            ! Satellite galaxy coordinates x, y, and z in Mpc
            gdata(i+1)%pos(1) = gdata(1)%pos(1) + r*sin(theta)*cos(phi)
            gdata(i+1)%pos(2) = gdata(1)%pos(2) + r*sin(theta)*sin(phi)
            gdata(i+1)%pos(3) = gdata(1)%pos(3) + r*cos(theta)
            ! Periodic wrapping of coordinates to stay within the bounding box
            call periodic_wrap( gdata(i+1)%pos, args%offset, args%boxsize )
            
            ! Satellite galaxy mass in Msun
            gdata(i+1)%mass = gdata(1)%mass * f

        end do
        
    end subroutine generate_galaxies

    elemental subroutine periodic_wrap(x, offset, width)
        !! Periodicall wrap the value to the interval [offset, offset+width] 

        real(c_double), intent(inout) :: x
        real(c_double), intent(in)    :: offset, width

        ! If width <= 0, then no wrapping is done: this can be used for forcing 
        ! no wrapping, if needed...
        if ( width > 0. ) then
            ! Using `modulo` (for mathematical modulo) instead of `mod` function
            ! (remainder of a division)
            x = offset + modulo(x - offset, width)
        end if

    end subroutine periodic_wrap

    function nfw_c(a) result(res)
        !! Return the value of inverse to the NFW mass function. 

        real(c_double), intent(in) :: a
        real(c_double) :: res, x, p1, p2

        integer, parameter :: f8 = c_double

        ! Using an approximate piecewise rational function for inverting the 
        ! c-A(c) relation. This fit works good upto A(c) ~ 10, with less that  
        ! 10% error. For getting the actual value, one should invert the NFW 
        ! mass function, A(c) = log(1+c) - c/(1+c).  
        x = log10(a)
        if ( a < 1e-03_c_double ) then
            res = 0.50018962_f8*x + 0.15241388_f8
        else if ( a < 2._c_double ) then
            p1  = 2699.40545133_f8 + 2921.07235917_f8*x - 1162.90566455_f8*x**2 
            p2  = 3705.23701996_f8 - 3065.75405505_f8*x -   61.92662277_f8*x**2
            res = p1 / p2
        else
            p1  = 47.25501938_f8 + 32.98237791_f8*x - 38.26172387_f8*x**2  
            p2  = 66.29326139_f8 - 87.37718415_f8*x + 29.86558497_f8*x**2
            res = p1 / p2
        end if
        res = 10._c_double**res
        
    end function nfw_c

    subroutine generate_galaxy_catalog(halo_path, glxy_path, fl, hmargs, bbox,       &
                                       pktab, np, lnma, lnmb, ns, filt, sigma_table, &
                                       seed, nthreads, error                         &
        )
        !! Generate a galaxy catalog using the given halo catalog. 

        character(PATH_LEN), intent(in) :: halo_path
        !! Path to the input halo catalog file: the file must be a binary stream of 
        !! `halodata_t` records in little endian byteorder.

        character(PATH_LEN), intent(in) :: glxy_path
        !! Path to the output galaxy catalog file: the file will be a binary stream 
        !! of `galaxydata_t` records in little endian byteorder.

        integer(c_int), intent(in) :: fl
        !! Log file unit: log messages are written to this file.

        type(hmargs_t), intent(in) :: hmargs
        !! Halo model parameters

        real(c_double), intent(in) :: bbox(3,2)
        !! Bounding box for the space containing all halos (used for periodic 
        !! wrapping of galaxy position).

        integer(c_int64_t), intent(in) :: np !! Power spectrum table size
        real(c_double)    , intent(in) :: pktab(2, np) 
        !! Matter power spectrum table. First column should be log(k in 1/Mpc) and the 
        !! second log(power in Mpc^3).

        real(c_double)    , intent(in) :: lnma !! Minimum mass in the table
        real(c_double)    , intent(in) :: lnmb !! Maximum mass in the table
        integer(c_int64_t), intent(in) :: ns   !! Size of the internal variance table 
        integer(c_int)    , intent(in) :: filt !! Filter function for smoothing

        real(c_double), intent(out) :: sigma_table(3, ns)
        !! Matter variance table calculated from power spectrum, using the given 
        !! specifications: First 2 columns are log(mass in Msun) and log(sigma). 

        integer(c_int64_t), intent(in) :: seed
        !! Seed value for random number generators

        integer(c_int), intent(in) :: nthreads
        !! Number of threads to use

        integer(c_int), intent(out) :: error
        !! Error code (0=success, 1=error)

        integer(c_int)     :: tid, fi, fo, iostat, next_progress
        integer(c_int64_t) :: total_halos, processed_halos, n_halos, total_galaxies 
        integer(c_int64_t) :: remaining_halos, chunk_size, rstate(nthreads)
        real(c_double)     :: progress
        type(halodata_t), allocatable :: hbuf(:) ! Halo data

        error = 1
        
        ! Creating cubic spline for variance interpolation
        call make_sigma_table_(hmargs, pktab, np, filt, lnma, lnmb, &
                               ns, sigma_table                      &
        ) 

        ! Initialising the random number generator. This RNG works on a state 
        ! private to the thread, with a seed offset by the main seed. So, the 
        ! generated RVs should be a different sequence on each thread.
        do tid = 1, nthreads
            call pcg32_init(rstate(tid), seed + 1000*tid)
        end do

        ! Opening the input halo catalog as binary stream and count the number 
        ! of halos available. Also opens a binary stream for the output galaxies. 
        call open_fs_(fi, fo, halo_path, glxy_path, total_halos, error, fl)
        if ( error /= 0 ) return 
        
        ! Set number of threads
        call omp_set_num_threads(nthreads) 
        write(fl, '(a,i0,a)') 'info: catalog generation using ',nthreads,' threads'

        ! Setting the chunk size multiple of no. of threads, so that halos are 
        ! assigned evenly over multiple threads.
        chunk_size = nthreads*1000 
        allocate( hbuf(chunk_size) ) ! allocate halo data buffer

        ! Loading halo data a chunks from the input file
        next_progress   = 5
        total_galaxies  = 0 ! number of galaxies generated
        processed_halos = 0 ! number of halos used 
        remaining_halos = total_halos
        do while ( processed_halos < total_halos )
            
            ! Loading halos
            n_halos = min(chunk_size, remaining_halos) ! actual chunk size 
            read(fi, iostat=iostat) hbuf(1:n_halos)
            if ( iostat /= 0 ) exit

            ! Generating galaxies 
            call distribute_catgen_(hbuf(1:n_halos), n_halos, sigma_table, ns, bbox, &
                                    hmargs, rstate, fo, nthreads, total_galaxies     &
            )
            processed_halos = processed_halos + n_halos
            remaining_halos = remaining_halos - n_halos
            progress        = 100*dble(processed_halos)/dble(total_halos)
            
            if ( floor(progress) >= next_progress ) then
                ! Log messages are written only at 5% increments:
                write(fl, '(a,3(i0,a),f0.2,a)')                                &
                    'info: generated ',total_galaxies,' galaxies from ',       &
                    processed_halos,' halos, ',remaining_halos,' remaining, ', &
                    progress,'% completed'
                next_progress = next_progress + 5
            end if
        end do

        deallocate( hbuf )
        close(fi); close(fo) ! close files

        error = 0

    end subroutine generate_galaxy_catalog

    subroutine make_sigma_table_(hmargs, pktab, np, filt, lnma, lnmb, ns, sigma_table)
        !! Generate the variance table from the given power spectrum table.

        type(hmargs_t)    , intent(in)  :: hmargs
        integer(c_int64_t), intent(in)  :: np, ns 
        integer(c_int)    , intent(in)  :: filt 
        real(c_double)    , intent(in)  :: lnma, lnmb, pktab(2,np)  
        real(c_double)    , intent(out) :: sigma_table(3,ns)

        real(c_double)     :: delta, lnm, lnr, var
        integer(c_int64_t) :: i

        ! Calculating the sigma table using the given power spectrum table 
        delta = (lnmb - lnma) / (ns - 1._c_double)
        lnm   = lnma
        do i = 1, ns
            lnr = lagrangian_r(hmargs, lnm)
            var = variance( lnr, 0_c_int, 0_c_int, filt, pktab, np, 2_c_int )
            sigma_table(1,i) = lnm
            sigma_table(2,i) = 0.5_c_double * log(var)
            lnm              = lnm + delta
        end do
        
        ! Creating cubic spline for interpolation
        call generate_cspline(ns, sigma_table) 

    end subroutine make_sigma_table_

    subroutine open_fs_(fi, fo, halo_path, glxy_path, total_halos, error, fl)
        !! Open a halo catalog file as binary stream of `halodata_t` records, and
        !! count the number of halos available in the catalog. Also open a binary 
        !! stream for the output (`galaxydata_t` records).

        character(PATH_LEN), intent(in)  :: halo_path, glxy_path
        integer(c_int)     , intent(in)  :: fl
        integer(c_int)     , intent(out) :: fi, fo, error
        integer(c_int64_t) , intent(out) :: total_halos

        integer(c_int64_t) :: file_size_bytes
        integer(c_int64_t), parameter :: item_size_bytes = c_sizeof(halodata_t( &
            0_c_int64_t,                                                        &
            [ 0._c_double, 0._c_double, 0._c_double ],                          &
            0._c_double                                                         &   
        )) !! Size of `halodata_t`: should be 40

        ! Open halo catalog file:
        fi = 10
        open(newunit=fi, file=trim(halo_path), access='stream', form='unformatted', &
             convert='little_endian', status='old', action='read', iostat=error     &
        )
        if ( error /= 0 ) then
            write(fl,'(a,a)') "error: failed to open file: ",trim(halo_path)
            return
        end if
        
        ! Get the number of halos from the file: since the input file is expected 
        ! to be a binary stream of `halodata_t` (size: 40 bytes), number of halos
        ! in the file can be calculated as `file_size_bytes / item_size_bytes` 
        inquire(fi, size=file_size_bytes)
        total_halos = file_size_bytes / item_size_bytes

        ! Open galaxy catalog file:
        fo = 11
        open(newunit=fo, file=trim(glxy_path), access='stream', form='unformatted',  &
             convert='little_endian', status='replace', action='write', iostat=error &
        )
        if ( error /= 0 ) then
            write(fl,'(a,a)') "error: failed to open file: ",trim(glxy_path)
            return
        end if
        
        write(fl,'(a,i0,a,a)') "info: found ",total_halos," halos in ",trim(halo_path)
        
    end subroutine open_fs_

    subroutine distribute_catgen_(hbuf, n_halos, sigma, ns, bbox, hmargs, &
                                  rstate, fo, nthreads, ngals             &
        )
        !! Generate galaxy catalog from the given halo data in parallel.

        type(halodata_t)  , intent(in)    :: hbuf(n_halos)   
        type(hmargs_t)    , intent(in)    :: hmargs
        integer(c_int64_t), intent(in)    :: n_halos, ns
        integer(c_int64_t), intent(inout) :: rstate(nthreads), ngals
        integer(c_int)    , intent(in)    :: fo, nthreads
        real(c_double)    , intent(in)    :: sigma(3,ns), bbox(3,2)

        integer(c_int)     :: nts, tid
        integer(c_int64_t) :: istart, istop, i, n, gsize 
        type(cgargs_t)     :: args 
        type(galaxydata_t), allocatable :: gbuf(:) ! Galaxy position and mass
        integer(c_int64_t), parameter   :: buffer_size = 10000

        !$omp parallel private(nts, tid, istart, istop, i, gbuf, n, gsize, args)

        nts = omp_get_num_threads() ! no. of threads (should be same as `nthreads`)
        tid = omp_get_thread_num()  ! thread ID
        call get_block_(n_halos, tid, nts, istart, istop)

        args%rstate       = rstate(tid+1) ! set RNG state
        args%boxsize(1:3) = bbox(1:3,2) - bbox(1:3,1) 
        args%offset(1:3)  = bbox(1:3,1) 

        ! Allocate galaxy buffer and metadata: 
        allocate( gbuf(buffer_size) ) 

        gsize = 0 ! size of data buffer: index to append is gsize+1
        do i = istart, istop

            ! Copy halo data to local args
            args%pos = hbuf(i)%pos         ! position
            args%lnm = log( hbuf(i)%mass ) ! mass
            args%s   = exp( interpolate(args%lnm, ns, sigma) ) ! matter variance
            
            ! Setting up 
            call setup_catalog_generation(hmargs, args)

            n = args%n_cen + args%n_sat ! total number of galaxies in this halo 
            if ( n < 1 ) cycle
            
            if ( gsize + n  <= buffer_size ) then
                ! Local buffer has not enough data for saving the new galaxy data:
                ! Current data is written to the main output buffer and the start 
                ! pointers are restted to the start. 

                !$omp critical
                call flush_to_buffer_( fo, gsize, ngals, gbuf(1:gsize) )
                !$omp end critical
            end if    
            
            ! Generating the galaxies
            gbuf(gsize+1:gsize+n)%halo_id = hbuf(i)%id ! assign parent halo
            call generate_galaxies( hmargs, args, n, gbuf(gsize+1:gsize+n) )

            gsize = gsize + n
        end do        
        if ( gsize > 0 ) then ! write remaining data to the buffer...
            !$omp critical
            call flush_to_buffer_( fo, gsize, ngals, gbuf(1:gsize) )
            !$omp end critical
        end if 

        deallocate(gbuf)

        ! Save final RNG state for continuing the sequence for next chunk
        rstate(tid+1) = args%rstate 

        !$omp end parallel

    end subroutine distribute_catgen_

    subroutine get_block_(total_size, tid, nthreads, istart, istop)
        !! Get bounds for a block allotted to a thread in a distributed calculation. 

        integer(c_int)    , intent(in)  :: tid, nthreads
        integer(c_int64_t), intent(in)  :: total_size 
        integer(c_int64_t), intent(out) :: istart, istop

        integer(c_int64_t) :: isize, rem

        if ( nthreads == 1 ) then ! serial calculation
            istart = 1
            isize  = total_size
            return
        end if
        
        ! multi-threaded calculation
        isize  = total_size / nthreads       
        rem    = modulo(total_size, nthreads)
        istart = tid*isize + min(tid, rem) + 1 
        istop  = istart + isize - 1
        if (tid < rem) istop = istop + 1
        
    end subroutine get_block_

    subroutine flush_to_buffer_(fo, current_size, total_size, gbuf)
        !! Append galaxy catalog data to the output buffer. NOTE: in distributed case, 
        !! use with `omp critical` lock for thread safety.

        integer(c_int)    , intent(in)    :: fo
        integer(c_int64_t), intent(inout) :: current_size, total_size
        type(galaxydata_t), intent(in)    :: gbuf(current_size)


        if ( current_size < 1 ) return 

        ! Writing as block of galaxydata_t records
        write(fo) gbuf
        ! NOTE: check if the written records include padding bytes (size with padding 48, 
        ! size without padding 41). Numpy dtype in the app assuming no padding - check
        ! if correct.
        
        ! Total no. of records in the buffer so far 
        total_size = total_size + current_size 
        
        ! Reset buffer 
        current_size = 0

    end subroutine flush_to_buffer_

! Wrapper for C/Python

    subroutine cgenerate_galaxy_catalog(halo_path_c, glxy_path_c, logs_path_c, hmargs,     &
                                        bbox, pktab, np, lnma, lnmb, ns, filt, mrsc_table, & 
                                        seed, nthreads, error                              &
        ) bind(c)
        !! Generate a galaxy catalog using the given halo catalog. (Wrapper around 
        !! `generate_galaxy_catalog` for use from C/Python)

        character(kind=c_char), intent(in) :: halo_path_c(*)
        !! Path to the input halo catalog file: the file must be a binary stream of 
        !! `halodata_t` records in little endian byteorder.

        character(kind=c_char), intent(in) :: glxy_path_c(*)
        !! Path to the output galaxy catalog file: the file will be a binary stream 
        !! of `galaxydata_t` records in little endian byteorder.

        character(kind=c_char), intent(in) :: logs_path_c(*)
        !! Path to the log file: log messages are written to this file.

        type(hmargs_t), intent(in) :: hmargs
        !! Halo model parameters

        real(c_double), intent(in) :: bbox(3,2)
        !! Bounding box for the space containing all halos (used for periodic 
        !! wrapping of galaxy position).

        integer(c_int64_t), intent(in), value :: np !! Power spectrum table size
        real(c_double)    , intent(in)        :: pktab(2, np) 
        !! Matter power spectrum table. First column should be log(k in 1/Mpc) and the 
        !! second log(power in Mpc^3).

        real(c_double)    , intent(in), value :: lnma !! Minimum mass in the table
        real(c_double)    , intent(in), value :: lnmb !! Maximum mass in the table
        integer(c_int64_t), intent(in), value :: ns   !! Size of the internal variance table 
        integer(c_int)    , intent(in), value :: filt !! Filter function for smoothing

        real(c_double), intent(out) :: mrsc_table(4, ns)
        !! A table of halo mass(Msun), lagrangian radius (Mpc), matter variance and halo
        !! concentration parameter values, in log format. This data can be used for any 
        !! later calculations.

        integer(c_int64_t), intent(in), value :: seed
        !! Seed value for random number generators

        integer(c_int), intent(in), value :: nthreads
        !! Number of threads to use

        integer(c_int), intent(out) :: error
        !! Error code (0=success, 1=error)

        integer(c_int)      :: fl
        integer(c_int64_t)  :: i
        character(PATH_LEN) :: halo_path, glxy_path, logs_path
        real(c_double), allocatable :: sigma_table(:,:)

        ! Converting the C strings (`const char*`) into fortran strings: Path has a
        ! maximum length limit (1024 characters), at which it is trimmed. Always make
        ! sure the string has correct length.  
        call c_to_f_string(halo_path_c, halo_path)
        call c_to_f_string(glxy_path_c, glxy_path)
        call c_to_f_string(logs_path_c, logs_path)

        ! Opening log file:
        fl = 8
        open(newunit=fl, file=logs_path, status="replace", action="write", iostat=error) ! log file
        if ( error /= 0 ) stop "error: cannot open log file"    
        
        allocate( sigma_table(3, ns) )

        ! Generate galaxy catalog
        call generate_galaxy_catalog(halo_path, glxy_path, fl,                 &
                                     hmargs, bbox, pktab, np, lnma, lnmb, ns,  &
                                     filt, sigma_table, seed, nthreads, error  &
        )

        ! Writing additional data - table of halo mass, radius, sigma value
        ! and concentration parameter to the end of the pipe.
        do i = 1, ns
            mrsc_table(1,i) = sigma_table(1,i) ! ln(mass in Msun)
            mrsc_table(2,i) = lagrangian_r(hmargs, sigma_table(1,i)) ! ln(radius in Mpc)
            mrsc_table(3,i) = sigma_table(2,i) ! ln(matter variance, sigma)
            mrsc_table(4,i) = log(halo_concentration(hmargs, exp(sigma_table(2,i)))) ! log(halo concentration) 
        end do

        ! Final step:
        deallocate( sigma_table ) 
        write(fl, '(a)') "END" ! Sending a sentinal to mark the end of log section
        close(fl)
        
    end subroutine cgenerate_galaxy_catalog

    subroutine c_to_f_string(cstr, fstr)
        !! Convert C string (null terminated) to allocatable Fortran string.
        character(kind=c_char), intent(in)  :: cstr(*) ! incoming C string
        character(PATH_LEN)   , intent(out) :: fstr
        integer(c_int64_t) :: n

        ! Scan the c string until a null character or maximum allowed length.
        fstr = ''
        n    = 0
        do while ( cstr(n+1) /= c_null_char .and. n <= PATH_LEN )
            n = n + 1
            fstr(n:n) = cstr(n)
        end do
    end subroutine c_to_f_string

end module galaxy_catalog_mod
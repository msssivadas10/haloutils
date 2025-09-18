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
        rho_h = rho_m ! Halo density (TODO: chek if the halo density is rho_m * self.Delta)

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

        real(c_double), intent(out) :: gdata(4,n)
        !! Galaxy positions (columns 1-3) and masses (column 4). 

        integer(c_int64_t) :: i
        real(c_double) :: m_halo, r_halo, c_halo, f, r, theta, phi, Ac, k1, k2, p

        if ( args%n_cen < 1 ) return ! No galaxies in this halo

        m_halo = exp(args%lnm) ! Halo mass in Msun
        r_halo = exp(args%lnr) ! Halo radius in Mpc
        c_halo = args%c        ! Halo concentration parameter

        ! Halo has a central galaxy: the position and mass of this galaxy is same
        ! as that of the parent halo.
        gdata(1:3,1) = args%pos(1:3) ! in Mpc
        gdata(4  ,1) = m_halo        ! in Msun

        if ( args%n_sat < 1 ) return ! No satellite galaxies in this halo
        do i = 1, args%n_sat
            
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
            gdata(1,i+1) = gdata(1,1) + r*sin(theta)*cos(phi)
            gdata(2,i+1) = gdata(2,1) + r*sin(theta)*sin(phi)
            gdata(3,i+1) = gdata(3,1) + r*cos(theta)
            ! Periodic wrapping of coordinates to stay within the bounding box
            call periodic_wrap( gdata(1:3,i+1), args%offset(1:3), args%boxsize(1:3) )
            
            ! Satellite galaxy mass in Msun
            gdata(4,i+1) = gdata(4,1) * f

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

! -- For External Use -- 
! Catalog generation for large catalogs: 

    subroutine generate_galaxy_catalog(fid, seed, nthreads, error) bind(c)
        !! Generate a galaxy catalog using the given halo catalog. 
        !!
        !! **Notes**:
        !! 
        !! This function is basically for use by the Python haloutils package (or 
        !! by an external C/Python code). So, the inputs passed through shared 
        !! files only, except the variables that are mostly different each time, 
        !! like the process ID and no. of threads. But, use from Fortran is also
        !! possible, if the inputs are passed correctly.
        !!
        !! Files required to run this:
        !! - `fid.vars.dat` containing values for various inputs.
        !! - `fid.hbuf.dat` containing the halo catalog as stream of `halodata_t`
        !!
        !! and, produce the output file:
        !! - `fid.gbuf.dat` containing the galaxy catalog as stream of `galaxydata_t`
        !! - `fid.log` (log file)
        !!    
        !! Inputs required are:
        !! - Halo model args as `hmargs_t`...
        !! - Bounding box `float64` array of shape (3, 2) 
        !! - Size of the power spectrum table (`int64`)
        !! - Filter function code (`int`, 0=tophat, 1=gaussian)
        !! - Size of the variance table (`int64`) 
        !! - Mass range for calculating variance values (`float64`), and 
        !! - Power spectrum table, as `float64` array of shape (pktab_size, 2)
        !! 
        !! which should appear in the `vars` file in specified order. Also, note that 
        !! all the arrays are expected to in C order.
        !!

        integer(c_int64_t), intent(in), value :: fid
        !! Unique ID for inter process communication. 
        
        integer(c_int64_t), intent(in), value :: seed
        !! Seed value for random number generators

        integer(c_int), intent(in), value :: nthreads
        !! Number of threads to use

        integer(c_int), intent(out) :: error
        !! Error code (0=success, 1=error)

        ! Input varables: shared through file
        real(c_double)     :: bbox(3, 2) !! Bounding box [min, max]
        type(hmargs_t)     :: hmargs     !! Halo model parameters
        integer(c_int64_t) :: ns         !! Size of the variance spline
        integer(c_int64_t) :: chunk_size
        real(c_double), allocatable :: sigma(:,:)
        !! A spline for interpolating matter variance as function of mass 
        !! in Msun (i.e., ln(sigma) as function of ln(m)).

        integer(c_int)     :: tid, fi, fo, fl, ierr
        integer(c_int64_t) :: rstate(nthreads)
        integer(c_int64_t) :: total_halos, processed_halos, n_halos, total_galaxies
        type(halodata_t), allocatable :: hbuf(:)   ! Halo data
        
        ! -- SETTING UP -- !
        fl = 8; call open_logs(fl, fid) ! opening log file
        
        ! Initialising the random number generator. This RNG works on a state 
        ! private to the thread, with a seed offset by the main seed. So, the 
        ! generated RVs should be a different sequence on each thread.
        do tid = 1, nthreads
            call pcg32_init(rstate(tid), seed + 1000*tid)
        end do
         
        ! Set number of threads
        call omp_set_num_threads(nthreads) 
        write(fl, '("galaxy catalog generation using ",i0," threads")') nthreads

        ! Setting the chunk size multiple of no. of threads, so that halos are 
        ! assigned evenly over multiple threads.
        chunk_size = nthreads*1000 
        allocate( hbuf(chunk_size) ) ! allocate halo data buffer

        ! -- CATALOG GENERATION (MULTI-THREADED) -- !

        ! Loading input variables from the shared workspace file
        call load_shared_data(fid, bbox, hmargs, ns, sigma)

        ! Opening I/O files:
        fo =  9; call open_stream(fo, fid, 'gbuf', 'w') ! catalog output file 
        fi = 10; call open_stream(fo, fid, 'hbuf', 'r') ! halo catalog file

        ! Get no. of halos in the catalog file:
        call get_halo_count(fi, total_halos) 
        write(fl, '("found ",i0," halos...")') total_halos
        
        ! Loading halo data a chunks from the input file
        processed_halos = 0
        total_galaxies  = 0 ! number of galaxies generated so far
        do while ( processed_halos < total_halos )
            
            ! Loading halos
            n_halos = min(chunk_size, total_halos - processed_halos) ! actual chunk size 
            read(fi, iostat=ierr) hbuf(1:n_halos)
            if ( ierr /= 0 ) exit

            ! Generating galaxies 
            call generate_galaxy_catalog2(hbuf(1:n_halos), n_halos, sigma, ns, & 
                                          bbox, hmargs, fo, nthreads, rstate,  &
                                          total_galaxies                       &
            )
            
            processed_halos = processed_halos + n_halos
            write(fl, '("generated ",i0," galaxies from ",i0," halos '//        &
                      '(remaining: ",i0,", completed: ",f5.2,"%)")'             &
            )   total_galaxies, processed_halos, total_halos - processed_halos, &
                100*dble(processed_halos)/dble(total_halos)
            
        end do

        ! Finialize:
        deallocate( hbuf, sigma )
        call close_stream(fi); call close_stream(fo); call close_logs(fl) ! close files
        
        error = 0 ! everything worked as expected :)

    end subroutine generate_galaxy_catalog

    subroutine load_shared_data(fid, bbox, hmargs, ns, sigma)
        !! Load the shared data from workspace file.

        integer(c_int64_t), intent(in)  :: fid   
        type(hmargs_t)    , intent(out) :: hmargs
        real(c_double)    , intent(out) :: bbox(3,2)
        integer(c_int64_t), intent(out) :: ns
        real(c_double)    , intent(out), allocatable :: sigma(:,:)

        integer(c_int)     :: fi, filt 
        integer(c_int64_t) :: i, pktab_size 
        real(c_double)     :: lnma, lnmb, lnm, delta_lnm, lnr, var
        real(c_double), allocatable :: pktab(:,:)  

        fi = 7; call open_stream(fi, fid, "vars", 'r') ! file unit for input

        ! The shared workspace memory is expected to have the following layout:
        read(fi) hmargs      ! first, the halo model args as `hmargs_t`...
        read(fi) bbox        ! then, the bounding box: float64, shape(3, 2), C order...
        read(fi) pktab_size  ! then, size of the power spectrum table: int64...
        read(fi) filt        ! then, filter function code: int (0=tophat, 1=gaussian)
        read(fi) ns          ! then, size of the sigma table: int64 
        read(fi) lnma, lnmb  ! then, mass range for calculating sigma values (float64) 
        
        ! Finally, power spectrum table, as float64 array of shape (pktab_size, 2), 
        ! in C order...
        allocate( pktab(2, pktab_size) )
        do i = 1, pktab_size
            read(fi) pktab(1:2, i)
        end do

        ! Calculating the sigma table using the given power spectrum table 
        allocate( sigma(3, ns) )
        delta_lnm = (lnmb - lnma) / (ns - 1._c_double)
        lnm       = lnma
        do i = 1, ns
            lnr = lagrangian_r(hmargs, lnm)
            var = variance( lnr, 0_c_int, 0_c_int, filt, pktab, pktab_size, 2_c_int )
            sigma(:, i) = [ lnm, 0.5_c_double*log(var) ]
            lnm = lnm + delta_lnm
        end do
        call generate_cspline(ns, sigma) ! creating cubic spline for interpolation

        deallocate( pktab ) 
        call close_stream(fi)
        
    end subroutine load_shared_data

    subroutine generate_galaxy_catalog2(hbuf, n_halos, sigma, ns, bbox, hmargs, &
                                        fo, nthreads, rstate, ngals             &
        )
        !! Generate galaxy catalog from halo catalog.

        type(halodata_t)  , intent(in)    :: hbuf(n_halos)   
        integer(c_int64_t), intent(in)    :: n_halos
        real(c_double)    , intent(in)    :: sigma(3,ns)
        integer(c_int64_t), intent(in)    :: ns
        real(c_double)    , intent(in)    :: bbox(3,2)
        type(hmargs_t)    , intent(in)    :: hmargs
        integer(c_int)    , intent(in)    :: fo   
        integer(c_int)    , intent(in)    :: nthreads
        integer(c_int64_t), intent(inout) :: rstate(nthreads), ngals

        integer(c_int64_t), parameter :: buffer_size = 10000

        type(cgargs_t)     :: args 
        integer(c_int)     :: nt, tid
        integer(c_int64_t) :: ng_thread(nthreads), istart, istop, i, ng, nh, p 
        integer(c_int64_t), allocatable :: meta(:,:) ! Halo index and galaxy count 
        real(c_double)    , allocatable :: gbuf(:,:) ! Galaxy position and mass

        ng_thread(1:nthreads) = 0_c_int64_t ! number of galaxies generated in a thread

        !$omp parallel &
        !$omp private(nt, tid, istart, istop, i, gbuf, ng, nh, p, args)

        nt  = omp_get_num_threads() ! no. of threads (should be same as `nthreads`)
        tid = omp_get_thread_num()  ! thread ID
        call get_block_range(n_halos, tid, nt, istart, istop)

        args%rstate       = rstate(tid+1) ! set RNG state
        args%boxsize(1:3) = bbox(1:3,2) - bbox(1:3,1) 
        args%offset(1:3)  = bbox(1:3,1) 

        ! Allocate galaxy buffer and metadata: 
        allocate( gbuf(4,buffer_size), meta(2,buffer_size) ) 

        p  = 0 ! size of data buffer: index to append is p+1
        nh = 0 ! size of metadata buffer: number of halos with galaxies
        do i = istart, istop

            ! Copy halo data to local args
            args%pos = hbuf(i)%pos         ! position
            args%lnm = log( hbuf(i)%mass ) ! mass
            args%s   = exp( interpolate(args%lnm, ns, sigma) ) ! matter variance
            
            ! Setting up 
            call setup_catalog_generation(hmargs, args)

            ng = args%n_cen + args%n_sat ! total number of galaxies in this halo 
            if ( ng < 1 ) cycle
            
            if ( p + ng  <= buffer_size ) then
                ! Local buffer has not enough data for saving the new galaxy data:
                ! Current data is written to the main output buffer and the start 
                ! pointers are restted to the start. 

                !$omp critical
                call flush_to_buffer( fo, nh, p, meta, gbuf, buffer_size )
                !$omp end critical
            end if    
            
            ! Generating the galaxies
            call generate_galaxies( hmargs, args, ng, gbuf(1:4,p+1:p+ng) )
            
            meta(:,nh+1)     = [ hbuf(i)%id, ng ]
            nh               = nh + 1
            p                = p  + ng
            ng_thread(tid+1) = ng_thread(tid+1) + ng
        end do        
        if ( nh > 0 ) then ! write remaining data to the buffer...
            !$omp critical
            call flush_to_buffer( fo, nh, p, meta, gbuf, buffer_size )
            !$omp end critical
        end if 
        deallocate(gbuf, meta)

        ! Save final RNG state for continuing the sequence for next chunk
        rstate(tid+1) = args%rstate 

        !$omp end parallel

        ngals = ngals + sum(ng_thread) ! total number of galaxies generated so far...
        
    end subroutine generate_galaxy_catalog2

    subroutine flush_to_buffer(fo, halo_count, galaxy_count, meta, gbuf, bufsize)
        !! Append galaxy catalog data to the output buffer. 
        !! NOTE: in distributed case, use with `omp critical` lock.

        integer(c_int)    , intent(in)    :: fo
        integer(c_int64_t), intent(inout) :: halo_count, galaxy_count
        integer(c_int64_t), intent(in)    :: bufsize, meta(2,bufsize)
        real(c_double)    , intent(in)    :: gbuf(4,bufsize)

        integer(c_int64_t) :: i, offset, p, gal_count, halo_id

        offset = 1
        do i = 1, halo_count
            halo_id   = meta(1,i) 
            gal_count = meta(2,i) ! no. of galaxies for this halo

            ! First item in each block is the central galaxy...
            write(fo) halo_id, gbuf(1:4,offset), 'c'
            
            ! Remaining items are for satellite galaxy...
            do p = offset+1, offset + gal_count
                write(fo) halo_id, gbuf(1:4,p), 's' 
            end do
            offset = offset + gal_count
        end do

        ! Reset buffers:
        halo_count   = 0
        galaxy_count = 0

    end subroutine flush_to_buffer

! Helper functions:

    subroutine open_logs(fu, fid)
        !! Open a log file (plain text).

        integer(c_int)    , intent(inout) :: fu
        integer(c_int64_t), intent(in)    :: fid

        integer(c_int) :: error
        character(256) :: fn

        write(fn, '(i0,".log")') fid ! filename for logs
        open(newunit=fu, file=fn, status="replace", action="write", &
             iostat=error                                           &
        ) ! log file
        if ( error /= 0 ) stop "error: cannot open log file"     
        
    end subroutine open_logs

    subroutine close_logs(fu)
        !! Close log file.
        integer(c_int), intent(inout) :: fu
    
        ! Sending a sentinal to mark the end of log file
        write(fu, '(a)') 'END' 
        close(fu)

    end subroutine close_logs

    subroutine open_stream(fu, fid, suffix, mode)
        !! Open a binary stream with little endian byteorder.

        character(len=4)  , intent(in)    :: suffix
        character(len=1)  , intent(in)    :: mode
        integer(c_int)    , intent(inout) :: fu
        integer(c_int64_t), intent(in)    :: fid
        
        integer(c_int) :: error
        character(256) :: fn

        write(fn,'(i0,".",a,".dat")') trim(suffix), fid ! filename
        select case( mode )
        case ( 'r' )
            ! Open in read mode
            open(newunit=fu, file=fn, access='stream', form='unformatted', &
                 convert='little_endian', status='old', action='read',     &
                 iostat=error                                              &
            )
        case ( 'w' )
            ! Open in write mode
            open(newunit=fu, file=fn, access='stream', form='unformatted',  &
                 convert='little_endian', status='replace', action='write', &
                 iostat=error                                               &
            )
        end select
        if ( error /= 0 ) stop "error: cannot open file: "//trim(fn) 
        
    end subroutine open_stream

    subroutine close_stream(fu)
        !! Close a binary file.
        integer(c_int), intent(inout) :: fu
        close(fu)
    end subroutine close_stream

    subroutine get_halo_count(fi, total_halos)
        !! Get the number of halos from the input catalog.

        integer(c_int)    , intent(in) :: fi
        integer(c_int64_t), intent(out) :: total_halos

        integer(c_int64_t) :: file_size_bytes
        integer(c_int64_t), parameter :: item_size_bytes = c_sizeof(halodata_t( &
            0_c_int64_t,                                                        &
            [ 0._c_double, 0._c_double, 0._c_double ],                          &
            0._c_double                                                         &   
        )) !! Size of `halodata_t`: should be 40

        ! Get the number of halos from the file: since the input file is expected 
        ! to be a binary stream of `halodata_t` (size: 40 bytes), number of halos
        ! in the file can be calculated as `file_size_bytes / item_size_bytes` 
        inquire(fi, size=file_size_bytes)
        total_halos = file_size_bytes / item_size_bytes
        
    end subroutine get_halo_count

    subroutine get_block_range(total_size, tid, nthreads, istart, istop)
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
        
    end subroutine get_block_range

end module galaxy_catalog_mod
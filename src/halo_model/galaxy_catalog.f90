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
    public :: setup_catalog_generation, generate_galaxies!, generate_galaxy_catalog
    
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

! Catalog generation for large catalogs:

    subroutine generate_galaxy_catalog(fid, seed, nthreads, error) bind(c)
        !! Generate a galaxy catalog using the given halo catalog. 

        integer(c_int64_t), intent(in), value :: fid
        !! Unique ID for inter process communication. 
        
        integer(c_int64_t), intent(in), value :: seed
        !! Seed value for random number generators

        integer(c_int), intent(in), value :: nthreads
        !! Number of threads to use

        integer(c_int), intent(out) :: error
        !! Error code (0=success, 1=error)

        integer(c_int64_t), parameter :: item_size_bytes = c_sizeof( &
            halodata_t(                                    &
                0_c_int64_t,                               &
                [ 0._c_double, 0._c_double, 0._c_double ], &
                0._c_double )                              &   
        ) !! Size of `halodata_t`: should be 40

        real(c_double)     :: bbox(3, 2) !! Bounding box [min, max]
        type(hmargs_t)     :: hmargs     !! Halo model parameters
        integer(c_int64_t) :: ns         !! Size of the variance spline
        integer(c_int64_t) :: chunk_size
        real(c_double), allocatable :: sigma(:,:)
        !! A spline for interpolating matter variance as function of mass 
        !! in Msun (i.e., ln(sigma) as function of ln(m)).

        character(len=256) :: fn
        integer(c_int)     :: tid, fi, fl, ierr
        integer(c_int64_t) :: file_size_bytes, rstate(nthreads)
        integer(c_int64_t) :: n_halos_total, n_halos_processed, n_halos, n_galaxies
        type(halodata_t), allocatable :: hbuf(:)   ! Halo data

        ! -- SETTING UP -- !

        ! Loading shared data from the workspace file. 
        ! NOTE: always make sure that the workspace file has the correct 
        ! layout: `hmargs_t, bbox, ns, sigma` and arrays are stored in C
        ! order - not fortran order.
        call load_shared_workspace_(fid, bbox, hmargs, ns, sigma)
        
        ! Initialising the random number generator. This RNG works on a state 
        ! private to the thread, with a seed offset by the main seed. So, the 
        ! generated RVs should be a different sequence on each thread.
        do tid = 1, nthreads
            call pcg32_init(rstate(tid), seed + 1000*tid)
        end do

        ! Allocate halo data buffer
        allocate( hbuf(chunk_size) )

        ! Opening log file:
        fl = 8 ! file unit for logs
        write(fn, '(i0,".log")') fid ! filename for logs
        open(newunit=fl, file=fn, status="replace", action="write") ! log file

        ! Creating galaxy catalog output file: 
        fi = 9 ! file unit for output
        write(fn, '(i0,".gbuf.dat")') fid ! filename for main output
        open(newunit=fi, file=fn, access='stream', form='unformatted', &
             convert='little_endian', status='replace', action='write' &
        )
        close(fi)

        ! Opening halo catalog file:
        fi = 9 ! file unit for input
        write(fn, '(i0,".hbuf.dat")') fid ! filename for input
        open(newunit=fi, file=fn, access='stream', form='unformatted', &
             convert='little_endian', status='old', action='read'      &
        ) ! input file

        ! -- CATALOG GENERATION (MULTI-THREADED) -- !

        ! Get the number of halos from the file: since the input file is expected 
        ! to be a binary stream of `halodata_t` (size: 40 bytes), number of halos
        ! in the file can be calculated as `file_size_bytes / item_size_bytes` 
        inquire(fi, size=file_size_bytes)
        n_halos_total = file_size_bytes / item_size_bytes
        
        if ( n_halos_total < 1 ) return
        write(fl, '("found ",i0," halos...")') n_halos_total
                
        ! Set number of threads
        call omp_set_num_threads(nthreads) 
        write(fl, '("galaxy catalog generation using ",i0," threads")') nthreads
        ! Setting the chunk size multiple of no. of threads, so that halos are 
        ! assigned evenly over multiple threads.
        chunk_size = nthreads*1000 

        ! Loading halo data a chunks from the input file
        n_halos_processed = 0
        n_galaxies        = 0_c_int64_t ! number of galaxies generated so far
        do while ( n_halos_processed < n_halos_total )
            
            ! Loading halos
            n_halos = min(chunk_size, n_halos_total - n_halos_processed) ! actual chunk size 
            read(fi, iostat=ierr) hbuf(1:n_halos)
            if ( ierr /= 0 ) exit

            ! Generating galaxies 
            call generate_galaxy_catalog2(hbuf(1:n_halos), n_halos, sigma, ns, bbox, &
                                          hmargs, fid, nthreads, rstate, n_galaxies  &
            )

            n_halos_processed = n_halos_processed + n_halos
            write(fl, '("generated ",i0," galaxies from ",i0," halos out of ",i0)') &
                n_galaxies, n_halos_processed, n_halos_total
            
        end do

        close(fi)
        deallocate( hbuf  )
        deallocate( sigma )

        write(fl, '(a)') 'END' ! sentinal to mark the end of log file
        close(fl) ! close log file
        
        error = 0 ! everything worked as expected :)

    end subroutine generate_galaxy_catalog

    subroutine generate_galaxy_catalog2(hbuf, n_halos, sigma, ns, bbox, hmargs, &
                                        fid, nthreads, rstate, ngals            &
        )
        !! Generate galaxy catalog from halo catalog.

        type(halodata_t)  , intent(in)    :: hbuf(n_halos)   
        integer(c_int64_t), intent(in)    :: n_halos
        real(c_double)    , intent(in)    :: sigma(3,ns)
        integer(c_int64_t), intent(in)    :: ns
        real(c_double)    , intent(in)    :: bbox(3,2)
        type(hmargs_t)    , intent(in)    :: hmargs
        integer(c_int64_t), intent(in)    :: fid   
        integer(c_int)    , intent(in)    :: nthreads
        integer(c_int64_t), intent(inout) :: rstate(nthreads), ngals

        integer(c_int64_t), parameter :: buffer_size = 10000

        type(cgargs_t)     :: args 
        integer(c_int)     :: nt, tid
        integer(c_int64_t) :: ng_thread(nthreads), istart, istop, i, ng, nh, p 
        real(c_double)    , allocatable :: gbuf(:,:) ! Galaxy position and mass
        integer(c_int64_t), allocatable :: meta(:,:) ! Halo index and galaxy count 

        ng_thread(1:nthreads) = 0_c_int64_t ! number of galaxies generated in a thread

        !$OMP PARALLEL PRIVATE(nt, tid, istart, istop, i, gbuf, ng, nh, p, args)

        nt  = omp_get_num_threads() ! Number of threads (should be same as `nthreads`)
        tid = omp_get_thread_num()  ! thread ID
        call get_block_range(n_halos, tid, nt, istart, istop)

        args%rstate       = rstate(tid+1) ! set RNG state
        args%boxsize(1:3) = bbox(1:3,2) - bbox(1:3,1) 
        args%offset(1:3)  = bbox(1:3,1) 

        ! Allocate galaxy buffer and metadata: 
        allocate( gbuf(4,buffer_size) ) ! (position, mass) buffer
        allocate( meta(2,buffer_size) ) ! (halo id, count) buffer

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
            
            ! Check if the buffer has enough space to append the data. If not
            ! enough space, current data is flushed to the main output buffer 
            ! and the start insertion pointer reseted.
            if ( p + ng  <= buffer_size ) then
                !$OMP CRITICAL
                call flush_to_output_buffer( fid, nh, p, meta(:,:nh), gbuf(:,p) )
                !$OMP END CRITICAL
                
                p  = 0 ! reset buffers
                nh = 0
            end if    
            
            ! Generating the galaxies
            call generate_galaxies( hmargs, args, ng, gbuf(1:4,p+1:p+ng) )
            
            meta(:,nh+1)     = [ hbuf(i)%id, ng ]
            nh               = nh + 1
            p                = p  + ng
            ng_thread(tid+1) = ng_thread(tid+1) + ng
        end do        
        if ( nh > 0 ) then ! write remaining data
            !$OMP CRITICAL
            call flush_to_output_buffer( fid, nh, p, meta(:,:nh), gbuf(:,p) )
            !$OMP END CRITICAL
        end if 
        deallocate(gbuf)
        deallocate(meta)

        ! Save final RNG state for continuing the sequence for next chunk
        rstate(tid+1) = args%rstate 

        !$OMP END PARALLEL

        ngals = ngals + sum(ng_thread) ! total number of galaxies generated so far...
        
    end subroutine generate_galaxy_catalog2

    subroutine get_block_range(total_size, tid, nthreads, istart, istop)
        !! Get bounds for a block allotted to a thread in a distributed calculation. 
        integer(c_int)    , intent(in)  :: tid, nthreads
        integer(c_int64_t), intent(in)  :: total_size 
        integer(c_int64_t), intent(out) :: istart, istop

        integer(c_int64_t) :: jsize, rem

        if ( nthreads == 1 ) then
            istart = 1
            jsize  = total_size
            return
        end if
        
        jsize  = total_size / nthreads       
        rem    = modulo(total_size, nthreads)
        istart = tid*jsize + min(tid, rem) + 1 
        istop  = istart + jsize - 1
        if (tid < rem) istop = istop + 1
        
    end subroutine get_block_range

    subroutine flush_to_output_buffer(fid, nh, ng, meta, gbuf)
        !! Append galaxy catalog data to the output buffer. 
        integer(c_int64_t), intent(in)  :: fid
        integer(c_int64_t), intent(in)  :: nh, ng, meta(2,nh)
        real(c_double)    , intent(in)  :: gbuf(4,ng)

        integer(c_int64_t) :: i, offset, p, gal_count, halo_id

        ! open output file

        offset = 1
        do i = 1, nh
            halo_id   = meta(1,i)
            gal_count = meta(2,i) ! no. of galaxies for this halo
            ! write halo_id, gbuf(1:4,offset), 'c' ! central galaxy data
            do p = offset+1, offset + gal_count
                ! write halo_id, gbuf(1:4,p), 's' ! satellite galaxy data
            end do
            offset = offset + gal_count
        end do

        ! close file
        
    end subroutine flush_to_output_buffer
    

end module galaxy_catalog_mod
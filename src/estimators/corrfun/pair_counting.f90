module pair_counting_mod
    !! Count pairs between two sets of points. 
    
    use omp_lib
    use iso_c_binding
    use iso_fortran_env, only: output_unit
    use spatial_hash_mod
    use pair_utils_mod
    implicit none

    private
    public :: count_pairs
    
contains

    subroutine count_pairs(pid, auto, periodic, box, nr, rbins, cnts, ncells, &
                           npts1, grid_info1, grid_data1, pos1,               &
                           npts2, grid_info2, grid_data2, pos2,               &
                           nthreads, verbose, error_code                      &
        ) bind(c)
        !! Count the number of pairs between two sets of points, given their grid
        !! hash representation. For each set of points, a grid hash is composed of
        !! the position buffer, list of indices of points in each cell (as stacked
        !! array) and an array of `(block_start, block_size)` tuples mapping the 
        !! cells to its block on the index stack.

        integer(c_int64_t), intent(in), value :: pid
        !! Process ID
        
        integer(c_int), intent(in), value :: auto
        !! Flag for taking auto correlation (if both sets are same)
     
        integer(c_int), intent(in), value :: periodic
        !! Flag for using periodic boundary conditions
        
        type(boxinfo_t), intent(in) :: box
        !! Details about the space (both sets must be in the same box)

        integer(c_int64_t), intent(in), value :: nr        
        real(c_double)    , intent(in)        :: rbins(nr) !! Distance bin edges
        integer(c_int64_t), intent(out)       :: cnts(nr)  !! Pair counts.
        ! For `nr` bin edges, there will be `nr-1` bins. So, `cnts(nr-1)` is 
        ! not used.

        integer(c_int64_t), intent(in), value :: ncells
        !! Number of cells in the grid (same for both sets).

         ! -- Points and grid for set-1:
        integer(c_int64_t), intent(in), value :: npts1              !! Number of points (1)    
        real(c_double)    , intent(in)        :: pos1(3,npts1)      !! Position buffer (1)    
        integer(c_int64_t), intent(in)        :: grid_data1(npts1)  !! Grid representation (1)
        type(cinfo_t)     , intent(in)        :: grid_info1(ncells) !! Grid details (1)        
        
        ! -- Points and grid for set-2:
        integer(c_int64_t), intent(in), value :: npts2              !! Number of points (2)      
        real(c_double)    , intent(in)        :: pos2(3,npts2)      !! Position buffer (2)      
        integer(c_int64_t), intent(in)        :: grid_data2(npts2)  !! Grid representation (2)  
        type(cinfo_t)     , intent(in)        :: grid_info2(ncells) !! Grid details (2)          
        
        integer(c_int), intent(in), value :: nthreads
        !! Number of threads to use

        integer(c_int), intent(in), value :: verbose
        !! Flag for controlling messages (0=no logs, 1=log to stdout, 2=log to pid.log)
        
        integer(c_int), intent(out) :: error_code
        !! Error code (0=success, 1=error)

        character(256)     :: fn, log_fn
        real(c_double)     :: r2bins(nr)
        integer(c_int)     :: fi, ios, fl
        integer(c_int64_t) :: file_size, n_pairs, n_pairs_total, n_pairs_processed
        integer(c_int64_t) :: chunk_size, j1, j2
        integer(c_int64_t), allocatable :: pairs(:,:)

        if ( verbose == 2 ) then
            ! Log messages are written to a special pid.log file
            fl = 1 

            ! Opening log file:
            write(log_fn, '(i0,".log")') pid ! filename for logs
            open(newunit=fl, file=log_fn, status="replace", action="write") ! log file

        else if ( verbose /= 0 ) then
            ! Log messages are written to stdout
            fl = output_unit
        end if

        error_code = 1

        call omp_set_num_threads(nthreads) ! set number of threads

        ! Precalculating cell pairs
        if ( verbose /= 0 ) write(fl, '("info: ",a)') 'listing cell pairs...'
        call enumerate_cell_pairs(pid, npts1, npts2, ncells, rbins(nr), box,  &
                                  periodic, grid_info1, grid_info2, nthreads, &
                                  error_code                                  &
        )
        if ( error_code /= 0 ) then
            if ( verbose /= 0 .and. error_code == 3 ) then
                write(  &
                    fl, &
                    '("error: gridsize (",2(i0,","),i0,") does not match with ",i0," cells")' &
                ) box%gridsize, ncells
            else if ( verbose /= 0 .and. error_code == 2 ) then
                write(fl, '("error: ",a)') 'incompatible cellsize and gridsize' 
            end if
            return
        end if 

        r2bins = rbins**2
        cnts   = 0_c_int64_t

        ! First, processing the 'normal' cell pairs stored in `pid.cplist.bin`.
        ! Pairs in these set are processed in parallel. 
        fi = 9 ! file unit for input
        write(fn, '(i0,".cplist.bin")') pid ! filename for input
        open(newunit=fi, file=fn, access='stream', form='unformatted', &
             convert='little_endian', status='old', action='read'      &
        ) ! input file
        inquire(fi, size=file_size)
        if ( file_size > 0 ) then

            ! Calculate the number if pairs: the file is a stream of int64 cell
            ! index pairs.
            n_pairs_total = file_size / (2*c_sizeof(1_c_int64_t))
            if ( verbose /= 0 ) then
                write(  &
                fl, &
                '("info: found ",i0," cell pairs in ",a," (filesize: ",i0," bytes)")' &
                ) n_pairs_total, trim(fn), file_size      
            end if

            ! Allocating storage for chunks
            chunk_size = n_pairs_total / nthreads
            if ( modulo(n_pairs_total, nthreads) > 0 ) chunk_size = chunk_size + 1 
            chunk_size = min(chunk_size, 10000) ! limit memory
            allocate( pairs(2, chunk_size) )
            
            ! Loading pairs as chunks, and processing items in parallel...
            n_pairs_processed = 0
            do while ( n_pairs_processed < n_pairs_total )
                n_pairs = min( chunk_size, n_pairs_total - n_pairs_processed )
                read(fi, iostat=ios) pairs(:, 1:n_pairs)
                if ( ios /= 0 ) exit ! EOF or error 
                
                ! Counting pairs
                call distributed_pair_count_cells(pid, auto, periodic, n_pairs, pairs, &
                                                  box, nr, r2bins, cnts, ncells,       &
                                                  npts1, grid_info1, grid_data1, pos1, &
                                                  npts2, grid_info2, grid_data2, pos2, &
                                                  nthreads                             &
                )
                
                n_pairs_processed = n_pairs_processed + n_pairs
                if ( verbose /= 0 ) then
                    write(fl, '("info: processed ",i0," of ",i0," cell pairs (",f6.2,"%)")') & 
                        n_pairs_processed,                        &
                        n_pairs_total,                            &
                        percent(n_pairs_processed, n_pairs_total) ! percentage completed
                end if
            end do

            deallocate( pairs )

        end if
        if ( verbose /= 0 ) write(fl, '("info: deleting ",a)') trim(fn)
        close(fi, status='delete') ! file is deleted on close

        ! Second, special cell pairs (with much larger counts compared to average)
        ! are processed. For these, parellization is over the point pairs, as this 
        ! is much efficient than over cell pairs.
        fi = 9 ! file unit for input
        write(fn, '(i0,".cplist.spl.bin")') pid ! filename for input
        open(newunit=fi, file=fn, access='stream', form='unformatted', &
             convert='little_endian', status='old', action='read'      &
        ) ! input file
        inquire(fi, size=file_size)
        if ( file_size > 0 ) then
            
            ! Calculate the number if pairs: the file is a stream of int64 cell
            ! index pairs.
            n_pairs_total = file_size / (2*c_sizeof(1_c_int64_t))
            if ( verbose /= 0 ) then
                write(  &
                    fl, &
                    '("info: found ",i0," cell pairs in ",a," (filesize: ",i0," bytes)")' &
                ) n_pairs_total, trim(fn), file_size      
            end if

            ! Loading pairs and processing point pairs in parallel...
            n_pairs_processed = 0
            do while ( n_pairs_processed < n_pairs_total )
                read(fi, iostat=ios) j1, j2
                if ( ios /= 0 ) exit ! EOF or error 
                
                ! Counting pairs
                call count_pairs_parallel(pid, auto, periodic, j1, j2, box,    &
                                          nr, r2bins, cnts, ncells,            &
                                          npts1, grid_info1, grid_data1, pos1, &
                                          npts2, grid_info2, grid_data2, pos2, &
                                          nthreads                             &
                )

                n_pairs_processed = n_pairs_processed + 1
                if (( verbose /= 0 ) .and.                      & 
                    ( modulo(n_pairs_processed, 1000) == 0 .or. &
                     n_pairs_processed == n_pairs_total )       &
                ) then
                    write(fl, '("info: processed ",i0," of ",i0," cell pairs (",f6.2,"%)")') & 
                        n_pairs_processed,                        &
                        n_pairs_total,                            &
                        percent(n_pairs_processed, n_pairs_total) ! percentage completed
                end if
            end do
            
        end if
        if ( verbose /= 0 ) write(fl, '("info: deleting ",a)') trim(fn)
        close(fi, status='delete') ! file is deleted on close

        if ( verbose == 2 ) close(fl) ! closing log file (if opened)
        error_code = 0
        
    end subroutine count_pairs

! Parallelized pair counting over cell pairs:

    subroutine distributed_pair_count_cells(pid, auto, periodic, n_pairs, pairs,  &
                                            box, nr, r2bins, cnts, ncells,        &
                                            npts1, grid_info1, grid_data1, pos1,  &
                                            npts2, grid_info2, grid_data2, pos2,  &
                                            nthreads                              &
        )
        !! Paralelly count pairs over a set of cell pairs.

        integer(c_int64_t), intent(in)    :: pid
        integer(c_int)    , intent(in)    :: auto, periodic, nthreads
        integer(c_int64_t), intent(in)    :: n_pairs, pairs(2,n_pairs)
        integer(c_int64_t), intent(in)    :: nr, ncells, npts1, npts2              
        real(c_double)    , intent(in)    :: r2bins(nr), pos1(3,npts1), pos2(3,npts2)
        integer(c_int64_t), intent(in)    :: grid_data1(npts1), grid_data2(npts2)  
        type(cinfo_t)     , intent(in)    :: grid_info1(ncells), grid_info2(ncells) 
        type(boxinfo_t)   , intent(in)    :: box
        integer(c_int64_t), intent(inout) :: cnts(nr)  

        integer(c_int)     :: tid
        integer(c_int64_t) :: p, j1, j2
        integer(c_int64_t), allocatable :: local_cnts(:)

        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(tid, p, j1, j2, local_cnts)

        allocate( local_cnts(nr) )
        local_cnts = 0_c_int64_t
        
        !$OMP DO SCHEDULE(static)
        do p = 1, n_pairs
            j1 = pairs(1, p)
            j2 = pairs(2, p)
            call count_pairs_serial(auto, periodic, j1, j2, box,         &
                                    nr, r2bins, local_cnts, ncells,      &
                                    npts1, grid_info1, grid_data1, pos1, &
                                    npts2, grid_info2, grid_data2, pos2  &
            )
        end do
        !$OMP END DO
        
        ! Temporarily save the counts
        tid = omp_get_thread_num() + 1 ! thread ID 
        call save_counts(pid, tid, nr, local_cnts) 
        deallocate( local_cnts )

        !$OMP END PARALLEL

        ! Load count data from each temporary file and update the global count 
        call reduce_counts(pid, nthreads, nr, cnts)
        
    end subroutine distributed_pair_count_cells

    subroutine count_pairs_serial(auto, periodic, j1, j2,              &
                                  box, nr, r2bins, cnts, ncells,       &
                                  npts1, grid_info1, grid_data1, pos1, &
                                  npts2, grid_info2, grid_data2, pos2  &
        )
        !! Sequencially count pairs between points from two cells j1 and j2.

        integer(c_int)    , intent(in)    :: auto, periodic
        integer(c_int64_t), intent(in)    :: j1, j2, nr, ncells, npts1, npts2              
        real(c_double)    , intent(in)    :: r2bins(nr), pos1(3,npts1), pos2(3,npts2)
        integer(c_int64_t), intent(in)    :: grid_data1(npts1) , grid_data2(npts2)  
        type(cinfo_t)     , intent(in)    :: grid_info1(ncells), grid_info2(ncells) 
        type(boxinfo_t)   , intent(in)    :: box
        integer(c_int64_t), intent(inout) :: cnts(nr) 

        real(c_double)     :: p1(3), p2(3)
        integer(c_int64_t) :: i1, i2, i1_start, i2_start, i1_stop, i2_stop

        ! Range of the block for cell j1
        i1_start = grid_info1(j1)%start
        i1_stop  = i1_start + grid_info1(j1)%count - 1
        
        ! Range of the block for cell j2
        i2_start = grid_info2(j2)%start
        i2_stop  = i2_start + grid_info2(j2)%count - 1

        do i1 = i1_start, i1_stop
            do i2 = i2_start, i2_stop
                if ( auto /= 0 .and. grid_data2(i2) <= grid_data1(i1) ) then
                    !! Autocorrelation: both sets are the same - only count unique 
                    !! pairs...
                    cycle
                endif 
                p1 = pos1( 1:3, grid_data1(i1) ) 
                p2 = pos2( 1:3, grid_data2(i2) )
                call locate_bin_and_update(p1, p2, box%boxsize,       &
                                           r2bins, cnts, nr, periodic &
                )
            end do
        end do
        
    end subroutine count_pairs_serial

! Parallelized pair counting over point pairs:

    subroutine count_pairs_parallel(pid, auto, periodic, j1, j2,         &
                                    box, nr, r2bins, cnts, ncells,       &
                                    npts1, grid_info1, grid_data1, pos1, &
                                    npts2, grid_info2, grid_data2, pos2, &
                                    nthreads                             &
        )
        !! Parallely count pairs between points from two cells j1 and j2.

        integer(c_int)    , intent(in)    :: auto, periodic, nthreads
        integer(c_int64_t), intent(in)    :: pid, j1, j2, nr, ncells, npts1, npts2              
        real(c_double)    , intent(in)    :: r2bins(nr), pos1(3,npts1), pos2(3,npts2)
        integer(c_int64_t), intent(in)    :: grid_data1(npts1) , grid_data2(npts2)  
        type(cinfo_t)     , intent(in)    :: grid_info1(ncells), grid_info2(ncells) 
        type(boxinfo_t)   , intent(in)    :: box
        integer(c_int64_t), intent(inout) :: cnts(nr)
        
        real(c_double)     :: p1(3), p2(3)
        integer(c_int)     :: tid
        integer(c_int64_t) :: i1, i2, i1_start, i2_start, i1_count, i2_count
        integer(c_int64_t) :: i_stop, i
        integer(c_int64_t), allocatable :: local_cnts(:)

        ! Range of the block for cell j1
        i1_start = grid_info1(j1)%start
        i1_count = grid_info1(j1)%count
        
        ! Range of the block for cell j2
        i2_start = grid_info2(j2)%start
        i2_count = grid_info2(j2)%count

        ! Total number of point pairs
        i_stop = i1_count * i2_count

        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(tid, i, i1, i2, p1, p2, local_cnts)

        allocate( local_cnts(nr) )
        local_cnts = 0_c_int64_t
        
        !$OMP DO SCHEDULE(static)
        do i = 1, i_stop
            i1 = modulo(i, i1_count) + i1_start 
            i2 = modulo(i / i1_count, i2_count) + i2_start
            
            if ( auto /= 0 .and. grid_data2(i2) <= grid_data1(i1) ) then
                !! Autocorrelation: both sets are the same - only count unique 
                !! pairs...
                cycle
            endif 
            p1 = pos1( 1:3, grid_data1(i1) ) 
            p2 = pos2( 1:3, grid_data2(i2) )
            call locate_bin_and_update(p1, p2, box%boxsize, r2bins, &
                                       local_cnts, nr, periodic     &
            )
        end do
        !$OMP END DO

        ! Temporarily save the counts:
        tid = omp_get_thread_num() + 1 ! thread ID 
        call save_counts(pid, tid, nr, local_cnts) 
        deallocate( local_cnts )

        !$OMP END PARALLEL
        
        ! Load count data from each temporary file and update the global count 
        call reduce_counts(pid, nthreads, nr, cnts)
        
    end subroutine count_pairs_parallel

! Helper functions:

    subroutine save_counts(pid, tid, nr, cnts)
        !! Write count data to a temporary file.
        integer(c_int64_t), intent(in) :: pid, nr
        integer(c_int)    , intent(in) :: tid
        integer(c_int64_t), intent(in) :: cnts(nr)  

        character(256) :: fn
        integer(c_int) :: fu

        fu  = 10 + tid ! file unit for this thread
        write(fn, '(i0,".",i0,".cnts.tmp")' ) pid, tid ! filename for this thread
        open(newunit=fu, file=fn, access='stream', form='unformatted',     &
            convert='little_endian', status='unknown', position='append',  &
            action='write'                                                 &
        )
        write(fu) cnts
        close(fu)
        
    end subroutine save_counts

    subroutine reduce_counts(pid, nthreads, nr, cnts)
        !! Accumulate counts from temporary data.
        integer(c_int64_t), intent(in)    :: pid, nr
        integer(c_int)    , intent(in)    :: nthreads
        integer(c_int64_t), intent(inout) :: cnts(nr)  

        character(256) :: fn
        integer(c_int) :: fu, tid
        integer(c_int64_t), allocatable :: local_cnts(:)

        allocate( local_cnts(nr)  )

        fu = 10
        do tid = 1, nthreads
            write(fn, '(i0,".",i0,".cnts.tmp")' ) pid, tid ! filename for this thread
            open(newunit=fu, file=fn, access='stream', form='unformatted', &
                 convert='little_endian', status='old', action='read'      &
            )
            read(fu) local_cnts
            cnts = cnts + local_cnts
            close(fu, status='delete') ! file is deleted on close 
        end do
        
        deallocate( local_cnts  )
        
    end subroutine reduce_counts

    subroutine locate_bin_and_update(p1, p2, boxsize, r2bins, cnts, nr, periodic) 
        !! Locate the bin for the given point pair using a binary search on the   
        !! bin edges array and update the count. Do nothing if the distance is 
        !! out of range.
        real(c_double)    , intent(in)    :: p1(3), p2(3), boxsize(3), r2bins(nr)
        integer(c_int)    , intent(in)    :: periodic
        integer(c_int64_t), intent(in)    :: nr
        integer(c_int64_t), intent(inout) :: cnts(nr)
        integer(c_int64_t) :: i, j, m
        real(c_double)     :: dist2, dx(3)

        ! Calculate the distance between points
        dx = p2 - p1
        if ( periodic /= 0 ) then
            ! Correcting the coordinate distances for periodic boundary
            dx = dx - boxsize*nint(dx / boxsize) 
        end if 
        dist2 = dot_product(dx, dx) ! squared distance between the points  
    
        ! Value is out of range:
        if ( dist2 < r2bins(1) .or. dist2 >= r2bins(nr) ) return

        ! Binary search to find the bin index such that r2bins(i) <= dist2 < r2bins(i+1)
        i = 1_c_int64_t
        j = nr
        do while ( j - i > 1 )
            m = (i + j) / 2
            if ( dist2 < r2bins(m) ) then
                j = m
            else
                i = m
            end if
        end do
        cnts(i) = cnts(i) + 1
    
    end subroutine locate_bin_and_update

    function percent(n, n_total) result(res)
        !! Convert to percentage.
        integer(c_int64_t), intent(in) :: n, n_total
        real(c_double) :: res
        res = 100*( dble(n) / dble(n_total) )        
    end function percent

end module pair_counting_mod
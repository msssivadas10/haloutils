module spatial_hash_mod
    !! Spatial hashing for fast pair counting. Also, contains routines
    !! for spatial statistics calculation.
    
    use omp_lib
    use iso_c_binding
    implicit none

    private
    public :: build_grid_hash, calc_gridsize, calc_cellsize

    type, public, bind(c) :: cinfo_t
        !! A struct containing information about a grid cell. This, used 
        !! along with a sequence of point index blocks, can be used as an
        !! efficient spatial hash.    
        integer(c_int64_t) :: start !! Start index of a block
        integer(c_int64_t) :: count !! Size of the block  
    end type cinfo_t

    type, public, bind(c) :: boxinfo_t
        !! A struct for storing the details about the box 
        real(c_double)     ::  boxsize(3) !! Size of the box
        real(c_double)     ::   origin(3) !! Coordinates of the origin - lower left corner
        integer(c_int64_t) :: gridsize(3) !! Number of cells on each direction
        real(c_double)     :: cellsize(3) !! Size of the cell along each direction
    end type boxinfo_t
    
contains

    subroutine calc_gridsize(boxsize, cellsize, gridsize) bind(c)
        !! Helper function to calculate the gridsize, given boxsize and cellsize.
        real(c_double)    , intent(in)  ::  boxsize(3) 
        real(c_double)    , intent(in)  :: cellsize(3) 
        integer(c_int64_t), intent(out) :: gridsize(3) 
        gridsize = floor(boxsize / cellsize, kind=c_int64_t)
    end subroutine calc_gridsize

    subroutine calc_cellsize(boxsize, gridsize, cellsize) bind(c)
        !! Helper function to calculate the cellsize, given boxsize and gridsize.
        real(c_double)    , intent(in)  ::  boxsize(3) 
        integer(c_int64_t), intent(in)  :: gridsize(3) 
        real(c_double)    , intent(out) :: cellsize(3) 
        cellsize = boxsize / gridsize
    end subroutine calc_cellsize

    subroutine build_grid_hash(pid, npts, pos, box, ncells, grid_info, &
                               grid_data, nthreads, error_code         &
        ) bind(c)
        !! Calculate a grid spatial hash for the positions for fast and efficient 
        !! pair counting.

        integer(c_int64_t), intent(in), value :: pid
        !! A unique positive integer value (Used for safe file IO). 

        integer(c_int64_t), intent(in), value :: npts
        !! Number of points in the catalog

        real(c_double), intent(in) ::  pos(3, npts)
        !! Position

        type(boxinfo_t), intent(in) :: box
        !! Details about the space

        integer(c_int64_t), intent(in), value :: ncells
        !! Number of cells in the grid - must be equal to `product(gridsize)`

        integer(c_int64_t), intent(out) :: grid_data(npts)
        !! Grid data - an array of cell blocks, where each block gives the 
        !! indices of the points in that cell. This together with `grid_info`
        !! and poistion buffer give the complete grid hash.  

        type(cinfo_t), intent(out) :: grid_info(ncells)
        !! Grid details - start index and size of each cell group. 

        integer(c_int), intent(in), value :: nthreads
        !! Number of threads to use

        integer(c_int), intent(out) :: error_code
        !! Error code (0=success, 1=error, 2=cellsize mismatch, 3=gridsize mismatch)

        integer(c_int64_t), allocatable :: cell_index(:)

        ! Check gridsize and total number of cells matching:
        error_code = 3
        if ( product(box%gridsize) /= ncells ) return

        ! Check gridsize and cellsize matching:
        error_code = 2
        if (any(abs(box%cellsize - box%boxsize / box%gridsize) > 1e-08_c_double)) return
        
        error_code = 1

        grid_info%count = 0_c_int64_t ! initialise all count to 0
        grid_info%start = 0_c_int64_t ! initialise all start to 0
        allocate( cell_index(npts) )
        
        ! Set number of threads
        call omp_set_num_threads(nthreads) 

        ! -- Step 1 -- 
        ! Calculate the flattened index of the grid cell that contain the point. Also, 
        ! calculate the cell histogram - number of points in each cell. This part is 
        ! parallelised over multiple threads.
        call estimate_counts(pid, npts, pos, box, ncells, grid_info, cell_index)

        ! -- Setp 2 -- 
        ! Prefix sum: calculating cell start index using counts. This is calculated in 
        ! parallel.
        call calculate_prefix_sum(ncells, grid_info, nthreads)
        
        ! -- Step 3 --
        ! Scattering points into a sorted index array. This will create an array of 
        ! indices, where points in a specific cell are grouped together. Using this
        ! along with the grid_info, one can map points to its corresponding cells. 
        call sort_cell_indices(npts, cell_index, ncells, grid_info, grid_data)
        
        deallocate( cell_index )
        error_code = 0

    end subroutine build_grid_hash

! Internal routines (private):

    subroutine estimate_counts(pid, npts, pos, box, ncells, grid_info, cell_index)
        !! A parallelized coutning routine to get number of points in each 
        !! cells of a grid. 

        integer(c_int64_t), intent(in)    :: pid               !! Process ID
        integer(c_int64_t), intent(in)    :: npts              !! Number of points
        real(c_double)    , intent(in)    :: pos(3, npts)      !! Position buffer
        type(boxinfo_t)   , intent(in)    :: box               !! Box info
        integer(c_int64_t), intent(in)    :: ncells            !! Number of cells
        type(cinfo_t)     , intent(inout) :: grid_info(ncells) !! Grid details
        integer(c_int64_t), intent(out)   :: cell_index(npts)  !! Indices of cell for the points 

        character(256)     :: fn
        integer(c_int)     :: fu, tid, nthreads
        integer(c_int64_t) :: cell(3), i, j
        integer(c_int64_t), allocatable :: local_count(:)
        
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(tid, cell, local_count, i, j, fu, fn)

        ! Get the number of parallel threads available
        nthreads = omp_get_num_threads()
        
        ! Allocate local histogram for this thread
        allocate( local_count(ncells) )
        local_count(:) = 0_c_int64_t
        
        !$OMP DO SCHEDULE(static)
        do i = 1, npts
            cell = int(( pos(:, i) - box%origin ) / box%cellsize) ! 3D cell index

            ! Apply bounds: all the cell indices must be within [0, gridsize-1].
            ! Any value outside that will be clipped. 
            cell = min( max( 0_c_int64_t, cell ), box%gridsize-1 )

            ! Flattening the index (0 based)
            cell_index(i) = cell(1) + box%gridsize(1)*( cell(2) + box%gridsize(2)*cell(3) )
            
            ! Increment count for this cell
            j = cell_index(i) + 1
            local_count(j) = local_count(j) + 1

        end do
        !$OMP END DO

        ! For a thread-safe accumulatiion of the counts from different threads, each 
        ! thread will store the count to a temporary file, then update using that.
        tid = omp_get_thread_num() + 1 ! thread ID 
        fu  = 10 + tid ! file unit for this thread
        write(fn, '(i0,".",i0,".tmp")' ) pid, tid ! filename for this thread
        open(newunit=fu, file=fn, access='stream', form='unformatted',     &
            convert='little_endian', status='unknown', position='append',  &
            action='write'                                                 &
        )
        write(fu) local_count
        close(fu)
        
        deallocate( local_count  )

        !$OMP END PARALLEL

        ! Load count data from each temporary file and update the global count 
        allocate( local_count(ncells)  )
        fu = 10
        do tid = 1, nthreads
            write(fn, '(i0,".",i0,".tmp")' ) pid, tid ! filename for this thread
            open(newunit=fu, file=fn, access='stream', form='unformatted', &
                 convert='little_endian', status='old', action='read'      &
            )
            read(fu) local_count
            !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(j)
            do j = 1, ncells
                ! Accumulating counts: No need to sync this part, as each thread 
                ! works on a seperate block.
                grid_info(j)%count = grid_info(j)%count + local_count(j) 
            end do
            !$OMP END PARALLEL DO 
            close(fu, status='delete') ! file is deleted on close 
        end do
        deallocate( local_count  )
        
    end subroutine estimate_counts

    subroutine calculate_prefix_sum(ncells, grid_info, nthreads)
        !! Parallelized prefix sum calculation (used for calculating cell start
        !! index). 

        integer(c_int64_t), intent(in)    :: ncells            !! Number of cells
        type(cinfo_t)     , intent(inout) :: grid_info(ncells) !! Grid details
        integer(c_int)    , intent(in)    :: nthreads          !! Number of threads

        integer(c_int) :: tid, nts
        integer(c_int64_t) :: j, jstart, jstop, blksize, rem
        integer(c_int64_t), allocatable :: partial(:)

        ! Allocate space for storing partial sums from threads
        allocate( partial(nthreads+1) )
        partial(:) = 0_c_int64_t

        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(tid, nts, j, jstart, jstop, blksize, rem)
        
        nts      = omp_get_num_threads() ! Number of threads (should be same as `nthreads`)
        tid      = omp_get_thread_num()  ! ID of this thread
        blksize  = ncells / nts          ! Size of the block allotted to this thread
        rem      = modulo(ncells, nts)
        jstart   = tid*blksize + min(tid, rem) + 1 
        jstop    = jstart + blksize - 1
        if (tid < rem) jstop = jstop + 1

        grid_info(jstart)%start = 0_c_int64_t
        do j = jstart+1, jstop
            grid_info(j)%start = grid_info(j-1)%start + grid_info(j-1)%count
        end do
        partial(tid+2) = grid_info(jstop)%start + grid_info(jstop)%count
        !$OMP BARRIER

        ! Prefix sum over chunk totals:
        !$OMP SINGLE
        partial(1) = 1_c_int64_t
        do j = 2, nts+1
            partial(j) = partial(j) + partial(j-1)
        end do
        !$OMP END SINGLE

        ! Final start index:
        do j = jstart, jstop
            grid_info(j)%start = grid_info(j)%start + partial(tid+1)
        end do

        !$OMP END PARALLEL

        deallocate( partial )
        
    end subroutine calculate_prefix_sum

    subroutine sort_cell_indices(npts, cell_index, ncells, grid_info, grid_data)
        !! Generate a sorted array of cells, that represent a grid over which 
        !! the points are placed, using the given grid details.

        integer(c_int64_t), intent(in)  :: npts              !! Number of points
        integer(c_int64_t), intent(in)  :: ncells            !! Number of cells
        type(cinfo_t)     , intent(in)  :: grid_info(ncells) !! Grid details
        integer(c_int64_t), intent(in)  :: cell_index(npts)  !! Indices of cell for the points 
        integer(c_int64_t), intent(out) :: grid_data(npts)   !! Cell array / grid

        integer(c_int64_t) :: i, j, p
        integer(c_int64_t), allocatable :: offset(:)

        allocate( offset(ncells) )
        offset(:) = 0_c_int64_t

        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, p)
        !$OMP DO SCHEDULE(static)
        do i = 1, npts
            j = cell_index(i) + 1
            !$OMP ATOMIC CAPTURE
            p         = offset(j)
            offset(j) = offset(j) + 1
            !$OMP END ATOMIC
            grid_data( grid_info(j)%start + p ) = i
        end do
        !$OMP END DO
        !$OMP END PARALLEL

        deallocate( offset )
        
    end subroutine sort_cell_indices
    
end module spatial_hash_mod
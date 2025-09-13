module pair_utils_mod
    !! Helper functions for pair counting.

    use omp_lib
    use iso_c_binding
    use spatial_hash_mod
    implicit none

    private
    public :: enumerate_cell_pairs, to_3d_index, to_flat_index
    
contains

    subroutine enumerate_cell_pairs(pid, npts1, npts2, ncells, rmax, box, periodic, &
                                    grid_info1, grid_info2, nthreads, error_code    &
        )
        !! List all possible cell pairs. Also, marks the pairs that needs special
        !! handling. Data are written to disk, to a file with name based on the 
        !! `pid` value. Cell pairs, where the counts are almost uniform will be 
        !! in `pid.cplist.bin` file as a binary stream of int64 index pairs (1-
        !! based flat index). Also, pairs of unusually populated cells are stored 
        !! seperately in `pid.cplist.spl.bin` in the same format.    
        
        integer(c_int64_t), intent(in), value :: pid
        !! Process ID
        
        integer(c_int), intent(in), value :: periodic
        !! Flag for using periodic boundary conditions

        real(c_double), intent(in), value :: rmax
        !! Maximum distance bin value

        integer(c_int64_t), intent(in), value :: ncells
        !! Number of cells in the grid (same for both sets).

        type(boxinfo_t), intent(in) :: box
        !! Details about the space (both sets must be in the same box)

        integer(c_int64_t), intent(in), value :: npts1
        !! Number of points in set-1

        type(cinfo_t), intent(in) :: grid_info1(ncells)
        !! Grid specification for set-1

        integer(c_int64_t), intent(in), value :: npts2
        !! Number of points in set-2

        type(cinfo_t), intent(in) :: grid_info2(ncells)
        !! Grid specification for set-2

        integer(c_int), intent(in), value :: nthreads
        !! Number of threads to use
        
        integer(c_int), intent(out) :: error_code
        !! Error code (0=success, 1=error, 2=cellsize mismatch, 3=gridsize mismatch)

        integer(c_int64_t) :: count_th(2), delta(3)

        ! Check gridsize and total number of cells matching:
        error_code = 3
        if ( product(box%gridsize) /= ncells ) return
        
        ! Check gridsize and cellsize matching:
        error_code = 2
        if (any(abs(box%cellsize - box%boxsize / box%gridsize) > 1e-08_c_double)) return
        
        error_code = 1
        
        ! Over-population thresholds: using 100 times average count.
        count_th = floor( 100*dble([ npts1, npts2 ]) / ncells, kind=c_int64_t )
        
        ! Specify the number of neighbours to check on each direction. If the
        ! 3D cell index is`c[i]` for axis i, then the neighbour cell indices 
        ! runs from `c[i] - delta[i]` to `c[i] + delta[i]` (BC applied). 
        delta = ceiling( rmax / box%cellsize ) 

        call omp_set_num_threads(nthreads) ! set number of threads
        call distribute_enumeration(pid, periodic, count_th, box%gridsize, &
                                    delta, ncells, grid_info1, grid_info2  &
        )
        call merge_pair_lists(pid, nthreads)
        
        error_code = 0
        
    end subroutine enumerate_cell_pairs

    subroutine distribute_enumeration(pid, periodic, count_th, gridsize, delta,  &
                                      ncells, grid_info1, grid_info2             &
        )
        !! Enumerate the cell pairs in parallel. Each thread will write the 
        !! pairs it found to a binary stream `pid.tid.cpstack.tmp`. Each item
        !! in the stream will be a tuple of cell indices (int64) and a flag
        !! if any of the cells are overpopulated (int32): i.e., a sequence 
        !! of `(i1,j2,f1), (i2,j2,f2)...`.

        integer(c_int64_t), intent(in) :: pid                !! Process ID
        integer(c_int)    , intent(in) :: periodic           !! Periodic BC flag
        integer(c_int64_t), intent(in) :: count_th(2)        !! Over-population threshold 
        integer(c_int64_t), intent(in) :: gridsize(3)        !! Size of the grid
        integer(c_int64_t), intent(in) :: delta(3)           !! Nearby-cell-range specifier
        integer(c_int64_t), intent(in) :: ncells             !! Number of cells
        type(cinfo_t)     , intent(in) :: grid_info1(ncells) !! Grid-1
        type(cinfo_t)     , intent(in) :: grid_info2(ncells) !! Grid-2

        character(256)     :: fn
        integer(c_int)     :: tid, fu, dens
        integer(c_int64_t) :: j1, j2, c1(3), c2(3), cstart(3), cstop(3), k, kstop

        !$OMP  PARALLEL DEFAULT(SHARED) &
        !$OMP& PRIVATE(tid, dens, j1, j2, c1, c2, cstart, cstop, k, kstop, fu, fn)
        
        ! Open a temp file to save the pairs found from this thread: 
        tid = omp_get_thread_num() + 1 ! thread ID 
        fu  = 10 + tid ! file unit for this thread
        write( fn, '(i0,".",i0,".cplist.tmp")' ) pid, tid ! filename for this thread
        open(newunit=fu, file=fn, access='stream', form='unformatted',      &
             convert='little_endian', status='unknown', position='append',  &
             action='write'                                                 &
        )

        !$OMP DO SCHEDULE(static)
        do j1 = 1, ncells

            if ( grid_info1(j1)%count < 1 ) cycle ! cell is empty
            
            ! Converting the flat cell index to 3D index
            c1 = to_3d_index( j1, gridsize )

            ! Find the range of indices for the neighbouring cells. 
            cstart = c1 - delta
            cstop  = c1 + delta
            if ( periodic == 0 ) then
                ! When periodic BC is not used, ignore the cells outside the 
                ! grid range, so that cell indices are in range [0, gridsize-1].
                cstart = min( max( 0_c_int64_t, cstart ), gridsize-1 )
                cstop  = min( max( 0_c_int64_t, cstop  ), gridsize-1 )
            end if

            if ( grid_info1(j1)%count > count_th(1) ) then
                ! This cell is over-populated, based on the given over-population 
                ! threshold (usually 1000 x average count). When parallely counting
                ! pairs over cells, threads handling those cells will take longer
                ! time to complete. So, for these cells, pair counting is done 
                ! over the points, which is the better way!    
                dens = 1_c_int
            else 
                dens = 0_c_int
            end if

            ! Walking through the neighbouring cells to make pairs:
            kstop = product(cstop - cstart + 1) ! number of neighbouring cells
            c2  = cstart
            do k = 1, kstop

                ! Converting the 3D cell index to a flat index
                if ( periodic /= 0 ) then
                    ! Using periodic boundary conditions: wrap around, if the cell  
                    ! index is outside the range. 
                    j2 = to_flat_index( modulo( c2, gridsize ), gridsize )
                else 
                    j2 = to_flat_index( c2, gridsize )
                end if

                if ( grid_info2(j2)%count > 0 ) then
                    ! The combined over-population flag will be set based on the  
                    ! count in this cell. The value written to the file is that of
                    ! the combined flag: `dens .or. dens_other`  
                    if ( grid_info2(j2)%count > count_th(2) ) then
                        ! This cell is over-populated, and keep for special handling. 
                        write(fu) j1, j2, 1_c_int
                    else
                        write(fu) j1, j2, dens   
                    end if 
                end if                
                
                call increment_counter3( c2, cstart, cstop ) ! to next cell...
            end do

        end do 
        !$OMP END DO

        close(fu)

        !$OMP END PARALLEL

    end subroutine distribute_enumeration

    subroutine increment_counter3(c, cstart, cstop)
        !! Increment a 3D index counter `c` running from `cstart` to `stop`.  
        !! This is used for flattening nested do loops.
        integer(c_int64_t), intent(in)    :: cstart(3), cstop(3)
        integer(c_int64_t), intent(inout) :: c(3)

        ! First index is the faster changing one. If one index reached the 
        ! stop value, it is reset to the start value and the next index is 
        ! incremented. This is repeated until the last index reach its stop 
        ! value...
        c(1) = c(1) + 1
        if ( c(1) > cstop(1) ) then
            c(1) = cstart(1)     
            c(2) = c(2) + 1
            if ( c(2) > cstop(2) ) then
                c(2) = cstart(2) 
                c(3) = c(3) + 1
            end if
        end if
    
    end subroutine increment_counter3

    subroutine merge_pair_lists(pid, nthreads)
        !! Merge pair lists from temp files created by threads into two
        !! files, one for cell pairs with normal population `pid.cplist.bin`
        !! and one for over-populated cell pairs `pid.cplist.spl.bin`. All 
        !! temporary files will be deleted. 
        integer(c_int64_t), intent(in) :: pid
        integer(c_int)    , intent(in) :: nthreads

        integer(c_int64_t) :: j1, j2
        integer(c_int)     :: tid, fu, fp1, fp2, iostat, dens
        character(256)     :: fn

        ! File for normal cell pair stack 
        fp1 = 8
        write( fn, '(i0,".cplist.bin")' ) pid 
        open(newunit=fp1, file=fn, access='stream', form='unformatted',     &
             convert='little_endian', status='replace', position='append',  &
             action='write'                                                 &
        )
        
        ! File for special cell pair stack
        fp2 = 9
        write( fn, '(i0,".cplist.spl.bin")' ) pid 
        open(newunit=fp2, file=fn, access='stream', form='unformatted',     &
             convert='little_endian', status='replace', position='append',  &
             action='write'                                                 &
        )
        
        ! Combining the individual stacks from threads
        do tid = 1, nthreads
            fu  = 10 + tid ! file unit for this thread
            write( fn, '(i0,".",i0,".cplist.tmp")' ) pid, tid ! filename for this thread
            open(newunit=fu, file=fn, access='stream', form='unformatted', &
                 convert='little_endian', status='old', action='read'      &
            )
            do 
                read(fu, iostat=iostat) j1, j2, dens
                if ( iostat /= 0 ) exit ! EOF or error
                if ( dens == 1 ) then
                    write(fp2) j1, j2 ! to over-populated pair list
                else
                    write(fp1) j1, j2 ! to normal pair list
                end if                 
            end do
            close(fu, status='delete') ! file is deleted on close 
        end do

        close(fp1)
        close(fp2)
        
    end subroutine merge_pair_lists

! Helper functions

    function to_3d_index(j, gridsize) result(cell)
        !! Calculate the 3D cell indices from a flat index. This 3D index 
        !! follows 0-based indexing, but the flattened index is 1-based.
        integer(c_int64_t), intent(in)  :: gridsize(3), j
        integer(c_int64_t) :: cell(3)
    
        cell(1) = modulo( j-1, gridsize(1) ); cell(3) = (j-1) / gridsize(1) 
        cell(2) = modulo( cell(3), gridsize(2) )
        cell(3) = cell(3) / gridsize(2)
        
    end function to_3d_index
    
    function to_flat_index(cell, gridsize) result(j)
        !! Calculate the flat index from 3D cell indices. This 3D index 
        !! follows 0-based indexing, but the flattened index is 1-based.
        integer(c_int64_t), intent(in) :: gridsize(3), cell(3)
        integer(c_int64_t) :: j
    
        j = cell(1) + gridsize(1)*( cell(2) + gridsize(2)*cell(3) ) + 1

    end function to_flat_index

end module pair_utils_mod
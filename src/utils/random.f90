module random_mod
    !! Random number generation using PCG32.

    use iso_c_binding
    implicit none
    
    private
    public :: pcg32_init, pcg32_random, uniform_rv, normal_rv, &
              poisson_rv, binomial_rv

    integer, parameter :: i64 = c_int64_t
    integer, parameter :: i32 = c_int32_t
    
    integer(i64), parameter :: pcg32_mul = int(z'5851f42d4c957f2d', kind=i64) 
    !! PCG32 multiplier (6364136223846793005u)

    integer(i64), parameter :: pcg32_inc = int(z'14057b7ef767814f', kind=i64) 
    !! PCG32 increment (1442695040888963407u)

    real(c_double), parameter :: pi = 3.141592653589793_c_double
    !! Pi

contains

    subroutine pcg32_init(rstate, seed)
        !! Initialize a PCG32 random number generator.

        integer(i64), intent(inout) :: rstate
        !! Random number generator state

        integer(i64), intent(in), value :: seed
        !! Seed value

        integer(i32) :: rv

        rstate = pcg32_inc + seed
        rv     = pcg32_random(rstate)
        
    end subroutine pcg32_init

    function pcg32_random(rstate) result(rword)
        !! Return a 32 bit random integer generated using PCG32 generator. 
        !! Range: -2147483648 to 2147483647 (i.e., 2^31 - 1)

        integer(i64), intent(inout) :: rstate
        !! Random number generator state

        integer(i32) :: rword
        integer(i64) :: state, rotation
        integer(i64), parameter :: mask32 = int(z'00000000ffffffff', kind=i64)

        state  = rstate ! Current state

        ! Select a random rotation amount - bits 63-59 of the current state
        rotation = shiftr(state, 59)

        ! Update the state
        rstate = state * pcg32_mul + pcg32_inc

        ! An xorshift mixes some high-order bits down
        state = xor(state, shiftr(state, 18)) 
        ! 32 bit rotation on bits 58-27
        state = iand(shiftr(state, 27), mask32)
        state = ior(shiftr(state, rotation), shiftl(state, 32_i64 - rotation)) ! rotation
        rword = int(state, kind=i32) ! Last 32 bits
        
    end function pcg32_random

    function uniform_rv(rstate) result(rv)
        !! Generate a uniform random number in [0, 1).

        integer(i64), intent(inout) :: rstate !! Random number generator state
        real(c_double) :: rv
        
        real(c_double), parameter :: shift = 2147483648.0_c_double
        real(c_double), parameter :: scale = 4294967295.0_c_double

        ! PCG32 generates random integers in the range from -2147483648 to 
        ! 2147483647. This coverted to a float in 0 to 1 (not incl.) range.
        rv = ( pcg32_random(rstate) + shift ) / scale

    end function uniform_rv

    function normal_rv(rstate, mu, std) result(rv)
        !! Generate a Normal random number.

        integer(i64), intent(inout) :: rstate 
        !! Random number generator state

        real(c_double), intent(in), value :: mu
        !! Mean

        real(c_double), intent(in), value :: std
        !! Standard deviation

        real(c_double) :: rv, u, v

        ! Random number generation using Box-Muller transform
        u  = uniform_rv(rstate) ! in 0 to 1 range
        v  = uniform_rv(rstate) ! in 0 to 1 range
        rv = mu + std * sqrt( -2*log(u) ) * cos( 2*pi*v ) ! normal random number

    end function normal_rv
    
    function poisson_rv(rstate, lam) result(rv)
        !! Generate a Poisson random number.

        integer(i64), intent(inout) :: rstate 
        !! Random number generator state

        real(c_double), intent(in), value :: lam
        !! Rate parameter

        integer(c_int64_t) :: rv
        real(c_double)     :: a, b, c, k, p, u, v, x, y, lhs, rhs

        if ( lam <= 30._c_double ) then
            ! For lam <= 30, random numbers are generated using Knuth's
            ! algorithm. This easy and exact, but has O(lam) expected 
            ! multiplications - fine for small lam.
            c  = exp(-lam)
            p  = 1._c_double
            rv = 0
            do 
               rv = rv + 1
               u  = uniform_rv(rstate) ! in 0 to 1 range
               p  = p*u
               if ( p <= c ) exit 
            end do
            rv = rv - 1
        else if ( lam <= 1e+05_c_double ) then
            ! Using Ahrens–Dieter or Atkinson’s method for medium lam, 
            ! because Knuth becomes slow as expected iterations ~ lam.
            ! This method uses normal/gamma approximations with rejection, 
            ! and is much faster.
            ! 
            ! Reference:
            ! - A. C. Atkinson, The Computer Generation of Poisson Random Variables. 
            !   Appl. Statist. (1979), 28, No. I, pp, 29-35.
            c = 0.767_c_double - 3.36_c_double / lam
            b = pi / sqrt(3*lam)
            a = b * lam
            k = log(c) - lam - log(b)
            do 
                u   = uniform_rv(rstate) ! in 0 to 1 range
                x   = (a - log((1._c_double - u) / u)) / b
                rv  = floor(x + 0.5_c_double)
                if ( rv < 0 ) cycle
                v   = uniform_rv(rstate) ! in 0 to 1 range
                y   = a - b*x
                lhs = y + log(v / (1._c_double + exp(y))**2)
                rhs = k + rv*log(lam) - log_gamma(rv + 1._c_double)
                if ( lhs <= rhs ) exit 
            end do
        else
            ! For very large lam > 10^5, normal approximation is used. 
            ! Normal random number is generated using a Box-Muller 
            ! transform.
            u  = uniform_rv(rstate) ! in 0 to 1 range
            v  = uniform_rv(rstate) ! in 0 to 1 range
            k  = sqrt( -2*log(u) ) * cos( 2*pi*v ) ! std. normal random number
            x  = lam + sqrt(lam)*k + 0.5_c_double
            rv = max( 0_c_int64_t, int(x, kind=c_int64_t) )  
        end if
        
    end function poisson_rv

    function binomial_rv(rstate, n, p) result(rv)
        !! Generate a Binomial random number.

        integer(i64), intent(inout) :: rstate 
        !! Random number generator state

        integer(c_int64_t), intent(in), value :: n
        !! Number of trials

        real(c_double), intent(in), value :: p
        !! Probability

        integer(c_int64_t) :: rv, i
        real(c_double) :: q, np, s, f, u, v, x, y

        if ( n < 30 ) then
            ! For small n, use a Direct Bernoulli sum, which is exact 
            ! and O(n) complexity.
            rv = 0
            do i = 1, n
                if ( uniform_rv(rstate) < p ) rv = rv + 1 
            end do
        else if ( n < 100000 ) then
            ! For medium n, use BTPE algorithm (Binomial Triangle-
            ! Parallelogram Exponential), which is exact and O(1)
            ! complexity (expected).
            q  = 1._c_double - p
            np = n*p
            s  = sqrt(np*q)
            f  = np + p
            do 
                u = uniform_rv(rstate) - 0.5_c_double ! in -0.5 to 0.5 range
                v = uniform_rv(rstate) ! in 0 to 1 range
                x = floor(np + s*( u / sqrt(v / (1._c_double - v)) ))
                if (x < 0._c_double .or. x > n) cycle
                y = x * log(np / x) + (n-p) * log((n - np) / (n - x)) 
                if ( v <= exp(y) ) then
                    rv = int(x, kind=c_int64_t)
                    return
                end if
            end do
        else
            ! For large n, use Normal approximation for performance.
            q  = 1._c_double - p
            np = n*p        ! mean
            s  = sqrt(np*q) !std. deviation
            u  = uniform_rv(rstate) ! in 0 to 1 range
            v  = uniform_rv(rstate) ! in 0 to 1 range
            x  = np + s * sqrt( -2*log(u) ) * cos( 2*pi*v ) ! normal random number
            rv = max( 0_c_int64_t, min( n, int(x + 0.5_c_double, kind=c_int64_t) ) ) 
        end if
        
    end function binomial_rv

end module random_mod

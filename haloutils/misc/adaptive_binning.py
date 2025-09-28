import numpy as np, numpy.typing as nt

def adaptive_bins(
        x           : nt.NDArray[np.float64], 
        f           : nt.NDArray[np.float64], 
        min_dx      : float, 
        target_area : float = None, 
    ) -> nt.NDArray[np.float64]:
    """
    Construct adaptive bins over [x[0], x[-1]] according to f(x).

    Parameters
    ----------
    x : array_like
        Monotonic grid points where f(x) is given.

    f : array_like
        Distribution values at x (not normalized).
    
    min_dx : float
        Minimum bin width.
    
    target_area : float, optional
        Desired integral per bin. If None, it is chosen so that
        the average bin width is ~min_dx.

    Returns
    -------
    bin_edges : np.ndarray
        Array of bin edges (length n_bins+1).
    """

    shift = min(x)
    x = np.asarray(x) - shift
    f = np.asarray(f)

    # cumulative integral of f
    dx    = np.diff(x)
    mid_f = 0.5*(f[1:] + f[:-1]) - min(f)
    cum   = np.concatenate(([0.0], np.cumsum(mid_f*dx)))
    cum   = cum / cum[-1]

    # set target area if not given
    if target_area is None:
        approx_bins = (x[-1]-x[0]) / min_dx
        target_area = 1. / approx_bins

    bin_edges = [x[0]]
    current_target = target_area

    # walk along the CDF
    for i in range(1, len(x)):
        while cum[i] >= current_target:
            # interpolate to find precise edge
            frac = (current_target - cum[i-1])/(cum[i]-cum[i-1] + 1e-300)
            edge = x[i-1] + frac*(x[i]-x[i-1])
            if edge - bin_edges[-1] >= min_dx:
                bin_edges.append(edge)
                current_target += target_area
            else:
                # skip if edge would be too close, wait for more area
                current_target += target_area
    # ensure last edge is included
    if bin_edges[-1] < x[-1]:
        bin_edges.append(x[-1])

    return np.array(bin_edges) + shift


# Example usage
if __name__ == "__main__":
    # Example: f(x) ~ exp(-x), x in [0,10]
    x = np.linspace(0, 10, 200)
    f = np.exp(-x)

    bins = adaptive_bins(x, f, min_dx=0.2)
    print("Bin edges:", bins)
    print("Number of bins:", len(bins)-1)

    import matplotlib.pyplot as plt
    plt.plot(x, f, '-', bins, np.exp(-bins), 'o')
    plt.show()
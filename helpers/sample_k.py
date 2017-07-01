def sample_k(vals, k):
    """
    """
    N = vals.shape[0]
    E = elem_sympoly(vals, k)[:,1:]
    sample = np.zeros(N, dtype=int)
    rem = k

    for elem, val in reversed(list(enumerate(vals))):

        # Check if we chose k elements
        if not rem:
            break
            
        # Compute conditional marginal of elem
        marg = val * E[rem-1, elem-1] / E[rem, elem]

        # Sample elem
        if np.random.rand() < marg:
            sample[elem] = 1
            rem -= 1
        
        # Check if all remaining elements will be chosen
        if elem == rem:
            sample[np.arange(rem)] = 1
            break

    return sample
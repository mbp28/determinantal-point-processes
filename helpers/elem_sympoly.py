def elem_sympoly(vals, k):
    """Uses Newton's identities to compute elementary symmetric polynomials."""
    N = vals.shape[0]
    E = np.zeros([k+1, N+1])
    E[0,] = 1
    
    for i in range(1, k+1):
        for j in range(1, N+1):
            E[i,j] = E[i, j-1] + vals[j-1] * E[i-1, j-1]
    
    return E[:,1:]
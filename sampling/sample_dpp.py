import numpy as np
from scipy.linalg import orth

def sample_dpp(vals, vecs, k=0, one_hot=False):
    """
    This function expects 
    
    Arguments: 
    vals: NumPy 1D Array of Eigenvalues of Kernel Matrix
    vecs: Numpy 2D Array of Eigenvectors of Kernel Matrix

    """
    n = vecs.shape[0] # number of items in ground set
    
    # k-DPP
    if k:
        index = sample_k() # sample_k, need to return index

    # Sample set size
    else:
        index = (np.random.rand(n) < (vals / (vals + 1)))
        k = np.sum(index)
    
    # Check for empty set
    if not k:
        return np.zeros(n) if one_hot else np.empty(0)
    
    # Check for full set
    if k == n:
        return np.ones(n) if one_hot else np.arange(k, dtype=float) 
    
    V = vecs[:, index]

    # Sample a set of k items 
    items = list()

    for i in range(k):
        p = np.sum(V**2, axis=1)
        p = np.cumsum(p / np.sum(p)) # item cumulative probabilities
        item = (np.random.rand() <= p).argmax()
        items.append(item)
        
        # Delete one eigenvector not orthogonal to e_item and find new basis
        j = (np.abs(V[item, :]) > 0).argmax() 
        Vj = V[:, j]
        V = orth(V - (np.outer(Vj,(V[item, :] / Vj[item])))) 
    
    items.sort()
    sample = np.array(items, dtype=float)    

    if one_hot:
        sample = np.zeros(n)
        sample[items] = np.ones(k)
    
    return sample 
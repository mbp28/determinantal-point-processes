import numpy as np

def sample_dual_dpp(B, C, k=0, one_hot=False):
    """
	B is the N x d feature matrix (L would be B*B', but is too big to work with)
	C is the decomposed covariance matrix, computed using:
	C = decompose_kernel(B'*B);
	k is (optionally) the size of the set to return.

    """
    n = B.shape[0] # number of items in ground set
    vals, vecs = np.linalg.eig(C)
    
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
    
    # Choose eigenvectors
    V = vecs[:, index]

    # Rescale eigenvectors
    V = np.apply_along_axis(lambda vec: vec / np.sqrt(vec.dot(C).dot(vec)), 0, V)

    # Sample a set of k items 
    items = list()

    for i in range(k):
    	# Choose an item 
        p = np.sum(np.dot(B,V)**2, axis=1)
        p = np.cumsum(p / np.sum(p)) # item cumulative probabilities
        item = (np.random.rand() <= p).argmax()
        items.append(item)
        
        # Choose an eigenvector j to delete
        S = B[item, :].dot(V)
        j = (np.abs(S) > 0).argmax() 
        Vj = V[:, j]
        Sj = S[j]

        # Update V
        V = np.delete(V, j, 1)
        S = np.delete(S, j)
        V = (V - (np.outer(Vj, S / Sj))) 
    
        # orthogonalize in the projected space
        for a in range(V.shape[1]):
        	for b in range(a):
        		V[:,a] = V[:,a] - (V[:,a].dot(C).dot(V[:,b])) * V[:,b]
        	V[:,a] = V[:,a] / np.sqrt(V[:,a].dot(C).dot(V[:,a]))

    items.sort()
    sample = np.array(items, dtype=float)    

    if one_hot:
        sample = np.zeros(n)
        sample[items] = np.ones(k)
    
    return sample 
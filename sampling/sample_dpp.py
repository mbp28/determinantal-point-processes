import numpy as np
from scipy.linalg import orth


def sample_dpp_bin(e_val,e_vec):
    """
    This function expects 



    """
    
    my_rand = np.random.rand(len(e_val))
    ind = (my_rand <= (e_val)/(1+e_val))
    k = sum(ind)
    if k == len(e_vec):
        return np.ones(len(e_vec),dtype=int) # check for full set
    if k == 0:
        return np.zeros(len(e_vec),dtype=int) 
    V = e_vec[:,np.array(ind)]

    # sample a set of k items 
    sample = np.zeros(len(e_vec),dtype=int)
    for l in range(k-1,-1,-1):
        p = np.sum(V**2,axis=1)
        
        p = np.cumsum(p / np.sum(p)) # item cumulative probabilities
        my_rand = np.random.rand()
        i = int((my_rand <= p).argmax()) # choose random item
        sample[i] = 1
        
        j = (np.abs(V[i,:])> 0.0).argmax() # pick an eigenvector not orthogonal to e_i
        Vj = V[:,j]
        V = orth(V - (np.outer(Vj,(V[i,:]/Vj[i]))))

    return sample
import numpy as np
from itertools import chain, combinations
import itertools


t = True
iteration = 1
m = 10
n = 20
while(t == True):
    print('current_iteration: ', iteration)
    t = False
    A = np.random.binomial(1,1/2,(m,n))
    A = A.astype(float)

    col_indices = list(range(n))
    r = 1
    for i in range(1, m+1):
        nchoosei = list(itertools.combinations(col_indices, i))
        for S in nchoosei:
            A_S = A[:,S]
            rank = np.linalg.matrix_rank(A_S)
            if rank < len(S):
                r = 0
                break
        if r == 0:
            t = True
            break
    
    iteration += 1

print(A)
for i in range(n):
    A[:,i] = A[:,i]/np.linalg.norm(A[:,i])
np.save('sensing_matrix.npy', A)
    
    



    

    


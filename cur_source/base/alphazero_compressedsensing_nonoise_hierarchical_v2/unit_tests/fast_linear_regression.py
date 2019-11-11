#Test fast linear regression via matrix inversion lemma
#Compare 2 different techniques
#1)using np.linalg.lstsq
#2)using matrix inversion lemma and storing (A^T A)^-1 to compute (B^T B)^-1, where
#B = [A | c]. IOW B is the matrix A with an additional column. 

import numpy as np
import time

param = {}
param['additional_columns'] = 70
param['m'] = 100
param['n'] = 1

#Generate the matrices A and corresponding vectors b to apply linear regression on
#matrices list contains list elements of the form [A,b].
matrices = [[np.random.normal(0,1,(param['m'], param['n'])), np.random.normal(0,1, (param['m'], 1))]]

for i in range(param['additional_columns']):
    c = np.random.normal(0,1,(param['m'], 1))
    A = np.hstack((matrices[-1][0],c))
    b = np.random.normal(0,1,(param['m'], 1))
    matrices.append([A, b])


#Solve linear regression. 

#1)via np.linalg.stsq
start = time.time()
for A, b in matrices[1:]:
    solution = np.linalg.lstsq(A, b)[0]
    #print(solution)
    #print('')
end = time.time()
print('For Method 1, the total time taken for ' + str(param['additional_columns']+1) + ' (A,b) pairs is: ', end-start)

#2)using matrix inversion lemma and saving (A^T*A) at every iteration
start = time.time()
cur_inverse = np.linalg.inv(np.matmul(matrices[0][0].transpose(), matrices[0][0]))
cur_ATb = np.matmul(matrices[0][0].transpose(), matrices[0][1])
solution = np.matmul(cur_inverse, cur_ATb)

for A, b in matrices[1:]:
    
    #1)update cur_inverse for next iteration using matrix inversion. See Algorithm 1 of saved pdf. 
    u1 = np.matmul(A[:,0:-1].transpose(), A[:,-1])
    u1 = np.reshape(u1, (u1.shape[0], 1))
    u2 = np.matmul(cur_inverse, u1)
    
    d = 1/(np.matmul(A[:,-1].transpose(), A[:,-1]) - np.matmul(np.matmul(u1.transpose(), cur_inverse), u1))
    u3 = d*u2
    F11_inverse = cur_inverse + d*np.outer(u2, u2)
    
    left = np.vstack((F11_inverse, -1*u3.transpose()))
    right = np.vstack((-1*u3, d))
    cur_inverse = np.hstack((left, right))
    
    #print(np.linalg.inv(np.matmul(A.transpose(), A)))
    #print('')
    #print(cur_inverse)
    
    #2)update cur_ATb
    cur_ATb = np.matmul(A.transpose(), b)
    
    #3)solve the regression problem
    solution = np.matmul(cur_inverse, cur_ATb)
    #print(solution)
    #print('')

end = time.time()

print('For Method 2, the total time taken for ' + str(param['additional_columns']+1) + ' (A,b) pairs is: ', end-start)
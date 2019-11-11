import numpy as np


A = np.random.normal(1, 3, (10,5))
UVU = np.linalg.svd(A)

print("before column normalization...")
print("A:", A)
print("singular values:", UVU[1])

#normalize columns
for column_index in range(A.shape[1]):
    column_norm = np.linalg.norm(A[:,column_index])
    print("column_norm: ", column_index, column_norm)
    A[:, column_index] = 1/column_norm*A[:, column_index]
    UVU[1][column_index] = 1/column_norm*UVU[1][column_index]
#compute new SVD
new_UVU = np.linalg.svd(A)

print("")
print("new singular values where each singular value is multiplied by 1/corresponding column norm", UVU[1])
print("")

print("after column normalization...")
print("normalized A:" ,A)
print("singular values:", new_UVU[1])


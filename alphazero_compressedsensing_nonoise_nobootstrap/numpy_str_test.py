#This small test shows that numpy.array_str is extremely slow!!!!!!!!
#The time needed to run METHOD1 is roughly slower by 1e02!!!!

import numpy as np
import time
np.set_printoptions(threshold=np.nan)

print('')
A = np.zeros(1501)
B = np.zeros(1501)
A[784] = 1



#METHOD 1--------------------------------------------------- 
start = time.time()
keyA = np.array_str(A)


keyB = np.array_str(B)
#print(keyA)
#print(keyB)
#print(len(keyA))
#print(len(keyB))

Dict = {}
Dict[keyA] = 6

if keyB in Dict:
    print('keyB is equal to keyA and is already in Dict')
end = time.time()

print('METHOD1: Total runtime of key search(conv. to string and key search) in seconds is: ' + str(end - start))
#------------------------------------------------------------

#METHOD 2----------------------------------------------------
start = time.time()
keyA = tuple(A.tolist())
keyB = tuple(B.tolist())
    
#Here we do not hash the numpy array. Instead we transform the numpy arrays into something else before ultimately transforming
#it into a string(or not)
hashed_valueA = hash(keyA)
print('The hashed value of keyA is: ' + str(hashed_valueA))
keyA_string = str(hashed_valueA)

Dict2 = {}
Dict2[keyA_string] = 6

hashed_valueB = hash(keyB)
keyB_string = str(hashed_valueB)

if keyB_string in Dict2:
    print('key B is equal to keyA and is already in Dict2')
end = time.time()
print('METHOD2: Total runtime of key search(conv. to string and key search) in seconds is: ' + str(end - start))
print('')
#-------------------------------------------------------------

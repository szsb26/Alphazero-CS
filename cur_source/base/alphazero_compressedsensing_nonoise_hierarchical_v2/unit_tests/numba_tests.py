from numba import njit, jit, vectorize, float64
import numpy as np
import time

def test_func1(v):
    for i in range(len(v)):
        v[i] += 100

#jit version, gives significant overhead in this simple example we have
@jit
def test_func2(v):
    for i in range(len(v)):
        v[i] += 100

#vectorize version
@vectorize([float64(float64)])
def test_func3(component_of_v):
    component_of_v += 100
    return component_of_v

#Test:
v = np.zeros(100)

#test function 1
start = time.time()
test_func1(v)
end = time.time()
total_time1 = end-start
print('')
print('Input:', v)
print("total time taken WITHOUT numba functionality:", total_time1)
print('')

#test function 3
start = time.time()
test_func3(v)
max = max(v)
end = time.time()
total_time2 = end-start
print('')
print('Input:', v)
print("total time taken WITH numba functionality(vectorize):", total_time2)
print('')
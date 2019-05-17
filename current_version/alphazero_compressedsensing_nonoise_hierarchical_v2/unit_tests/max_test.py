import numpy as np
import time

#Test max with custom max

def custom_max(v):
    max = float("-inf")
    for i in range(len(v)):
        if v[i] > max:
            max = v[i]
    
    return max
    
#Test
v = np.random.normal(0,1,10000)

start = time.time()
max = max(v)
end = time.time()

#print("v:", v)
print("max:",max)
print("time to compute using python max:", end - start)

start = time.time()
max = custom_max(v)
end = time.time()

#print("v:", v)
print("max:", max)
print("time to compute using custom max function:", end - start)
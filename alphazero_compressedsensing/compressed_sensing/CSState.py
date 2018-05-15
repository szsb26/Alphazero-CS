import numpy as np

#CSState is an object which stores information about the current state. The input is a numpy array of 0 and 1's, where
#the last index denotes whether the stopping action has been taken. For indices i which
#are not the last index, self.action_indices[i] = 0 means column has not yet been taken 
#while self.action_indices[i] = 1 means column has already been taken. 
#DO NOT STORE A HERE AS IT IS MEMORY INTENSIVE

class State():
    def __init__(self, actions_indicator): 
        self.action_indices = actions_indicator 
        self.num_col = np.sum(self.action_indices[0:self.action_indices.size-1])
        
    def getcolIndices(self):
    	S = []
    	for i in range(self.action_indices.size-1):
    		if self.action_indices[i] == 1:
    			S.append(i)
    	return S
  	      
  
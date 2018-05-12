from Game import Game
import numpy as np

#CSState is an object which stores information about the current state. The input is a numpy array of 0 and 1's, where
#the last index denotes whether the stopping action has been taken. self.col[i] = 0 means column has not yet been taken
#while self.col[i] = 1 means column has already been taken. 

class State():
    def __init__(self, columns_indicator):
        self.col = columns_indicator 
        self.num_col = np.sum(self.col[0:self.col.size])
  	      
  
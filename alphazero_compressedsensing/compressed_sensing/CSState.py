from Game import Game
import numpy as np

#CSState is an object which stores information about the current state. 

class State():
    def __init__(self, columns_indicator):
        self.col = columns_indicator #columns_indicator is a 0, 1 numpy vector where columns_indicator[i] = 0 
        					 #means the ith column has not been taken while 1 means ith column has been taken.
  		
  	      
  
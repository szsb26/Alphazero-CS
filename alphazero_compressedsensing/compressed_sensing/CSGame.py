from Game import Game
from CSState import State
import numpy as np

class CSGame(Game):
    def __init__(self, A, y, k = 1): 
    #A and y are both assumed to be numpy arrays/vectors
        self.sensing_matrix = A
        self.obs_vector = y
        self.sparsity_limit = k

    def getInitBoard(self): 
    #Get the initial state at beginning of CS Game
    	col_size = self.sensing_matrix.shape[1]
    	Initial_State = State(np.zeros(col_size + 1)) #The plus one is for the STOP action. If Initial_State.col[col_size] = 1, then algorithm
    													#chooses to stop, and the next state is now a terminating state
    	return Initial_State

    def getActionSize(self): 
    #return number of all actions (equal to column size of A + 1)
    	return self.sensing_matrix.shape[1]+1 

    def getNextState(self, state, action): 
    #input is a State object and an action(integer). Output is the next state as a CSState object. 
    	if state.col[-1] = 1:
    		print ('Already at terminal state, no more actions can be taken.')
    
        elif state.col[action] == 1:
        	print (str(action) + ' action already taken, invalid move.')
        	return
        else:
        	nextstate_col_ind = np.array(state.col)
        	nextstate_col_ind[action] = 1
        	next_state = State(nextstate_col_ind)
        	return next_state 

    def getValidMoves(self, state): 
    #input is a State object. Output is a binary numpy vector for valid moves,
    #where b[i] = 1 implies ith column can be taken. 
    
        valid_moves = np.zeros(self.sensing_matrix.shape[1]+1)
        for i in range(self.sensing_matrix.shape[1]+1):
        	if state.col[i] == 0:
        		valid_moves[i] = 1
		
		return valid_moves
		
    def getGameEnded(self, state): 
    	#check if state is a terminal state or not. 
    	#terminal state is decided by satisfying any one of the 3 conditions
    	#	1)how many columns we have taken
    	#	2)STOP action has been chosen state.col[-1] = 1
    	#	3)whether Ax = y
    	#   return reward of ||x||_0 + ||Ax - y||_2^2 where x is the solution to min_z||A_S*z - y||_2^2,
    	#   where S is the set of columns chosen.
    	#rewards at the terminal state should be negative because we are doing a minimization problem vs alphago maximization problem   
    	
    	A_S = np.matmul(self.sensing_matrix, state.col[0:state.col.size])
    	x = np.linalg.lstsq(A_S, self.obs_vector)
    	error = np.linalg.norm(np.matmul(A_S, x) - y)
    	epsilon = 1e-10
    	gamma = 1
        
        if state.num_col = self.sparsity_limit or state.col[-1] = 1 or error<epsilon:
        	reg = gamma*np.linalg.norm(np.matmul(A_S,x))
        	r = - state.num_col - reg
        	
        return r
        
    def stringRepresentation(self, state):
        return np.array_str(state.col)

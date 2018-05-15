from CSState import State
import numpy as np

class CSGame():
    def __init__(self, A, y, k = 1): 
    #A and y are both assumed to be numpy arrays/vectors
        self.sensing_matrix = A
        self.obs_vector = y
        self.sparsity_limit = k #sparsity limit assumed to be less than/equal to m

    def getInitBoard(self): 
    #Get the initial state at beginning of CS Game
    	col_size = self.sensing_matrix.shape[1] #column size
    	Initial_State = State(np.zeros(col_size + 1)) #The plus one is for the STOP action. If Initial_State.action_indices[col_size] = 1, then algorithm
    													#chooses to stop, and the next state is now a terminating state
    	return Initial_State

    def getActionSize(self): 
    #return number of all actions (equal to column size of A + 1)
    	return self.sensing_matrix.shape[1]+1 

    def getNextState(self, state, action): 
    #input is a State object and an action(integer) which is less than column size of A + 1. 
    #Output is the next state as a CSState object. 
    	if state.action_indices[-1] == 1: #Last entry of the state.action_indices numpy vector is the stop action. 
    		print ('Already at terminal state, no more actions can be taken.')
    		return state	
    	elif state.action_indices[action] == 1:
    		print (str(action) + ' action already taken, invalid move.')
    		return state
    	else:
        	nextstate_col_ind = np.array(state.action_indices)
        	nextstate_col_ind[action] = 1
        	next_state = State(nextstate_col_ind)
        	return next_state 

    def getValidMoves(self, state): 
    #input is a State object. Output is a binary numpy vector for valid moves,
    #where b[i] = 1 implies ith column can be taken. Total vector size is n+1, where n is number of columns of sensing matrix A
    	valid_moves = np.zeros(self.sensing_matrix.shape[1]+1)
    	for i in range(self.sensing_matrix.shape[1]+1):
    		if state.action_indices[i] == 0:
    			valid_moves[i] = 1
    	return valid_moves
		
    def getGameEnded(self, state): 
    	#check if state is a terminal state or not. 
    	#terminal state is decided by satisfying any one of the 3 conditions
    	#	1)how many columns we have taken
    	#	2)STOP action has been chosen state.action_indices[-1] = 1
    	#	3)whether Ax = y
    	#   return reward of -(||x||_0 + ||Ax - y||_2^2) where x is the solution to min_z||A_S*z - y||_2^2,
    	#   where S is the set of columns chosen.
    	#	rewards at the terminal state should be negative because we are doing a minimization problem vs alphago maximization problem   
    	
    	S = state.getcolIndices()
    	A_S = self.sensing_matrix[:,S] 
    	x = np.linalg.lstsq(A_S, self.obs_vector) # x is a length 4 list which contains (solution, residuals, rank of A_S, singular values of A_S)
    	#residuals is a (1,) numpy array. residuals returns an empty np array if A_S is a a)not full rank, 
    	#b)square matrix or fatter.
    	epsilon = 1e-5
    	gamma = 1
    	
    	if not x[1]: #Case in which residual returns an empty size 1 array, which implies state.num_col = number of rows of sensing matrix
    		r = - state.num_col
    		return r
    	elif state.num_col == self.sparsity_limit or state.action_indices[-1] == 1 or x[1] < epsilon:
    		r = -state.num_col - gamma*x[1]
    		return r
    	else:
    		print("input state is not a terminal state")
    		return 0
        
    def stringRepresentation(self, state):
        return np.array_str(state.action_indices)

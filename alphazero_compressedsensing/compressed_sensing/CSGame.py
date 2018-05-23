from Game_Args import Game_args
from CSState import State
import numpy as np

class CSGame(): 

    def getInitBoard(self, args): #Game_args is an object
    #Get the initial state at beginning of CS Game
    	Initial_State = State(np.zeros(args['n'] + 1)) #The plus one is for the STOP action.
    	return Initial_State

    def getActionSize(self, args): 
    #return number of all actions (equal to column size of A + 1)
    	return args['n']+1 

    def getNextState(self, state, action): 
    #input is a State object and an action(integer) which is less than column size of A + 1. 
    #Output is the next state as a CSState object with new self.action_indices and self.col_indices.
    	if state.action_indices[-1] == 1: #Last entry of the state.action_indices numpy vector is the stop action. 
    		print ('Already at terminal state, no more actions can be taken.')
    		return state	
    	elif state.action_indices[action] == 1:
    		print (str(action) + ' action already taken, invalid move.')
    		return state
    	else:
        	nextstate_action_indices = np.array(state.action_indices)
        	nextstate_action_indices[action] = 1
        	
        	#Check if the action taken was a stopping action and construct next state's column indices
        	if action < state.action_indices.size-1: #Note that state.action_indices[state.action_indices.size-1] = stopping action
        		nextstate_col_indices = copy.deepcopy(state.col_indices).append(action)
        	else: #if stop action was taken, then the next state's taken columns are the same
        		nextstate_col_indices = copy.deepcopy(state.col_indices)
        	
        	next_state = State(nextstate_action_indices, nextstate_col_indices)
        	
        	return next_state 

    def getValidMoves(self, state): #O(n) operation
    #input is a State object. Output is a binary numpy vector for valid moves,
    #where b[i] = 1 implies ith column can be taken. Total vector size is n+1, where n is number of columns of sensing matrix A
    	valid_moves = np.zeros(state.action_indices.size)
    	for i in range(state.action_indices.size):
    		if state.action_indices[i] == 0:
    			valid_moves[i] = 1
    	return valid_moves
		
    def getGameEnded(self, state, Game_args): 
    	#check if state is a terminal state or not. Look at state.computeTermReward
    	#terminal state is decided by satisfying any one of the 3 conditions
    	#	1)how many columns we have taken
    	#	2)STOP action has been chosen state.action_indices[-1] = 1
    	#	3)whether Ax = y
    	#	rewards at the terminal state should be negative because we are doing a minimization problem vs alphago maximization problem   
    	
    	state.computeTermReward(Game_args)
    	return state.termreward
        
    def stringRepresentation(self, state): #convert state.action_indices into a string for hashing in MCTS
        return np.array_str(state.action_indices)

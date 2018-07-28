from Game_Args import Game_args
from CSState import State
import numpy as np
import copy

class CSGame(): 

    def getInitBoard(self, args, Game_args): #Game_args is an object
    #Get the initial state at beginning of CS Game and compute its features and terminal reward
        Initial_State = State(np.zeros(args['n'] + 1)) #The plus one is for the STOP action.
        return Initial_State

    def getActionSize(self, args): 
    #return number of all actions (equal to column size of A + 1)
        return args['n']+1 

    def getNextState(self, state, action): 
    #input is a State object and an action(integer) which is less than column size of A + 1. 
    #Output is the next state as a CSState object with new self.action_indices and self.col_indices.
    #This method is only called in MCTS.py for when s is NOT A LEAF and NOT A TERMINAL STATE, since MCTS search stops when terminal node is met
        
        
        #Construct new col_indices for next state. If action was stopping action, DO NOT add to col_indices
        
        if action != state.action_indices.size - 1: #Note that state.action_indices[state.action_indices.size - 1] is the stopping action
            nextstate_col_indices = copy.deepcopy(state.col_indices)
            nextstate_col_indices.append(action)
        else: #if stopping action was chosen as best action, then the column indices of next state is the same as current state
            nextstate_col_indices = copy.deepcopy(state.col_indices)
        #Construct action_indices for next state
        nextstate_action_indices = np.array(state.action_indices)
        nextstate_action_indices[action] = 1
        #Construct the next state object
        next_state = State(nextstate_action_indices, nextstate_col_indices)
        
        #FOR TESTING--------
        #print('The Next State has the following action indices and currently held columns respectively:')
        #print(next_state.action_indices)
        #print(next_state.col_indices)
        #print('')
        #-------------------
        
        
        return next_state 

    def getValidMoves(self, state): #O(n) operation
    #input is a State object. Output is a binary numpy vector of size n+1 for valid moves,
    #where b[i] = 1 implies i+1 column can be taken(for i in [0,1,...,n-1].b[n]= 0 or 1 denotes whether
    #the stopping action can be taken. n is number of columns of sensing matrix A. 
        valid_moves = np.zeros(state.action_indices.size)
        #Determine valid moves for all columns
        for i in range(state.action_indices.size-1):
            if state.action_indices[i] == 0:
                valid_moves[i] = 1
        #Determine if the stopping action is a valid move. This is determined by whether the current state has residual of ||A_Sx-y||_2^2. If residual has nonzero norm, then stopping action cannot be taken and is set to 0.
        #Note that getValidMoves is called only in MCTS.search, and compute_x_S_and_res has been called before getValidMoves is called, so the below is well defined.
        
        ATtimesres_norm = np.linalg.norm(state.feature_dic['col_res_IP'])**2
        if ATtimesres_norm > 1e-5:
            valid_moves[state.action_indices.size-1] = 0
        return valid_moves
        
    def getGameEnded(self, state, args, Game_args): 
        #check if state is a terminal state or not. Look at state.computeTermReward
        #terminal state is decided by satisfying any one of the 3 conditions
        #   1)how many columns we have taken
        #   2)STOP action has been chosen state.action_indices[-1] = 1
        #   3)whether Ax = y
        #   rewards at the terminal state should be negative because we are doing a minimization problem vs alphago maximization problem   
        state.computeTermReward(args, Game_args)
        return state.termreward
        
    def stringRepresentation(self, state): #convert state.action_indices into a string for hashing in MCTS
        action_indices_tuples = tuple(state.action_indices.tolist())
        action_indices_hashkey = hash(action_indices_tuples)
        action_indices_dictkey = str(action_indices_hashkey)
        return action_indices_dictkey
        #return np.array_str(state.action_indices) 
from Game_Args import Game_args
from CSState import State
import numpy as np
import copy

class CSGame(): 

    def getInitBoard(self, args, Game_args, identifier = None): #Game_args is an object
    #Get the initial state at beginning of CS Game and compute its features and terminal reward
        Initial_State = State(np.zeros(args['n'] + 1), identifier = identifier) #The plus one is for the STOP action.
        return Initial_State

    def getActionSize(self, args): 
    #return number of all actions (equal to column size of A + 1)
        return args['n']+1 

    def getNextState(self, state, action, Game_args, compute_inverse): 
    #input is a State object and an action(integer) which is less than column size of A + 1. 
    #Output is the next state as a CSState object with new self.action_indices and self.col_indices.
    #This method is only called in MCTS.py for when s is NOT A LEAF and NOT A TERMINAL STATE, since MCTS search stops when terminal node is met
        
        #Construct new col_indices for next state. If action was stopping action, DO NOT add to col_indices
        
        if action != state.action_indices.size - 1: #Note that state.action_indices[state.action_indices.size - 1] is the stopping action
            #nextstate_col_indices = copy.deepcopy(state.col_indices)
            nextstate_col_indices = state.col_indices[:] #makes a shallow copy of state.col_indices. Note that state.col_indices contains integers, which are immutable, so it is sufficient to use shallow copy here.
            nextstate_col_indices.append(action)
        else: #if stopping action was chosen as best action, then the column indices of next state is the same as current state
            #nextstate_col_indices = copy.deepcopy(state.col_indices)
            nextstate_col_indices = state.col_indices[:]
        #Construct action_indices for next state
        nextstate_action_indices = np.array(state.action_indices)
        nextstate_action_indices[action] = 1
        #Construct the next state object
        next_state = State(nextstate_action_indices, nextstate_col_indices, state.identifier)
        
        #if compute_inverse == 1, compute the new (A^T * A)^-1 and save it in next_state.inverse
        if compute_inverse == 1:
            if action == state.action_indices.size - 1: #if stopping action is chosen, then the inverse and AT*b are the same as previous state.
                next_state.inverse = state.inverse
                next_state.ATy = state.ATy
            elif len(next_state.col_indices) > 1: #use information from state to compute the inverse for next_state
                #1)Compute the new inverse of next_state
                #Get the old matrix and new columns respectively
                A_prev = Game_args.sensing_matrix[:, state.col_indices]
                new_c = Game_args.sensing_matrix[:,action]
        
                u1 = np.matmul(A_prev.transpose(), new_c)
                u1 = np.reshape(u1, (u1.shape[0], 1))
                u2 = np.matmul(state.inverse, u1)
                
                d = 1/(np.matmul(new_c.transpose(), new_c) - np.matmul(np.matmul(u1.transpose(), state.inverse), u1))
                u3 = d*u2
                F11_inverse = state.inverse + d*np.outer(u2, u2)
                
                left = np.vstack((F11_inverse, -1*u3.transpose()))
                right = np.vstack((-1*u3, d))
                
                next_state.inverse = np.hstack((left, right))
                
                #2)Compute A^T*b of next_state
                bottom = np.matmul(new_c, Game_args.obs_vector) #should be just a float
                next_state.ATy = np.vstack((state.ATy, bottom))
            
            else: #If there is only one column chosen, then we just go ahead and compute
                A_S = Game_args.sensing_matrix[:, next_state.col_indices]
                next_state.inverse = np.linalg.inv(np.matmul(A_S.transpose(), A_S)) #should just be a number here for 1 column case
                next_state.ATy = np.matmul(A_S.transpose(), Game_args.obs_vector) #should just be a number here for 1 column case
        
        return next_state 

    def getValidMoves(self, state): #O(n) operation
    #input is a State object. Output is a binary numpy vector of size n+1 for valid moves,
    #where b[i] = 1 implies i+1 column can be taken(for i in [0,1,...,n-1].b[n]= 0 or 1 denotes whether
    #the stopping action can be taken. n is number of columns of sensing matrix A. 
        num_actions = state.action_indices.size
        valid_moves = np.zeros(num_actions)
        #Determine valid moves for all columns except the stopping action by flipping action_indices vector. 
        valid_moves[0:num_actions] = 1 - state.action_indices[0:num_actions]
        #Determine if the stopping action is a valid move. This is determined by whether the current state has residual of ||A_Sx-y||_2^2. If residual has nonzero norm, then stopping action cannot be taken and is set to 0.
        #Note that getValidMoves is called only in MCTS.search, and compute_x_S_and_res has been called before getValidMoves is called, so the below is well defined.
        
        ATtimesres_norm = np.linalg.norm(state.feature_dic['col_res_IP'])**2
        if ATtimesres_norm > 1e-5:
            valid_moves[num_actions-1] = 0
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
        
    def keyRepresentation(self, state): #convert state.action_indices into a string for hashing in MCTS
        action_indices_list = state.action_indices.tolist()
        action_indices_list.append(state.identifier)
        hashkey_tuples = tuple(action_indices_list)
        state_dictkey = hash(hashkey_tuples)
        
        #save the key in State object
        state.keyRep = state_dictkey
        
        return state_dictkey
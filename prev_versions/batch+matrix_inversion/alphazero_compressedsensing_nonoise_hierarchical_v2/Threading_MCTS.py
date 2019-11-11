import threading
import numpy as np
import math
#FOR TESTING---------
import time
#END TESTING---------

EPS = 1e-8

#This class handles doing a single search among every (MCTS_object, States_list) pair in the list MCTS_States_list
class Threading_MCTS():
    
    def __init__(self, args, nnet):
        self.nnet = nnet #needed for prediction
        self.args = args
        
        #exit_flag is an event object which will control when the threaded_search function STOPS.
        #When each threaded_search stops, we access the MCTS_object for the appropriate neural network inputs
        
    def getActionProbs (self, MCTS_States_list):
    # calls parallel_search numMCTS times and outputs prob. dist. for each (MCTS object, [States list]) pair in MCTSandStates  
    # method is called by method Coach.advanceEpisodes
        
        #parallel search numMCTSsims times
        for i in range(self.args['numMCTSSims']):
            #TESTING-------------------------------
            #print('')
            #print('CURRENT BATCH SIMULATION:', i)
            #print('')
            #END TESTING---------------------------
            self.parallel_search(MCTS_States_list)
        
        #Once numMCTSsims parallel_searches have been done, all edge weights in each MCTS_obj have been updated, so we retrieve the probabilities
        #given by N(s,a). Put these new probabilities into a list.
        
        actionProbs = []
        
        for MCTS_object, State_list in MCTS_States_list:
            
            s = MCTS_object.game.keyRepresentation(State_list[-1])
            
            temp_counts = [MCTS_object.Nsa[(s,a)] if (s,a) in MCTS_object.Nsa else 0 for a in range(MCTS_object.game.getActionSize(self.args))]
            total_sum = float(sum(temp_counts))
            probs = [x/total_sum for x in temp_counts]
            actionProbs.append(probs)
                
        return actionProbs 
        
    def parallel_search (self, MCTS_States_list):
    #1)For each MCTS_object in MCTS_States_list, traverse down from the current root to a leaf
    #2)For each MCTS_object, States_list in MCTS_States_list, update the leaf
    #3)For each MCTS_object, update the edges which were traversed in step 1. 
    
        #FOR TESTING-----------------
        #print('------------------------------------------------------------')
        #print('BEGINNING SINGLE PARALLEL_SEARCH.....')
        #print('')
        #END TESTING-----------------
        
        #Conduct a search on every (MCTS_object, States_list) pair
        
        #FOR TESTING------------------
        #print('')
        #print('BEGINNING TRAVERSING DOWN TO LEAF FOR EACH MCTS_object, States_list pair...')
        #END TESTING------------------
        
        for MCTS_object, States_list in MCTS_States_list:
        
            current_root = States_list[-1]
            
            #reinitialize search_path to be empty(since this is a new search from root to leaf)
            MCTS_object.search_path = []
            
            #start recursive search to leaf. After traversetoLeaf is called, the search path traveled
            #for MCTS_object is saved in MCTS_object.search_path.
            self.search_traversetoLeaf(MCTS_object, current_root)
        
        #After all MCTS_object.search_path have been computed, make a batch prediction by 
        #compiling each MCTS_object.search_path[-1] into a single query.
        
        pas_matrix, v_matrix = self.nnet.batch_predict(MCTS_States_list)
        
        #FOR TESTING-------------------------
        #print('')
        #print('The batch prediction returned the following:')
        #print("pas_matrix: ", pas_matrix)
        #print("v_matrix: ", v_matrix)
        #print('')
        #END TESTING--------------------------
            
        #Save the batch predictions into each MCTS_object and continue with MCTS search by updating the leaf node
        #of each MCTS_object. pas_matrix and v_matrix saves the predictions from searches which end
        #on a leaf and NOT a terminal node. Hence, as we loop through MCTS_States_list, we should skip
        #pairs in which the search ended on a terminal state. Hence, the if check in the loop below.
        
        i = 0

        for MCTS_object, States_list in MCTS_States_list:
            last_state = MCTS_object.search_path[-1]

            if MCTS_object.Es[last_state.keyRep] == 0:
                MCTS_object.batchquery_prediction = [pas_matrix[i,:], v_matrix[i]]
                
                #We only need to update the last state in search path if the search 
                #ended on a leaf(and not a terminal node) 
                
                self.search_updateLeaf(MCTS_object)
                i += 1
                
            #Note that for each MCTS_object, we need to update edge weights of 
            #search path no matter if we ended on a  terminal node or not during search
            self.search_updateTraversedEdges(MCTS_object)
    
    
    def search_traversetoLeaf(self, MCTS_object, State):
        #traverse from root to leaf and store the search path into
        #MCTS_object.search_path. MCTS_object.search_path is in the form of 
        #[(state, a), (state2, a2), ..., leaf or terminal state]

        s = MCTS_object.game.keyRepresentation(State)
        
        #BASE CASES FOR TRAVERSE TO LEAF RECURSIVE SEARCH
        
        #1)Compute the terminal reward for state if not computed before
        if s not in MCTS_object.Es: # Note that MCTS_object.Es[s] not defined is NOT THE SAME as MCTS_object.Es[s] = 0
            MCTS_object.Es[s] = MCTS_object.game.getGameEnded(State, self.args, MCTS_object.game_args)
        
        #2)Check if the current state we are on is terminal or not. If terminal, 
        #attach the node to our search path and return. 
        if MCTS_object.Es[s] != 0:
            MCTS_object.search_path.append(State)
            
            return
            
        #3)Check if we are at leaf by checking if MCTS_object.Ps[s] is well defined
        if s not in MCTS_object.Ps:
            #Compute the features of the leaf 
            State.compute_x_S_and_res(self.args, MCTS_object.game_args)
            #Save the features of this leaf to the MCTS_object for batch prediction later. Note that
            #from compute_x_S_and_res, we first save features in state object, and then we assign it in MCTS_object.features_s[s]
            MCTS_object.features_s[s] = State.feature_dic
            #
            MCTS_object.search_path.append(State)
            return 
        
        #RECURSIVE CASE. If MCTS_object.Es[s] == 0 and MCTS_object.Ps[s] is well defined, then we are not yet at a leaf, so
        #we continue the search. 
        valids = MCTS_object.Vs[s] #retrieve numpy vector of valid moves
        
        cur_best = -float('inf') #temp variable which holds the current highest UCB value
        best_act = -1 #temp variable which holds the current best action with largest UCB. Initialized to -1.
        
        for a in range(MCTS_object.game.getActionSize(self.args)): #iterate over all possible actions. 
            if valids[a]:
                if (s,a) in MCTS_object.Qsa:
                    u = MCTS_object.Qsa[(s,a)] + self.args['cpuct']*MCTS_object.Ps[s][a]*math.sqrt(MCTS_object.Ns[s])/(1+MCTS_object.Nsa[(s,a)]) #note here that MCTS_object.Ns[s] is number of times s 
                    #was visited. Note that MCTS_object.Ns[s] = sum over b of MCTS_object.Nsa[(s,b)], so the equation above is equal to surag nair's notes.
                else:
                    #This line occurs if (s,a) is not in MCTS_object.Qsa, which means that if we take action a, then the next node next_s must be a leaf. This (s,a) will be added to MCTS_object.Qsa below and be assigned value of v.
                    u = self.args['cpuct']*MCTS_object.Ps[s][a]*math.sqrt(MCTS_object.Ns[s] + EPS)    
                
                #find the largest uct value
                if u > cur_best: 
                    cur_best = u
                    best_act = a

        #append the (state,action) tuple to the search path
        MCTS_object.search_path.append((State,best_act))
        #get the next state and continue recursive search by calling traversetoLeaf
        next_s = MCTS_object.game.getNextState(State, best_act, MCTS_object.game_args, 1)
        self.search_traversetoLeaf(MCTS_object, next_s) #traverse from root to a leaf or terminal node using recursive search. 
        

    def search_updateLeaf(self, MCTS_object):
    #This should be run after search_traversetoLeaf and batch_predict has been run on ALL MCTS_objects.
    #Note that the leaf only needs to be updated assuming traversetoLeaf landed on a leaf and NOT a terminal node.
    #From MCTS_object.search_path, we need to update the following variables in MCTS_object:
    #1)Compute MCTS_object.Ps[leaf] using batch neural network prediction.
    #2)Store valid moves for leaf in MCTS_object.Vs[leaf]
    #3)Initialize visit count to 0 in MCTS_object.Ns[leaf]
    
        leaf_state = MCTS_object.search_path[-1]
        leaf_key = leaf_state.keyRep
    
        #retrieve the computed MCTS_object.Ps[leaf_key]
        MCTS_object.Ps[leaf_key] = MCTS_object.batchquery_prediction[0]
    
        valids = MCTS_object.game.getValidMoves(leaf_state) #returns a numpy vector of 0 and 1's which indicate valid moves from the set of all actions
        MCTS_object.Ps[leaf_key] = MCTS_object.Ps[leaf_key]*valids      # masking(hiding) invalid moves(this element wise product between two equally sized vectors creates a vector of probabilities of valid moves) the neural network may predict. 
        sum_Ps_leaf = np.sum(MCTS_object.Ps[leaf_key])    
    
        #final assignment for MCTS_object.Ps[leaf_key]
        if sum_Ps_leaf > 0:
            MCTS_object.Ps[leaf_key] /= sum_Ps_leaf    # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable
                
            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you have overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
            print("All valid moves were masked, do workaround.")
                
            MCTS_object.Ps[leaf_key] = MCTS_object.Ps[leaf_key] + valids #These two lines makes all valid moves equally probable. 
            MCTS_object.Ps[leaf_key] /= np.sum(MCTS_object.Ps[leaf_key])
        
        #augment MCTS_object.Ps[leaf_key] with prior knowledge of the true solution x. EQUATE BETA TO 1 FOR TESTING!!!
        #------------------------------------------------------------------------------------------------
        x_I = np.ceil(abs(MCTS_object.game_args.sparse_vector)) #Since x is always between -1 to 1, x_I is the indicator vector corresponding to x
        x_I = np.append(x_I, 0) #append the stopping action to x_I, so x_I is the indicator for the support of x and includes a 1 for the stopping action. 
        valid_xI = x_I*valids #component-wise multiplication of indicator x_I and valids
        MCTS_object.Ps[leaf_key] = MCTS_object.args['beta']*MCTS_object.Ps[leaf_key] + (1 - self.args['beta']) * (1/np.sum(valid_xI)) * valid_xI
        #------------------------------------------------------------------------------------------------
            
        MCTS_object.Vs[leaf_key] = valids #Store the valids for leaf
        MCTS_object.Ns[leaf_key] = 0    #Initialize visit count of leaf to 0. We will update this in search_updateTraversedEdges instead.

                
    def search_updateTraversedEdges(self, MCTS_object):
    #For each traversed edge (s,a) in the search, we update the following:
    #MCTS_object.Qsa[(s,a)]
    #MCTS_object.Nsa[(s,a)]
        
        #Propagate the true reward up search path if search path ended on a terminal node. Otherwise, propagate up the output of the neural network
        #Note that v is a 1 by 1 np array, so hence the [0] at the end.
        if MCTS_object.Es[MCTS_object.search_path[-1].keyRep] == 0:
            v = MCTS_object.batchquery_prediction[1][0]
        else: #if last state visited in the MCTS simulation is a terminal node. 
            v = MCTS_object.Es[MCTS_object.search_path[-1].keyRep]
        
        
        #Update weights of all edges in the search_path. Also increment node values. Note that the loop omits the last element because the last element is a state and not a pair.
        for (State, a) in MCTS_object.search_path[:-1]:
            #Note that State.keyRep should be well defined since every State in each (State, a) pair have had search_traversetoLeaf called on it,
            #which calls game.keyRepresentation
            s = State.keyRep
            if (s,a) in MCTS_object.Qsa:
                #FOR TESTING----------------------
                #print("")
                #print('(s,a) IS IN MCTS_object.Qsa !!!!!!!!')
                #print("MCTS_object.identifier:", MCTS_object.identifier)
                #print("current_root:", MCTS_object.search_path[0][0].col_indices)
                #print("(s, a):", State.col_indices, a)
                #print("(s, a): ", s, a)
                #print("State.inverse: ", State.inverse)
                #print("State.ATy: ", State.ATy)
                #Check that the State.inverse*State.ATy is indeed the solution matching np.linalg.lstsq
                #if State.col_indices != []:
                #    print("regression solution from product of inverse and ATy: ", np.matmul(State.inverse, State.ATy))
                #    x = np.linalg.lstsq(MCTS_object.game_args.sensing_matrix[:, State.col_indices], MCTS_object.game_args.obs_vector)
                #    print("regression solution and residual from np.linalg.lstsq: ", x[0], x[1])
                #print("BEFORE updating Qsa, Nsa for (s,a) %%%%%%%%%")
                #print("v:", v)
                #print("MCTS_object.Qsa[(s,a)]:", MCTS_object.Qsa[(s,a)])
                #print("MCTS_object.Nsa[(s,a)]:", MCTS_object.Nsa[(s,a)])
                #END TESTING----------------------
                
                MCTS_object.Qsa[(s,a)] = (MCTS_object.Nsa[(s,a)]*MCTS_object.Qsa[(s,a)] + v)/(MCTS_object.Nsa[(s,a)]+1) #The v in this equation could be the true terminal reward OR the predicted reward from NN, depending on whether the search ended on a leaf which is also a terminal node. 
                MCTS_object.Nsa[(s,a)] += 1
                
                #FOR TESTING----------------------
                #print("AFTER updating Qsa, Nsa for (s,a) %%%%%%%%%")
                #print("v:", v)
                #print("MCTS_object.Qsa[(s,a)]:", MCTS_object.Qsa[(s,a)])
                #print("MCTS_object.Nsa[(s,a)]:", MCTS_object.Nsa[(s,a)])
                #print("Other Statistics.....")
                #print("MCTS_object.Ps[s]:", MCTS_object.Ps[s])
                #print("")
                #END TESTING----------------------
                

            else: #if (s,a) is not in dictionary MCTS_object.Qsa, that means (s,a) has never been visited before. These are edges connected to leaves!! IOW N(s,a) = 0. Hence, by the formula 3 lines above, MCTS_object.Qsa[(s,a)] = v.
                #FOR TESTING----------------------
                #print("")
                #print('(s,a) NOT IN MCTS_object.Qsa !!!!!!!!')
                #print("MCTS_object.identifier:", MCTS_object.identifier)
                #print("current_root:", MCTS_object.search_path[0][0].col_indices)
                #print("(s, a): ", State.col_indices, a)
                #print("(s, a): ", s, a)
                #print("State.inverse: ", State.inverse)
                #print("State.ATy: ", State.ATy)
                #Check that the State.inverse*State.ATy is indeed the solution matching np.linalg.lstsq
                #if State.col_indices != []:
                #    print("regression solution from product of inverse and ATy: ", np.matmul(State.inverse, State.ATy))
                #    x = np.linalg.lstsq(MCTS_object.game_args.sensing_matrix[:, State.col_indices], MCTS_object.game_args.obs_vector)
                #    print("regression solution and residual from np.linalg.lstsq: ", x[0], x[1])
                #END TESTING----------------------
                
                MCTS_object.Qsa[(s,a)] = v 
                MCTS_object.Nsa[(s,a)] = 1
                
                #FOR TESTING----------------------
                #print("AFTER updating Qsa, Nsa for (s,a) %%%%%%%%%")
                #print("v:", v)
                #print("MCTS_object.Qsa[(s,a)]:", MCTS_object.Qsa[(s,a)])
                #print("MCTS_object.Nsa[(s,a)]:", MCTS_object.Nsa[(s,a)])
                #print("Other Statistics.....")
                #print("MCTS_object.Ps[s]:", MCTS_object.Ps[s])
                #END TESTING----------------------
        
            MCTS_object.Ns[s] += 1
        
        #FOR TESTING--------------------------
        #last_state = MCTS_object.search_path[-1]
        #print('')
        #print('last state col indices: ', last_state.col_indices)
        #print('last state action indices: ', last_state.action_indices)
        #last_state_key = last_state.keyRep
        #print('last state key rep: ', last_state_key)
        #print("State.inverse: ", last_state.inverse)
        #print("State.ATy: ", last_state.ATy)
        #Check that the last_state.inverse*last_state.ATy is indeed the solution matching np.linalg.lstsq
        #if last_state.col_indices != []:
        #    print("regression solution from product of inverse and ATy: ", np.matmul(last_state.inverse, last_state.ATy))
        #    x = np.linalg.lstsq(MCTS_object.game_args.sensing_matrix[:, last_state.col_indices], MCTS_object.game_args.obs_vector)
        #    print("regression solution and residual from np.linalg.lstsq: ", x[0], x[1])
        #print('The termreward currently stored for the last state is: ', last_state.termreward)
        #last_state.computeTermReward(MCTS_object.args, MCTS_object.game_args)
        #if last_state_key in MCTS_object.Ps:
        #    print('last state updated Ps[s]: ', MCTS_object.Ps[last_state_key])
        #print('')
        #print('Generated vector y is: ', MCTS_object.game_args.obs_vector)
        #print('')
        #print('------------------------------------------------------------')
        #END TESTING--------------------------
  
    
 
        
    
    
    
    
    

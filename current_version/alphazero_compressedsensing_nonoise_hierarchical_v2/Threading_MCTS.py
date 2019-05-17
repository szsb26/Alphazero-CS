import threading
import numpy as np
import math
#FOR TESTING---------
import time
#END TESTING---------

EPS = 1e-8

#This class handles doing a single search among every (MCTS_object, States_list) pair in the list MCTS_States_list
class Threading_MCTS():
    
    def __init__(self, args, nnet, skip_nnet = None):
        self.nnet = nnet #needed for prediction
        self.skip_nnet = skip_nnet
        self.args = args
        
        #exit_flag is an event object which will control when the threaded_search function STOPS.
        #When each threaded_search stops, we access the MCTS_object for the appropriate neural network inputs
        
    def getActionProbs (self, MCTS_States_list, temp=1):
    # calls parallel_search numMCTS times and outputs prob. dist. for each (MCTS object, [States list]) pair in MCTSandStates  
        
        #parallel search numMCTSsims times
        for i in range(self.args['numMCTSSims']):
            #TESTING-------------------------------
            #print('')
            #print('CURRENT BATCH SIMULATION:', i)
            #print('')
            #END TESTING---------------------------
            self.parallel_search(MCTS_States_list)
        
        #Once numMCTSsims parallel_searches have been done, all weights in each MCTS_obj have been updated, so we retrieve the probabilities
        #given by N(s,a). Put these new probabilities into a list.
        
        actionProbs = []
        
        for pair in MCTS_States_list:
            MCTS_obj = pair[0]
            State_list = pair[1]
            
            s = MCTS_obj.game.keyRepresentation(State_list[-1])
            
            temp_counts = [MCTS_obj.Nsa[(s,a)] if (s,a) in MCTS_obj.Nsa else 0 for a in range(MCTS_obj.game.getActionSize(self.args))]
            
            if temp==0:
                bestA = np.argmax(temp_counts)
                probs = [0] * len(temp_counts)
                probs[bestA] = 1
                actionProbs.append(probs)
            
            else:
                temp_counts = [x**(1./temp) for x in temp_counts]
                probs = [x/float(sum(temp_counts)) for x in temp_counts]
                actionProbs.append(probs)
                
        return actionProbs 
        
    def parallel_search (self, MCTS_States_list):
    #1)parallel search conducts one search across all MCTS objects and updates the weights of each MCTS object
    #2)Each pair in MCTS_States_list is in the form (MCTS object, [State Objects])
    #3)We use threaded_search above on each MCTS object
        
        #Conduct a search on every (MCTS_object, States_list) pair
        for pair in MCTS_States_list:
        
            MCTS_object = pair[0]
            States_list = pair[1]
            current_root = States_list[-1]
            
            #reinitialize search_path to be empty. 
            MCTS_object.search_path = []
            
            #start recursive search to leaf. After traversetoLeaf is called, the search path traveled
            #for MCTS_object is saved in MCTS_object.search_path.
            self.search_traversetoLeaf(MCTS_object, current_root)
        
        #After all MCTS_object.search_path have been computed, make a batch prediction by 
        #compiling each MCTS_object.search_path[-1] into a single query.
        
        pas_matrix, v_matrix = self.nnet.batch_predict(MCTS_States_list)
            
        #Save the batch predictions into each MCTS_object and continue with MCTS search by updating the leaf node
        #of each MCTS_object. pas_matrix and v_matrix saves the predictions from searches which end
        #on a leaf and NOT a terminal node. Hence, as we loop through MCTS_States_list, we should skip
        #pairs in which the search ended on a terminal state. Hence, the if check in the loop below.
        
        i = 0

        for pair in MCTS_States_list:
            MCTS_object = pair[0]
            last_state = MCTS_object.search_path[-1]

            if MCTS_object.Es[last_state.keyRep] == 0:
                MCTS_object.batchquery_prediction = [pas_matrix[i,:], v_matrix[i]]
            
                #TESTING-------------------------
                #search_path = []
                #for j in range(len(MCTS_object.search_path)-1):
                #    state = MCTS_object.search_path[j][0].col_indices
                #    action = MCTS_object.search_path[j][1]
                #    search_path.append((state,action))
                #search_path.append(MCTS_object.search_path[-1].col_indices)
                #print('')
                #print('MCTS_object identifier where search ended on LEAF node: ', MCTS_object.identifier)
                #print('generated sparsity for this MCTS object is: ', MCTS_object.game_args.game_iter)
                #print('original sparse vector x is: ', MCTS_object.game_args.sparse_vector)
                #print('MCTS_obj ' + str(MCTS_object.identifier) + ' search path to leaf was: ', search_path)
                #print('MCTS_obj ' + str(MCTS_object.identifier) + ' pas prediction: ', pas_matrix[i, :])
                #print('MCTS_obj ' + str(MCTS_object.identifier) + ' pas prediction size: ', pas_matrix[i, :].size)
                #print('MCTS_obj ' + str(MCTS_object.identifier) + ' v prediction: ', v_matrix[i])
                #print('MCTS_obj ' + str(MCTS_object.identifier) + ' v prediction size: ', v_matrix[i].size)
                #print('')
                #END TESTING---------------------
            
                #We only need to update the last state in search path if the search 
                #ended on a leaf(and not a terminal node) 
                self.search_updateLeaf(MCTS_object)
                i += 1
            #TESTING-------------------
            #else:
                #search_path = []
                #for j in range(len(MCTS_object.search_path)-1):
                #    state = MCTS_object.search_path[j][0].col_indices
                #    action = MCTS_object.search_path[j][1]
                #    search_path.append((state,action))
                #search_path.append(MCTS_object.search_path[-1].col_indices)
                #print('')
                #print('MCTS_object had search path which ended on terminal node')
                #print('generated sparsity for this MCTS object is: ', MCTS_object.game_args.game_iter)
                #print('original sparse vector x is: ', MCTS_object.game_args.sparse_vector)
                #print('MCTS_obj ' + str(MCTS_object.identifier) + ' search path to leaf was: ', search_path)
                #print('')
            #--------------------------
            #Note that for each MCTS_object, we need to update edge weights of 
            #search path no matter if we ended on a  terminal node or not during search
            self.search_updateTraversedEdges(MCTS_object)
            
            #TESTING-------------------
            #print('')
            #print('TREE STATISTICS AFTER UPDATING TRAVERSED EDGES:')
            #print('MCTS_object identifier:', MCTS_object.identifier)
            #print('MCTS_object.Qsa:', MCTS_object.Qsa.values())
            #print('MCTS_object.Nsa:', MCTS_object.Nsa.values())
            #print('MCTS_object.Ns:', MCTS_object.Ns.values())
            #print('MCTS_object.Ps:', MCTS_object.Ps.values())
            #print('MCTS_object.Es:', MCTS_object.Es.values())
            #print('')
            #END TESTING---------------
    
    
    def search_traversetoLeaf(self, MCTS_object, State):
        #traverse from root to leaf and store the search path into
        #MCTS_object.search_path. MCTS_object.search_path is in the form of 
        #[(state, a), (state2, a2), ..., leaf or terminal state]

        s = MCTS_object.game.keyRepresentation(State)
        
        #BASE CASES FOR TRAVERSE TO LEAF RECURSIVE SEARCH
        
        #1)Compute the terminal reward for state if not computed before
        if s not in MCTS_object.Es:
            MCTS_object.Es[s] = MCTS_object.game.getGameEnded(State, self.args, MCTS_object.game_args)
        
        #2)Check if the current state we are on is terminal or not. If terminal, 
        #attach the node to our search path and return. 
        if MCTS_object.Es[s] != 0:
            MCTS_object.search_path.append(State)
            
            return
        #3)We are now at a leaf
        if s not in MCTS_object.Ps:
            #Compute the features of the leaf
            State.compute_x_S_and_res(self.args, MCTS_object.game_args)
            #Save the features of this leaf to the MCTS_object for batch prediction later.
            MCTS_object.features_s[s] = State.feature_dic
            #
            MCTS_object.search_path.append(State)
            return 
        
        
        #RECURSIVE CASE
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
            
        MCTS_object.Vs[leaf_key] = valids #Store the valids for leaf
        MCTS_object.Ns[leaf_key] = 0    #Initialize visit count of leaf to 0. We will update this in search_updateTraversedEdges instead.

                
    def search_updateTraversedEdges(self, MCTS_object):
    #For each traversed edge (s,a) in the search, we update the following:
    #MCTS_object.Qsa[(s,a)]
    #MCTS_object.Nsa[(s,a)]
        
        #Propagate the true reward up search path if search path ended on a terminal node. Otherwise, propagate up the output of the neural network
        if MCTS_object.Es[MCTS_object.search_path[-1].keyRep] == 0:
            v = MCTS_object.batchquery_prediction[1]
        else: #if last state visited in the MCTS simulation is a terminal node.
            v = MCTS_object.Es[MCTS_object.search_path[-1].keyRep]
        
        
        #Update weights of all edges in the search_path. Also increment node values. Note that the loop omits the last element because the last element is a state and not a pair. 
        for (State, a) in MCTS_object.search_path[:-1]:
            #Note that State.keyRep should be well defined since every State in each (State, a) pair have had search_traversetoLeaf called on it,
            #which calls game.keyRepresentation
            s = State.keyRep
            if (s,a) in MCTS_object.Qsa:
                MCTS_object.Qsa[(s,a)] = (MCTS_object.Nsa[(s,a)]*MCTS_object.Qsa[(s,a)] + v)/(MCTS_object.Nsa[(s,a)]+1) #The v in this equation could be the true terminal reward OR the predicted reward from NN, depending on whether the search ended on a leaf which is also a terminal node. 
                MCTS_object.Nsa[(s,a)] += 1

            else: #if (s,a) is not in dictionary MCTS_object.Qsa, that means (s,a) has never been visited before. These are edges connected to leaves!! IOW N(s,a) = 0. Hence, by the formula 3 lines above, MCTS_object.Qsa[(s,a)] = v.
                MCTS_object.Qsa[(s,a)] = v 
                MCTS_object.Nsa[(s,a)] = 1
        
            MCTS_object.Ns[s] += 1
  
    
 
        
    
    
    
    
    

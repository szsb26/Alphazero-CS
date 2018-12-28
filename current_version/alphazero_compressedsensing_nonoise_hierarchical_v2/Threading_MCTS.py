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
        
            #TESTING-------------------------
            #print('--------------------------------------STARTING parallel_search ' + str(i) + '--------------------------------------')
            #END TESTING---------------------
            
            self.parallel_search(MCTS_States_list)
            
            #FOR TESTING------------------------
            #print('FINAL STATISTICS FOR EACH MCTS OBJECT AFTER RUNNING PARALLEL SEARCH: ' + str(i))
            #for pair in MCTS_States_list:
                #MCTS_obj = pair[0]
                #State_list = pair[1]
                #print('')
                #print('Statistics for MCTS object ' + str(MCTS_obj.identifier))
                #for pair in MCTS_obj.search_path[:-1]:
                    #State = pair[0]
                    #action = pair[1]
                    #print( 'Sequential state action pairs in search path is: ', (State.col_indices, State.keyRep, action))
                #print('The last state in the search path is: ', (MCTS_obj.search_path[-1].col_indices, MCTS_obj.search_path[-1].keyRep))
                #print('MCTS Nsa: ', MCTS_obj.Nsa)
                #print('MCTS Qsa: ', MCTS_obj.Qsa)
                #print('MCTS Ns: ', MCTS_obj.Ns)
                #print('MCTS Ps: ', MCTS_obj.Ps)
                #print('MCTS Vs: ', MCTS_obj.Vs)
            #END TESTING------------------------
            
            
            #TESTING-------------------------
            #print('--------------------------------------COMPLETING parallel_search ' + str(i) + '--------------------------------------')
            #print('')
            #END TESTING---------------------
        
        
        #Once numMCTSsims parallel_searches have been done, all weights in each MCTS_obj have been updated, so we retrieve the probabilities
        #given by N(s,a). Put these new probabilities into a list.
        
        actionProbs = []
        
        for pair in MCTS_States_list:
            MCTS_obj = pair[0]
            State_list = pair[1]
            
            s = MCTS_obj.game.keyRepresentation(State_list[-1])
            
            temp_counts = [MCTS_obj.Nsa[(s,a)] if (s,a) in MCTS_obj.Nsa else 0 for a in range(MCTS_obj.game.getActionSize(self.args))]
            
            #FOR TESTING--------------------------------------
            #print('The probability label for MCTS object ' + str(MCTS_obj.identifier) + ' at current root ' + str(State_list[-1].action_indices) + ' is: ')
            #print(temp_counts)
            #END TESTING--------------------------------------
            
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
            current_root = pair[1][-1]
            
            #reinitialize search_path to be empty. Note that this line cannot be put in traversetoLeaf because traversetoLeaf is recursive
            MCTS_object.search_path = []
            
            #start recursive search to leaf. After traversetoLeaf is called, the search path traveled
            #for MCTS_object is saved in MCTS_object.search_path.
            self.search_traversetoLeaf(MCTS_object, current_root)
        
        #After all MCTS_object.search_path have been computed, make a batch prediction by 
        #compiling each MCTS_object.search_path[-1] into a single query.
        
        pas_matrix, v_matrix = self.nnet.batch_predict(MCTS_States_list)
        
        #TESTING-------------------------
        #print('')
        #print('Got here too')
        #print('pas_matrix: ', pas_matrix)
        #print('pas_matrix size: ', pas_matrix.shape)
        #print('v_matrix:', v_matrix)
        #print('v_matrix:', v_matrix.shape)
        #print('')
        #END TESTING---------------------
            
        #Save the batch predictions into each MCTS_object and continue with MCTS search by updating the leaf node
        #of each MCTS_object
        
        i = 0
        for pair in MCTS_States_list:
            MCTS_object = pair[0]
            if MCTS_object.Es[MCTS_object.search_path[-1].keyRep] == 0:
                MCTS_object.batchquery_prediction = [pas_matrix[i,:], v_matrix[i]]
            
                #TESTING-------------------------
                #print('MCTS_obj ' + str(i) + ' pas prediction: ', pas_matrix[i, :])
                #print('MCTS_obj ' + str(i) + ' pas prediction size: ', pas_matrix[i, :].size)
                #print('MCTS_obj ' + str(i) + ' v prediction: ', v_matrix[i])
                #print('MCTS_obj ' + str(i) + ' v prediction size: ', v_matrix[i].size)
                #END TESTING---------------------
            
                #We only need to update the Leaf if the search ended on a Leaf(and not a terminal node) 
                self.search_updateLeaf(MCTS_object)
                i += 1
            
            self.search_updateTraversedEdges(MCTS_object)
            
        
        #Once MCTS trees have received the prediction, set flag to True, so all threads above continue to run, which completes a
        #single search on all MCTS tree objects in MCTS_States_list
    
        #Wait until all threads have completed. Essentially waits for each single search on each MCTS object completes.
            
        #FOR TESTING---------------------
        #print('')
        #print('Current number of running threads: ', threading.active_count())
        #print('')
        #END TESTING---------------------
        
    
    #-------------------------------------------------------------------------
    #search_traversetoLeaf(self, MCTS_object, State), search_updateLeaf(self, MCTS_object), search_updateTraversedEdges(self, MCTS_object)
    #constitute a single Monte Carlo Tree Search on the state object State. 
    
    def search_traversetoLeaf(self, MCTS_object, State):
        #FOR TESTING--------------------------------------------
        #print('------------------------beginning traversaltoLeaf for MCTS_object: ' + str(MCTS_object.identifier) + '------------------------')
        #END TESTING--------------------------------------------
    
    #recursive search down to a Leaf and store the search path in MCTS_object.search_path. 
    #nnet.batch_predict will take in MCTS_States_list as input, and each the MCTS_object in each pair in MCTS_States_list must have
    #MCTS_object.search_path well defined.
    
        #Note that calling  MCTS_object.game.keyRepresentation(State) actually saves the key representation in State.keyRep self variable.
        s = MCTS_object.game.keyRepresentation(State)
        

        #Check if terminal reward has been computed or not. If not, compute it via getGamEnded and save in MCTS_object.Es dictionary
        if s not in MCTS_object.Es:
            MCTS_object.Es[s] = MCTS_object.game.getGameEnded(State, self.args, MCTS_object.game_args)
            
            #FOR TESTING--------------------------------------------
            #print('terminal node has reward: ', MCTS_object.Es[s])
            #END TESTING--------------------------------------------
        
        #1) CHECK IF CURRENT SEARCHED STATE IS A TERMINAL STATE OR NOT
        #Check if the terminal reward is nonzero. If the terminal reward is nonzero, then we have arrived at a terminal state, so append s
        #to MCTS_object.search_path. Furthermore, since we have arrived at a terminal state, our search ends, and we also do not need to have the NN
        #run a prediction on this particular (MCTS_object, States_list) pair!!
        if MCTS_object.Es[s] != 0:
            #if the state we arrive at is a terminal node, there is no need to call the NN to expand it. Instead, save the terminal reward to State.termreward.
            #When we check MCTS_States_list later, we will see that State.termreward is no longer None.
            State.termreward = MCTS_object.Es[s]
            MCTS_object.search_path.append(State)
            
            #FOR TESTING--------------------------------------------
            #print('------------------------completing traversaltoLeaf for MCTS_object: ' + str(MCTS_object.identifier) + '------------------------')
            #END TESTING--------------------------------------------
            
            return
            
        #2) IF s IS A LEAF AND NOT A TERMINAL NODE
        if s not in MCTS_object.Ps:
            #Compute features and save them in State.feature_dic. This needs to be computed and saved
            #(in MCTS_object.features_s[s] so nnet.batch_predict can access this information when 
            #predicting for the leaf.
            State.compute_x_S_and_res(self.args, MCTS_object.game_args)
            #Since we are not making a prediction on the state immediately, we need to save the computed features in the MCTS_object
            MCTS_object.features_s[s] = State.feature_dic
            
            #FOR TESTING_------------------------------------
            #print('MCTS object: ', MCTS_object.identifier)
            #print('MCTS search leaf: ', State.action_indices)
            #print(MCTS_object.features_s[State.keyRep])
            #print(State.keyRep)
            #END TESTING ------------------------------------
            
            
            #FOR TESTING--------------------------------------------
            #print('------------------------completing traversaltoLeaf for MCTS_object: ' + str(MCTS_object.identifier) + '------------------------')
            #END TESTING--------------------------------------------
            
            #append s to MCTS_object.search_path
            MCTS_object.search_path.append(State)
            return 
        
        
        #3) IF NEITHER CASE 1 OR CASE 2, CONTINUE RECURSIVE SEARCH ONTO NEXT NODE VIA UCT RULE
        valids = MCTS_object.Vs[s] #retrieve numpy vector of valid moves
        
        cur_best = -float('inf') #temp variable which holds the current highest UCB value
        best_act = -1 #temp variable which holds the current best action with largest UCB. Initialized to -1.
        
        for a in range(MCTS_object.game.getActionSize(self.args)): #iterate over all possible actions. 
            if valids[a]:
                if (s,a) in MCTS_object.Qsa:
                    u = MCTS_object.Qsa[(s,a)] + self.args['cpuct']*MCTS_object.Ps[s][a]*math.sqrt(MCTS_object.Ns[s])/(1+MCTS_object.Nsa[(s,a)]) #note here that MCTS_object.Ns[s] is number of times s 
                    #was visited. Note that MCTS_object.Ns[s] = sum over b of MCTS_object.Nsa[(s,b)], so the equation above is equal to surag nair's notes.
                else:
                    u = self.args['cpuct']*MCTS_object.Ps[s][a]*math.sqrt(MCTS_object.Ns[s] + EPS)     # Q = 0 ? This line occurs if (s,a) is not in MCTS_object.Qsa, which means that if we take action a, then the next node next_s must be a leaf. This (s,a) will be added to MCTS_object.Qsa below and be assigned value of v.
                    
                if u > cur_best: #because this is equivalent to taking the max, this is why our rewards are negative!!!!!
                    cur_best = u
                    best_act = a

        a = best_act 
        #append the (state,action) tuple to the search path
        MCTS_object.search_path.append((State,a))
        #get the next state and continue recursive search by calling traversetoLeaf
        next_s = MCTS_object.game.getNextState(State, a)
        self.search_traversetoLeaf(MCTS_object, next_s) #traverse from root to a leaf or terminal node using recursive search. 
        

    def search_updateLeaf(self, MCTS_object):
    #This should be run after search_traversetoLeaf and batch_predict has been run on ALL MCTS_objects.
    #Note that the Leaf only needs to be updated assuming traversetoLeaf landed on a leaf and NOT a terminal node.
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
            
        MCTS_object.Vs[leaf_key] = valids #Since s is a leaf, intialize and store valid moves for leaf
        MCTS_object.Ns[leaf_key] = 0    #Initialize visit count of leaf to 0. We will update this in search_updateTraversedEdges instead.

        
                
    def search_updateTraversedEdges(self, MCTS_object):
    #For each traversed edge (s,a) in the search, we update the following:
    #MCTS_object.Qsa[(s,a)]
    #MCTS_object.Nsa[(s,a)]
        
        #Propagate the true reward up search path if search path ended on a terminal node. Otherwise, propagate up the output of the neural network
        if MCTS_object.Es[MCTS_object.search_path[-1].keyRep] == 0:
            v = MCTS_object.batchquery_prediction[1]
        else:
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
    #-------------------------------------------------------------------------   
    
 
        
    
    
    
    
    

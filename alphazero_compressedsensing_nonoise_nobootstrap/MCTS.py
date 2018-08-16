import math
import numpy as np
#np.set_printoptions(threshold=np.nan) #allows correct conversion from numpy array to string. Otherwise string is truncated with '...'
import time
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree. Note that for each game, a single tree is built. We do not construct new trees for each state/move during a single game.
    """

    def __init__(self, game, nnet, args, Game_args):
        self.game = game    #MCTS stores a a single game(as game object). Look at Game.py. 
                            #The game object stores essential information like valid moves, get next state(which is another game object, etc...
        
        self.nnet = nnet
        self.args = args    # args is a dictionary which contains user specified parameters which determines the complexity of the entire algorithm
        self.game_args = Game_args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited, which is equivalent to sum_b(self.Nsa[(s,b)])
        self.Ps = {}        # stores initial policy (returned by neural net). In our problem, the policy is a vector whose size is equal to #columns of A + 1 (+1 for the stopping action). 

        self.Es = {}        # stores game.getGameEnded ended for board s. IOW, stores all states and their terminal rewards. 0 if state not terminal, and true reward if state is terminal.
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1): #temp = 1, the default option, has getActionProb return a probability distribution. The next action is chosen at random from this distribution. 
        """
        This function performs numMCTSSims simulations of MCTS(expands tree numMCTSSims times) starting from
        canonicalBoard. We will be using this function to do self play. Uses search method below

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #do numMCTSSims number of MCTS searches/simulations
        #self.search updates the all the self variables above
        for i in range(self.args['numMCTSSims']):
            
            #FOR TESTING----------------------------
            #start = time.time()
            #END TESTING----------------------------
            self.search(canonicalBoard)
            #FOR TESTING----------------------------
            #end = time.time()
            #print('Total time for MCTS simulation iteration ' + str(i) + ' is: ' + str(end-start))
            #END TESTING----------------------------
        
        #print('')
        #print('MCTS STATISTICS after conducting numMCTSSims... for current state ' + str(canonicalBoard.action_indices))
        #print('self.Qsa: ' + str(self.Qsa))
        #for key in self.Nsa:
            #print(self.Nsa[key])
        #print('self.Ns: ' + str(self.Ns))
        #print('self.Ps: ' + str(self.Ps))
        #print('self.Es: ' + str(self.Es))
        #print('self.Vs: ' + str(self.Vs))
        #print('')
        
        #Count the number of times an action was taken from the canonicalBoard state as root node
        #and construct a list of how many times each action was taken
        s = self.game.stringRepresentation(canonicalBoard)
        #counts is a list of integer values
        #Namely, it contains values for number of times action a was investigated given state s(canonicalBoard from above)
        #otherwise, if an action was never investigated, then set it as 0. 
        #counts should sum to numMCTSsims 
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize(self.args))]
        
        #FOR TESTING-------------------------------------------------------
        #print('current counts is: ' + str(counts))
        #print('sum of counts is: ' + str(sum(counts)))
        #END TESTING-------------------------------------------------------
        
        if temp==0:
            #return the index of the list counts with largest value
            bestA = np.argmax(counts)
            #construct a zero vector with length of counts
            probs = [0]*len(counts)
            probs[bestA]=1
            #probs is a zero vector with a single 1 in bestA index.
            #print('MCTS probability dist. of next move is: ' + str(probs))
            return probs #here probs is a vector of 0 and 1s!!!
        
        #1./temp is a remnant of Python 2. For ex 1/5 = 0, but 1./5 = 0.2
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        #print('MCTS probability dist. of next move is: ' + str(probs))
        #returns a probability vector with floating values
        return probs #here probs is a vector which sums to 1!!!!


    def search(self, canonicalBoard): 
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the negative of the value of the current canonicalBoard. 
        """
        #BEGIN SEARCH ON GAME STATE canonicalBoard. canonicalBoard is root of MCTS tree
        #FOR TESTING----------------------------
        #print('The columns taken of the current state are: ' + str(canonicalBoard.col_indices))
        #END TESTING----------------------------
        
        #FOR TESTING----------------------------
        #start = time.time()
        #END TESTING----------------------------
        s = self.game.stringRepresentation(canonicalBoard)#get the string representation of the state canonicalBoard in the game and save as s. Required for hashing in steps below:
        
        #FOR TESTING----------------------------
        #end = time.time()
        #print('state action indices to string conversion time is: ' + str(end-start))
        #END TESTING----------------------------
        
        #----------CHECK IF canonicalBoard IS A TERMINAL STATE OR NOT----------
        #1)self.Es[s] is updated here since s is a terminal state
        #2)Note that terminal states are NOT expanded, so it is conceivable that MCTS search may search the same terminal node more than once. However,  
        #as the number of visits increases, the UCB for every action taken up to that terminal node will decrease(Nsa increases, where s is the node leading up to terminal node, so UCB decreases).
        #Since UCB decreases, this allows exploration of other actions. 
        #3)terminal states are forever leaves.
        if s not in self.Es: #if s is not in self.Es, hash s into Es and determine whether s is a terminal state or not. 
            #FOR TESTING----------------------------
            #start = time.time()
            #END TESTING----------------------------
            self.Es[s] = self.game.getGameEnded(canonicalBoard, self.args, self.game_args) #game.getGameEnded returns either -alpha||x_S||_0 + gamma*||A_Sx-y||_2^2 or 0. Note that once getGameEnded is called, canonicalBoard.termreward is set.
            #FOR TESTING----------------------------
            #end = time.time()
            #print('time it takes to compute terminal reward and insert into self.Es[s] of MCTS tree is: ' + str(end-start))
            #END TESTING----------------------------
        if self.Es[s]!=0: #Now that we have hashed s into self.Es, check whether the reward returned above is 0 or not. If not zero, return the true terminal reward from getGameEnded above and search stops.
            return self.Es[s] #FIRST RETURN. If we arrived at a terminal node, then we return the TRUE REWARD instead of the predicted v of the neural network. 
        #----------------------------------------------------------------------
        
        #-------------IF s IS A LEAF, THEN:---------------------
        #1)Call neural network to return probability distribution outcome v
        #2)Using step one, initialize the self variables of s by saving numpy vector of valid moves and times visited.
        #3)return v.
        #4)self.Ps[s], self.Vs[s], and self.Ns[s] are all initialized.
        #FOR TESTING----------------------------
        #start_leaf = time.time()
        #END TESTING----------------------------
        if s not in self.Ps: #If s is not a key in self.Ps(which hashes s to its next available game states, then s must be a leaf.
            # leaf node
            #Prepare canonicalBoard for feeding into NN
            
            #COMPUTE self.feature_dic and compute NN representation so we can call nnet.predict below
            #FOR TESTING----------------------------
            #start = time.time()
            #END TESTING----------------------------
            canonicalBoard.compute_x_S_and_res(self.args, self.game_args)
            canonicalBoard.converttoNNInput()
            #FOR TESTING----------------------------
            #end = time.time()
            #print('Time it takes to compute leaf state features and convert to NN input is: ' + str(end-start))
            #END TESTING----------------------------
            #FOR TESTING----------------------------
            #start = time.time()            
            #END TESTING----------------------------
            self.Ps[s], v = self.nnet.predict(canonicalBoard) #neural network takes in position s and returns a prediction(which is p_theta vector and v_theta (numpy vector). Look at own notes)
            #FOR TESTING----------------------------
            Ps_test, v_test = self.nnet.predict(canonicalBoard)
            #end = time.time()
            #print('Time it takes for NN to predict on leaf state is: ' + str(end - start))
            #print('Current leaf we are on is: ' + str(canonicalBoard.col_indices))
            #print('The output of the leaf when evaluated by NN is: ')
            #print('predicted prob. dist. : ' + str(self.Ps[s]))
            #print('predicted reward: ' + str(v))
            #END TESTING----------------------------
            #FOR TESTING----------------------------
            #start = time.time()
            #END TESTING----------------------------
            valids = self.game.getValidMoves(canonicalBoard) #returns a numpy vector of 0 and 1's which indicate valid moves from the set of all actions
            #FOR TESTING----------------------------
            #print('valid moves for leaf are: ' + str(valids))
            #END TESTING----------------------------
            self.Ps[s] = self.Ps[s]*valids      # masking(hiding) invalid moves(this inner product creates a vector of probabilities of valid moves) the neural network may predict. 
            sum_Ps_s = np.sum(self.Ps[s])       # probabilities may not add up to 1 anymore after hiding invalid moves(since NN may predict nonzero prob for illegal moves). Hence, renormalize such that
                                                # valid actions sum up to 1.
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you have overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                
                #FOR TESTING-------------------------------------------
                #print('A_S matrix is: ', self.game_args.sensing_matrix[:, canonicalBoard.col_indices])
                #print('rank(A_S) is: ', np.linalg.matrix_rank(self.game_args.sensing_matrix[:, canonicalBoard.col_indices]))
                #print('actual prob. output of neural network: ', Ps_test)
                #print('actual v output of neural network: ', v_test)
                #print('Ps[s]: ' + str(self.Ps[s]))
                #print('')
                #print('valids: ' + str(valids))
                #print('')
                #print('PROBLEM STATE DURING MCTS SEARCH (AT LEAF):')
                #print('---------------------------------------------')
                #print('action indices: ', canonicalBoard.action_indices)
                #print('')
                #print('col indices: ', canonicalBoard.col_indices)
                #print('')
                #print('feature_dic: ', canonicalBoard.feature_dic)
                #print('')
                #print('nn_input: ', canonicalBoard.nn_input)
                #print('')
                #print('true y: ', self.game_args.obs_vector)
                #print('')
                #print('true sparse x: ', self.game_args.sparse_vector)
                #print('')
                #print('reward: ', canonicalBoard.termreward)
                #print('---------------------------------------------')
                #return
                #END TESTING-------------------------------------------
                
                self.Ps[s] = self.Ps[s] + valids #These two lines makes all valid moves equally probable. 
                self.Ps[s] /= np.sum(self.Ps[s])
            #FOR TESTING----------------------------
            #end = time.time()
            #print('Time it takes to compute self.Ps[s] of leaf is: ' + str(end-start))
            #END TESTING----------------------------
            
            self.Vs[s] = valids #Since s is a leaf, store the newly found valid moves and set amount of times s was visited as zero.
            self.Ns[s] = 0
            
            #FOR TESTING----------------------------
            #end_leaf = time.time()
            #print('Time it takes to compute all operations given we are at leaf: ' + str(end_leaf - start_leaf))
            #END TESTING----------------------------
            
            return v #SECOND RETURN
        #--------------------------------------------------------
        
        #-----------IF CURRENT NODE s IS NOT A LEAF, THEN CONTINUE SEARCH------------------------
        #1)search from a root to a leaf via UCB using recursive search
        #2)Once we arrive at a leaf, due to the recursive nature, we update self.Qsa and self.Nsa dictionaries in a bottom up fashion. 
        #3)Note that if s is not a leaf, it has been a leaf before, so self.Vs[s], self.Ps[s], self.Ns[s] are all well defined.
        
        #FOR TESTING----------------------------
        #start_notleaf = time.time()
        #END TESTING----------------------------
        
        valids = self.Vs[s] #retrieve numpy vector of valid moves
        
        cur_best = -float('inf') #temp variable which holds the current highest UCB value
        best_act = -1 #temp variable which holds the current best action with largest UCB. Initialized to -1.
        for a in range(self.game.getActionSize(self.args)): #iterate over all possible actions. 
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['cpuct']*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)]) #note here that self.Ns[s] is number of times s 
                    #was visited. Note that self.Ns[s] = sum over b of self.Nsa[(s,b)], so the equation above is equal to surag nair's notes.
                else:
                    u = self.args['cpuct']*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ? This line occurs if (s,a) is not in self.Qsa, which means that if we take action a, then the next node next_s must be a leaf. This (s,a) will be added to self.Qsa below and be assigned value of v.
                    
                if u > cur_best: #because this is equivalent to taking the max, this is why our rewards are negative!!!!!
                    cur_best = u
                    best_act = a

        a = best_act #define action with highest UCB computed above as a. a is chosen over valids. Note that best_act is a single action we take when traversing a single depth of MCTS tree.
        next_s = self.game.getNextState(canonicalBoard, a) #returns next state object
        
        #FOR TESTING----------------------------
        #end_notleaf = time.time()
        #print('Time it takes to determine which neighbor to traverse to if current node we are on is not a leaf: ' + str(end_notleaf-start_notleaf))
        #END TESTING----------------------------
        
        v = self.search(next_s) #traverse from root to a leaf or terminal node using recursive search. 
        #-------------------------------------------------------------------------------------
        #because we recursively search in the lines above, the below snippet updates self.Qsa and self.Nsa of all visited nodes from the bottom to the root.
        #Because we reach a leaf sooner or later, the middle section is executed and v below is well defined!
        #formulas below match what we have in notebook.
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1) #The v in this equation could be the true terminal reward OR the predicted reward from NN, depending on whether the search ended on a leaf which is also a terminal node. 
            self.Nsa[(s,a)] += 1

        else: #if (s,a) is not in dictionary self.Qsa, that means (s,a) has never been visited before. These are edges connected to leaves!! IOW N(s,a) = 0. Hence, by the formula 3 lines above, self.Qsa[(s,a)] = v.
            self.Qsa[(s,a)] = v 
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v #THIRD RETURN
        
import math
import numpy as np
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
        self.Ps = {}        # stores initial policy (returned by neural net). In our problem, the policy is a vector whose size is equal to #columns of A. 

        self.Es = {}        # stores game.getGameEnded ended for board s. IOW, stores all states and their rewards. 0 if state not terminal, and true reward if state is terminal.
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS(expands tree numMCTSSims times) starting from
        canonicalBoard. We will be using this function to do self play. Uses search method below

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #do numMCTSSims number of MCTS searches/simulations
        #print(self.game_args.sensing_matrix)
        #print(self.game_args.obs_vector)
        #print(canonicalBoard.action_indices)
        for i in range(self.args['numMCTSSims']):
            #FOR TESTING---------------------  
            #print(i)
            #--------------------------------
            self.search(canonicalBoard)
        
        #Count the number of times an action was taken from the canonicalBoard state as root node
        #and construct a list of how many times each action was taken
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize(self.args))]
        
        if temp==0:
            #return the index of the list counts with largest value
            bestA = np.argmax(counts)
            #construct a zero vector with length of counts
            probs = [0]*len(counts)
            probs[bestA]=1
            #probs is a zero vector with a single 1 in bestA index.
            return probs
        
        #1./temp is a remnant of Python 2. For ex 1/5 = 0, but 1./5 = 0.2
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        #returns a probability vector with floating values
        return probs


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

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        #BEGIN SEARCH ON GAME STATE canonicalBoard. canonicalBoard is root of MCTS tree
        s = self.game.stringRepresentation(canonicalBoard)#get the string representation of the state canonicalBoard in the game and save as s. Required for hashing in steps below:
        
        #----------CHECK IF canonicalBoard IS A TERMINAL STATE OR NOT----------
        #1)self.Es[s] is updated here since s is a terminal state
        #2)Note that terminal states are NOT expanded, so it is conceivable that MCTS search may search the same terminal node more than once. However,  
        #as the number of visits increases, the UCB for every action taken up to that terminal node will decrease(Nsa increases, where s is the node leading up to terminal node, so UCB decreases).
        #Since UCB decreases, this allows exploration of other actions. 
        if s not in self.Es: #if s is not in self.Es, hash s into Es and determine whether s is a terminal state or not. 
            self.Es[s] = self.game.getGameEnded(canonicalBoard, self.args, self.game_args) #game.getGameEnded returns either -sparsity + ||A_Sx-y||_2^2 or 0. Note that once getGameEnded is called, canonicalBoard.termreward is set.
        if self.Es[s]!=0: #Now that we have hashed s into self.Es, check whether the reward returned above is 0 or not. If not zero, return the computed terminal value from getGameEnded above and search stops.
            return self.Es[s]
        #----------------------------------------------------------------------
        
        #-------------IF s IS A LEAF, THEN:---------------------
        #1)Call neural network to return probability distribution outcome v
        #2)Using step one, initialize the self variables of s by saving numpy vector of valid moves and times visited.
        #3)return v.
        #4)self.Ps[s], self.Vs[s], and self.Ns[s] are all initialized.
        if s not in self.Ps: #If s is not a key in self.Ps(which hashes s to its next available game states, then s must be a leaf.
            # leaf node
            #Prepare canonicalBoard for feeding into NN
            
            #COMPUTE self.feature_dic and compute NN representation
            canonicalBoard.compute_x_S_and_res(self.args, self.game_args)
            canonicalBoard.converttoNNInput()
                        
            self.Ps[s], v = self.nnet.predict(canonicalBoard) #neural network takes in position s and returns a prediction(which is p_theta vector and v_theta (numpy vector). Look at own notes)
            valids = self.game.getValidMoves(canonicalBoard) #returns a numpy vector of 0 and 1's which indicate valid moves from the set of all actions
            self.Ps[s] = self.Ps[s]*valids      # masking(hiding) invalid moves(this inner product creates a vector of probabilities of valid moves)
            sum_Ps_s = np.sum(self.Ps[s])       # probabilities may not add up to 1 anymore after hiding invalid moves(since NN may predict nonzero prob for illegal moves). Hence, renormalize such that
                                                # valid actions sum up to 1.
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids #Since s is a leaf, store the newly found valid moves and set amount of times s was visited as zero.
            self.Ns[s] = 0
            return v
        #--------------------------------------------------------
        
        #-----------IF CURRENT NODE s IS NOT A LEAF, THEN CONTINUE SEARCH------------------------
        #1)search from a root to a leaf via UCB using recursive search
        #2)Once we arrive at a leaf, due to the recursive nature, we update self.Qsa and self.Nsa dictionaries in a bottom up fashion. 
        #3)Note that if s is not a leaf, it has been a leaf before, so self.Vs[s], self.Ps[s], self.Ns[s] are all well defined.
        
        valids = self.Vs[s] #retrieve numpy vector of valid moves
        cur_best = -float('inf') #temp variable which holds the current highest UCB value
        best_act = -1 #temp variable which holds the current best action with largest UCB. Initialized to -1.
        for a in range(self.game.getActionSize(self.args)): #iterate over all possible actions. 
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['cpuct']*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)]) #note here that self.Ns[s] is number of times s 
                    #was visited. Note that self.Ns[s] = sum over b of self.Nsa[(s,b)], so the equation above is equal to surag nair's notes.
                else:
                    u = self.args['cpuct']*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best: #because this is equivalent to taking the max, this is why our rewards are negative!!!!!
                    cur_best = u
                    best_act = a

        a = best_act #define action with highest UCB computed above as a. a is chosen over valids. Note that best_act is a single action we take when traversing a single depth of MCTS tree.
        #FOR TESTING---------------------------
        #print('The action with highest UCB is: ' + str(a)) 
        #--------------------------------------
        next_s = self.game.getNextState(canonicalBoard, a) #returns next board state(game object)
        #next_s = self.game.getCanonicalForm(next_s, next_player) #canonical form does not matter for one player games. This line may be unnecessary.

        v = self.search(next_s) #traverse from root to a leaf or terminal node using recursive search. 
        
        #because we recursively search in the line above, the below snippet updates self.Qsa and self.Nsa of all visited nodes from the bottom to the root.
        #Because we reach a leaf sooner or later, the middle section is executed and v below is well defined!
        #formulas below match what we have in notebook.
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else: #if (s,a) is not in dictionary self.Qsa, that means i(s,a) has never been visited before. IOW N(s,a) = 0. Hence, by the formula 3 lines above, self.Qsa[(s,a)] = v.
            self.Qsa[(s,a)] = v 
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v
        #-------------------------------------------------------------------------------------
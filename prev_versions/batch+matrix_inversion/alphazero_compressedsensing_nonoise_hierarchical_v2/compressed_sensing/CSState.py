import numpy as np
from math import exp #to calculate the transformed feature data

#CSState is an object which stores as much information as possible about the current state. 
#All the variables are initialized to None for memory sake when conducting MCTS search, as only action_indices
#is needed when conducting MCTS search

class State():
    def __init__(self, actions_indicator, col_indices = [], identifier = None): #default of col_indices is None if not specified
        #action_indicator initalizes variables which are used for MCTS search. All other variables are
        #initialized when Neural Network computes features
        self.action_indices = actions_indicator #numpy array of 0 and 1's indicating available actions. 
        self.identifier = identifier #identifies which MCTS tree state belongs to.
        self.keyRep = None #Assuming game.keyRepresentation is called, the key is saved here. Should be an integer. 
        self.col_indices = col_indices
        #feature dictionary
        self.feature_dic = {} #contains all relevant feature data used in NN. Use methods below to compute these
        #compute terminal reward, 0 if not a terminal state.
        self.termreward = None
        #labels(if there are any)
        pi_as_label = np.zeros(actions_indicator.size)
        #pi_as_label[-1] = 1
        self.pi_as = pi_as_label
        self.z = None #The computed label for state
        #NN_input format for prediction(dont need labels for states we wish to predict)
        self.nn_input = None
        self.inverse = None #stores (A^T * A)^-1 for this state wrt to columns chosen. We can compute this from previous state
        self.ATy = None #stores A^T * b. We can compute this from previous state
        
    def computecolStats(self): #O(n) operation, where n is the length of the list
        S = []
        for i in range(self.action_indices.size-1):
            if self.action_indices[i] == 1:
                S.append(i)
        
        self.col_indices = S
        
    def compute_x_S_and_res(self, args, Game_args): #compute feature vectors for feeding into NN. Labels are returned by computeTermReward. 
        if self.col_indices: #If self.col_indices is not an empty list(meaning that we are not at the start state in which we have not chosen any columns)
            #FEATURE 1:
            if args['x_l2'] == True:
                x_S = np.matmul(self.inverse, self.ATy)
                x_S = x_S.flatten() #flatten x_S shape from [|S|, 1) to (|S|,) for computations below

                opt_sol_l2 = np.zeros(args['n'])
                i = 0
                for k in self.col_indices:
                    opt_sol_l2[k] = x_S[i]
                    i += 1
            
                self.feature_dic['x_l2']=opt_sol_l2

                
            #FEATURE 2:
            if args['lambda'] == True: 
                residual = Game_args.obs_vector - np.matmul(Game_args.sensing_matrix[:, self.col_indices], x_S)
                
                col_res_IP = np.matmul(Game_args.sensing_matrix.transpose(), residual)
                self.feature_dic['col_res_IP'] = col_res_IP

        else: #If column indices is empty, this means we have not chosen any columns, so current l2 solution is 0, and col_res_IP is A^T*y
            if args['x_l2'] == True:
                self.feature_dic['x_l2'] = np.zeros(args['n'])
            if args['lambda'] == True:
                self.feature_dic['col_res_IP'] = np.matmul(Game_args.sensing_matrix.transpose(), Game_args.obs_vector)
        
    def computeTermReward(self, args, Game_args): 
    #determine whether terminal state conditions are met. If any of the terminal state conditions are met, return terminal value, which is negative
    #See Game.getGameEnded. 
    #1)Game.getGameEnded() is called in MCTS.search to verify if a state/node we are currently at in MCTS search is a terminal state or not.
    #2)Game.getGameEnded() is also called in Coach.executeEpisode, when self play is being conducted. For each game state we enter in self-play, we call Game.getGameEnded() to verify if we are at a terminal state or not.
    #  If we are at a terminal state, then we stop and convert all the states w'eve visited to training samples with the labels being the terminal rewards. 
    #3)If self.termreward = 0, then the state is NOT a terminal state. Only nonzero self.termrewards should be labels for training the neural network. 
        if self.col_indices: #If self.col_indices is not an empty list
            S = self.col_indices #note that when we compute the termreward for initial state, THIS WILL RETURN AN ERROR because col_indices of initial state is []   
            A_S = Game_args.sensing_matrix[:,S]
            x_S = np.matmul(self.inverse, self.ATy)
            #Note that product.shape = (7, 1)
            product = np.matmul(A_S, x_S)
            product = product.flatten()
            #Note that product.shape = (7, ) and Game_args.obs_vector.shape = (7, )
            residual = Game_args.obs_vector - product
            res_norm_squared = np.linalg.norm(residual)**2
            
            #if terminal state, compute the reward.
            if len(self.col_indices) == Game_args.game_iter or self.action_indices[-1] == 1 or res_norm_squared < args['epsilon']: #Game_args.game_iter is set every time we call Game_args.generateNewObsVec
                self.termreward = - args['alpha']*len(self.col_indices) - args['gamma']*res_norm_squared
               
            #ow, reward is 0 if state is not a terminal state
            else:
                self.termreward = 0 #not terminal state
                
        elif self.action_indices[-1] == 1: #If self.col_indices is an empty list, but stopping action was taken, then reward is exactly equal to the negative of squared norm of y * gamma
            self.termreward = -args['gamma']*np.linalg.norm(Game_args.obs_vector)**2
            
        else:#If self.col_indices is an empty list, and self.action_indices[-1] != 1, then this implies we are at initial state. Set self.termreward to zero.
            self.termreward = 0
            
    
    def converttoNNInput(self): 
    #convert data in features dictionary into format recognizable by NN for prediction. This method is used in MCTS search method, where we output the p_as and z for searching for the next node to go to and backpropagating the reward. 
    #features_dic MUST ALREADY BE COMPUTED    
        NN_input_X = []
        for key in self.feature_dic:
            feature_data = self.feature_dic[key]
            feature_data = np.reshape(feature_data, (1, feature_data.size)) #reshape to (1, feature_data.size) for single predictions. Must be of this form for model.predict
            NN_input_X.append(feature_data)
        
        self.nn_input = NN_input_X
        

        
        

        
    
          
  
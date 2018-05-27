import numpy as np

#CSState is an object which stores as much information as possible about the current state. 
#All the variables are initialized to None for memory sake when conducting MCTS search, as only action_indices
#is needed when conducting MCTS search

class State():
    def __init__(self, actions_indicator, col_indices = []): #default of col_indices is None if not specified
        #action_indicator initalizes variables which are used for MCTS search. All other variables are
        #initialized when Neural Network computes features
        self.action_indices = actions_indicator #numpy array of 0 and 1's indicating available actions. 
        self.col_indices = col_indices
        #feature dictionary
        self.feature_dic = {} #contains all relevant feature data used in NN. Use methods below to compute these
        #compute terminal reward, 0 if not a terminal state.
        self.termreward = None
        #labels(if there are any)
        self.pi_as = None
        self.z = None
        #NN_input format for prediction(dont need labels for states we wish to predict)
        self.nn_input = None
        
    def computecolStats(self): #O(n) operation, where n is the length of the list
        S = []
        for i in range(self.action_indices.size-1):
            if self.action_indices[i] == 1:
                S.append(i)
        
        self.col_indices = S
        
    def compute_x_S_and_res(self, args, Game_args): #compute feature vectors for feeding into NN
        if self.col_indices: #If self.col_indices is not an empty list(meaning that we are not at the start state in which we have not chosen any columns)
            #FEATURE 1:
            if args['x_l2'] == True:
                S = self.col_indices #Assume self.col_indices has already been computed from MCTS state creation.
                A_S = Game_args.sensing_matrix[:,S]
                x = np.linalg.lstsq(A_S, Game_args.obs_vector) #x[0] contains solution, x[1] contains the sum squared residuals, x[2] contains rank, x[3] contains singular values
                opt_sol_l2 = np.zeros(Game_args.sensing_matrix.shape[1])
                i = 0
                for k in S:
                    opt_sol_l2[k] = x[0][i]
                    i += 1
            
                self.feature_dic['x_l2']=opt_sol_l2
            
            else:
                print('selected feature set to false in args')
            #FEATURE 2:
            if args['lambda'] == True: 
                if not x[1]: #If the residual x[1] is an empty list [], then A_Sx - y is solved exactly.
                #WE ASSUME A HAS FULL RANK. Hence, in this case, the residual is a np vector of zeros
                    col_res_IP = np.zeros(Game_args.sensing_matrix[1])
                else:
                    residual_vec = Game_args.obs_vector - np.matmul(A_S, x[0])
                    col_res_IP = np.matmul(Game_args.sensing_matrix.transpose(), residual_vec)
                    col_res_IP = np.absolute(col_res_IP)
        
                self.feature_dic['col_res_IP'] = col_res_IP
            else:               
                print('selected feature set to false in args')
        else: #If column indices is empty, this means we have not chosen any columns
            self.feature_dic['x_l2'] = np.zeros(Game_args.sensing_matrix.shape[1])
            self.feature_dic['col_res_IP'] = np.matmul(Game_args.sensing_matrix.transpose(), Game_args.obs_vector)
        
    def computeTermReward(self, args, Game_args): #determine whether terminal state conditions are met. If any of the terminal state conditions are met, return terminal value, which is negative
        if self.col_indices: #If self.col_indices is not an empty list
            S = self.col_indices #note that when we compute the termreward for initial state, THIS WILL RETURN AN ERROR because col_indices of initial state is []   
            A_S = Game_args.sensing_matrix[:,S]
            x = np.linalg.lstsq(A_S, Game_args.obs_vector)
        
            if not x[1]: #Case in which residual returns an empty size 1 array, which implies state.num_col = number of rows of sensing matrix
                self.termreward = - len(self.col_indices)
            elif len(self.col_indices) == Game_args.game_iter or self.action_indices[-1] == 1 or x[1] < args['epsilon']:
                self.termreward = - len(self.col_indices) - args['gamma']*x[1]
            else:
                self.termreward = 0 #not terminal state
        else: #If self.col_indices is an empty list, that means we have not chosen any columns, so we are definitely not at a terminal node. Set self.termreward to 0.
            self.termreward = 0
            
    
    def converttoNNInput(self): #convert data in features dictionary into format recognizable by NN for prediction. This method is used in MCTS search method, where we output the p_as and z for searching for the next node to go to and backpropagating the reward. 
        
        NN_input_X = []
        for key in self.feature_dic:
            feature_data = self.feature_dic[key]
            feature_data = np.reshape(feature_data, (1, feature_data.size)) #reshape to (1, feature_data.size) for single predictions. Must be of this form for model.predict
            NN_input_X.append(feature_data)
        
        self.nn_input = NN_input_X
        

        
        

        
    
          
  
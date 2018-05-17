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
    	
    def compute_x_S_and_res(self, Game_args): 
    	#Computes 2 feature vectors related to l2 minimization
    	#Game_args is an object which contains information about A and y. 
    	#FEATURE VECTOR 1: solution to argmin_x ||y - A_Sx||_2^2, expanded to a n-dim vector opt_sol_l2 st 
		#A*opt_sol_12 = A_S*x. (n is column dimension)
    	S = self.col_indices #Assume self.col_indices has already been computed from above
    	A_S = Game_args.sensing_matrix[:,S]
    	x = np.linalg.lstsq(A_S, Game_args.obs_vector) #x[0] contains solution, x[1] contains the sum squared residuals, x[2] contains rank, x[3] contains singular values
    	opt_sol_l2 = np.zeros(Game_args.sensing_matrix.shape[1])
    	i = 0
    	for k in S:
    		opt_sol_l2[k] = x[0][i]
    		i += 1
    		
    	self.feature_dic['x_l2']=opt_sol_l2
    	
    	#FEATURE VECTOR 2: vector of inner product of columns of sensing matrix with
    	#residual vector y-A_S*x. Call this vector lambda. (AKA lambda = A^T(y-A_S*x))
    	if not x[1]: #If the residual x[1] is an empty list [], then A_Sx - y is solved exactly.
    	#WE ASSUME A HAS FULL RANK. Hence, in this case, the residual is a np vector of zeros
    		col_res_IP = np.zeros(Game_args.sensing_matrix[1])
    	else:
    		residual_vec = Game_args.obs_vector - np.matmul(A_S, x[0])
    		col_res_IP = np.matmul(Game_args.sensing_matrix.transpose(), residual_vec)
    		col_res_IP = np.absolute(col_res_IP)
    	
    	self.feature_dic['col_res_IP'] = col_res_IP
    	
    def computeTermReward(self, Game_args):
    
    	S = self.col_indices
    	A_S = Game_args.sensing_matrix[:,S]
    	x = np.linalg.lstsq(A_S, Game_args.obs_vector)
    	epsilon = 1e-5
    	gamma = 1
    	
    	if not x[1]: #Case in which residual returns an empty size 1 array, which implies state.num_col = number of rows of sensing matrix
    		self.termreward = - len(self.col_indices)
    	elif len(self.col_indices) == Game_args.game_iter or self.action_indices[-1] == 1 or x[1] < epsilon:
    		self.termreward = - len(self.col_indices) - gamma*x[1]
    	else:
    		self.termreward = 0 #not terminal state
    
    def converttoNNInput(self): #convert data in features dictionary into format recognizable by NN for prediction
    	
    	NN_input_X = []
    	for key in self.feature_dic:
    		self.feature_dic[key] = feature_data
    		NN_input_X.append(feature_data)
    	
    	self.nn_input = NN_input_X
    	
def ConstructTraining(states): 
#INPUT:states is a list of state objects
#
#OUTPUT:(X,Y) training data saved into a file. X is a list of np.arrays, where each array corresponds to
#data from a single feature in the feature dictionary, and the number of rows in each array equals the number 
#of states/training_samples. Y is a length 2 list containing labels, where Y[0] is a numpy array where each row is pi_as,
#and Y[1] is a numpy vector which contains the terminal reward. Again the number of rows equals 
#the number of states in the input list 
#
#FUNCTION:For a given list of states, output (X,Y) into a file. May need to do this incrementally, as (X,Y)
#may not fit into RAM all at once. 

	pass
    	
		
		

    	
    
  	      
  
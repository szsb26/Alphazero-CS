from CS_NNet import NetArch
import numpy as np
from keras.models import Model, model_from_json

class NNetWrapper(): 
	def __init__(self, args, game): #remember that game object stores the matrix A and y
		self.args = args
		self.nnet = NetArch(args, game) #self.nnet is a NetArch object
		
	def statetoFeatures(self, state, game):
	#INPUT: a state object(used in MCTS) and a game instance(which includes the information about A and y)
	#OUTPUT: a single training input as a list; where each element in the list is a numpy array corresponding to a single feature
	#NOTES: transforms a single state object into input recognizable by Neural Network. Labels are not determined here
	#since the labels (pi_as and z) are ultimately decided by MCTS. The code here is CRUCIAL
	#for Neural Network to learn well. Unlike Go, the features here will need to be handcrafted for
	#the neural network to learn well. 
		
		#FEATURE VECTOR 1: solution to argmin_x ||y - A_Sx||_2^2, expanded to a n-dim vector opt_sol_l2 st 
		#A*opt_sol_12 = A_S*x. (n is column dimension)
		S = state.getcolIndices()
		A_S = game.sensing_matrix[:,S]
		x = np.linalg.lstsq(A_S, game.obs_vector) #x is a list of size 4, x[0] contains solution, x[1] is sum of residual value in l2, x[2] = rank(A_S), x[3] = singular values
		opt_sol_l2 = np.zeros(game.sensing_matrix.shape[1])
		i = 0
		for k in S:
			opt_sol_l2[k] = x[0][i]
			i += 1
			
		#FEATURE VECTOR 2: vector of inner product of columns of sensing matrix with 
		#residual vector y-A_S*x. Call this vector lambda. (AKA lambda = A^T(y-A_S*x))
		if not x[1]: #If the residual x[1] is an empty list [], then A_Sx - y is solved exactly. 
		#WE ASSUME A HAS FULL RANK. Hence, in this case, the residual is a np vector of zeros
			col_res_IP = np.zeros(game.sensing_matrix[1])
		else:
			residual_vec = game.obs_vector - np.matmul(A_S, x[0])
			col_res_IP = np.matmul(game.sensing_matrix.transpose(), residual_vec)
			col_res_IP = np.absolute(col_res_IP) #take absolute value entrywise
		
		#CONCATENATE ALL FEATURE VECTORS TOGETHER INTO A LIST. THIS IS A SINGLE TRAINING EXAMPLE. 
		training_sample = [opt_sol_l2, col_res_IP] 
		
		return training_sample
				
	def convertStates(self, states, game):
	#INPUT: a list of state objects (no labels here)
	#OUTPUT: a list of 2D arrays, where the row indices of each array indicate the number of training samples,
	#		and the columns of each array correspond to the values which describe a single feature. Directly fed into NN
	
		#Initialize for loop	
		state = states.pop(0)
		X = self.statetoFeatures(state, game)
		#Stack each training sample(training sample is a list containing feature vectors as its elements)
		for state in states:
			conv_state = self.statetoFeatures(state, game) #convert every state in states into a training sample recognizable by NN
			for i in range(len(X)): #numpy array: 
				X[i] = np.stack((X[i], conv_state[i]))	
			
		return X	
	
	def convertLabels(self, labels):
	#INPUT:
	#OUTPUT: list of labels, directly fed into NN
		
		Y = 0
		
		return Y
	
	def train(self, X, Y): #Take in the final training and labels
	#INPUT: training and labels X,Y respectively
	#OUTPUT: None
		
		self.nnet.model.fit(X,Y, epochs = self.args['epochs'], batch_size = self.args['batch_size'])
		
	def predict(self, conv_state): 
	#INPUT: Converted state(using statetoFeatures above)
	#OUTPUT: p_as and v

		p_as, v = self.nnet.model.predict(conv_state)
		return p_as, v
	
	def save_checkpoint(self, folder, filename):
	#INPUT: folder and filename 
	#OUTPUT: None
	#FUNCTION: save the current model and its weights in some folder
		self.nnet.model.save_weights(folder + filename + '_weights.h5')
		model_json = self.nnet.model.to_json()
		with open(folder + filename + '_model.json', 'w') as json_file:
			json_file.write(model_json)
			
	def load_checkpoint(self, folder, filename):
	#INPUT: folder and filename
	#OUTPUT: load a model and its weights with given folder and filename into self.nnet
		#Load the model
		json_file = open(folder + filename + '_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.nnet.model = model_from_json(loaded_model_json)
		#Load the weights
		self.nnet.model.load_weights(folder + filename + '_weights.h5')
		

from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras import regularizers


class NetArch():
	def __init__(self, args, Game_args):
	#INPUTS:
	#args(args is from main.py) and current game
	#OUTPUT:
	#the Neural Network architecture model. Look to pg 78 in notes for a pictorial representation of NN architecture.
		
		self.args = args
		self.output_dim_pas = Game_args.sensing_matrix[1]+1
	
		hidden_neurons = self.args['neurons_per_layer']
		#Create the input and first hidden layer based on which features are initialized in args
		Inputs = []
		HL1 = []
		#Depending on which features are given in args, create the input size of NN
		if self.args['x_l2'] == True:
			xl2_input1 = Input(shape = (Game_args.sensing_matrix.shape[1],)) #corresponds to a combination of feature a and b (pg 78)
			Inputs.append(xl2_input1)
		if self.args['lambda'] == True:
			colresIP_input2 = Input(shape = (Game_args.sensing_matrix.shape[1],)) #corresponds to feature b1 in notes  (pg 78)
			Inputs.append(colresIP_input2)
		
		
		#Build first layer (not fully connected) by iteratively building sets of neurons corresponding to an input feature. 
		#Concatenate all of these neurons to form the first hidden layer x. 
		for input in Inputs:
			 HL1_set = Dense(hidden_neurons, activation = 'relu')(input)
			 HL1 = HL1.append(HL1_set)
		
		x = concatenate(HL1, axis = -1)
	
		#Define the subsequent fully connected hidden layers depending on self.args['num_layers']
		for i in range(self.args['num_layers']-1):
			x = Dense(hidden_neurons, activation = 'relu')(x)
			
		#Define the two outputs, where activation is softmax and identity, since reward in mcts is
		#sparsity + l2 regularization term.
		self.p_as = Dense(self.output_dim_pas, activation = 'softmax', name = 'p_as')(x) 
		self.v = Dense(1, name = 'v')(x)
		
		self.model = Model(inputs = Inputs, outputs = [self.p_as, self.v])
		self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(self.args['lr']))


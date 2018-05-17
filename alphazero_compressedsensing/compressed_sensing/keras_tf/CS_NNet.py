from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras import regularizers


class NetArch():
	def __init__(self, args, game):
	#INPUTS:
	#args used to determine hyperparameters of neural network (args from NNet.py)
	#game object used to determine the output dimension(number of actions for softmax output (game is an object from CSGame.py)
	#NEED TO CHANGE THIS IN THE FUTURE TO ACCOMODATE LARGER NUMBER OF FEATURES WITHOUT HARDCODING
	#OUTPUT:
	#the Neural Network architecture model. Look to pg 78 in notes for a pictorial representation of NN architecture.
		
		self.args = args
		self.output_dim_pas = game.getActionSize()
		
		hidden_neurons = self.args['neurons_per_layer']
		xl2_input1 = Input(shape = (game.sensing_matrix.shape[1],)) #corresponds to a combination of feature a and b (pg 78)
		colresIP_input2 = Input(shape = (game.sensing_matrix.shape[1],)) #corresponds to feature b1 in notes  (pg 78)
		#Initialize construction of NN by building the first hidden layer and concatenating the output of first hidden layer
		HL1_xL2 = Dense(hidden_neurons, activation = 'relu')(xl2_input1)
		HL1_colresIP = Dense(hidden_neurons, activation = 'relu')(colresIP_input2)
		x = concatenate([HL1_xL2, HL1_colresIP], axis = -1)
		#Define the hidden layers
		for i in range(self.args['num_layers']-1):
			x = Dense(hidden_neurons, activation = 'relu')(x)
		#Define the two outputs, where activation is softmax and identity, since reward in mcts is
		#sparsity + l2 regularization term.
		self.p_as = Dense(self.output_dim_pas, activation = 'softmax', name = 'p_as')(x) 
		self.v = Dense(1, name = 'v')(x)
		
		self.model = Model(inputs = [xl2_input1, colresIP_input2], outputs = [self.p_as, self.v])
		self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(self.args['lr']))



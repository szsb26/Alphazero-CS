from CS_NNet import NetArch
import numpy as np
from keras.models import Model, model_from_json

class NNetWrapper(): 
	#Game_args is an object, while args is just a dictionary found in main.py. When we initialize a 
	#NNetWrapper object, we specify these two args from main.py and Game_Args.py
	def __init__(self, args, Game_args): 
		self.args = args
		self.nnet = NetArch(args, Game_args) #self.nnet is a NetArch object
				
	def train(self, X, Y): #Take in the final training and labels
	#INPUT: training and labels X,Y respectively
	#OUTPUT: None
		
		self.nnet.model.fit(X,Y, epochs = self.args['epochs'], batch_size = self.args['batch_size'])
		
	def predict(self, state): 
	#INPUT: state object. Note that state.col_indices, state.feature_dic and state.converttoNNInput 
	#must all be computed before NNetWrapper.predict can make a meaningful prediction. 
	#OUTPUT: p_as and v
		
		p_as, v = self.nnet.model.predict(state.nn_input)	
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
		

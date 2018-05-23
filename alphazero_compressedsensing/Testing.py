import sys
#Add Class files to the sys search path for importing later on
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing/keras_tf')
from CSGame import CSGame
from CSState import State
from Game_Args import Game_args
from NNet import NNetWrapper
from Coach import Coach
import numpy as np

args = {
	#Compressed Sensing Parameters, Ax = y, where A is of size m by n
	'matrix_type': 'sdnormal',	#type of random matrix
	'x_type': 'sdnormal',
	'm': 5,	#row dimension of A
	'n':15,	#column dimension of A
	'sparsity':2,
	#---------------------------------------------------------------
	#General Alphazero Parameters
	'training_samples': 100000, #dictates how many training_samples are generated per iteration of alphazero algorithm
	'save_into_csv_batch': 1000, #dictates how many training pairs we save at a time into csv file in case of memory overflow
	'numIters': 1000, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
	#numEps and maxlenOfQueue controls the size of iterationTrainExamples, which contains all the states(up to maxlenOfQueue) of numEps number of generated self-play games.
	'numEps': 100, #dictates how many self play games are played each iteration of the algorithm
	'maxlenOfQueue':500, #dictates total number of game states saved(NOT games). 
	'numItersForTrainExamplesHistory': 2, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
	'checkpoint': '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/training_data',
	'load_folder_(folder)': '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/training_data',
	'load_folder_(filename)': 'best.pth.tar',
	'arenaCompare': 40, #number of games played to compare pmcts and nmcts
	'updateThreshold': 0.6, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
	'load_model': False, 
	#---------------------------------------------------------------
	#NN Parameters
	'lr': 0.001,
    'num_layers': 2,
    'neurons_per_layer':100,
    'epochs': 10,
    'batch_size': 64, #dictates the batch_size when training 
    'num_channels': 512,
    'num_features' : 2,
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,		#solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,	#the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 1, 
    'numMCTSSims': 25,
    'tempThreshold': 15,	#dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
}

#Only if "python main.py" is executed in command line will the below execute
if __name__=='__main__':

	#Initialize game, nnet, args, and Game_args
	#Initialize Game_args
	Game_args = Game_args()
	Game = CSGame()
	nnet = NNetWrapper(args)
	
	#if 'load_model' = True, then load the current best model with its weights. Currently set to false because we havent
	#trained anything yet
	if args['load_model']:
		nnet.load_checkpoint(args['load_folder_(folder)'], args['load_folder_(filename)'])

	c = Coach(Game, nnet, args, Game_args)
	if args['load_model']:
		print("Load trainExamples from file")
		c.loadTrainExamples()
	c.learn() #primary command which starts alphazero!!!!


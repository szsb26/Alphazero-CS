import sys
#add arguments of algorithm into system path so that we can import them
sys.path.insert(0, '/Users/sichenzhong/Desktop/Sichen/Graduate School/ML/NN_MCTS_CS/python src/alphazero_compressedsensing')
sys.path.insert(0, '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing')

#args dictionary which dictates behavior of NN, and MCTS
args = {
	#---------------------------------------------------------------
	#General Alphazero Parameters
	'training_samples': 100000, #dictates how many training_samples are generated per iteration of alphazero algorithm
	'save_into_csv_batch': 1000, #dictates how many training pairs we save at a time into csv file in case of memory overflow
	'numIters': 1000, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
	'numEps': 100, #dictates how many games are played each iteration of the algorithm
	'maxlenOfQueue':500,
	#---------------------------------------------------------------
	#NN Parameters
    'lr': 0.001,
    'num_layers': 2,
    'neurons_per_layer':100,
    'epochs': 10,
    'batch_size': 64, #dictates the batch_size when training 
    'num_channels': 512,
    'num_features' : 2,
    'x_l2' : True,		#solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,	#the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 1, 
    'numMCTSSims': 25,
    'tempThreshold': 15,	#dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
}

#Game_args is an object, which contains parameters for the underlying sensing matrix A and y.
#Game_args is its own object because we may need to generate different A or y for different games.
#Hence, these functions are contained in Game_args object as class methods. 
Game_args = Game_args()
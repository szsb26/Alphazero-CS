import sys
#add arguments of algorithm into system path so that we can import them
sys.path.insert(0, '/Users/sichenzhong/Desktop/Sichen/Graduate School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing')
from Coach import Coach
from MCTS import MCTS
sys.path.insert(0, '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing')
from CSGame import CSGame
from CSState import State
from Game_Args import Game_args
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing/keras_tf')
from NNet import NNetWrapper


#args dictionary which dictates behavior of NN, and MCTS, and Alphazero.
args = {
    #Compressed Sensing Parameters, Ax = y, where A is of size m by n
    'matrix_type': 'sdnormal',  #type of random matrix
    'x_type': 'sdnormal',
    'm': 5, #row dimension of A
    'n':15, #column dimension of A
    'sparsity':5, #dictates the maximum sparsity of x when generating the random vector x. Largest number of nonzeros of x is sparsity-1. sparsity cannot be greater than m above. 
    #---------------------------------------------------------------
    #General Alphazero Parameters
    'numIters': 3, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
    'numEps': 100, #dictates how many games are played each iteration of the algorithm
    'maxlenOfQueue':10000, #dictates total number of game states saved(not games). 
    'numItersForTrainExamplesHistory': 2, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
    'checkpoint': '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/training_data', #filepath for SAVING newly generated self play training data
    'load_training': False, #If set to True, then load latest batch of self play games for training. 
    'load_folder_(folder)': '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/training_data', #filepath for LOADING the latest set of training data
    'load_folder_(filename)': 'best.pth.tar', #filename for LOADING the latest generated set of training data
    'arenaCompare': 40, #number of games played in the arena to compare 2 networks pmcts and nmcts
    'updateThreshold': 0.55, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
    #---------------------------------------------------------------
    #NN Parameters
    'lr': 0.001,    #learning rate of NN
    'num_layers': 2,    #number of hidden layers after the 1st hidden layer
    'neurons_per_layer':100,    #number of neurons per hidden layer
    'epochs': 10,   #number of training epochs
    'batch_size': 64, #dictates the batch_size when training 
    'num_features' : 2, #number of self-designed features used in the input
    'load_nn_model' : False, #If set to True, load the best network (best.json and best.h5)
    'network_checkpoint' : '/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/network_checkpoint', #filepath for SAVING the temp neural network model/weights, checkpoint networks model/weights, and the best networks model/weights
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,      #solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,    #the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 1, 
    'numMCTSSims': 25,
    'tempThreshold': 15,    #dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
    'gamma': 1, #note that reward for a terminal state is -||x||_0 - gamma*||A_S*x-y||_2^2. 
    'epsilon': 1e-5, #If x is the optimal solution to l2, and the residual of l2 regression ||A_Sx-y||_2^2 is less than epsilon, then the state corresponding to indices S is a terminal state in MCTS. 
}

#START ALPHAZERO:
#Initialize Game_args, nnet, Game, and Alphazero
Game_args = Game_args()
nnet = NNetWrapper(args)
if args['load_nn_model'] == True:
    filename = 'best'
    nnet.load_checkpoint(args['network_checkpoint'], filename)
Game = CSGame()
Alphazero = Coach(Game, nnet, args, Game_args)
if args['load_training'] == True:
    print('Load trainExamples from file')
    Alphazero.loadTrainExamples()
#Start Training Alphazero
Alphazero.learn()

    

import sys, os
print(sys.version)
#FOR TESTING---------------------------
import numpy as np
np.set_printoptions(threshold=np.nan)
#END TESTING---------------------------

#add arguments of algorithm into system path so that we can import them
sys.path.insert(0, os.getcwd())
from Coach import Coach
from MCTS import MCTS
sys.path.insert(0, os.getcwd() + '/compressed_sensing')
from CSGame import CSGame
from CSState import State
from Game_Args import Game_args
sys.path.insert(0, os.getcwd() + '/compressed_sensing/keras_tf')
from NNet import NNetWrapper


#args dictionary which dictates behavior of NN, and MCTS, and Alphazero.
args = {
    #Compressed Sensing Parameters, Ax = y, where A is of size m by n
    'fixed_matrix': True, #fixes a single matrix across entire alphazero algorithm. If set to False, then self play games generated in each iteration have different sensing matrices. The below options will not run if this is set to False.
        'load_existing_matrix': True, #If we are using a fixed_matrix, then this option toggles whether to load an existing matrix from args['fixed_matrix_filepath'] or generate a new one. If loading an existing matrix, the matrix must be saved as name 'sensing_matrix.npy'
            'matrix_type': 'sdnormal',  #type of random matrix generated if(assuming we are not loading existing matrix)
    'x_type': 'uniform01',  #type of entries generated for sparse vector x when playing games of self-play
    'm': 25, #row dimension of A
    'n': 250, #column dimension of A
    'sparsity':12, #dictates the maximum sparsity of x when generating the random vector x. Largest number of nonzeros of x is sparsity-1. sparsity cannot be greater than m above. 
    'fixed_matrix_filepath': os.getcwd() + '/fixed_sensing_matrix', #If args['fixed_matrix'] above is set to True, then this parameter determines where the fixed sensing matrix is saved or where the existing matrix is loaded from. 
    #---------------------------------------------------------------
    #General Alphazero Training Parameters
    'numIters': 100, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
    'numEps': 200, #dictates how many self play games are played each iteration of the algorithm
    'maxlenOfQueue':10000, #dictates total number of game states saved(not games). 
    'numItersForTrainExamplesHistory': 1, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
    'checkpoint': os.getcwd() + '/training_data', #filepath for SAVING newly generated self play training data
    'load_training': False, #If set to True, then load latest batch of self play games for training. 
        'load_folder_(folder)': os.getcwd() + '/training_data', #filepath for LOADING the latest set of training data
        'load_folder_(filename)': 'best.pth.tar', #filename for LOADING the latest generated set of training data. Currently, this must be saved as 'best.pth.tar'
    'Arena': False, #determines whether model selection/arena is activated or not. Below options will not be run if this is set to False.
        'arenaCompare': 100, #number of games played in the arena to compare 2 networks pmcts and nmcts
        'updateThreshold': 0.55, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
    #---------------------------------------------------------------
    #NN Parameters
    'lr': 0.001,    #learning rate of NN, relevant for NetArch(), NetArch1()
    'num_layers': 2,    #number of hidden layers after the 1st hidden layer, only relevant for NetArch()
    'neurons_per_layer':500,    #number of neurons per hidden layer
    'epochs': 10,   #number of training epochs. If There are K self play states, then epochs is roughly K/batch_size. Note further that K <= numEps*sparsity. epochs determines the number of times weights are updated.
    'batch_size': 200, #dictates the batch_size when training 
    'num_features' : 2, #number of self-designed features used in the input
    'load_nn_model' : True, #If set to True, load the best network (best_model.json and best_weights.h5)
    'network_checkpoint' : os.getcwd() + '/network_checkpoint', #filepath for SAVING the temp neural network model/weights, checkpoint networks model/weights, and the best networks model/weights
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,      #solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,    #the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 2, #controls the amount of exploration at each depth of MCTS tree.
    'numMCTSSims': 1000, #For each move, numMCTSSims is equal to the number of MCTS simulations in finding the next move during self play. Smallest value of numMCTSsims is 2.
    'tempThreshold': 12,    #dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
    'gamma': 1, #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller gamma is, the more likely algorithm is going to choose stopping action earlier(when ||x||_0 is small). gamma enforces how much we want to enforce Ax is close to y. 
                #choice of gamma is heavily dependent on the distribution of our signal and the distribution of entries of A. gamma should be apx. bigger than m/||A_Sx^* - y||_2^2, where y = Ax, and x^* is the solution to the l2 regression problem.
    'alpha':1e-5,  #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller alpha is, the more weight the algorithm gives in selecting a sparse solution.
    'epsilon': 1e-5, #If x is the optimal solution to l2, and the residual of l2 regression ||A_Sx-y||_2^2 is less than epsilon, then the state corresponding to indices S is a terminal state in MCTS. 
}

#START ALPHAZERO TRAINING:
#Initialize Game_args, nnet, Game, and Alphazero
Game_args = Game_args()
nnet = NNetWrapper(args)
if args['load_nn_model'] == True:
    filename = 'best'
    nnet.load_checkpoint(args['network_checkpoint'], filename)
Game = CSGame()
Alphazero_train = Coach(Game, nnet, args, Game_args)

#FOR TESTING-----------------------------------------------------
weights = Alphazero_train.nnet.nnet.model.get_weights()
min_max = []
for layer_weights in weights:
    print('number of weights in current array in list (output as matrix size): ', layer_weights.shape)
    layer_weights_min = np.amin(layer_weights)
    layer_weights_max = np.amax(layer_weights)
    min_max.append([layer_weights_min, layer_weights_max])
print('')
print('The smallest and largest weights of each layer are: ')
for pair in min_max:
    print(pair)
    print('')
#END TESTING-----------------------------------------------------                      

if args['load_training'] == True:
    print('Load trainExamples from file')
    Alphazero_train.loadTrainExamples()
#Start Training Alphazero
Alphazero_train.learn()

    

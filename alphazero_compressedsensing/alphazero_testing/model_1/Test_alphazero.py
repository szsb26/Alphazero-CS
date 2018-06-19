#Load a trained neural network and MCTS object for prediction.
#This module tests the predictive capabilites of a trained neural network together with MCTS. 
#bootstrapped NN with MCTS should do better than both 1)bootstrapped NN and 2)OMP
import os
import sys
import csv
import numpy as np
from sklearn.linear_model import orthogonal_mp
import matplotlib
import matplotlib.pyplot as plt
two_up = two_up =  os.path.abspath(os.path.join(os.getcwd(),"../.."))
sys.path.insert(0, two_up)
from Coach import Coach
from MCTS import MCTS
sys.path.insert(0, two_up + '/compressed_sensing')
from CSGame import CSGame
from Game_Args import Game_args
sys.path.insert(0, two_up + '/compressed_sensing/keras_tf')
from NNet import NNetWrapper

args = {
    #Compressed Sensing Parameters, Ax = y, where A is of size m by n
    'fixed_matrix': True, #fixes a single matrix across entire alphazero algorithm. If set to True, then self play games generated in each iteration have different sensing matrices. The below options will not run if this is set to False.
        'load_existing_matrix': True, #If we are using a fixed_matrix, then this option toggles whether to load an existing matrix from args['fixed_matrix_filepath'] or generate a new one. If loading an existing matrix, the matrix must be saved as name 'sensing_matrix.npy'
            'matrix_type': 'sdnormal',  #type of random matrix generated if(assuming we are not loading existing matrix)
    'x_type': 'uniform01',  #type of entries generated for sparse vector x
    'm': 7, #row dimension of A
    'n':15, #column dimension of A
    'sparsity':7, #dictates the maximum sparsity of x when generating the random vector x. Largest number of nonzeros of x is sparsity-1. sparsity cannot be greater than m above. 
    'fixed_matrix_filepath': os.getcwd() + '/fixed_sensing_matrix', #If args['fixed_matrix'] above is set to True, then this parameter determines where the fixed sensing matrix is saved or where the existing matrix is loaded from. 
    #---------------------------------------------------------------
    #General Alphazero Training Parameters
    'numIters': 100, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
    'numEps': 10000, #dictates how many self play games are played each iteration of the algorithm
    'maxlenOfQueue':10000, #dictates total number of game states saved(not games). 
    'numItersForTrainExamplesHistory': 3, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
    'checkpoint': os.getcwd() + '/training_data', #filepath for SAVING newly generated self play training data
    'load_training': False, #If set to True, then load latest batch of self play games for training. 
        'load_folder_(folder)': os.getcwd() + '/training_data', #filepath for LOADING the latest set of training data
        'load_folder_(filename)': 'best.pth.tar', #filename for LOADING the latest generated set of training data. Currently, this must be saved as 'best.pth.tar'
    'Arena': False, #determines whether model selection/arena is activated or not. Below options will not be run if this is set to False.
        'arenaCompare': 100, #number of games played in the arena to compare 2 networks pmcts and nmcts
        'updateThreshold': 0.55, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
    #---------------------------------------------------------------
    #NN Parameters
    'lr': 0.001,    #learning rate of NN
    'num_layers': 2,    #number of hidden layers after the 1st hidden layer
    'neurons_per_layer':200,    #number of neurons per hidden layer
    'epochs': 10,   #number of training epochs. If There are K self play states, then epochs is roughly K/batch_size. Note further that K <= numEps*sparsity. epochs determines the number of times weights are updated.
    'batch_size': 50000, #dictates the batch_size when training 
    'num_features' : 2, #number of self-designed features used in the input
    'load_nn_model' : True, #If set to True, load the best network (best_model.json and best_weights.h5)
    'network_checkpoint' : os.getcwd() + '/network_checkpoint', #filepath for SAVING the temp neural network model/weights, checkpoint networks model/weights, and the best networks model/weights
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,      #solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,    #the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 1, #controls the amount of exploration at each depth of MCTS tree.
    'numMCTSSims': 500, #For each move, numMCTSSims is equal to the number of MCTS simulations in finding the next move during self play. 
    'tempThreshold': 5,    #dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
    'gamma': 1, #note that reward for a terminal state is -||x||_0 - gamma*||A_S*x-y||_2^2. The smaller gamma is, the more likely algorithm is going to choose stopping action earlier(when ||x||_0 is small). gamma enforces how much we want to enforce Ax is close to y. 
    'epsilon': 1e-5, #If x is the optimal solution to l2, and the residual of l2 regression ||A_Sx-y||_2^2 is less than epsilon, then the state corresponding to indices S is a terminal state in MCTS. 
}


#INITIALIZE ALPHAZERO FOR PREDICTION
#--------------------------------------------------------------
#initialize Game_args
#load sensing_matrix into game_args
game_args = Game_args()
matrix_filename = 'sdnormal0.npy'
A = np.load(matrix_filename)
game_args.sensing_matrix = A
#initialize neural network wrapper object
#load weights and model we wish to predict with using nnet.load_checkpoint
nnet = NNetWrapper(args)
model_filename = 'OMPbootstrapNN'
nnet.load_checkpoint(os.getcwd(), model_filename)
#initialize a new game object
new_game = CSGame()
#Initialize Alphazero
Alphazero = Coach(new_game, nnet, args, game_args)
#---------------------------------------------------------------
#Alphazero IS NOW READY FOR PREDICTION
#Initialize accuracy vector
alphazero_accuracy = np.zeros(args['m'])
OMP_accuracy = np.zeros(args['m'])
#For every signal in testData, initiate a single game of self play via Coach.executeEpisode

for s in range(1, args['sparsity']-5):
    obsy_filepath = 'testData/' + str(s) + 'sparse' + '/' + str(s) + 'sparse_obsy.csv'
    sparsex_filepath = 'testData/' + str(s) + 'sparse' + '/' + str(s) + 'sparse_x.csv'
    with open(obsy_filepath) as obsy, open(sparsex_filepath) as sparsex:
        reader_y = csv.reader(obsy)
        reader_x = csv.reader(sparsex)
        counter_s = 0 #accuracy counter for alphazero for a fixed s
        counter_OMP_s = 0 #accuracy counter for OMP for a fixed s
        num_samples_s = 0
        for y, x in zip(reader_y, reader_x):
            y = list(map(float,y))
            y = np.asarray(y)
            x = list(map(float, x))
            x = np.asarray(x)
            #Initialize game_args and MCTS object(MCTS needs to be reinitialized everytime a new (y,x) pair is generated)
            Alphazero.game_args.obs_vector = y
            Alphazero.game_args.sparse_vector = x
            Alphazero.mcts = MCTS(Alphazero.game, Alphazero.nnet, Alphazero.args, Alphazero.game_args)
            #trainExamples is a list of states visited in the course of a game. Each state has well defined state.pi_as, state.z, and state.feature_dic
            trainExamples = Alphazero.executeEpisode() #MCTS.getActionProb is called here
            #compute the final predicted signal by looking at the last state
            last_state = trainExamples[-1]
            print('')
            print('(y,x) ITER ' + str(num_samples_s) + ':' + '///////////////////////////////////////////////')
            state_index = 0
            for state in trainExamples:
                print('')
                print('STATE ' + str(state_index) + ' statistics:' + '----------------------------------')
                print('action_indices: ' + str(state.action_indices))
                print('columns_taken: ' + str(state.col_indices)) 
                print('feature_dic: ')
                print(str(state.feature_dic))
                print('z: ' + str(state.z))
                print('pi_as(prediction of MCTS, NOT NN): ' + str(state.pi_as))
                state_index += 1
            print('/////////////////////////////////////////////////////////////')
            print('')
            predicted_x = last_state.feature_dic['x_l2']
            predicted_OMPx = orthogonal_mp(A, y, n_nonzero_coefs = s)
            
            error_alphazero = np.linalg.norm(Alphazero.game_args.sparse_vector - predicted_x)**2
            error_OMP = np.linalg.norm(Alphazero.game_args.sparse_vector - predicted_OMPx)**2
            
            if error_alphazero < 1e-03:
                counter_s += 1
            
            if error_OMP < 1e-03:
                counter_OMP_s += 1
                
            num_samples_s += 1
        
    #For fixed s, compute the recovery accuracy    
    alphazero_fixeds_recacc = counter_s/num_samples_s
    OMP_fixeds_recacc = counter_OMP_s/num_samples_s
    #Update alphazero_accuracy vector
    alphazero_accuracy[s] = alphazero_fixeds_recacc    
    OMP_accuracy[s] = OMP_fixeds_recacc    
    
print('')
print('---------------------------------------------------')
print('Recovery of Alphazero is: ' )
print(str(alphazero_accuracy))
print('Recovery of OMP is: ' )
print(str(OMP_accuracy))
print('----------------------------------------------------')
print('')

plt.plot(alphazero_accuracy)
plt.plot(OMP_accuracy)
plt.legend(['alphazero', 'OMP'], loc = 'upper left')
plt.title('alphazero iteration 0 vs OMP')
plt.ylabel('accuracy')
plt.xlabel('sparsity')
plt.savefig('alphazero_it0_vs_OMP.png')

            








